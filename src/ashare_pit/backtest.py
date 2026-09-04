#!/usr/bin/env python3
"""A 股滚动 Point-in-Time 回测引擎（默认股票池：沪深 300）。

每个再平衡日都会重新选股，不用当前成分倒推历史。财务数据只允许使用已超过
法定披露截止日的报告，价格序列截断到调仓日，估值取调仓日及以前的最新 TTM PE。
所有因子先做横截面标准化，再按配置权重合成。真实摩擦版本同时处理佣金、滑点、
印花税、涨停/停牌过滤、换手统计和可选的 AUM 市场冲击。

注意：法定截止日是保守可用日近似，不等于逐份公告的真实时间戳；当前全 A 名单
仍包含退市股票缺失造成的幸存者偏差。``--self-test`` 可用合成数据离线验证引擎。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from bisect import bisect_right
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import capacity as cn_capacity
from . import data_source as cn
from . import portfolio_opt as opt
from . import size as cn_size
from . import universe_csi500 as csi


def round2(value: float) -> float:
    return round(value, 2)


def window_metrics(dates: list[date], values: list[float]) -> dict[str, Any]:
    """Total/annualized return, volatility, Sharpe (rf=0), and max drawdown."""
    returns = [values[i] / values[i - 1] - 1 for i in range(1, len(values))]
    total = values[-1] / values[0] - 1
    years = max((dates[-1] - dates[0]).days / 365.25, 1e-9)
    cagr = (values[-1] / values[0]) ** (1 / years) - 1
    stdev = statistics.stdev(returns) if len(returns) > 2 else 0.0
    ann_vol = stdev * math.sqrt(252)
    sharpe = statistics.mean(returns) / stdev * math.sqrt(252) if stdev > 0 else 0.0
    peak = values[0]
    max_drawdown = 0.0
    for value in values:
        peak = max(peak, value)
        max_drawdown = max(max_drawdown, 1 - value / peak)
    return {
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "trading_days": len(dates),
        "total_return_percent": round2(total * 100),
        "annualized_return_percent": round2(cagr * 100),
        "annualized_volatility_percent": round2(ann_vol * 100),
        "sharpe_ratio_rf0": round2(sharpe),
        "max_drawdown_percent": round2(max_drawdown * 100),
    }

DEFAULT_FACTOR_WEIGHTS = {
    "growth": 0.20,
    "quality": 0.20,
    "value": 0.15,
    "momentum": 0.20,
    "low_volatility": 0.15,
    "liquidity": 0.10,
}

# 策略配置直接决定参与合成的因子，因此可安全加入成长加速度、小盘等扩展因子。
# `diagnostic_tilted` is data-driven from ashare_pit.factor_diagnostics (full-market run):
# low_volatility was the only industry-robust positive factor, small-cap (低流动性)
# strongly outperformed, while quality/momentum had negative IC — so it overweights
# low_vol + small_size and drops quality/momentum.
STRATEGY_PROFILES: dict[str, dict[str, float]] = {
    "balanced": dict(DEFAULT_FACTOR_WEIGHTS),
    "quality_value_lowvol": {
        "growth": 0.10, "quality": 0.25, "value": 0.25,
        "momentum": 0.10, "low_volatility": 0.20, "liquidity": 0.10,
    },
    "momentum_quality": {
        "growth": 0.15, "quality": 0.20, "value": 0.10,
        "momentum": 0.30, "low_volatility": 0.15, "liquidity": 0.10,
    },
    "growth_acceleration": {
        "growth": 0.15, "growth_acceleration": 0.25, "quality": 0.20,
        "value": 0.10, "momentum": 0.15, "low_volatility": 0.10, "liquidity": 0.05,
    },
    "diagnostic_tilted": {
        "low_volatility": 0.45, "small_size": 0.35, "value": 0.10, "growth": 0.10,
    },
    # Price/volume alphas that screened strong & industry-robust (max_ret IR 1.09,
    # reversal_20d 0.76). Deliberately excludes size/illiquidity so any OOS excess
    # over the equal-weight benchmark is orthogonal to its small-cap beta.
    "reversal_lottery": {
        "max_ret_20d": 0.35, "reversal_20d": 0.30, "volume_trend": 0.15,
        "reversal_5d": 0.10, "low_volatility": 0.10,
    },
    # 国泰君安短周期价量四因子等权复合 (§3.2). Meant to be run on the synthetic 中证500
    # pool with --construction optimize + --benchmark 000905 to reproduce the paper's
    # industry/style-neutral setup. Needs OHLCV, so pass --required to demand the
    # price/volume factors (not the fundamentals default).
    "gtja_pv": {
        "pv_divergence": 0.25, "opening_gap": 0.25,
        "abnormal_volume": 0.25, "amplitude_divergence": 0.25,
    },
    # GTJA191 full alpha set: equal-weighted composite of all available GTJA191 factors.
    # This is a placeholder — the actual factor list is passed via --gtja-factors CLI
    # and injected into raw_factors_at at runtime. The profile just reserves the name.
    # Use with --required "" --construction optimize --universe csi500_synth.
    "gtja191": {},
}


# 候选价量因子必须先通过因子诊断和 walk-forward，才可进入正式策略。
CANDIDATE_FACTORS = ["reversal_5d", "reversal_20d", "max_ret_20d", "illiquidity", "volume_trend"]

# 国泰君安短周期价量体系 4 个代表因子 (§3.2)。需 OHLCV,旧 close+amount 缓存下返回
# None,须 `--refresh` 重抓后才有值。与上面 5 个合并即完整候选池。
GTJA_FACTORS = ["pv_divergence", "opening_gap", "abnormal_volume", "amplitude_divergence"]
CANDIDATE_FACTORS = CANDIDATE_FACTORS + GTJA_FACTORS


def get_strategy_weights(name: str) -> dict[str, float]:
    if name not in STRATEGY_PROFILES:
        available = ", ".join(STRATEGY_PROFILES)
        raise ValueError(f"Unknown strategy '{name}'. Available: {available}")
    return STRATEGY_PROFILES[name]


Bar = dict[str, Any]  # {'date': 'YYYY-MM-DD', 'close': float, 'amount': float}


# ---------------------------------------------------------------------------
# 价格辅助函数：ISO 日期可按字符串排序，输入序列预先按日期升序排列
# ---------------------------------------------------------------------------
def _cut(bars: list[Bar], as_of: str) -> int:
    """Count of bars with date <= as_of (bars pre-sorted ascending). O(log n)."""
    return bisect_right(bars, as_of, key=lambda b: b["date"])


def rows_until(bars: list[Bar], as_of: str) -> list[Bar]:
    return bars[: _cut(bars, as_of)]


def close_asof(bars: list[Bar], as_of: str) -> float | None:
    i = _cut(bars, as_of)
    return bars[i - 1]["close"] if i > 0 else None


def _daily_returns(closes: list[float]) -> list[float]:
    return [closes[i] / closes[i - 1] - 1 for i in range(1, len(closes)) if closes[i - 1] > 0]


def momentum_12m_1m(bars: list[Bar], as_of: str) -> float | None:
    rows = rows_until(bars, as_of)
    if len(rows) < 252:
        return None
    base = rows[-252]["close"]
    recent = rows[-21]["close"]
    return recent / base - 1 if base > 0 else None


def trailing_vol(bars: list[Bar], as_of: str, lookback: int = 252) -> float | None:
    rows = rows_until(bars, as_of)
    if len(rows) < lookback:
        return None
    rets = _daily_returns([r["close"] for r in rows[-lookback:]])
    return statistics.stdev(rets) * math.sqrt(252) if len(rets) >= 60 else None


def avg_amount(bars: list[Bar], as_of: str, lookback: int = 63) -> float | None:
    rows = rows_until(bars, as_of)[-lookback:]
    amounts = [r["amount"] for r in rows if r.get("amount")]
    return statistics.mean(amounts) if amounts else None


# --- 候选价量因子：统一定向为数值越高、预期收益越高，最终方向由 IC 检验确认 ---
def reversal(bars: list[Bar], as_of: str, lookback: int) -> float | None:
    """−(trailing return over `lookback` days): short-term reversal (loser → bounce)."""
    rows = rows_until(bars, as_of)
    if len(rows) < lookback + 1:
        return None
    base = rows[-(lookback + 1)]["close"]
    return -(rows[-1]["close"] / base - 1) if base > 0 else None


def max_daily_return(bars: list[Bar], as_of: str, lookback: int = 20) -> float | None:
    """−max(daily return, `lookback`): lottery/MAX effect (high max → overpriced)."""
    rows = rows_until(bars, as_of)
    if len(rows) < lookback + 1:
        return None
    rets = _daily_returns([r["close"] for r in rows[-(lookback + 1):]])
    return -max(rets) if rets else None


def amihud_illiquidity(bars: list[Bar], as_of: str, lookback: int = 20) -> float | None:
    """Mean |ret|/成交额 (Amihud): higher illiquidity → higher expected return."""
    rows = rows_until(bars, as_of)[-(lookback + 1):]
    vals = []
    for prev, cur in zip(rows, rows[1:]):
        amt = cur.get("amount")
        if prev["close"] > 0 and amt and amt > 0:
            vals.append(abs(cur["close"] / prev["close"] - 1) / amt)
    return statistics.mean(vals) * 1e8 if vals else None


def volume_trend(bars: list[Bar], as_of: str, short: int = 5, long: int = 60) -> float | None:
    """−(recent 成交额 surge vs baseline): heavy recent volume often precedes fade."""
    amts = [r["amount"] for r in rows_until(bars, as_of) if r.get("amount")]
    if len(amts) < long:
        return None
    long_avg = statistics.mean(amts[-long:])
    return -(statistics.mean(amts[-short:]) / long_avg - 1) if long_avg > 0 else None


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation; None if <3 points or a series is constant."""
    n = len(xs)
    if n < 3:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sx, sy = statistics.pstdev(xs), statistics.pstdev(ys)
    if sx == 0 or sy == 0:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    return cov / (sx * sy)


# --- 国泰君安《基于短周期价量特征的多因子选股体系》4 个代表因子 (§3.2 举例1-4) ---
# 这组因子需要完整 OHLCV；旧缓存只有 close/amount 时返回 None，避免伪造输入。
def gtja_pv_divergence(bars: list[Bar], as_of: str, lookback: int = 6) -> float | None:
    """价量背离: −corr(close, volume) over `lookback` days (low corr → higher return).

    Report uses vwap; our hfq amount is the close×volume proxy so vwap≡close,
    hence close is the price series here."""
    rows = rows_until(bars, as_of)[-lookback:]
    if len(rows) < lookback:
        return None
    closes = [r.get("close") for r in rows]
    vols = [r.get("volume") for r in rows]
    if any(c is None for c in closes) or any(v is None for v in vols):
        return None
    c = _pearson(closes, vols)
    return -c if c is not None else None


def gtja_opening_gap(bars: list[Bar], as_of: str) -> float | None:
    """开盘缺口: open / prevClose − 1 (short-term momentum, higher gap → higher return)."""
    rows = rows_until(bars, as_of)
    if len(rows) < 2:
        return None
    op, prev_close = rows[-1].get("open"), rows[-2].get("close")
    if op is None or not prev_close or prev_close <= 0:
        return None
    return op / prev_close - 1


def gtja_abnormal_volume(bars: list[Bar], as_of: str, lookback: int = 20) -> float | None:
    """异常成交量: −volume / mean(volume, `lookback`) (volume spike → reversal)."""
    rows = rows_until(bars, as_of)[-lookback:]
    vols = [r.get("volume") for r in rows]
    if len(vols) < lookback or any(v is None for v in vols):
        return None
    avg = statistics.mean(vols)
    return -(vols[-1] / avg) if avg > 0 else None


def gtja_amplitude_divergence(bars: list[Bar], as_of: str, lookback: int = 6) -> float | None:
    """量幅背离: −corr(high/low, volume) over `lookback` days (low corr → higher return)."""
    rows = rows_until(bars, as_of)[-lookback:]
    if len(rows) < lookback:
        return None
    amps, vols = [], []
    for r in rows:
        hi, lo, v = r.get("high"), r.get("low"), r.get("volume")
        if hi is None or not lo or lo <= 0 or v is None:
            return None
        amps.append(hi / lo)
        vols.append(v)
    c = _pearson(amps, vols)
    return -c if c is not None else None


def pe_asof(pe_series: list[dict[str, Any]], as_of: str) -> float | None:
    valid = [p["pe_ttm"] for p in pe_series if p["date"] <= as_of and p.get("pe_ttm")]
    return valid[-1] if valid else None


# ---------------------------------------------------------------------------
# 横截面打分
# ---------------------------------------------------------------------------
def zscores(values: dict[str, float | None]) -> dict[str, float]:
    valid = [v for v in values.values() if v is not None]
    if len(valid) < 2:
        return {k: 0.0 for k in values}
    mean = statistics.mean(valid)
    try:
        sd = statistics.stdev(valid)
    except (statistics.StatisticsError, AttributeError):
        sd = 0.0
    if sd == 0 or not math.isfinite(sd):
        return {k: 0.0 for k in values}
    return {k: (v - mean) / sd if v is not None else 0.0 for k, v in values.items()}


def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def raw_factors_at(
    codes: list[str],
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    gtja_factors: list[str] | None = None,
) -> dict[str, dict[str, float | None]]:
    """Per-code raw factor values (pre-z-score), point-in-time as of `as_of`.

    When `gtja_factors` is a non-empty list, the named GTJA191 alphas are computed
    in batch via ashare_pit.gtja191 and merged into the per-code dict (None when unavailable)."""
    out: dict[str, dict[str, float | None]] = {}
    for code in codes:
        bars = prices.get(code) or []
        mom = momentum_12m_1m(bars, as_of)
        vol = trailing_vol(bars, as_of)
        adv = avg_amount(bars, as_of)
        available = sorted(
            (e for e in financials.get(code, []) if e["avail_date"] <= as_of),
            key=lambda e: e["period"],
        )
        cur_growth = _avg_growth(available[-1]["financials"]) if available else None
        prev_growth = _avg_growth(available[-2]["financials"]) if len(available) >= 2 else None
        fin = available[-1]["financials"] if available else {}
        pe_val = pe_asof(pe.get(code, []), as_of)
        liquidity = math.log10(adv) if adv and adv > 0 else None
        out[code] = {
            "growth": cur_growth,
            "growth_acceleration": (cur_growth - prev_growth)
            if (cur_growth is not None and prev_growth is not None) else None,
            "quality": _num(fin.get("roe_percent")),
            "value": (1.0 / pe_val) if (pe_val and pe_val > 0) else None,
            "momentum": mom,
            "low_volatility": (-vol if vol is not None else None),
            "liquidity": liquidity,
            "small_size": (-liquidity if liquidity is not None else None),
            "reversal_5d": reversal(bars, as_of, 5),
            "reversal_20d": reversal(bars, as_of, 20),
            "max_ret_20d": max_daily_return(bars, as_of, 20),
            "illiquidity": amihud_illiquidity(bars, as_of, 20),
            "volume_trend": volume_trend(bars, as_of, 5, 60),
            "pv_divergence": gtja_pv_divergence(bars, as_of, 6),
            "opening_gap": gtja_opening_gap(bars, as_of),
            "abnormal_volume": gtja_abnormal_volume(bars, as_of, 20),
            "amplitude_divergence": gtja_amplitude_divergence(bars, as_of, 6),
        }
    if gtja_factors:
        from . import gtja191
        gtja = gtja191.compute_factors(codes, as_of, prices, factor_names=gtja_factors, min_bars=20)
        for code in codes:
            if code in gtja:
                out[code].update(gtja[code])
    return out


def _avg_growth(fin: dict[str, Any]) -> float | None:
    """Mean of revenue/net-profit YoY growth for one report (None if neither)."""
    vals = [x for x in (_num(fin.get("revenue_yoy_percent")),
                        _num(fin.get("net_profit_yoy_percent"))) if x is not None]
    return statistics.mean(vals) if vals else None


def compute_composite(
    codes: list[str],
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    required: tuple[str, ...] = ("growth", "quality", "momentum"),
    gtja_factors: list[str] | None = None,
) -> dict[str, float]:
    """Composite factor z-score per eligible code (higher = better); {} if none.

    The single scoring primitive: `rank_candidates` sorts its output, and the
    constrained optimizer (ashare_pit.portfolio_opt) consumes it directly as the alpha."""
    raw = raw_factors_at(codes, as_of, prices, financials, pe, gtja_factors=gtja_factors)
    eligible = {c: f for c, f in raw.items() if all(f[k] is not None for k in required)}
    if not eligible:
        return {}
    z_by_factor = {
        factor: zscores({c: f[factor] for c, f in eligible.items()})
        for factor in weights
    }
    return {
        c: sum(w * z_by_factor[factor][c] for factor, w in weights.items())
        for c in eligible
    }


def rank_candidates(
    codes: list[str],
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    required: tuple[str, ...] = ("growth", "quality", "momentum"),
    gtja_factors: list[str] | None = None,
) -> list[str]:
    """Full cross-sectional ranking of eligible codes (best first)."""
    composite = compute_composite(codes, as_of, prices, financials, pe, weights, required,
                                 gtja_factors=gtja_factors)
    return sorted(composite, key=lambda c: composite[c], reverse=True)


def select_at(
    codes: list[str],
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    top_n: int,
    required: tuple[str, ...] = ("growth", "quality", "momentum"),
    gtja_factors: list[str] | None = None,
) -> list[str]:
    return rank_candidates(codes, as_of, prices, financials, pe, weights, required,
                          gtja_factors=gtja_factors)[:top_n]


# ---------------------------------------------------------------------------
# A 股交易摩擦
# ---------------------------------------------------------------------------
def _limit_threshold(code: str) -> float:
    """Daily price-limit as a return (leave a small margin below the exact cap)."""
    code = str(code).zfill(6)
    if code.startswith("688") or code.startswith("300"):
        return 0.195   # 科创板/创业板 ±20%
    return 0.095       # 主板 ±10%(未区分 ST ±5%)


def _stamp_bps(d: date) -> float:
    """印花税(卖出单边):2023-08-28 起由 10bp 降为 5bp。"""
    return 10.0 if d < date(2023, 8, 28) else 5.0


def buyable(code: str, bars: list[Bar], as_of: str, max_stale_days: int = 7) -> bool:
    """Can we actually buy this name at the rebalance close? (no 停牌 / 涨停)"""
    rows = rows_until(bars, as_of)
    if not rows:
        return False
    last = rows[-1]
    if (date.fromisoformat(as_of) - date.fromisoformat(last["date"])).days > max_stale_days:
        return False  # 停牌 / 数据过旧
    if len(rows) >= 2 and rows[-2]["close"] > 0:
        day_return = last["close"] / rows[-2]["close"] - 1
        if day_return >= _limit_threshold(code):
            return False  # 涨停,买不进
    return True


# ---------------------------------------------------------------------------
# 滚动回测模拟
# ---------------------------------------------------------------------------
def parse_rebalance_freq(freq: str) -> tuple[str, int | None]:
    """Parse rebalance frequency string -> (freq_type, N)."""
    if freq == "quarterly":
        return ("quarterly", None)
    if freq == "monthly":
        return ("monthly", None)
    if freq == "weekly":
        return ("weekly", None)
    if freq.endswith("d") and freq[:-1].isdigit():
        n = int(freq[:-1])
        if n < 1:
            raise ValueError(f"Rebalance step N must be >= 1, got {n}")
        return ("nd", n)
    raise ValueError(
        f"Unknown rebalance frequency '{freq}'. "
        "Supported: quarterly, monthly, weekly, or Nd (e.g. 2d, 5d)."
    )


def periods_per_year(freq: str) -> int:
    """Rebalance periods per calendar year for Sharpe annualization."""
    freq_type, n = parse_rebalance_freq(freq)
    if freq_type == "quarterly":
        return 4
    if freq_type == "monthly":
        return 12
    if freq_type == "weekly":
        return 52
    return max(252 // n, 1)  # nd


def rebalance_dates(calendar: list[date], freq: str = "quarterly") -> set[date]:
    """First trading day of each rebalance period in the calendar."""
    freq_type, n = parse_rebalance_freq(freq)
    dates: set[date] = set()
    if freq_type == "nd":
        return set(calendar[::n])
    seen: set = set()
    for d in calendar:
        if freq_type == "quarterly":
            key = (d.year, (d.month - 1) // 3)
        elif freq_type == "monthly":
            key = (d.year, d.month)
        else:  # weekly
            key = d.isocalendar()[:2]
        if key not in seen:
            seen.add(key)
            dates.add(d)
    return dates


def _optimize_rebalance(
    pool: list[str],
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    weights: dict[str, float],
    required: tuple[str, ...],
    snapshot: dict[str, dict[str, float]],
    neutralize: tuple[str, ...],
    weight_cap: float,
    ridge: float,
    limit_filter: bool,
    max_stale_days: int,
    old_w: dict[str, float],
    turnover_cap: float | None = None,
    gtja_factors: list[str] | None = None,
) -> dict[str, float]:
    """Optimizer target weights for one rebalance: composite alpha → industry/size-
    neutral, benchmark-relative (cap-weighted pool) constrained portfolio."""
    cand = [c for c in pool
            if (not limit_filter) or buyable(c, prices.get(c, []), as_of, max_stale_days)]
    comp = compute_composite(cand, as_of, prices, financials, pe, weights, required,
                            gtja_factors=gtja_factors)
    mcap: dict[str, float] = {}
    industry: dict[str, str] = {}
    for c in comp:
        mc = cn_size.market_cap_at(c, as_of, prices, financials, snapshot)
        if not mc:
            continue  # need mcap for both the benchmark weight and the size axis
        mcap[c] = mc
        pit = cn.pit_financials(financials.get(c, []), as_of)
        ind = pit["financials"].get("industry") if pit else None
        if ind:
            industry[c] = str(ind)
    names = list(mcap)
    if not names:
        return {}
    total = sum(mcap.values())
    bench_w = {c: mcap[c] / total for c in names}
    size = {c: math.log(mcap[c]) for c in names}
    alpha = {c: comp[c] for c in names}
    return opt.optimize_weights(
        names, alpha, bench_w, industry, size,
        w_cap=weight_cap, ridge=ridge, neutralize=neutralize, w_prev=old_w,
        turnover_cap=turnover_cap,
    )


def run_rolling_backtest(
    codes: list[str],
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
    benchmark: list[dict[str, Any]],
    start: date,
    end: date,
    weights: dict[str, float],
    top_n: int,
    commission_bps: float = 0.0,
    slippage_bps: float = 0.0,
    stamp_tax: bool = False,
    limit_filter: bool = False,
    max_stale_days: int = 7,
    return_curve: bool = False,
    rebalance_freq: str = "quarterly",
    universe_mode: str = "allA",
    construction: str = "equal",
    neutralize: tuple[str, ...] = ("industry", "size"),
    weight_cap: float = 0.03,
    ridge: float = 1.0,
    snapshot: dict[str, dict[str, float]] | None = None,
    universe_rows: list[dict[str, str]] | None = None,
    required: tuple[str, ...] = ("growth", "quality", "momentum"),
    csi500_drop_top: int = 300,
    csi500_take: int = 500,
    csi500_min_bars: int = 250,
    turnover_cap: float | None = None,
    gtja_factors: list[str] | None = None,
    portfolio_aum: float | None = None,
    impact_coefficient: float = 0.5,
) -> dict[str, Any]:
    """Rolling PIT backtest. Frictions off by default (set them for realism).

    Portfolio construction (aligns the GTJA reproduction with the paper):
      universe_mode  'allA' (current 全A) | 'csi500_synth' (synthetic PIT 中证500)
      construction   'equal' (rank→top-N→equal) | 'optimize' (industry/size-neutral QP)
    The optimize / csi500_synth paths need `snapshot` (cn.load_share_snapshot) for
    point-in-time market cap; `universe_rows` ([{code,name}]) supplies ST filtering."""
    if (universe_mode == "csi500_synth" or construction == "optimize") and not snapshot:
        raise ValueError(
            "universe_mode='csi500_synth' / construction='optimize' need a spot snapshot; "
            "pass snapshot=cn.load_share_snapshot() (run once with network, then cached)."
        )
    if portfolio_aum is not None and portfolio_aum <= 0:
        raise ValueError("portfolio_aum must be positive when capacity analysis is enabled")
    if impact_coefficient < 0:
        raise ValueError("impact_coefficient must be non-negative")
    bench_in_window = [b for b in benchmark if start.isoformat() <= b["date"] <= end.isoformat()]
    calendar = [date.fromisoformat(b["date"]) for b in bench_in_window]
    if len(calendar) < 40:
        raise ValueError("Not enough benchmark trading days in the window.")
    rebals = rebalance_dates(calendar, freq=rebalance_freq)
    one_way = (commission_bps + slippage_bps) / 10000.0

    value = 1.0
    holdings: dict[str, float] = {}  # code -> shares
    selections: list[dict[str, Any]] = []
    total_cost_drag = 0.0
    total_explicit_cost_drag = 0.0
    total_market_impact_drag = 0.0
    skipped_untradable = 0
    turnover_observations: list[dict[str, float]] = []
    capacity_observations: list[dict[str, Any]] = []
    values: list[float] = []
    dates: list[date] = []
    name_rows = universe_rows or [{"code": c, "name": ""} for c in codes]
    cur_pool: list[str] | None = None   # synthetic CSI500, refreshed on index cadence
    cur_pool_key: Any = None

    for d in calendar:
        iso = d.isoformat()
        # Only holdings need daily marking; candidate prices are fetched on
        # rebalance days below. Marking the full universe every day was O(days ×
        # universe × bars) and dominated runtime.
        price_now = {c: close_asof(prices.get(c, []), iso) for c in holdings}
        if holdings:
            mark = sum(sh * (price_now.get(c) or 0.0) for c, sh in holdings.items())
            if mark > 0:
                value = mark
        if d in rebals and value > 0:
            # 1) Universe for this rebalance (synthetic CSI500 refreshed semiannually).
            if universe_mode == "csi500_synth":
                key = (d.year, d.month > 6)
                if cur_pool is None or key != cur_pool_key:
                    cur_pool = csi.build_synthetic_csi500(
                        iso, prices, financials, snapshot, name_rows,
                        drop_top=csi500_drop_top, take=csi500_take, min_bars=csi500_min_bars)
                    cur_pool_key = key
                pool = cur_pool
            else:
                pool = codes
            old_w = {c: (sh * (price_now.get(c) or 0.0)) / value for c, sh in holdings.items()}
            # 2) Target weights: constrained optimize, or rank→top-N→equal.
            if construction == "optimize":
                new_w = _optimize_rebalance(
                    pool, iso, prices, financials, pe, weights, required, snapshot,
                    neutralize, weight_cap, ridge, limit_filter, max_stale_days, old_w,
                    turnover_cap, gtja_factors=gtja_factors)
                selected = list(new_w)
            else:
                ranked = rank_candidates(pool, iso, prices, financials, pe, weights, required,
                                        gtja_factors=gtja_factors)
                selected = []
                for c in ranked:
                    if limit_filter and not buyable(c, prices.get(c, []), iso, max_stale_days):
                        skipped_untradable += 1
                        continue
                    selected.append(c)
                    if len(selected) >= top_n:
                        break
                new_w = {c: 1.0 / len(selected) for c in selected} if selected else {}
            for c in selected:
                if c not in price_now:
                    price_now[c] = close_asof(prices.get(c, []), iso)
            # 3) Turnover-based cost, then re-hold at target weights (equal path is
            #    bit-identical to the old per-name/len formula since w = 1/len).
            if new_w:
                names = set(old_w) | set(new_w)
                delta_weights = {c: new_w.get(c, 0.0) - old_w.get(c, 0.0) for c in names}
                sell_turn = sum(max(old_w.get(c, 0) - new_w.get(c, 0), 0) for c in names)
                buy_turn = sum(max(new_w.get(c, 0) - old_w.get(c, 0), 0) for c in names)
                gross_turn = buy_turn + sell_turn
                turnover_observations.append({
                    "gross": gross_turn,
                    "one_way": gross_turn / 2.0,
                    "buy": buy_turn,
                    "sell": sell_turn,
                })
                stamp = (_stamp_bps(d) / 10000.0) if stamp_tax else 0.0
                explicit_cost = buy_turn * one_way + sell_turn * (one_way + stamp)
                market_impact = 0.0
                capacity = None
                if portfolio_aum and portfolio_aum > 0:
                    adv = {c: avg_amount(prices.get(c, []), iso, 63) for c in names}
                    volatility = {c: trailing_vol(prices.get(c, []), iso, 63) for c in names}
                    capacity = cn_capacity.estimate_rebalance_impact(
                        delta_weights,
                        adv,
                        volatility,
                        portfolio_aum * value,
                        impact_coefficient,
                    )
                    market_impact = capacity["estimated_nav_impact_fraction"]
                    capacity_observations.append(capacity)
                cost = explicit_cost + market_impact
                if cost >= 1.0:
                    raise ValueError(
                        f"Estimated rebalance cost is {cost:.2%} on {iso}; "
                        "reduce AUM or recalibrate the impact coefficient."
                    )
                total_explicit_cost_drag += explicit_cost * value
                total_market_impact_drag += market_impact * value
                total_cost_drag += cost * value
                value *= (1 - cost)
                holdings = {c: (value * w) / px for c, w in new_w.items()
                            if (px := price_now.get(c)) and px > 0}
                selection = {
                    "date": iso,
                    "selected": selected,
                    "gross_turnover": round(gross_turn, 6),
                    "one_way_turnover": round(gross_turn / 2.0, 6),
                }
                if capacity:
                    selection["capacity"] = {
                        "estimated_nav_impact_bps": capacity["estimated_nav_impact_bps"],
                        "liquidity_coverage": capacity["liquidity_coverage"],
                        "participation_rate": capacity["participation_rate"],
                    }
                selections.append(selection)
        values.append(value)
        dates.append(d)

    port = window_metrics(dates, values)
    base = bench_in_window[0]["close"]
    bench_vals = [b["close"] / base for b in bench_in_window]
    bench_metrics = window_metrics(
        [date.fromisoformat(b["date"]) for b in bench_in_window], bench_vals
    )
    active = round(
        port["annualized_return_percent"] - bench_metrics["annualized_return_percent"], 2
    )
    one_way_turnovers = [x["one_way"] for x in turnover_observations]
    gross_turnovers = [x["gross"] for x in turnover_observations]
    turnover_summary = {
        "definition": "one_way = 0.5 * sum(abs(target_weight - prior_weight))",
        "observations": len(turnover_observations),
        "mean_one_way": round(statistics.mean(one_way_turnovers), 6) if one_way_turnovers else None,
        "median_one_way": round(statistics.median(one_way_turnovers), 6) if one_way_turnovers else None,
        "max_one_way": round(max(one_way_turnovers), 6) if one_way_turnovers else None,
        "mean_gross": round(statistics.mean(gross_turnovers), 6) if gross_turnovers else None,
    }
    capacity_summary = None
    if capacity_observations:
        p95_values = [x["participation_rate"]["p95"] for x in capacity_observations
                      if x["participation_rate"]["p95"] is not None]
        max_values = [x["participation_rate"]["max"] for x in capacity_observations
                      if x["participation_rate"]["max"] is not None]
        coverages = [x["liquidity_coverage"] for x in capacity_observations]
        capacity_summary = {
            "model": "square_root_impact",
            "initial_aum": portfolio_aum,
            "impact_coefficient": impact_coefficient,
            "mean_liquidity_coverage": round(statistics.mean(coverages), 6),
            "mean_p95_participation_rate": round(statistics.mean(p95_values), 6) if p95_values else None,
            "max_participation_rate": round(max(max_values), 6) if max_values else None,
            "cumulative_market_impact_drag_fraction": round(total_market_impact_drag, 6),
        }
    result = {
        "frictions": {
            "commission_bps_per_side": commission_bps,
            "slippage_bps_per_side": slippage_bps,
            "stamp_tax": stamp_tax,
            "limit_up_suspension_filter": limit_filter,
            "market_impact_model": "square_root" if portfolio_aum else None,
        },
        "rebalances": len(selections),
        "buys_skipped_untradable": skipped_untradable,
        "total_cost_drag_fraction": round(total_cost_drag, 4),
        "total_explicit_cost_drag_fraction": round(total_explicit_cost_drag, 6),
        "total_market_impact_drag_fraction": round(total_market_impact_drag, 6),
        "turnover": turnover_summary,
        "capacity": capacity_summary,
        "rebalance_diagnostics": [
            {
                "date": selection["date"],
                "gross_turnover": selection["gross_turnover"],
                "one_way_turnover": selection["one_way_turnover"],
                **({"capacity": selection["capacity"]} if "capacity" in selection else {}),
            }
            for selection in selections
        ],
        "portfolio": port,
        "benchmark": bench_metrics,
        "active_annualized_return_percent": active,
        "latest_selection": selections[-1] if selections else None,
    }
    if return_curve:
        result["equity_curve"] = [[d.isoformat(), round(v, 6)] for d, v in zip(dates, values)]
        result["benchmark_curve"] = [[b["date"], round(v, 6)] for b, v in zip(bench_in_window, bench_vals)]
    return result


# ---------------------------------------------------------------------------
# 数据读取与完整运行
# ---------------------------------------------------------------------------
def load_all_data(
    index_code: str,
    price_start: str,
    cache_dir: str | None,
    refresh: bool,
    fin_start_year: int,
    fin_end_year: int,
    full_market: bool = False,
    skip_pe: bool = False,
    benchmark_code: str = "880008",
) -> dict[str, Any]:
    # Offline fast path (no --refresh): read the whole cache in one shot via
    # load_market_snapshot, which globs prices by widest cached series and so never
    # rebuilds the end=today key that silently cache-misses and re-downloads all 5527
    # codes. No network, no per-stock sleep(0.3). Falls through to the fetch loop only
    # when prices aren't cached at all (cold cache / first build).
    if not refresh:
        snap = cn.load_market_snapshot(cache_dir, with_prices=True, with_pe=not skip_pe,
                                        benchmark_code=benchmark_code)
        if full_market:
            codes = [row["code"] for row in snap["universe"]]
        else:
            codes = cn.load_index_constituents(index_code, cache_dir, refresh)
        prices = {c: snap["prices"][c] for c in codes if snap["prices"].get(c)}
        if prices:
            pe = {} if skip_pe else {c: snap["pe"][c] for c in codes if snap["pe"].get(c)}
            failed = [c for c in codes if c not in prices]
            print(f"  [snapshot] offline load: {len(prices)}/{len(codes)} codes with prices, "
                  f"{len(failed)} missing (no network)", flush=True)
            return {"codes": codes, "prices": prices, "pe": pe,
                    "financials": snap["financials"], "benchmark": snap["benchmark"],
                    "failed": failed, "universe_rows": snap["universe"]}
    if full_market:
        uni_rows = cn.load_universe(cache_dir, refresh)
        codes = [row["code"] for row in uni_rows]
    else:
        codes = cn.load_index_constituents(index_code, cache_dir, refresh)
        uni_rows = [{"code": c, "name": ""} for c in codes]
    periods = cn.report_periods(fin_start_year, fin_end_year)
    financials = cn.build_financial_timeline(periods, cache_dir, refresh, codes=set(codes))
    prices: dict[str, list[Bar]] = {}
    pe: dict[str, list[dict[str, Any]]] = {}
    price_failed: list[str] = []
    total = len(codes)
    for i, code in enumerate(codes):
        try:
            prices[code] = cn.load_hfq_prices(code, start=price_start, cache_dir=cache_dir, refresh=refresh)
        except Exception:
            price_failed.append(code)
        if not skip_pe:
            try:
                pe[code] = cn.load_pe_ttm(code, cache_dir=cache_dir, refresh=refresh)
            except Exception:
                pass  # PE optional (value factor); a miss must not drop a good price
        time.sleep(0.3)  # politeness: avoid Sina/Baidu rate limiting on many sequential calls
        if (i + 1) % 100 == 0 or i + 1 == total:
            print(f"  data pull {i + 1}/{total} (price_failed={len(price_failed)})", flush=True)
    benchmark = cn.load_benchmark(benchmark_code, cache_dir, refresh)
    return {"codes": codes, "prices": prices, "pe": pe, "financials": financials,
            "benchmark": benchmark, "failed": price_failed, "universe_rows": uni_rows}


def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.data_cache_dir or None
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    data = load_all_data(
        args.index, args.price_start, cache_dir, args.refresh, args.fin_start_year, end.year,
        full_market=args.full_market, skip_pe=args.skip_pe, benchmark_code=args.benchmark,
    )
    usable = [c for c in data["codes"] if data["prices"].get(c)]
    weights = get_strategy_weights(args.strategy)
    neutralize = tuple(x for x in (args.neutralize or "").split(",") if x.strip())
    required = tuple(x.strip() for x in args.required.split(",") if x.strip())
    gtja_factors = None
    if args.strategy == "gtja191":
        from . import gtja191
        if args.gtja_factors:
            gtja_factors = [f.strip() for f in args.gtja_factors.split(",")]
        else:
            gtja_factors = gtja191.GTJA191_NAMES
        weights = {f: 1.0 / len(gtja_factors) for f in gtja_factors}
        if args.gtja_ic_sign:
            with open(args.gtja_ic_sign) as f_ic:
                ic_data = json.load(f_ic)
            ic_map = {item["factor"]: item["mean_ic"] for item in ic_data["all_factors"]}
            n = len(gtja_factors)
            weights = {f: (1.0 if ic_map.get(f, 0) > 0 else -1.0) / n for f in gtja_factors}
    snapshot = None
    if args.universe == "csi500_synth" or args.construction == "optimize":
        snapshot = cn.load_share_snapshot(cache_dir, refresh=args.refresh_shares)
    # Benchmark = 通达信 880008 全A等权指数 (real index, no survivorship bias); its
    # daily dates also serve as the trading calendar.
    common = dict(
        codes=usable, prices=data["prices"], financials=data["financials"], pe=data["pe"],
        benchmark=data["benchmark"], start=start, end=end,
        weights=weights, top_n=args.top_n, rebalance_freq=args.rebalance_freq,
        universe_mode=args.universe, construction=args.construction, neutralize=neutralize,
        weight_cap=args.weight_cap, ridge=args.ridge, snapshot=snapshot,
        universe_rows=data.get("universe_rows"), required=required,
        csi500_min_bars=args.csi500_min_bars, turnover_cap=args.turnover_cap,
        gtja_factors=gtja_factors,
    )
    frictionless = run_rolling_backtest(**common)
    realistic = run_rolling_backtest(
        **common,
        commission_bps=args.commission_bps,
        slippage_bps=args.slippage_bps,
        stamp_tax=True,
        limit_filter=True,
        max_stale_days=args.max_stale_days,
        portfolio_aum=(args.portfolio_aum_millions * 1_000_000
                       if args.portfolio_aum_millions else None),
        impact_coefficient=args.impact_coefficient,
    )
    decay = round(
        frictionless["portfolio"]["annualized_return_percent"]
        - realistic["portfolio"]["annualized_return_percent"],
        2,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "framework": "a_share_rolling_point_in_time",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "universe": {"source": args.universe if args.universe != "allA"
                     else ("full_market" if args.full_market else args.index),
                     "benchmark": args.benchmark,
                     "constituents": len(data["codes"]),
                     "usable": len(usable), "price_fetch_failed": len(data["failed"])},
        "config": {"top_n": args.top_n, "strategy": args.strategy, "factor_weights": weights,
                   "construction": args.construction,
                   "neutralize": neutralize if args.construction == "optimize" else None,
                   "weight_cap": args.weight_cap if args.construction == "optimize" else None,
                   "ridge": args.ridge if args.construction == "optimize" else None,
                   "turnover_cap": args.turnover_cap if args.construction == "optimize" else None,
                   "portfolio_aum_millions": args.portfolio_aum_millions,
                   "impact_coefficient": args.impact_coefficient if args.portfolio_aum_millions else None,
                   "rebalance_freq": args.rebalance_freq, "required": required},
        "frictionless": frictionless,
        "realistic": realistic,
        "annualized_return_decay_percent": decay,
        "note": (
            "PIT: fundamentals gated by statutory disclosure deadline, prices truncated at "
            "each rebalance. Realistic run adds commission+slippage, sell-side stamp tax "
            "(10bp→5bp from 2023-08-28), and skips limit-up/suspended names at rebalance. "
            "When --portfolio-aum-millions is set, it also adds a volatility-scaled square-root "
            "market-impact estimate and reports ADV participation. "
            "Benchmark dates serve as the trading calendar. "
            "When benchmark=880008: real real-time-edited index incl. delisted names "
            "→ no benchmark-side survivorship bias; strategy universe is the current 全A "
            "list (不含退市股) → asymmetry / survivorship tailwind."
        ),
    }


# ---------------------------------------------------------------------------
# 离线自检（合成数据，不访问网络）
# ---------------------------------------------------------------------------
def _synth_bars(start: date, days: int, drift: float, seed: int) -> list[Bar]:
    import random
    rng = random.Random(seed)
    bars: list[Bar] = []
    px = 100.0
    d = start
    made = 0
    while made < days:
        if d.weekday() < 5:  # weekday
            px *= (1 + drift + rng.gauss(0, 0.012))
            bars.append({"date": d.isoformat(), "close": round(px, 3),
                         "amount": 1e8 * (1 + rng.random())})
            made += 1
        d = date.fromordinal(d.toordinal() + 1)
    return bars


def self_test() -> None:
    # Winners (high drift) vs losers (flat). PIT selection should favour winners.
    codes = [f"C{i:02d}" for i in range(10)]
    prices = {}
    for i, c in enumerate(codes):
        drift = 0.0012 if i < 5 else 0.0000  # first 5 are winners
        prices[c] = _synth_bars(date(2019, 1, 1), 900, drift, seed=i + 1)
    # Financials: winners have higher growth/ROE, available from 2020 annual (avail 2021-04-30).
    financials = {}
    for i, c in enumerate(codes):
        g = 30.0 if i < 5 else 2.0
        roe = 20.0 if i < 5 else 5.0
        financials[c] = [
            {"period": "20191231", "avail_date": "2020-04-30",
             "financials": {"revenue_yoy_percent": g, "net_profit_yoy_percent": g, "roe_percent": roe}},
            {"period": "20201231", "avail_date": "2021-04-30",
             "financials": {"revenue_yoy_percent": g, "net_profit_yoy_percent": g, "roe_percent": roe}},
        ]
    pe = {c: [{"date": "2019-01-01", "pe_ttm": 15.0 + i}] for i, c in enumerate(codes)}

    # 1) PIT no-lookahead: a report is unusable before its avail_date.
    entries = financials["C00"]
    assert cn.pit_financials(entries, "2020-04-15") is None, "年报截止前不可用"
    assert cn.pit_financials(entries, "2020-05-15")["period"] == "20191231"
    print("[PASS] PIT 财务门控:年报截止前不可用,截止后可用")

    # 2) Selection favours winners once price history + financials exist.
    as_of = "2021-06-30"
    selected = select_at(codes, as_of, prices, financials, pe, DEFAULT_FACTOR_WEIGHTS, top_n=5)
    winners = {f"C{i:02d}" for i in range(5)}
    overlap = len(set(selected) & winners)
    assert overlap >= 4, f"选股应偏向赢家,实际重合 {overlap}/5: {selected}"
    print(f"[PASS] 横截面选股偏向高成长/高动量赢家(重合 {overlap}/5)")

    # 3) Rolling backtest runs end to end and beats a flat benchmark.
    bench = _synth_bars(date(2019, 1, 1), 900, 0.0002, seed=999)
    bench = [{"date": b["date"], "close": b["close"]} for b in bench]
    common = dict(codes=codes, prices=prices, financials=financials, pe=pe, benchmark=bench,
                  start=date(2021, 1, 1), end=date(2022, 6, 30),
                  weights=DEFAULT_FACTOR_WEIGHTS, top_n=5)
    frictionless = run_rolling_backtest(**common)
    assert frictionless["rebalances"] >= 4, "应有多次季度再平衡"
    assert frictionless["portfolio"]["annualized_return_percent"] > frictionless["benchmark"]["annualized_return_percent"]
    print(f"[PASS] 滚动回测跑通:{frictionless['rebalances']} 次再平衡,"
          f"组合年化 {frictionless['portfolio']['annualized_return_percent']}% > "
          f"基准 {frictionless['benchmark']['annualized_return_percent']}%")

    # 4) Frictions reduce return (cost drag).
    realistic = run_rolling_backtest(**common, commission_bps=2.5, slippage_bps=10.0,
                                     stamp_tax=True, limit_filter=True)
    assert realistic["portfolio"]["annualized_return_percent"] <= frictionless["portfolio"]["annualized_return_percent"]
    assert realistic["total_cost_drag_fraction"] > 0
    print(f"[PASS] 摩擦拖累收益:无摩擦 {frictionless['portfolio']['annualized_return_percent']}% "
          f"-> 真实 {realistic['portfolio']['annualized_return_percent']}% "
          f"(累计成本拖累 {realistic['total_cost_drag_fraction']})")

    capacity_small = run_rolling_backtest(
        **common, commission_bps=2.5, slippage_bps=10.0, stamp_tax=True,
        limit_filter=True, portfolio_aum=10_000_000, impact_coefficient=0.5,
    )
    capacity_large = run_rolling_backtest(
        **common, commission_bps=2.5, slippage_bps=10.0, stamp_tax=True,
        limit_filter=True, portfolio_aum=100_000_000, impact_coefficient=0.5,
    )
    assert capacity_large["total_market_impact_drag_fraction"] > capacity_small["total_market_impact_drag_fraction"]
    assert capacity_large["portfolio"]["annualized_return_percent"] < capacity_small["portfolio"]["annualized_return_percent"]
    assert capacity_large["turnover"]["observations"] == capacity_large["rebalances"]
    print("[PASS] 容量压力:更高 AUM 产生更高 ADV 参与率/市场冲击,并降低净收益")

    # 4b) Nd rebalance frequency produces more rebalances than quarterly.
    nd2 = run_rolling_backtest(**common, rebalance_freq="2d")
    assert nd2["rebalances"] > frictionless["rebalances"], (
        f"2d ({nd2['rebalances']}) should rebalance more than quarterly ({frictionless['rebalances']})"
    )
    print(f"[PASS] 2d 再平衡次数 {nd2['rebalances']} >> 季度 {frictionless['rebalances']}")

    # 5) Strategy profiles: weights ~1, unknown errors, balanced == old default.
    for name, w in STRATEGY_PROFILES.items():
        if name == "gtja191":
            continue  # weights set dynamically at runtime
        assert abs(sum(w.values()) - 1.0) < 1e-9, f"策略 {name} 权重和应≈1: {sum(w.values())}"
    assert STRATEGY_PROFILES["balanced"] == DEFAULT_FACTOR_WEIGHTS, "balanced 应等于旧默认权重"
    try:
        get_strategy_weights("nope")
    except ValueError as exc:
        assert "Available" in str(exc)
    else:
        raise AssertionError("未知策略应报错")
    print(f"[PASS] {len(STRATEGY_PROFILES)} 套策略权重均≈1;未知策略报错;balanced 等于旧默认")

    # 6) growth_acceleration computed when two reports are available.
    accel_fin = {"X": [
        {"period": "20191231", "avail_date": "2020-04-30",
         "financials": {"revenue_yoy_percent": 10.0, "net_profit_yoy_percent": 10.0}},
        {"period": "20201231", "avail_date": "2021-04-30",
         "financials": {"revenue_yoy_percent": 30.0, "net_profit_yoy_percent": 30.0}},
    ]}
    accel = raw_factors_at(["X"], "2021-06-30", {"X": prices["C00"]}, accel_fin, {})["X"]
    assert accel["growth"] == 30.0 and accel["growth_acceleration"] == 20.0, accel
    only_one = raw_factors_at(["X"], "2020-06-30", {"X": prices["C00"]}, accel_fin, {})["X"]
    assert only_one["growth"] == 10.0 and only_one["growth_acceleration"] is None, only_one
    assert accel["small_size"] == -accel["liquidity"], "small_size 应为 −liquidity"
    print(f"[PASS] growth_acceleration:两期财报算出 {accel['growth_acceleration']}(单期为 None);small_size=−liquidity")

    # 7) candidate price/volume alphas: PIT + orientation.
    up = [{"date": (date(2021, 1, 1) + timedelta(days=i)).isoformat(),
           "close": 100.0 + i, "amount": 1e9} for i in range(25)]
    cand = raw_factors_at(["U"], up[-1]["date"], {"U": up}, {}, {})["U"]
    assert cand["reversal_5d"] is not None and cand["reversal_5d"] < 0, "上涨序列 → reversal 应为负"
    assert cand["reversal_20d"] is not None and cand["max_ret_20d"] is not None, cand
    assert cand["illiquidity"] is not None and cand["volume_trend"] is None, "volume_trend 需 60 日, 25 日应为 None"
    assert all(f in cand for f in CANDIDATE_FACTORS), "候选因子应全部出现在 raw_factors_at"
    # 国君 4 因子在旧 close+amount 缓存下应优雅返回 None(缺 OHLCV)。
    assert all(cand[f] is None for f in GTJA_FACTORS), "缺 OHLCV 时国君 4 因子应为 None"
    print(f"[PASS] 候选因子:9 个价量 alpha 已接入,PIT 定向正确(reversal 上涨为负 {cand['reversal_5d']:.4f})")

    # 8) 国君 4 因子:给足 OHLCV 后应算出且定向正确。close↑与 volume↑同向、
    #    振幅(high/low)↑与 volume↑同向 → 两个背离因子(−corr)为负;末日放量、跳空高开。
    ohlcv = []
    for i in range(22):
        c = 100.0 + i
        w = 1.0 + 0.01 * (i + 1)
        ohlcv.append({"date": (date(2021, 2, 1) + timedelta(days=i)).isoformat(),
                      "open": c, "high": c * w, "low": c / w, "close": c,
                      "volume": 1000.0 * (i + 1), "amount": c * 1000.0 * (i + 1)})
    g = raw_factors_at(["G"], ohlcv[-1]["date"], {"G": ohlcv}, {}, {})["G"]
    assert g["opening_gap"] is not None and g["opening_gap"] > 0, "跳空高开 → 缺口为正"
    assert g["abnormal_volume"] is not None and g["abnormal_volume"] < 0, "末日放量 → 异常成交量为负"
    assert g["pv_divergence"] is not None and g["pv_divergence"] < 0, "价量同向 → 价量背离为负"
    assert g["amplitude_divergence"] is not None and g["amplitude_divergence"] < 0, "量幅同向 → 量幅背离为负"
    print(f"[PASS] 国君 4 因子:补齐 OHLCV 后正确计算(量幅背离 {g['amplitude_divergence']:.3f})")

    # 9) csi500_synth pool + constrained optimize construction (paper alignment).
    ocodes = [f"O{i:02d}" for i in range(12)]
    oprices = {c: _synth_bars(date(2019, 1, 1), 900, 0.0004 * (i % 3), seed=100 + i)
               for i, c in enumerate(ocodes)}
    ofin = {c: [{"period": "20201231", "avail_date": "2021-04-30",
                 "financials": {"revenue_yoy_percent": 8.0 + i, "net_profit_yoy_percent": 8.0 + i,
                                "roe_percent": 6.0 + i, "net_profit": 1.0e8 * (i + 1), "eps": 1.0,
                                "industry": ("甲" if i % 2 == 0 else "乙")}}]
            for i, c in enumerate(ocodes)}
    osnap = {c: {"price": 10.0 * (i + 1), "mktcap": None} for i, c in enumerate(ocodes)}
    orows = [{"code": c, "name": ""} for c in ocodes]
    obench = [{"date": b["date"], "close": b["close"]}
              for b in _synth_bars(date(2019, 1, 1), 900, 0.0002, seed=777)]

    # Missing snapshot must raise, not silently mis-price.
    try:
        run_rolling_backtest(codes=ocodes, prices=oprices, financials=ofin, pe={}, benchmark=obench,
                             start=date(2021, 1, 1), end=date(2022, 6, 30),
                             weights=DEFAULT_FACTOR_WEIGHTS, top_n=8, construction="optimize")
    except ValueError as exc:
        assert "snapshot" in str(exc)
    else:
        raise AssertionError("optimize 缺 snapshot 应报错")

    ocommon = dict(codes=ocodes, prices=oprices, financials=ofin, pe={}, benchmark=obench,
                   start=date(2021, 1, 1), end=date(2022, 6, 30), weights=DEFAULT_FACTOR_WEIGHTS,
                   top_n=8, universe_mode="csi500_synth", construction="optimize",
                   neutralize=("industry", "size"), weight_cap=0.5, ridge=1.0, snapshot=osnap,
                   universe_rows=orows, csi500_drop_top=2, csi500_take=8)
    res_opt = run_rolling_backtest(**ocommon)
    assert res_opt["rebalances"] >= 2, f"应有多次再平衡: {res_opt['rebalances']}"
    assert "annualized_return_percent" in res_opt["portfolio"]

    # One rebalance solved directly: Σw≈1, industry & size active exposure ≈ 0.
    as_of = "2021-06-30"
    pool = csi.build_synthetic_csi500(as_of, oprices, ofin, osnap, orows, drop_top=2, take=8)
    nw = _optimize_rebalance(pool, as_of, oprices, ofin, {}, DEFAULT_FACTOR_WEIGHTS,
                             ("growth", "quality", "momentum"), osnap, ("industry", "size"),
                             0.5, 1.0, False, 7, {})
    mcap = {c: cn_size.market_cap_at(c, as_of, oprices, ofin, osnap) for c in pool}
    tot = sum(mcap.values())
    bench_w = {c: mcap[c] / tot for c in pool}
    industry = {c: ("甲" if int(c[1:]) % 2 == 0 else "乙") for c in pool}
    size = {c: math.log(mcap[c]) for c in pool}
    exp = opt.active_exposures(nw, bench_w, industry, size)
    assert abs(exp["sum_w"] - 1.0) < 1e-5, exp
    assert exp["max_abs_industry_active"] < 1e-3, f"行业中性: {exp}"
    assert abs(exp["size_active_exposure"]) < 1e-3, f"规模中性: {exp}"
    print(f"[PASS] csi500_synth+optimize:{res_opt['rebalances']} 次再平衡跑通;"
          f"单期行业残差 {exp['max_abs_industry_active']:.1e}、规模主动暴露 {exp['size_active_exposure']:.1e}≈0")
    print("ALL PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share rolling point-in-time backtest.")
    parser.add_argument("--self-test", action="store_true", help="Offline synthetic-data test.")
    parser.add_argument("--index", default="000300", help="Universe index code (default 沪深300).")
    parser.add_argument("--benchmark", default="880008",
                        help="Benchmark index code: 880008, 000300, 000905, etc. (default 880008 全A等权).")
    parser.add_argument("--full-market", action="store_true",
                        help="Use the full A-share list (~5500) as the universe instead of --index members.")
    parser.add_argument("--skip-pe", action="store_true",
                        help="Skip per-stock PE fetch (halves requests; value factor degrades to neutral).")
    parser.add_argument("--strategy", default="balanced",
                        help=f"Factor weight profile. Available: {', '.join(STRATEGY_PROFILES)}.")
    parser.add_argument("--top-n", type=int, default=30)
    parser.add_argument("--start", default="2021-01-01", help="Backtest start (PE history begins ~2021).")
    parser.add_argument("--end", default=None, help="Backtest end (default today).")
    parser.add_argument("--price-start", default="20180101", help="hfq price fetch start (need >=1y before backtest).")
    parser.add_argument("--fin-start-year", type=int, default=2018)
    parser.add_argument("--commission-bps", type=float, default=2.5, help="佣金/边(realistic run).")
    parser.add_argument("--slippage-bps", type=float, default=10.0, help="滑点/边(realistic run).")
    parser.add_argument("--max-stale-days", type=int, default=7, help="超过此天数无行情视为停牌,不可买.")
    parser.add_argument("--rebalance-freq", default="quarterly",
                        help="Rebalance frequency: quarterly, monthly, weekly, or Nd (e.g. 2d, 5d).")
    parser.add_argument("--universe", default="allA", choices=["allA", "csi500_synth"],
                        help="Stock pool: allA (current 全A) or csi500_synth (synthetic PIT 中证500).")
    parser.add_argument("--construction", default="equal", choices=["equal", "optimize"],
                        help="Portfolio construction: equal (rank→top-N→等权) or optimize "
                             "(industry/style-neutral constrained QP, aligns with the paper).")
    parser.add_argument("--neutralize", default="industry,size",
                        help="Comma list of neutral axes for --construction optimize: industry,size[,...].")
    parser.add_argument("--weight-cap", type=float, default=0.03,
                        help="Max single-name weight for --construction optimize (auto-raised to stay feasible).")
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="L2 ridge in the optimizer objective (spreads weight, avoids corner solutions).")
    parser.add_argument("--turnover-cap", type=float, default=None,
                        help="Per-rebalance turnover cap Σ|w−w_prev| for --construction optimize. "
                             "Set it → paper's LP (max expected return, hard turnover limit); this is "
                             "what keeps T+2 rebalancing viable after A-share frictions.")
    parser.add_argument("--portfolio-aum-millions", type=float, default=None,
                        help="Optional initial AUM in CNY millions. Enables ADV participation and "
                             "square-root market-impact estimates in the realistic run.")
    parser.add_argument("--impact-coefficient", type=float, default=0.5,
                        help="Square-root impact coefficient (default 0.5).")
    parser.add_argument("--csi500-min-bars", type=int, default=250,
                        help="Min bar history for csi500_synth eligibility (次新 filter; lower to start "
                             "earlier when price data has little runway, e.g. 120 for ~6 months).")
    parser.add_argument("--gtja-factors", default=None,
                        help="Comma-separated GTJA191 factor names for --strategy gtja191. "
                             "Default: all 187 available factors. "
                             "Example: --gtja-factors alpha001,alpha002,alpha004")
    parser.add_argument("--gtja-ic-sign", default=None,
                        help="Path to IC screening JSON (cn_gtja191_ic_screening.json) for sign-"
                             "flipped weights: positive IC → positive weight, negative IC → flipped. "
                             "Without this, all GTJA factors are equal-weighted.")
    parser.add_argument("--required", default="growth,quality,momentum",
                        help="Comma list of factors required non-None to be eligible "
                             "(pass '' for pure price/volume strategies).")
    parser.add_argument("--refresh-shares", action="store_true",
                        help="Force re-fetch of the spot share/price snapshot (else cached).")
    parser.add_argument("--data-cache-dir", default=cn.CACHE_DIR_DEFAULT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", help="Optional path to write JSON result.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    if args.strategy not in STRATEGY_PROFILES:
        parser.error(f"Unknown strategy '{args.strategy}'. Available: {', '.join(STRATEGY_PROFILES)}")
    result = run(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
