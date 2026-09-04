#!/usr/bin/env python3
"""A 股 Point-in-Time 因子诊断。

在不改变选股逻辑的前提下，逐期计算横截面 Spearman IC、IC/IR、正向比例、五分组
收益和分层单调性，并比较行业中性前后的 IC。IC 时间序列进一步使用 Newey-West、
循环移动区块 bootstrap 和 BH-FDR，避免把序列相关或批量筛选产生的假阳性误认为
稳定因子。数据读取、因子值和前向价格全部复用回测引擎，保证口径只有一个来源。
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import date, datetime, timezone
from typing import Any

from . import backtest as bt
from . import data_source as cn
from . import quant_stats as qstats

FACTORS = list(bt.DEFAULT_FACTOR_WEIGHTS.keys())
MIN_CROSS_SECTION = 30  # skip a factor's cross-section if fewer valid stocks


# ---------------------------------------------------------------------------
# 秩统计
# ---------------------------------------------------------------------------
def _ranks(values: list[float]) -> list[float]:
    """Average ranks (1-based), ties share the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _pearson(a: list[float], b: list[float]) -> float | None:
    n = len(a)
    if n < 3:
        return None
    ma, mb = statistics.mean(a), statistics.mean(b)
    cov = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return cov / math.sqrt(va * vb)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation; None if undefined (too few / no variance)."""
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    return _pearson(_ranks(xs), _ranks(ys))


# ---------------------------------------------------------------------------
# 截面构建：每个再平衡日保存因子、前向收益和行业
# ---------------------------------------------------------------------------
def build_sections(
    rebals: list[date],
    codes: list[str],
    prices: dict[str, list[dict[str, Any]]],
    financials: dict[str, list[dict[str, Any]]],
    pe: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """One entry per rebalance interval [T, T_next): PIT factors + forward returns."""
    sections: list[dict[str, Any]] = []
    for i in range(len(rebals) - 1):
        t, t_next = rebals[i], rebals[i + 1]
        iso, iso_next = t.isoformat(), t_next.isoformat()
        raw = bt.raw_factors_at(codes, iso, prices, financials, pe)
        fwd: dict[str, float] = {}
        industry: dict[str, str] = {}
        for c in codes:
            bars = prices.get(c, [])
            p0 = bt.close_asof(bars, iso)
            p1 = bt.close_asof(bars, iso_next)
            if p0 and p1 and p0 > 0:
                fwd[c] = p1 / p0 - 1
            pit = cn.pit_financials(financials.get(c, []), iso)
            ind = pit["financials"].get("industry") if pit else None
            if ind:
                industry[c] = str(ind)
        sections.append({"start": t, "end": t_next, "factors": raw, "fwd": fwd, "industry": industry})
    return sections


# ---------------------------------------------------------------------------
# IC 汇总与稳健统计
# ---------------------------------------------------------------------------
def _verdict(mean_ic: float, ir: float, positive_ratio: float) -> str:
    if mean_ic >= 0.05 and ir >= 0.5:
        return "strong"
    if mean_ic >= 0.03 and positive_ratio >= 0.55:
        return "weak_positive"
    if positive_ratio < 0.50:
        return "unstable"
    return "neutral"


def compute_ic(
    sections: list[dict[str, Any]],
    factors: list[str],
    min_cross: int = MIN_CROSS_SECTION,
) -> dict[str, Any]:
    per_factor: dict[str, list[float]] = {f: [] for f in factors}
    for sec in sections:
        fwd = sec["fwd"]
        for f in factors:
            xs, ys = [], []
            for c, fac in sec["factors"].items():
                v, r = fac.get(f), fwd.get(c)
                if v is not None and r is not None:
                    xs.append(v)
                    ys.append(r)
            if len(xs) < min_cross:
                continue
            ic = spearman(xs, ys)
            if ic is not None:
                per_factor[f].append(ic)

    summary: dict[str, Any] = {}
    p_values: dict[str, float] = {}
    for f, ics in per_factor.items():
        if not ics:
            summary[f] = {"n_periods": 0, "verdict": "insufficient_data"}
            continue
        mean_ic = statistics.mean(ics)
        std_ic = statistics.stdev(ics) if len(ics) > 1 else 0.0
        ir = mean_ic / std_ic if std_ic > 0 else 0.0
        positive_ratio = sum(1 for x in ics if x > 0) / len(ics)
        robust = qstats.summarize_signal(ics)
        nw = robust["newey_west"]
        if nw["p_value"] is not None:
            p_values[f] = float(nw["p_value"])
        summary[f] = {
            "mean_ic": round(mean_ic, 4),
            "std_ic": round(std_ic, 4),
            "ir": round(ir, 3),
            "positive_ratio": round(positive_ratio, 3),
            "n_periods": len(ics),
            "verdict": _verdict(mean_ic, ir, positive_ratio),
            "ic_series": [round(x, 6) for x in ics],
            "newey_west": nw,
            "moving_block_bootstrap": robust["moving_block_bootstrap"],
        }
    adjusted = qstats.benjamini_hochberg(p_values)
    for factor, result in adjusted.items():
        summary[factor]["multiple_testing"] = {
            "method": "Benjamini-Hochberg",
            "fdr": 0.05,
            **result,
        }
    return summary


# ---------------------------------------------------------------------------
# 分层回测
# ---------------------------------------------------------------------------
def _curve_metrics(interval_rets: list[float], total_years: float,
                    periods_per_year: int) -> dict[str, float]:
    equity = [1.0]
    for r in interval_rets:
        equity.append(equity[-1] * (1 + r))
    cagr = equity[-1] ** (1 / max(total_years, 1e-9)) - 1
    if len(interval_rets) > 1:
        sd = statistics.stdev(interval_rets)
        sharpe = statistics.mean(interval_rets) / sd * math.sqrt(periods_per_year) if sd > 0 else 0.0
    else:
        sharpe = 0.0
    peak, mdd = equity[0], 0.0
    for v in equity:
        peak = max(peak, v)
        mdd = max(mdd, 1 - v / peak)
    return {
        "annualized_return_percent": round(cagr * 100, 2),
        "sharpe": round(sharpe, 2),
        "max_drawdown_percent": round(mdd * 100, 2),
    }


def compute_quantiles(
    sections: list[dict[str, Any]],
    factors: list[str],
    quantiles: int = 5,
    min_cross: int = MIN_CROSS_SECTION,
    periods_per_year: int = 4,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in factors:
        group_rets: list[list[float]] = [[] for _ in range(quantiles)]
        total_days = 0
        n_periods = 0
        for sec in sections:
            valid = [
                (fac.get(f), sec["fwd"].get(c))
                for c, fac in sec["factors"].items()
                if fac.get(f) is not None and sec["fwd"].get(c) is not None
            ]
            if len(valid) < max(min_cross, quantiles):
                continue
            valid.sort(key=lambda t: t[0])  # ascending factor value
            n = len(valid)
            for g in range(quantiles):
                lo, hi = g * n // quantiles, (g + 1) * n // quantiles
                grp = valid[lo:hi]
                group_rets[g].append(statistics.mean([r for _, r in grp]) if grp else 0.0)
            total_days += (sec["end"] - sec["start"]).days
            n_periods += 1

        if n_periods == 0:
            out[f] = {"n_periods": 0, "verdict": "insufficient_data"}
            continue
        years = total_days / 365.25
        groups = [_curve_metrics(group_rets[g], years, periods_per_year) for g in range(quantiles)]
        ann = [g["annualized_return_percent"] for g in groups]
        trend = spearman([float(i) for i in range(quantiles)], ann)
        out[f] = {
            "n_periods": n_periods,
            "group_annualized_return_percent": ann,
            "group_sharpe": [g["sharpe"] for g in groups],
            "group_max_drawdown_percent": [g["max_drawdown_percent"] for g in groups],
            "long_short_spread_annualized_percent": round(ann[-1] - ann[0], 2),
            "monotonic_trend": round(trend, 3) if trend is not None else None,
            "monotonic": bool(trend is not None and trend >= 0.6),
        }
    return out


# ---------------------------------------------------------------------------
# 行业内标准化后的中性 IC
# ---------------------------------------------------------------------------
def compute_industry_neutral_ic(
    sections: list[dict[str, Any]],
    factors: list[str],
    min_cross: int = MIN_CROSS_SECTION,
) -> dict[str, Any]:
    if not any(sec["industry"] for sec in sections):
        return {"status": "skipped_no_industry_data"}

    by_factor: dict[str, Any] = {}
    for f in factors:
        raw_ics: list[float] = []
        neutral_ics: list[float] = []
        for sec in sections:
            entries = [
                (c, fac.get(f), sec["fwd"].get(c), sec["industry"].get(c))
                for c, fac in sec["factors"].items()
                if fac.get(f) is not None and sec["fwd"].get(c) is not None
            ]
            if len(entries) < min_cross:
                continue
            raw_ic = spearman([e[1] for e in entries], [e[2] for e in entries])
            if raw_ic is not None:
                raw_ics.append(raw_ic)

            grouped: dict[str, list[Any]] = defaultdict(list)
            for e in entries:
                if e[3]:
                    grouped[e[3]].append(e)
            neu_vals, neu_ret = [], []
            for grp in grouped.values():
                if len(grp) < 2:
                    continue
                vals = [e[1] for e in grp]
                mean, sd = statistics.mean(vals), statistics.stdev(vals)
                if sd == 0:
                    continue
                for e in grp:
                    neu_vals.append((e[1] - mean) / sd)
                    neu_ret.append(e[2])
            if len(neu_vals) >= min_cross:
                neu_ic = spearman(neu_vals, neu_ret)
                if neu_ic is not None:
                    neutral_ics.append(neu_ic)
        by_factor[f] = {
            "raw_mean_ic": round(statistics.mean(raw_ics), 4) if raw_ics else None,
            "industry_neutral_mean_ic": round(statistics.mean(neutral_ics), 4) if neutral_ics else None,
            "n_periods_neutral": len(neutral_ics),
        }
    return {"status": "computed", "by_factor": by_factor}


# ---------------------------------------------------------------------------
# 运行编排
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.data_cache_dir or None
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    data = bt.load_all_data(
        args.index, args.price_start, cache_dir, args.refresh,
        args.fin_start_year, end.year, full_market=args.full_market, skip_pe=args.skip_pe,
        benchmark_code=args.benchmark,
    )
    usable = [c for c in data["codes"] if data["prices"].get(c)]
    bench_in = [b for b in data["benchmark"] if start.isoformat() <= b["date"] <= end.isoformat()]
    calendar = [date.fromisoformat(b["date"]) for b in bench_in]
    freq = args.rebalance_freq
    rebals = sorted(bt.rebalance_dates(calendar, freq=freq))
    sections = build_sections(rebals, usable, data["prices"], data["financials"], data["pe"])

    factors = [f.strip() for f in args.factors.split(",")] if args.factors else FACTORS
    ic_summary = compute_ic(sections, factors)
    ppy = bt.periods_per_year(freq)
    quantile_backtest = compute_quantiles(sections, factors, args.quantiles, periods_per_year=ppy)
    industry_neutral = compute_industry_neutral_ic(sections, factors)

    freq_label = freq
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "framework": "a_share_factor_diagnostics",
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "universe": {
            "source": "full_market" if args.full_market else args.index,
            "benchmark_index": args.benchmark,
            "constituents": len(data["codes"]),
            "usable": len(usable),
            "price_fetch_failed": len(data["failed"]),
            "rebalance_dates": len(rebals),
            "ic_periods": len(sections),
            "quantiles": args.quantiles,
            "min_cross_section": MIN_CROSS_SECTION,
        },
        "ic_summary": ic_summary,
        "quantile_backtest": quantile_backtest,
        "industry_neutral": industry_neutral,
        "caveats": [
            f"IC 用 T→下一再平衡日的前向收益,横截面 Spearman;{freq_label}频率,样本期数={len(sections)}。",
            "IC 均值同时报告 Newey-West HAC t 检验、移动区块 bootstrap 95% 区间与 Benjamini-Hochberg 5% FDR。",
            "当前用最新指数成分名单,含成分股幸存者偏差(PIT 只解决财报滞后,不解决退市)。",
            "百度 PE 历史约 2021 起,value 因子早期可能缺失。",
            "流动性用 close×volume 代理(后复权价跨股票失真),权重最低。",
            "long-short 仅作因子有效性诊断,未计交易成本/做空成本,不是可交易策略。",
            "行业取自业绩报表 所处行业 字段,粗颗粒;缺失个股不参与行业中性。",
        ],
    }


# ---------------------------------------------------------------------------
# 离线自检（合成截面，不访问网络）
# ---------------------------------------------------------------------------
def self_test() -> None:
    # Rank stats sanity.
    assert spearman([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]) > 0.999, "完全同序 IC≈1"
    assert spearman([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]) < -0.999, "完全反序 IC≈-1"
    assert abs(_pearson([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    print("[PASS] Spearman/Pearson 秩相关方向与边界正确")

    # Synthetic cross-sections: a hidden signal drives forward return; 'pos' factor
    # equals it, 'neg' is its negation, 'rand' is independent noise.
    import random
    rng = random.Random(0)
    factors = ["pos", "neg", "rand"]
    sections: list[dict[str, Any]] = []
    for _ in range(8):
        facs: dict[str, dict[str, float]] = {}
        fwd: dict[str, float] = {}
        for i in range(40):
            c = f"C{i:02d}"
            signal = rng.gauss(0, 1)
            facs[c] = {"pos": signal, "neg": -signal, "rand": rng.gauss(0, 1)}
            fwd[c] = 0.05 * signal + rng.gauss(0, 0.02)
        sections.append({"start": date(2021, 1, 1), "end": date(2021, 4, 1),
                         "factors": facs, "fwd": fwd, "industry": {}})

    ic = compute_ic(sections, factors, min_cross=30)
    assert ic["pos"]["mean_ic"] > 0.3, f"正向因子 IC 应为正: {ic['pos']}"
    assert ic["neg"]["mean_ic"] < -0.3, f"反向因子 IC 应为负: {ic['neg']}"
    assert abs(ic["rand"]["mean_ic"]) < 0.15, f"随机因子 IC 应≈0: {ic['rand']}"
    assert ic["pos"]["verdict"] == "strong" and ic["neg"]["verdict"] == "unstable"
    assert ic["pos"]["newey_west"]["p_value"] < 0.05
    assert ic["pos"]["multiple_testing"]["significant"]
    assert ic["pos"]["moving_block_bootstrap"]["ci"][0] > 0
    print(f"[PASS] IC 方向正确:pos {ic['pos']['mean_ic']} / neg {ic['neg']['mean_ic']} / "
          f"rand {ic['rand']['mean_ic']}")

    quant = compute_quantiles(sections, factors, quantiles=5, min_cross=30)
    pos_groups = quant["pos"]["group_annualized_return_percent"]
    assert pos_groups[-1] > pos_groups[0], f"强因子最高分组应优于最低分组: {pos_groups}"
    assert quant["pos"]["monotonic"], f"强因子分层应单调: {quant['pos']}"
    assert quant["pos"]["long_short_spread_annualized_percent"] > 0
    print(f"[PASS] 分层回测:pos 五分组年化 {pos_groups} 单调递增,"
          f"多空 {quant['pos']['long_short_spread_annualized_percent']}%")

    ind = compute_industry_neutral_ic(sections, factors, min_cross=30)
    assert ind["status"] == "skipped_no_industry_data", "无行业字段应跳过"
    # With industry present, computes both raw and neutral IC.
    for sec in sections:
        sec["industry"] = {c: ("A" if i % 2 == 0 else "B") for i, c in enumerate(sec["factors"])}
    ind2 = compute_industry_neutral_ic(sections, factors, min_cross=30)
    assert ind2["status"] == "computed"
    assert ind2["by_factor"]["pos"]["raw_mean_ic"] > 0.3
    assert ind2["by_factor"]["pos"]["industry_neutral_mean_ic"] is not None
    print("[PASS] 行业中性:无行业跳过;有行业时输出 raw 与 neutral IC")
    print("ALL PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="A-share factor diagnostics (IC/IR + quantile).")
    parser.add_argument("--self-test", action="store_true", help="Offline synthetic-data test.")
    parser.add_argument("--index", default="000300", help="Universe index code (default 沪深300).")
    parser.add_argument("--benchmark", default="880008",
                        help="Benchmark index code: 880008, 000300, 000905, etc. (default 880008 全A等权).")
    parser.add_argument("--full-market", action="store_true",
                        help="Use the full A-share list (~5500) as the universe instead of --index members.")
    parser.add_argument("--skip-pe", action="store_true",
                        help="Skip per-stock PE fetch; the value factor degrades to insufficient_data.")
    parser.add_argument("--start", default="2021-01-01", help="Diagnostics start (PE history begins ~2021).")
    parser.add_argument("--end", default=None, help="End (default today).")
    parser.add_argument("--top-n", type=int, default=30, help="Kept for CLI parity; unused by diagnostics.")
    parser.add_argument("--price-start", default="20180101", help="hfq price fetch start.")
    parser.add_argument("--fin-start-year", type=int, default=2018)
    parser.add_argument("--quantiles", type=int, default=5)
    parser.add_argument("--rebalance-freq", default="quarterly",
                        help="Rebalance frequency for IC computation: quarterly, monthly, weekly, or Nd (e.g. 2d, 5d).")
    parser.add_argument("--factors", default=None,
                        help="Comma-separated factor names to diagnose (default: the 6 core factors). "
                             f"Candidates: {', '.join(bt.CANDIDATE_FACTORS)}.")
    parser.add_argument("--data-cache-dir", default=cn.CACHE_DIR_DEFAULT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", help="Optional path to write JSON result.")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    result = run(args)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
