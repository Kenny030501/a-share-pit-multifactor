#!/usr/bin/env python3
"""用财务数据和现价锚点近似重建 A 股历史时点总市值。

总股本近似为“净利润 / 每股收益”，并且只读取当时已超过保守可用日的财报；历史
不复权价则通过当前现价锚定后复权序列：

       raw_t = price_now × hfq_t / hfq_now

该方法优于直接比较不同股票的后复权价，但仍会受到历史累计分红差异影响，因此
属于公开数据约束下的近似，不是商业级历史市值数据库。
"""

from __future__ import annotations

import math
import statistics
from bisect import bisect_right
from typing import Any

Bar = dict[str, Any]


# ---------------------------------------------------------------------------
# 价格辅助函数
# ---------------------------------------------------------------------------
def _cut(bars: list[Bar], as_of: str) -> int:
    return bisect_right(bars, as_of, key=lambda b: b["date"])


def hfq_asof(bars: list[Bar], as_of: str) -> float | None:
    i = _cut(bars, as_of)
    return bars[i - 1]["close"] if i > 0 else None


# ---------------------------------------------------------------------------
# 从当时可用财报近似总股本
# ---------------------------------------------------------------------------
def _num(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def shares_from_report(fin: dict[str, Any]) -> float | None:
    """总股本 ≈ 净利润 / 每股收益; None if either missing or eps≈0."""
    np_ = _num(fin.get("net_profit"))
    eps = _num(fin.get("eps"))
    if np_ is None or eps is None or abs(eps) < 1e-9:
        return None
    sh = np_ / eps
    return sh if sh > 0 else None


def pit_shares(entries: list[dict[str, Any]], as_of: str) -> float | None:
    """Total shares from the most recent report available at `as_of` that yields a
    valid net_profit/eps estimate (scanning back if the latest lacks eps)."""
    available = sorted(
        (e for e in entries if e["avail_date"] <= as_of),
        key=lambda e: e["period"],
    )
    for e in reversed(available):
        sh = shares_from_report(e.get("financials", {}))
        if sh is not None:
            return sh
    return None


# ---------------------------------------------------------------------------
# 历史时点总市值
# ---------------------------------------------------------------------------
def unadj_price_at(bars: list[Bar], as_of: str, price_now: float | None) -> float | None:
    """raw_t = price_now × hfq_t / hfq_now (hfq_now = latest cached hfq close)."""
    if not price_now or price_now <= 0 or not bars:
        return None
    hz = hfq_asof(bars, as_of)
    h_now = bars[-1].get("close")
    if hz is None or not h_now or h_now <= 0:
        return None
    return price_now * hz / h_now


def market_cap_at(
    code: str,
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    snapshot: dict[str, dict[str, float]],
    shares_override: float | None = None,
) -> float | None:
    """Point-in-time total market cap (元), or None if inputs are missing."""
    snap = snapshot.get(code)
    if not snap:
        return None
    raw = unadj_price_at(prices.get(code) or [], as_of, snap.get("price"))
    if raw is None:
        return None
    sh = shares_override if shares_override is not None else pit_shares(financials.get(code, []), as_of)
    if sh is None or sh <= 0:
        return None
    return sh * raw


def avg_market_cap(
    code: str,
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    snapshot: dict[str, dict[str, float]],
    lookback: int = 250,
) -> float | None:
    """Mean daily total market cap over the last `lookback` bars ending at `as_of`.

    Shares are held at their `as_of` value across the window (they only step at the
    ~4 report dates/yr, so this is a tight approximation and one pit_shares call)."""
    snap = snapshot.get(code)
    if not snap:
        return None
    price_now = snap.get("price")
    bars = prices.get(code) or []
    if not price_now or price_now <= 0 or not bars:
        return None
    h_now = bars[-1].get("close")
    if not h_now or h_now <= 0:
        return None
    window = bars[: _cut(bars, as_of)][-lookback:]
    hfqs = [b["close"] for b in window if b.get("close")]
    if not hfqs:
        return None
    sh = pit_shares(financials.get(code, []), as_of)
    if sh is None or sh <= 0:
        return None
    return sh * price_now / h_now * statistics.mean(hfqs)


# ---------------------------------------------------------------------------
# 离线自检（合成数据）
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # shares = net_profit / eps, sign-safe for loss-makers.
    assert abs(shares_from_report({"net_profit": 8.232e10, "eps": 65.66}) - 1.254e9) / 1.254e9 < 0.01
    assert abs(shares_from_report({"net_profit": -8.85e10, "eps": -7.45}) - 1.188e10) / 1.188e10 < 0.02
    assert shares_from_report({"net_profit": 100.0, "eps": 0.0}) is None, "eps=0 应返回 None"
    assert shares_from_report({"net_profit": None, "eps": 1.0}) is None
    print("[PASS] shares_from_report: 净利润/EPS 近真实, 亏损股定号正确, 缺失优雅返回 None")

    # pit_shares scans back when the latest report lacks eps, and respects avail_date.
    entries = [
        {"period": "20231231", "avail_date": "2024-04-30",
         "financials": {"net_profit": 1.0e9, "eps": 1.0}},              # -> 1e9 shares
        {"period": "20240930", "avail_date": "2024-10-31",
         "financials": {"net_profit": 8.0e8, "eps": None}},             # unusable (no eps)
    ]
    assert pit_shares(entries, "2024-03-01") is None, "首份年报截止前无股本"
    assert abs(pit_shares(entries, "2024-06-01") - 1.0e9) < 1, "只有年报可用"
    assert abs(pit_shares(entries, "2024-12-01") - 1.0e9) < 1, "Q3 无 eps → 回退年报"
    print("[PASS] pit_shares: 按可用日选最新、缺 eps 时回退、无前视")

    # Unadjusted-price anchor: raw_t = price_now × hfq_t/hfq_now recovers the true raw
    # even when hfq has drifted far above raw (here hfq_now = 10× raw_now).
    bars = [{"date": f"2024-01-0{i}", "close": 1000.0 * (1 + 0.1 * i)} for i in range(1, 6)]
    price_now = bars[-1]["close"] / 10.0  # real price is 1/10 of hfq now
    raw_mid = unadj_price_at(bars, "2024-01-03", price_now)
    assert abs(raw_mid - bars[2]["close"] / 10.0) < 1e-6, "锚定后不复权价还原正确"
    assert unadj_price_at(bars, "2023-01-01", price_now) is None, "早于首根无价"
    print(f"[PASS] unadj_price_at: 现价锚定还原不复权价 (中段 {raw_mid:.2f})")

    # market_cap_at combines both; avg over window ~ point value on a slow drift.
    prices = {"X": bars}
    fin = {"X": [{"period": "20231231", "avail_date": "2024-01-01",
                  "financials": {"net_profit": 1.0e9, "eps": 1.0}}]}
    snap = {"X": {"price": price_now, "mktcap": None}}
    mc = market_cap_at("X", "2024-01-05", prices, fin, snap)
    assert mc is not None and abs(mc - 1.0e9 * price_now) < 1e-3, mc
    amc = avg_market_cap("X", "2024-01-05", prices, fin, snap, lookback=5)
    assert amc is not None and amc > 0
    assert market_cap_at("X", "2024-01-05", prices, fin, {}) is None, "无快照 → None"
    print(f"[PASS] market_cap_at / avg_market_cap: 组合股本×不复权价 (末日市值 {mc/1e8:.2f} 亿)")
    print("ALL PASS")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="PIT market cap for A-shares.")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        _self_test()
    else:
        p.error("Use --self-test.")
