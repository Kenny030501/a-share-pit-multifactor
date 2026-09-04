#!/usr/bin/env python3
"""基于市值规则构造近似的历史中证 500 样本空间。

每个换样时点先剔除 ST、上市历史不足和长期无新行情的股票，再按过去约一年平均
总市值排序，剔除最大的一组并选取随后 500 只。该方法避免直接把“当前指数成分”
倒推历史，但它仍依赖上游提供的证券名单；若名单缺少已退市股票，幸存者偏差依然
存在。因此这是规则近似，不是指数委员会真实历史成分。
"""

from __future__ import annotations

from datetime import date
from typing import Any

from . import size as cn_size

Bar = dict[str, Any]


def is_st(name: str) -> bool:
    return "ST" in (name or "").upper()


def build_synthetic_csi500(
    as_of: str,
    prices: dict[str, list[Bar]],
    financials: dict[str, list[dict[str, Any]]],
    snapshot: dict[str, dict[str, float]],
    universe: list[dict[str, str]],
    drop_top: int = 300,
    take: int = 500,
    min_bars: int = 250,
    lookback: int = 250,
    max_stale_days: int = 30,
) -> list[str]:
    """Return the synthetic 中证500 constituent codes as of `as_of`."""
    as_of_d = date.fromisoformat(as_of)
    sized: list[tuple[str, float]] = []
    for row in universe:
        code = row["code"]
        if is_st(row.get("name", "")):
            continue
        bars = prices.get(code) or []
        n = cn_size._cut(bars, as_of)
        if n < min_bars:  # 次新 / insufficient history
            continue
        last_date = bars[n - 1]["date"]
        if (as_of_d - date.fromisoformat(last_date)).days > max_stale_days:
            continue  # 停牌 / 退市
        mc = cn_size.avg_market_cap(code, as_of, prices, financials, snapshot, lookback)
        if mc is None or mc <= 0:
            continue
        sized.append((code, mc))
    sized.sort(key=lambda x: x[1], reverse=True)
    return [code for code, _ in sized[drop_top: drop_top + take]]


# ---------------------------------------------------------------------------
# 离线自检（合成数据）
# ---------------------------------------------------------------------------
def _flat_bars(n: int, end: str, close: float = 100.0) -> list[Bar]:
    """`n` daily bars ending at `end` (constant close so avg_market_cap == price_now)."""
    end_d = date.fromisoformat(end)
    return [{"date": date.fromordinal(end_d.toordinal() - (n - 1 - i)).isoformat(),
             "close": close} for i in range(n)]


def _self_test() -> None:
    as_of = "2023-06-30"
    universe = [{"code": f"C{i:02d}", "name": f"股票{i:02d}"} for i in range(20)]
    prices, financials, snapshot = {}, {}, {}
    for i, row in enumerate(universe):
        c = row["code"]
        prices[c] = _flat_bars(300, as_of)                       # enough history, fresh
        financials[c] = [{"period": "20221231", "avail_date": "2023-04-30",
                          "financials": {"net_profit": 1.0e8, "eps": 1.0}}]  # shares = 1e8
        snapshot[c] = {"price": 1000.0 - i * 10.0, "mktcap": None}  # strictly decreasing size

    # drop top 3, take next 5 → ranks 4..8 by size = C03..C07.
    pool = build_synthetic_csi500(as_of, prices, financials, snapshot, universe,
                                  drop_top=3, take=5, min_bars=250)
    assert pool == [f"C{i:02d}" for i in range(3, 8)], f"合成池应为 C03..C07, 实为 {pool}"
    print(f"[PASS] 市值规则: 剔除最大3只、取随后5只 → {pool}")

    # ST excluded even when largest; membership shifts down by one.
    uni_st = [dict(r) for r in universe]
    uni_st[0]["name"] = "ST退市测"
    pool_st = build_synthetic_csi500(as_of, prices, financials, snapshot, uni_st,
                                     drop_top=3, take=5, min_bars=250)
    assert "C00" not in pool_st, "ST 应被剔除"
    assert pool_st == [f"C{i:02d}" for i in range(4, 9)], f"ST剔除后应下移一位, 实为 {pool_st}"
    print(f"[PASS] ST 剔除: C00(最大)被排除, 池下移 → {pool_st}")

    # Recent IPO (few bars) and all-future data are excluded (PIT: no peeking ahead).
    p2 = dict(prices)
    p2["C10"] = _flat_bars(100, as_of)                            # 次新: 100 < 250 bars
    p2["C11"] = _flat_bars(300, "2025-01-01")                     # all bars AFTER as_of
    pool2 = build_synthetic_csi500(as_of, p2, financials, snapshot, universe,
                                   drop_top=0, take=20, min_bars=250)
    assert "C10" not in pool2 and "C11" not in pool2, "次新与未来数据应被剔除"
    print("[PASS] PIT: 次新(<250根) 与 as_of 之后才有数据的股票均被剔除")

    # Suspended (stale) name is dropped.
    p3 = dict(prices)
    p3["C05"] = _flat_bars(300, "2023-01-01")                     # last bar ~180d before as_of
    pool3 = build_synthetic_csi500(as_of, p3, financials, snapshot, universe,
                                   drop_top=0, take=20, min_bars=250, max_stale_days=30)
    assert "C05" not in pool3, "长期停牌(数据过旧)应被剔除"
    print("[PASS] 停牌剔除: 距 as_of 超 max_stale_days 无行情者被排除")
    print("ALL PASS")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Synthetic PIT CSI500 sample space.")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        _self_test()
    else:
        p.error("Use --self-test.")
