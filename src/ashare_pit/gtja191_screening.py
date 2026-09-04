#!/usr/bin/env python3
"""在 2012-2017 样本上执行 GTJA191 因子 IC 初筛。

为控制运行时间，固定随机种子抽取股票和再平衡日期；输出 Newey-West、移动区块
bootstrap 与 BH-FDR。该阶段只生成候选因子，不能替代独立测试窗。
"""

from __future__ import annotations

import json
import math
import random
import statistics
from collections import defaultdict
from datetime import date
from pathlib import Path

from . import backtest as bt
from . import data_source as cn
from . import gtja191
from . import quant_stats as qstats

SAMPLE_SIZE = 200
MIN_BARS = 120
SAMPLED_REBALS_MAX = 50  # limit for speed
REBAL_FREQ = "2d"
START = date(2012, 7, 1)
END = date(2017, 4, 30)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3: return None
    n = len(xs)
    def ranks(vals):
        order = sorted(range(n), key=lambda i: vals[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and vals[order[j+1]] == vals[order[i]]: j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j+1): r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    cov = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    vx = sum((x-mx)**2 for x in rx)
    vy = sum((y-my)**2 for y in ry)
    return cov / math.sqrt(vx*vy) if vx > 0 and vy > 0 else None


def summarize_factor_ics(factor_ics: dict[str, list[float]]) -> list[dict[str, object]]:
    """汇总 IC 序列并在全部有效因子之间统一执行 FDR 校正。"""
    results: list[dict[str, object]] = []
    p_values: dict[str, float] = {}
    for fname, ics in sorted(factor_ics.items()):
        if len(ics) < 10:
            continue
        mean_ic = statistics.mean(ics)
        std_ic = statistics.stdev(ics) if len(ics) > 1 else 0.0
        ir = mean_ic / std_ic if std_ic > 0 else 0.0
        robust = qstats.summarize_signal(ics)
        nw = robust["newey_west"]
        if nw["p_value"] is not None:
            p_values[fname] = float(nw["p_value"])
        results.append({
            "factor": fname,
            "mean_ic": round(mean_ic, 4),
            "std_ic": round(std_ic, 4),
            "ir": round(ir, 3),
            "positive_ratio": round(sum(1 for x in ics if x > 0) / len(ics), 3),
            "n_periods": len(ics),
            "ic_series": [round(x, 6) for x in ics],
            "newey_west": nw,
            "moving_block_bootstrap": robust["moving_block_bootstrap"],
        })
    adjusted = qstats.benjamini_hochberg(p_values)
    for row in results:
        correction = adjusted.get(str(row["factor"]))
        row["multiple_testing"] = {
            "method": "Benjamini-Hochberg",
            "fdr": 0.05,
            **(correction or {"q_value": None, "significant": False}),
        }
    return sorted(results, key=lambda row: abs(float(row["mean_ic"])), reverse=True)

def main():
    rng = random.Random(42)

    # Load all stock prices (snapshot mode: offline, fast)
    print("Loading data...", flush=True)
    snap = cn.load_market_snapshot(with_prices=True, with_pe=False, benchmark_code="000905")
    all_codes = [row["code"] for row in snap["universe"]]

    # Filter to codes with long enough history
    eligible = []
    for code in all_codes:
        bars = snap["prices"].get(code, [])
        if len(bars) >= 300:  # need substantial history for rolling windows
            # Check first bar date
            if bars[0]["date"] <= "2012-01-15":  # earliest bar is 2012-01-04
                eligible.append(code)
    print(f"Eligible codes with 2012 data: {len(eligible)}", flush=True)

    # Sample
    if len(eligible) > SAMPLE_SIZE:
        sample = rng.sample(eligible, SAMPLE_SIZE)
    else:
        sample = eligible
    print(f"Sample size: {len(sample)}", flush=True)

    # Build rebalance calendar
    bench = snap["benchmark"]
    bench_in = [b for b in bench if START.isoformat() <= b["date"] <= END.isoformat()]
    calendar = [date.fromisoformat(b["date"]) for b in bench_in]
    rebals = sorted(bt.rebalance_dates(calendar, freq=REBAL_FREQ))
    print(f"Rebalance dates: {len(rebals)}", flush=True)

    # Sample rebals to limit runtime
    if len(rebals) > SAMPLED_REBALS_MAX:
        step = len(rebals) // SAMPLED_REBALS_MAX
        sampled_rebals = rebals[::step][:SAMPLED_REBALS_MAX]
    else:
        sampled_rebals = rebals
    print(f"Sampled rebal dates: {len(sampled_rebals)}", flush=True)

    prices = {c: snap["prices"][c] for c in sample}

    # Compute factors and IC for each rebalance
    factor_ics: dict[str, list[float]] = defaultdict(list)
    n_done = 0
    for t in sampled_rebals:
        iso = t.isoformat()
        # Forward return: T → next rebalance
        t_idx = rebals.index(t)
        if t_idx + 1 >= len(rebals): continue
        t_next = rebals[t_idx + 1]
        iso_next = t_next.isoformat()

        # Compute GTJA191 factors for sampled stocks
        result = gtja191.compute_factors(sample, iso, prices, min_bars=MIN_BARS)

        # Forward returns
        fwd = {}
        for code in sample:
            bars = prices.get(code, [])
            p0 = bt.close_asof(bars, iso)
            p1 = bt.close_asof(bars, iso_next)
            if p0 and p1 and p0 > 0:
                fwd[code] = p1 / p0 - 1

        # Compute IC for each factor
        for fname in gtja191.GTJA191_NAMES:
            xs, ys = [], []
            for code in sample:
                v = result.get(code, {}).get(fname)
                r = fwd.get(code)
                if v is not None and r is not None and not (isinstance(v, float) and math.isnan(v)):
                    xs.append(v)
                    ys.append(r)
            if len(xs) >= 30:
                ic = spearman(xs, ys)
                if ic is not None:
                    factor_ics[fname].append(ic)

        n_done += 1
        if n_done % 10 == 0:
            print(f"  IC progress: {n_done}/{len(sampled_rebals)}", flush=True)

    results = summarize_factor_ics(factor_ics)
    fdr_factors = [row for row in results if row["multiple_testing"]["significant"]]

    output = {
        "sample_size": len(sample),
        "rebalance_freq": REBAL_FREQ,
        "n_rebalance_dates": len(sampled_rebals),
        "window": f"{START} → {END}",
        "top_ic_factors": results[:30],
        "top_fdr_factors": fdr_factors[:30],
        "all_factors": results,
        "statistical_method": {
            "mean_test": "Newey-West HAC",
            "confidence_interval": "circular moving-block bootstrap",
            "multiple_testing": "Benjamini-Hochberg FDR 5%",
        },
        "caveats": [
            "The stock sample is drawn from a current security list and retains delisting survivorship bias.",
            "Only 50 rebalance dates are sampled for speed; this is a screening stage, not a final strategy test.",
            "Any selected factor must be frozen and retested on an untouched period before being called out-of-sample evidence.",
        ],
    }

    out_path = PROJECT_ROOT / "results" / "cn_gtja191_ic_screening.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print("\nTop 20 factors by |IC|:", flush=True)
    for r in results[:20]:
        print(f"  {r['factor']}: IC={r['mean_ic']:+.4f} IR={r['ir']:+.3f} "
              f"pos={r['positive_ratio']:.0%} n={r['n_periods']} "
              f"BH-q={r['multiple_testing']['q_value']}", flush=True)

    print(f"\nSaved to {out_path}", flush=True)

if __name__ == "__main__":
    main()
