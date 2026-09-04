#!/usr/bin/env python3
"""GTJA191 因子的严格 T+2 回测入口。

根据 IC 方向合成因子，在近似中证 500 股票池上执行行业/规模中性组合优化，并同时
输出无摩擦与真实摩擦结果。该脚本仍使用同一研究窗口筛选因子，属于探索性复现，
不能替代预注册后的独立样本外检验。

用法：``python scripts/run_gtja191_backtest.py [min_abs_ic] [top_n] [turnover_cap]``。
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

from ashare_pit import backtest as bt
from ashare_pit import data_source as cn


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = str(ROOT / "data" / "interim" / "cache")
SCREENING_PATH = ROOT / "results" / "cn_gtja191_ic_screening.json"
START = date(2012, 7, 1)
END = date(2017, 4, 30)
BENCHMARK = "000905"


def load_ic_signed_weights(min_abs_ic: float = 0.02) -> tuple[dict[str, float], list[str]]:
    """Load IC screening results, return {factor: signed_weight} and ordered factor list.

    Positive IC → positive weight (factor direction matches paper).
    Negative IC → negative weight (flip factor before compositing).
    """
    with SCREENING_PATH.open(encoding="utf-8") as f:
        screening = json.load(f)

    all_ic = {f["factor"]: f["mean_ic"] for f in screening["all_factors"]}
    selected = {k: v for k, v in all_ic.items() if abs(v) >= min_abs_ic}
    n = len(selected)
    weights = {k: (1.0 if v > 0 else -1.0) / n for k, v in selected.items()}
    factor_list = sorted(weights.keys())

    n_pos = sum(1 for v in selected.values() if v > 0)
    n_neg = sum(1 for v in selected.values() if v < 0)
    print(f"IC screening → {n} factors with |IC| ≥ {min_abs_ic} "
          f"(positive={n_pos}, negative/sign-flipped={n_neg})")
    for f in factor_list[:10]:
        print(f"  {f}: IC={all_ic[f]:+.4f} → weight={weights[f]:+.4f}")
    if len(factor_list) > 10:
        print(f"  ... and {len(factor_list) - 10} more")
    return weights, factor_list


def run_one(
    prices: dict, financials: dict, benchmark: list,
    share_snap: dict[str, dict[str, float]],
    weights: dict[str, float], gtja_factors: list[str],
    top_n: int, turnover_cap: float,
    friction: bool,
) -> dict[str, Any]:
    label = "realistic" if friction else "frictionless"
    print(f"\n--- {label} ---")

    result = bt.run_rolling_backtest(
        codes=list(prices.keys()),
        prices=prices,
        financials=financials,
        pe={},
        benchmark=benchmark,
        start=START,
        end=END,
        weights=weights,
        top_n=top_n,
        rebalance_freq="2d",
        universe_mode="csi500_synth",
        construction="optimize",
        neutralize=("industry", "size"),
        weight_cap=0.03,
        ridge=1.0,
        snapshot=share_snap,
        required=(),
        csi500_min_bars=120,
        turnover_cap=turnover_cap,
        gtja_factors=gtja_factors,
        commission_bps=3.0 if friction else 0.0,
        slippage_bps=10.0 if friction else 0.0,
        stamp_tax=friction,
        limit_filter=friction,
    )

    p = result.get("portfolio", {})
    b = result.get("benchmark", {})
    print(f"  组合年化 {p.get('annualized_return_percent', 0):.2f}%  "
          f"Sharpe {p.get('sharpe_ratio_rf0', 0):.2f}  "
          f"回撤 {p.get('max_drawdown_percent', 0):.1f}%")
    print(f"  基准年化 {b.get('annualized_return_percent', 0):.2f}%  "
          f"Sharpe {b.get('sharpe_ratio_rf0', 0):.2f}")
    active = result.get("active_annualized_return_percent", 0)
    print(f"  超额 {active:+.2f}%  "
          f"再平衡 {result.get('rebalances', '?')}次  "
          f"成本拖累 {result.get('total_cost_drag_fraction', 0):.4%}")

    sel = result.get("latest_selection") or {}
    if sel:
        print(f"  末期持仓 {len(sel.get('selected', []))}只")
    return result


def main():
    min_abs_ic = float(sys.argv[1]) if len(sys.argv) > 1 else 0.02
    top_n = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    turnover_cap = float(sys.argv[3]) if len(sys.argv) > 3 else 0.15

    print("=== GTJA191 严格 T+2 回测 ===")
    print(f"窗口: {START} ~ {END} | 基准: {BENCHMARK} | T+2")
    print(f"min_abs_ic={min_abs_ic} top_n={top_n} turnover_cap={turnover_cap}")

    # 1. IC-signed weights
    weights, gtja_factors = load_ic_signed_weights(min_abs_ic)

    # 2. Load data
    print("\nLoading market snapshot...")
    snap = cn.load_market_snapshot(
        cache_dir=CACHE_DIR, with_prices=True, with_pe=False,
        benchmark_code=BENCHMARK,
    )
    prices = snap["prices"]
    financials = snap.get("financials", {})
    benchmark = snap["benchmark"]
    print(f"  prices: {len(prices)} codes, benchmark: {len(benchmark)} days, "
          f"financials: {len(financials)} codes")

    print("Loading share snapshot (offline)...")
    share_snap = cn.load_share_snapshot(CACHE_DIR)
    print(f"  {len(share_snap)} codes")

    # 3. Run
    fl = run_one(prices, financials, benchmark, share_snap,
                 weights, gtja_factors, top_n, turnover_cap, friction=False)
    rl = run_one(prices, financials, benchmark, share_snap,
                 weights, gtja_factors, top_n, turnover_cap, friction=True)

    # 4. Compare
    decay = (fl.get("portfolio", {}).get("annualized_return_percent", 0)
             - rl.get("portfolio", {}).get("annualized_return_percent", 0))
    print(f"\n=== 摩擦衰减: {decay:.2f}%/年 ===")

    # 5. Save
    output = {
        "generated_at_utc": fl.get("generated_at_utc", ""),
        "framework": "a_share_rolling_point_in_time",
        "window": {"start": START.isoformat(), "end": END.isoformat()},
        "config": {
            "strategy": "gtja191_ic_signed",
            "n_factors": len(gtja_factors),
            "min_abs_ic": min_abs_ic,
            "factor_weights": weights,
            "top_n": top_n,
            "turnover_cap": turnover_cap,
            "construction": "optimize",
            "neutralize": ["industry", "size"],
            "weight_cap": 0.03,
            "ridge": 1.0,
            "rebalance_freq": "2d",
            "universe": "csi500_synth",
            "benchmark": BENCHMARK,
        },
        "frictionless": {
            "portfolio": fl.get("portfolio", {}),
            "benchmark": fl.get("benchmark", {}),
            "active_annualized_return_percent": fl.get("active_annualized_return_percent", 0),
            "rebalances": fl.get("rebalances", 0),
            "latest_selection": fl.get("latest_selection", []),
        },
        "realistic": {
            "portfolio": rl.get("portfolio", {}),
            "benchmark": rl.get("benchmark", {}),
            "active_annualized_return_percent": rl.get("active_annualized_return_percent", 0),
            "rebalances": rl.get("rebalances", 0),
            "total_cost_drag_fraction": rl.get("total_cost_drag_fraction", 0),
            "latest_selection": rl.get("latest_selection", []),
        },
        "annualized_return_decay_percent": round(decay, 2),
    }

    out_path = ROOT / "results" / "cn_backtest_gtja191_ic_signed.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
