"""A 股策略的锚定扩张式 walk-forward 样本外验证。

每一折只在历史训练窗比较候选策略，再把选中的策略用于紧随其后的未见测试窗；
各测试段拼接为连续样本外曲线。训练和测试共享同一个回测引擎及真实摩擦口径，
不存在单独的“乐观测试路径”。固定策略结果只作参照，若规格未在测试前冻结，
不得称为无偏样本外证据。
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta, timezone
from typing import Any

from . import backtest as bt
from . import data_source as cn

OBJECTIVE_KEY = {
    "sharpe": ("portfolio", "sharpe_ratio_rf0"),
    "cagr": ("portfolio", "annualized_return_percent"),
    "active": ("active_annualized_return_percent",),
}


def add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError:  # Feb 29 → Feb 28
        return d.replace(year=d.year + n, day=28)


def anchored_splits(
    gstart: date, gend: date, min_train_years: int, test_years: int
) -> list[tuple[date, date, date, date]]:
    """生成训练起点固定、训练终点扩张且测试窗互不重叠的时间切分。"""
    folds: list[tuple[date, date, date, date]] = []
    train_end = add_years(gstart, min_train_years)
    while train_end < gend:
        test_end = min(add_years(train_end, test_years), gend)
        test_start = train_end + timedelta(days=1)
        if (test_end - test_start).days < 120:  # need a non-trivial test window
            break
        folds.append((gstart, train_end, test_start, test_end))
        train_end = test_end
    return folds


def _objective_value(result: dict[str, Any], objective: str) -> float:
    keys = OBJECTIVE_KEY[objective]
    v: Any = result
    for k in keys:
        v = v[k]
    return float(v)


def evaluate(
    data: dict[str, Any], weights: dict[str, float], start: date, end: date,
    top_n: int, frictions: dict[str, Any], return_curve: bool = False,
    rebalance_freq: str = "quarterly",
) -> dict[str, Any]:
    """Realistic-friction rolling backtest of one weight set over [start, end]."""
    usable = [c for c in data["codes"] if data["prices"].get(c)]
    return bt.run_rolling_backtest(
        codes=usable, prices=data["prices"], financials=data["financials"],
        pe=data["pe"], benchmark=data["benchmark"], start=start, end=end,
        weights=weights, top_n=top_n,
        commission_bps=frictions["commission_bps"], slippage_bps=frictions["slippage_bps"],
        stamp_tax=True, limit_filter=True, max_stale_days=frictions["max_stale_days"],
        return_curve=return_curve, rebalance_freq=rebalance_freq,
    )


def stitch(segments: list[list[list[Any]]]) -> tuple[list[date], list[float]]:
    """把每折从约 1.0 起步的净值段首尾相接为连续样本外曲线。"""
    dates: list[date] = []
    values: list[float] = []
    cum = 1.0
    for seg in segments:
        if not seg:
            continue
        base = seg[0][1]
        for iso, v in seg:
            dates.append(date.fromisoformat(iso))
            values.append(cum * v / base)
        cum = values[-1]
    return dates, values


def _metrics_brief(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "annualized_return_percent": m["annualized_return_percent"],
        "sharpe_ratio_rf0": m["sharpe_ratio_rf0"],
        "max_drawdown_percent": m["max_drawdown_percent"],
    }


def walk_forward(
    data: dict[str, Any], strategies: dict[str, dict[str, float]], gstart: date, gend: date,
    min_train_years: int, test_years: int, top_n: int, frictions: dict[str, Any],
    objective: str, rebalance_freq: str = "quarterly",
) -> dict[str, Any]:
    splits = anchored_splits(gstart, gend, min_train_years, test_years)
    if not splits:
        raise ValueError("Window too short for the requested train/test split.")

    folds_out: list[dict[str, Any]] = []
    oos_segments: list[list[list[Any]]] = []
    for i, (trs, tre, tes, tee) in enumerate(splits, 1):
        train = {name: evaluate(data, w, trs, tre, top_n, frictions, rebalance_freq=rebalance_freq)
                 for name, w in strategies.items()}
        ranking = sorted(strategies, key=lambda n: _objective_value(train[n], objective), reverse=True)
        selected = ranking[0]
        test_sel = evaluate(data, strategies[selected], tes, tee, top_n, frictions,
                            return_curve=True, rebalance_freq=rebalance_freq)
        oos_segments.append(test_sel["equity_curve"])
        folds_out.append({
            "fold": i,
            "train": {"start": trs.isoformat(), "end": tre.isoformat()},
            "test": {"start": tes.isoformat(), "end": tee.isoformat()},
            "train_ranking": [{
                "strategy": n,
                "train_sharpe": train[n]["portfolio"]["sharpe_ratio_rf0"],
                "train_cagr": train[n]["portfolio"]["annualized_return_percent"],
                "train_active": train[n]["active_annualized_return_percent"],
            } for n in ranking],
            "selected_strategy": selected,
            "test_result": {
                **_metrics_brief(test_sel["portfolio"]),
                "active_annualized_return_percent": test_sel["active_annualized_return_percent"],
            },
        })

    # Adaptive OOS curve stitched across folds.
    adaptive_dates, adaptive_values = stitch(oos_segments)
    adaptive = bt.window_metrics(adaptive_dates, adaptive_values)

    # Reference: each fixed strategy over the combined test span, plus benchmark.
    combined_start, combined_end = splits[0][2], splits[-1][3]
    reference: dict[str, Any] = {}
    benchmark_metrics: dict[str, Any] | None = None
    for name, w in strategies.items():
        r = evaluate(data, w, combined_start, combined_end, top_n, frictions, rebalance_freq=rebalance_freq)
        reference[name] = {
            **_metrics_brief(r["portfolio"]),
            "active_annualized_return_percent": r["active_annualized_return_percent"],
        }
        if benchmark_metrics is None:
            benchmark_metrics = r["benchmark"]

    bench_ann = benchmark_metrics["annualized_return_percent"]
    adaptive_active = round(adaptive["annualized_return_percent"] - bench_ann, 2)
    hit = sum(1 for f in folds_out if f["test_result"]["active_annualized_return_percent"] > 0)

    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "framework": "a_share_walk_forward_validation",
        "window": {"start": gstart.isoformat(), "end": gend.isoformat()},
        "config": {
            "scheme": "anchored_expanding",
            "min_train_years": min_train_years,
            "test_years": test_years,
            "objective": objective,
            "top_n": top_n,
            "strategies": list(strategies),
            "frictions": frictions,
            "universe": {"source": data.get("source", "?"),
                         "usable": len([c for c in data["codes"] if data["prices"].get(c)])},
        },
        "folds": folds_out,
        "oos_summary": {
            "combined_test_window": {"start": combined_start.isoformat(), "end": combined_end.isoformat()},
            "n_folds": len(folds_out),
            "adaptive_selection": {
                **_metrics_brief(adaptive),
                "active_annualized_return_percent": adaptive_active,
            },
            "benchmark": _metrics_brief(benchmark_metrics),
            "reference_fixed_strategies": reference,
            "selection_beat_benchmark_folds": f"{hit}/{len(folds_out)}",
        },
        "verdict": _verdict(adaptive_active, reference, folds_out),
    }


def _verdict(adaptive_active: float, reference: dict[str, Any], folds: list[dict[str, Any]]) -> str:
    dt = reference.get("diagnostic_tilted", {}).get("active_annualized_return_percent")
    parts = [f"自适应选择 OOS 超额 {adaptive_active:+.2f}%/年"]
    if dt is not None:
        parts.append(f"固定 diagnostic_tilted OOS 超额 {dt:+.2f}%/年")
    picks = {f["selected_strategy"] for f in folds}
    parts.append(f"各折选中: {', '.join(f['selected_strategy'] for f in folds)}")
    parts.append("稳定选中同一策略" if len(picks) == 1 else "选择在折间漂移(过拟合/不稳定信号)")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
def _self_test() -> None:
    # 1) anchored_splits: chronological, non-overlapping, train precedes test.
    folds = anchored_splits(date(2021, 1, 1), date(2026, 7, 8), 3, 1)
    assert folds, "should produce folds"
    for trs, tre, tes, tee in folds:
        assert trs < tre < tes < tee, ("bad fold order", trs, tre, tes, tee)
        assert tes > tre, "test must start after train ends (no overlap)"
    for a, b in zip(folds, folds[1:]):
        assert b[0] == folds[0][0], "anchored: train_start fixed"
        assert b[1] >= a[3], "train expands to previous test_end"
    print(f"[PASS] anchored_splits: {len(folds)} 折, 训练锚定/扩张, 测试无重叠")

    # 2) add_years Feb-29 guard.
    assert add_years(date(2024, 2, 29), 1) == date(2025, 2, 28)
    print("[PASS] add_years 处理闰日 2/29 → 2/28")

    # 3) stitch chains segments continuously.
    seg1 = [["2024-01-01", 1.0], ["2024-06-30", 1.1]]
    seg2 = [["2025-01-01", 1.0], ["2025-06-30", 1.2]]
    d, v = stitch([seg1, seg2])
    assert v == [1.0, 1.1, 1.1, round(1.1 * 1.2, 6)], v
    assert d[0] == date(2024, 1, 1) and d[-1] == date(2025, 6, 30)
    print(f"[PASS] stitch: 两段拼接连续, 末值 {v[-1]} (=1.1×1.2)")

    # 4) objective selection picks train argmax.
    mock = {
        "a": {"portfolio": {"sharpe_ratio_rf0": 0.5, "annualized_return_percent": 3.0}, "active_annualized_return_percent": -1.0},
        "b": {"portfolio": {"sharpe_ratio_rf0": 1.2, "annualized_return_percent": 1.0}, "active_annualized_return_percent": 2.0},
    }
    assert max(mock, key=lambda n: _objective_value(mock[n], "sharpe")) == "b"
    assert max(mock, key=lambda n: _objective_value(mock[n], "cagr")) == "a"
    assert max(mock, key=lambda n: _objective_value(mock[n], "active")) == "b"
    print("[PASS] 目标函数 sharpe/cagr/active 各自取训练窗 argmax")

    print("ALL PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward OOS validation (A-share PIT).")
    parser.add_argument("--self-test", action="store_true", help="Offline logic test (no data).")
    parser.add_argument("--index", default="000300", help="Universe index code (default 沪深300).")
    parser.add_argument("--benchmark", default="880008",
                        help="Benchmark index code: 880008, 000300, 000905, etc. (default 880008 全A等权).")
    parser.add_argument("--full-market", action="store_true", help="Use full-A universe.")
    parser.add_argument("--skip-pe", action="store_true", help="Drop the PE-based value factor.")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default=None, help="Default: last benchmark date.")
    parser.add_argument("--min-train-years", type=int, default=3)
    parser.add_argument("--test-years", type=int, default=1)
    parser.add_argument("--objective", default="sharpe", choices=list(OBJECTIVE_KEY),
                        help="Train-window selection objective (default sharpe).")
    parser.add_argument("--price-start", default="20180101")
    parser.add_argument("--fin-start-year", type=int, default=2018)
    parser.add_argument("--commission-bps", type=float, default=2.5)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--max-stale-days", type=int, default=7)
    parser.add_argument("--rebalance-freq", default="quarterly",
                        help="Rebalance frequency: quarterly, monthly, weekly, or Nd (e.g. 2d, 5d).")
    parser.add_argument("--data-cache-dir", default=cn.CACHE_DIR_DEFAULT)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output", help="Optional path to write JSON result.")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return

    cache_dir = args.data_cache_dir or None
    gstart = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    data = bt.load_all_data(
        args.index, args.price_start, cache_dir, args.refresh, args.fin_start_year, end.year,
        full_market=args.full_market, skip_pe=args.skip_pe, benchmark_code=args.benchmark,
    )
    data["source"] = "full_market" if args.full_market else args.index
    # Clamp the global end to the last date the data actually covers.
    last_bench = date.fromisoformat(data["benchmark"][-1]["date"])
    gend = min(end, last_bench)

    frictions = {
        "commission_bps": args.commission_bps,
        "slippage_bps": args.slippage_bps,
        "max_stale_days": args.max_stale_days,
    }
    result = walk_forward(
        data, bt.STRATEGY_PROFILES, gstart, gend,
        args.min_train_years, args.test_years, args.top_n, frictions, args.objective,
        rebalance_freq=args.rebalance_freq,
    )

    print(json.dumps(result["oos_summary"], ensure_ascii=False, indent=2))
    print("\nVERDICT:", result["verdict"])
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[written] {args.output}")


if __name__ == "__main__":
    main()
