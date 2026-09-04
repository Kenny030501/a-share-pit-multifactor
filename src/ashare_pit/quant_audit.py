#!/usr/bin/env python3
"""无需原始行情库的量化研究证据审计。

从已提交的紧凑 JSON 重建因子显著性、样本外泛化和交易执行三条证据链；旧版
汇总缺少原始 IC 序列时会明确降级为按期独立近似，不伪装成 HAC 结果。
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scipy.stats import t as student_t

from . import quant_stats as qstats

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required result is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _legacy_mean_test(item: dict[str, Any]) -> dict[str, Any]:
    n = int(item.get("n_periods", 0))
    mean = item.get("mean_ic")
    std = item.get("std_ic")
    if n < 2 or mean is None or not std:
        return {"t_stat": None, "p_value": None, "method": "insufficient_data"}
    standard_error = float(std) / math.sqrt(n)
    t_stat = float(mean) / standard_error
    p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 8),
        "method": "independent_period_approximation_from_legacy_summary",
    }


def audit_factors(data: dict[str, Any], source: str) -> list[dict[str, Any]]:
    """审计一组确认性因子，并统一补充多重检验结果。"""
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for factor, item in data["ic_summary"].items():
        if item.get("n_periods", 0) < 2:
            continue
        if item.get("newey_west", {}).get("p_value") is not None:
            test = {
                "t_stat": item["newey_west"]["t_stat"],
                "p_value": item["newey_west"]["p_value"],
                "method": "newey_west_hac",
            }
        else:
            test = _legacy_mean_test(item)
        if test["p_value"] is not None:
            p_values[factor] = float(test["p_value"])
        quantile = data.get("quantile_backtest", {}).get(factor, {})
        rows.append({
            "source": source,
            "factor": factor,
            "mean_ic": item.get("mean_ic"),
            "ir": item.get("ir"),
            "n_periods": item.get("n_periods"),
            "t_stat": test["t_stat"],
            "p_value": test["p_value"],
            "test_method": test["method"],
            "monotonic": quantile.get("monotonic"),
            "long_short_spread_annualized_percent": quantile.get("long_short_spread_annualized_percent"),
        })
    adjusted = qstats.benjamini_hochberg(p_values)
    for row in rows:
        result = adjusted.get(row["factor"])
        row["q_value_bh"] = result["q_value"] if result else None
        row["significant_fdr_5pct"] = result["significant"] if result else False
    return rows


def audit_legacy_factor_screen(data: dict[str, Any]) -> dict[str, Any]:
    """对旧版大规模因子筛选执行可复现的 BH-FDR 近似审计。"""
    rows: list[dict[str, Any]] = []
    p_values: dict[str, float] = {}
    for item in data["all_factors"]:
        test = _legacy_mean_test(item)
        if test["p_value"] is None:
            continue
        factor = item["factor"]
        p_values[factor] = float(test["p_value"])
        rows.append({
            "factor": factor,
            "mean_ic": item["mean_ic"],
            "ir": item["ir"],
            "n_periods": item["n_periods"],
            "t_stat": test["t_stat"],
            "p_value": test["p_value"],
        })
    adjusted = qstats.benjamini_hochberg(p_values)
    for row in rows:
        result = adjusted[row["factor"]]
        row["q_value_bh"] = result["q_value"]
        row["significant_fdr_5pct"] = result["significant"]
    rows.sort(key=lambda row: (row["q_value_bh"], -abs(row["mean_ic"])))
    significant = [row for row in rows if row["significant_fdr_5pct"]]
    return {
        "factors_with_minimum_history": len(rows),
        "significant_fdr_5pct_independence_approximation": len(significant),
        "top_corrected_candidates": significant[:15],
        "method_caveat": "Legacy summary approximation; rerun the screen for HAC and block-bootstrap inference.",
    }


def audit_walk_forward(data: dict[str, Any]) -> dict[str, Any]:
    """计算每折训练到测试的泛化缺口和跑赢基准次数。"""
    folds: list[dict[str, Any]] = []
    for fold in data["folds"]:
        selected = fold["selected_strategy"]
        train = next(x for x in fold["train_ranking"] if x["strategy"] == selected)
        test_active = float(fold["test_result"]["active_annualized_return_percent"])
        train_active = float(train["train_active"])
        folds.append({
            "fold": fold["fold"],
            "selected_strategy": selected,
            "train_active_annualized_percent": train_active,
            "test_active_annualized_percent": test_active,
            "generalization_gap_percent": round(test_active - train_active, 2),
            "beat_benchmark": test_active > 0,
        })
    reference = data["oos_summary"]["reference_fixed_strategies"]
    best_fixed_name = max(reference, key=lambda name: reference[name]["active_annualized_return_percent"])
    return {
        "folds": folds,
        "mean_generalization_gap_percent": round(statistics.mean(x["generalization_gap_percent"] for x in folds), 2),
        "beat_benchmark_folds": f"{sum(x['beat_benchmark'] for x in folds)}/{len(folds)}",
        "adaptive_oos": data["oos_summary"]["adaptive_selection"],
        "best_fixed_reference": {"strategy": best_fixed_name, **reference[best_fixed_name]},
        "best_fixed_reference_caveat": "A fixed reference is descriptive unless it was frozen before the test window.",
    }


def _execution_case(data: dict[str, Any]) -> dict[str, Any]:
    return {
        "turnover_cap": data["config"].get("turnover_cap"),
        "frictionless_active_annualized_percent": data["frictionless"]["active_annualized_return_percent"],
        "realistic_active_annualized_percent": data["realistic"]["active_annualized_return_percent"],
        "annualized_return_decay_percent": data["annualized_return_decay_percent"],
        "cumulative_explicit_cost_drag_fraction": data["realistic"]["total_cost_drag_fraction"],
    }


def build_audit(results_dir: Path) -> dict[str, Any]:
    fundamental = _load(results_dir / "cn_factor_diagnostics_fullmarket.json")
    gtja = _load(results_dir / "cn_factor_diagnostics_gtja_t2_full.json")
    walk = _load(results_dir / "cn_walkforward.json")
    controlled = _load(results_dir / "cn_backtest_gtja_strict_t2.json")
    uncontrolled = _load(results_dir / "cn_backtest_gtja_t2_noturnover.json")
    gtja191_screen = _load(results_dir / "cn_gtja191_ic_screening.json")
    factor_rows = audit_factors(fundamental, "fundamental_2021_2026") + audit_factors(gtja, "gtja_t2_2012_2017")
    execution = {
        "turnover_controlled": _execution_case(controlled),
        "turnover_uncontrolled": _execution_case(uncontrolled),
    }
    execution["realistic_active_improvement_percent"] = round(
        execution["turnover_controlled"]["realistic_active_annualized_percent"]
        - execution["turnover_uncontrolled"]["realistic_active_annualized_percent"],
        2,
    )
    return {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "framework": "quant_research_evidence_audit",
        "factor_significance": factor_rows,
        "gtja191_multiple_testing_audit": audit_legacy_factor_screen(gtja191_screen),
        "walk_forward": audit_walk_forward(walk),
        "execution": execution,
        "capacity_status": {
            "status": "engine_ready_not_backfilled",
            "reason": "Saved public runs predate the AUM-aware market-impact module.",
            "rerun_flag": "--portfolio-aum-millions",
        },
        "evidence_boundaries": [
            "旧版 IC 汇总只能进行按期独立的 t 检验近似；重新运行会直接输出 Newey-West HAC、区块 bootstrap 区间和 BH-FDR。",
            "策略股票池使用当前 A 股证券名单，因此仍有退市股票缺失造成的幸存者偏差。",
            "财务数据使用法定披露截止日作为保守可用日，而不是逐份交易所公告的真实时间戳。",
            "固定 reversal_lottery 参考只有在测试窗前冻结规格时，才能视为无偏样本外证据。",
            "容量输出来自模型，投入生产前需要用订单级真实滑点进行校准。",
        ],
    }


def _bar_svg(title: str, groups: list[str], series: list[tuple[str, list[float], str]], path: Path) -> None:
    width, height = 940, 480
    left, right, top, bottom = 90, 30, 70, 90
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [value for _, vals, _ in series for value in vals]
    low = min(0.0, min(values))
    high = max(0.0, max(values))
    pad = max((high - low) * 0.15, 1.0)
    low, high = low - pad, high + pad

    def y(value: float) -> float:
        return top + (high - value) / (high - low) * plot_h

    group_w = plot_w / max(len(groups), 1)
    bar_w = min(54.0, group_w / (len(series) + 1))
    zero_y = y(0.0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<defs><pattern id="diagonal" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#0072B2"/><path d="M-2 2 L2 -2 M0 8 L8 0 M6 10 L10 6" stroke="#FFFFFF" stroke-width="2"/></pattern><pattern id="dots" width="8" height="8" patternUnits="userSpaceOnUse"><rect width="8" height="8" fill="#E69F00"/><circle cx="2" cy="2" r="1.5" fill="#111111"/></pattern></defs>',
        '<rect width="100%" height="100%" fill="#FFFFFF"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="Arial,sans-serif" font-size="21" font-weight="700">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{zero_y:.2f}" x2="{width-right}" y2="{zero_y:.2f}" stroke="#222222" stroke-width="1.5"/>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.2f}" x2="{width-right}" y2="{yy:.2f}" stroke="#D9D9D9" stroke-width="1"/>')
        parts.append(f'<text x="{left-10}" y="{yy+5:.2f}" text-anchor="end" font-family="Arial,sans-serif" font-size="13">{value:.1f}</text>')
    for group_index, group in enumerate(groups):
        center = left + group_w * (group_index + 0.5)
        start = center - bar_w * len(series) / 2
        parts.append(f'<text x="{center:.2f}" y="{height-bottom+32}" text-anchor="middle" font-family="Arial,sans-serif" font-size="13">{html.escape(group)}</text>')
        for series_index, (_, vals, fill) in enumerate(series):
            value = vals[group_index]
            x = start + series_index * bar_w
            yy = y(value)
            rect_y, rect_h = min(yy, zero_y), abs(zero_y - yy)
            parts.append(f'<rect x="{x:.2f}" y="{rect_y:.2f}" width="{bar_w-5:.2f}" height="{max(rect_h,1):.2f}" fill="{fill}" stroke="#111111"/>')
            label_y = yy - 7 if value >= 0 else yy + 17
            parts.append(f'<text x="{x+(bar_w-5)/2:.2f}" y="{label_y:.2f}" text-anchor="middle" font-family="Arial,sans-serif" font-size="12" font-weight="700">{value:+.2f}</text>')
    legend_x = left
    for name, _, fill in series:
        parts.append(f'<rect x="{legend_x}" y="{height-35}" width="22" height="15" fill="{fill}" stroke="#111111"/>')
        parts.append(f'<text x="{legend_x+30}" y="{height-22}" font-family="Arial,sans-serif" font-size="13">{html.escape(name)}</text>')
        legend_x += 190
    parts.append('</svg>')
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_assets(audit: dict[str, Any], assets_dir: Path) -> list[Path]:
    assets_dir.mkdir(parents=True, exist_ok=True)
    gtja = [x for x in audit["factor_significance"] if x["source"] == "gtja_t2_2012_2017"]
    factor_path = assets_dir / "factor-ic-significance.svg"
    _bar_svg(
        "GTJA short-horizon factors: mean rank IC x 100",
        [x["factor"].replace("_", " ") for x in gtja],
        [("Mean IC", [100 * float(x["mean_ic"]) for x in gtja], "url(#diagonal)")],
        factor_path,
    )
    folds = audit["walk_forward"]["folds"]
    oos_path = assets_dir / "walk-forward-generalization.svg"
    _bar_svg(
        "Walk-forward active return: train versus unseen test",
        [f"Fold {x['fold']}" for x in folds],
        [
            ("Train active %", [x["train_active_annualized_percent"] for x in folds], "url(#diagonal)"),
            ("Test active %", [x["test_active_annualized_percent"] for x in folds], "url(#dots)"),
        ],
        oos_path,
    )
    execution = audit["execution"]
    cost_path = assets_dir / "turnover-cost-control.svg"
    _bar_svg(
        "T+2 active return before and after explicit trading frictions",
        ["Turnover cap 0.15", "No effective cap"],
        [
            ("Frictionless active %", [
                execution["turnover_controlled"]["frictionless_active_annualized_percent"],
                execution["turnover_uncontrolled"]["frictionless_active_annualized_percent"],
            ], "url(#diagonal)"),
            ("Realistic active %", [
                execution["turnover_controlled"]["realistic_active_annualized_percent"],
                execution["turnover_uncontrolled"]["realistic_active_annualized_percent"],
            ], "url(#dots)"),
        ],
        cost_path,
    )
    return [factor_path, oos_path, cost_path]


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "是" if value else "否"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_markdown(audit: dict[str, Any], asset_prefix: str = "assets") -> str:
    walk = audit["walk_forward"]
    execution = audit["execution"]
    lines = [
        "# 量化研究证据审计",
        "",
        "> 目的：把因子显著性、样本外泛化和交易执行放在同一张证据链中；不以样本内收益替代可交易结论。",
        "",
        "## 一页结论",
        "",
        f"- 锚定扩张 walk-forward 中，自适应选择仅有 **{walk['beat_benchmark_folds']}** 折跑赢基准，平均泛化缺口为 **{walk['mean_generalization_gap_percent']:+.2f}%/年**。",
        f"- T+2 策略加入换手上限后，真实超额相对无有效上限方案改善 **{execution['realistic_active_improvement_percent']:+.2f}%/年**。",
        "- 现有公开 IC 文件是旧版汇总，显著性只能从均值、标准差和期数近似；重新运行诊断后会直接输出 Newey-West、区块 bootstrap 和 BH-FDR。",
        "",
        f"![GTJA 因子 IC]({asset_prefix}/factor-ic-significance.svg)",
        "",
        f"![样本外泛化]({asset_prefix}/walk-forward-generalization.svg)",
        "",
        f"![换手与成本]({asset_prefix}/turnover-cost-control.svg)",
        "",
        "## 因子统计审计",
        "",
        "| 样本 | 因子 | Mean IC | IR | 期数 | t | p | BH q | FDR 5% | 分层单调 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in audit["factor_significance"]:
        lines.append(
            f"| {row['source']} | `{row['factor']}` | {_fmt(row['mean_ic'], 4)} | {_fmt(row['ir'])} | "
            f"{row['n_periods']} | {_fmt(row['t_stat'])} | {_fmt(row['p_value'], 4)} | "
            f"{_fmt(row['q_value_bh'], 4)} | {_fmt(row['significant_fdr_5pct'])} | {_fmt(row['monotonic'])} |"
        )
    screen = audit["gtja191_multiple_testing_audit"]
    lines.extend([
        "",
        "旧版结果的 t/p 假设期与期独立，只用于审计历史输出；正式重跑以 HAC 与 block bootstrap 为准。",
        "",
        "## GTJA191 多重筛选审计",
        "",
        f"旧筛选中有 **{screen['factors_with_minimum_history']}** 个因子满足至少 10 期有效 IC；在偏乐观的按期独立近似下，**{screen['significant_fdr_5pct_independence_approximation']}** 个通过 BH-FDR 5%。这些只能作为下一轮预注册候选。",
        "",
        "| 因子 | Mean IC | IR | t | p | BH q |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in screen["top_corrected_candidates"][:10]:
        lines.append(
            f"| `{row['factor']}` | {_fmt(row['mean_ic'], 4)} | {_fmt(row['ir'])} | "
            f"{_fmt(row['t_stat'])} | {_fmt(row['p_value'], 4)} | {_fmt(row['q_value_bh'], 4)} |"
        )
    lines.extend([
        "",
        "## Walk-forward 泛化",
        "",
        "| 折 | 训练窗选中策略 | 训练超额% | 测试超额% | 泛化缺口% | 跑赢基准 |",
        "|---:|---|---:|---:|---:|---|",
    ])
    for fold in walk["folds"]:
        lines.append(
            f"| {fold['fold']} | `{fold['selected_strategy']}` | {fold['train_active_annualized_percent']:+.2f} | "
            f"{fold['test_active_annualized_percent']:+.2f} | {fold['generalization_gap_percent']:+.2f} | "
            f"{_fmt(fold['beat_benchmark'])} |"
        )
    best = walk["best_fixed_reference"]
    lines.extend([
        "",
        f"固定策略参考中 `{best['strategy']}` 的 OOS 超额为 **{best['active_annualized_return_percent']:+.2f}%/年**；若该规格并非在测试窗前冻结，只能视为描述性参考，不能当作无偏样本外结论。",
        "",
        "## 交易执行与容量",
        "",
        "| 设定 | 无摩擦超额% | 真实超额% | 年化衰减% | 累计显式成本拖累 |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, key in [("换手上限 0.15", "turnover_controlled"), ("无有效换手上限", "turnover_uncontrolled")]:
        row = execution[key]
        lines.append(
            f"| {label} | {row['frictionless_active_annualized_percent']:+.2f} | "
            f"{row['realistic_active_annualized_percent']:+.2f} | {row['annualized_return_decay_percent']:.2f} | "
            f"{row['cumulative_explicit_cost_drag_fraction']:.3f} |"
        )
    lines.extend([
        "",
        "回测引擎现已支持 `--portfolio-aum-millions`：假设一日执行，按 63 日 ADV、63 日波动率和平方根冲击模型输出参与率及容量拖累。旧公开结果尚未回填 AUM 场景，因此不伪造容量数字。",
        "",
        "## 证据边界",
        "",
    ])
    lines.extend(f"- {item}" for item in audit["evidence_boundaries"])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit saved factor, OOS and execution evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output", help="Optional JSON output path.")
    parser.add_argument("--markdown", help="Optional Markdown report path.")
    parser.add_argument("--assets-dir", help="Optional directory for accessible SVG charts.")
    args = parser.parse_args()

    audit = build_audit(Path(args.results_dir))
    if args.output:
        Path(args.output).write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.assets_dir:
        write_assets(audit, Path(args.assets_dir))
    if args.markdown:
        asset_prefix = "assets"
        if args.assets_dir:
            asset_prefix = os.path.relpath(Path(args.assets_dir), Path(args.markdown).parent).replace(os.sep, "/")
        Path(args.markdown).write_text(render_markdown(audit, asset_prefix), encoding="utf-8")
    print(json.dumps({
        "factors_audited": len(audit["factor_significance"]),
        "walk_forward_hit_rate": audit["walk_forward"]["beat_benchmark_folds"],
        "mean_generalization_gap_percent": audit["walk_forward"]["mean_generalization_gap_percent"],
        "turnover_control_improvement_percent": audit["execution"]["realistic_active_improvement_percent"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
