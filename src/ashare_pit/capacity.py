#!/usr/bin/env python3
"""流动性参与率与平方根市场冲击诊断。"""

from __future__ import annotations

import math
import statistics
from typing import Any


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def estimate_rebalance_impact(
    delta_weights: dict[str, float],
    adv_by_code: dict[str, float | None],
    annual_volatility_by_code: dict[str, float | None],
    portfolio_value: float,
    impact_coefficient: float = 0.5,
) -> dict[str, Any]:
    """估算单次调仓的 ADV 参与率与净值冲击。

    冲击率采用 ``系数 × 日波动率 × sqrt(成交额 / ADV)``；缺少成交额或波动率的
    股票不会被静默填零，而是计入覆盖率缺口。
    """
    if portfolio_value <= 0:
        raise ValueError("portfolio_value must be positive")
    if impact_coefficient < 0:
        raise ValueError("impact_coefficient must be non-negative")

    gross_trade_weight = sum(abs(float(w)) for w in delta_weights.values())
    covered_trade_weight = 0.0
    impact_fraction = 0.0
    participations: list[float] = []
    missing: list[str] = []
    for code, delta in delta_weights.items():
        trade_weight = abs(float(delta))
        if trade_weight <= 0:
            continue
        adv = adv_by_code.get(code)
        annual_vol = annual_volatility_by_code.get(code)
        if not adv or adv <= 0 or annual_vol is None or annual_vol < 0:
            missing.append(code)
            continue
        participation = trade_weight * portfolio_value / adv
        daily_volatility = annual_vol / math.sqrt(252.0)
        impact_rate = impact_coefficient * daily_volatility * math.sqrt(max(participation, 0.0))
        impact_fraction += trade_weight * impact_rate
        covered_trade_weight += trade_weight
        participations.append(participation)

    coverage = covered_trade_weight / gross_trade_weight if gross_trade_weight > 0 else 1.0
    return {
        "model": "square_root_impact",
        "impact_coefficient": impact_coefficient,
        "execution_horizon_days": 1,
        "adv_window_days": 63,
        "volatility_window_days": 63,
        "portfolio_value": portfolio_value,
        "gross_trade_weight": round(gross_trade_weight, 6),
        "liquidity_coverage": round(coverage, 6),
        "estimated_nav_impact_fraction": round(impact_fraction, 8),
        "estimated_nav_impact_bps": round(impact_fraction * 10000.0, 4),
        "participation_rate": {
            "median": round(statistics.median(participations), 6) if participations else None,
            "p95": round(_percentile(participations, 0.95), 6) if participations else None,
            "max": round(max(participations), 6) if participations else None,
        },
        "missing_liquidity_names": sorted(missing),
    }


def capacity_scenarios(
    delta_weights: dict[str, float],
    adv_by_code: dict[str, float | None],
    annual_volatility_by_code: dict[str, float | None],
    portfolio_values: list[float],
    impact_coefficient: float = 0.5,
) -> list[dict[str, Any]]:
    """在相同目标权重下比较不同资产规模的容量压力。"""
    return [
        estimate_rebalance_impact(
            delta_weights,
            adv_by_code,
            annual_volatility_by_code,
            value,
            impact_coefficient,
        )
        for value in portfolio_values
    ]
