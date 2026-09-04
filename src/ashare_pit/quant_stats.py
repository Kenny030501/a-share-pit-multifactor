#!/usr/bin/env python3
"""因子时间序列的稳健统计工具。"""

from __future__ import annotations

import math
import random
import statistics
from typing import Iterable

from scipy.stats import t as student_t


def automatic_newey_west_lag(n: int) -> int:
    """按样本长度选择 Newey-West 最大滞后阶数。"""
    if n < 3:
        return 0
    return min(n - 1, max(1, int(4 * (n / 100.0) ** (2 / 9))))


def newey_west_mean_test(values: Iterable[float], lags: int | None = None) -> dict[str, float | int | None]:
    """对序列均值执行 Newey-West HAC t 检验，修正序列相关造成的标准误偏低。"""
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if n < 2:
        return {"n": n, "lags": 0, "standard_error": None, "t_stat": None, "p_value": None}

    mean = statistics.mean(xs)
    centered = [x - mean for x in xs]
    used_lags = automatic_newey_west_lag(n) if lags is None else max(0, min(int(lags), n - 1))
    long_run_variance = sum(x * x for x in centered) / n
    for lag in range(1, used_lags + 1):
        covariance = sum(centered[i] * centered[i - lag] for i in range(lag, n)) / n
        weight = 1.0 - lag / (used_lags + 1.0)
        long_run_variance += 2.0 * weight * covariance

    variance_of_mean = max(long_run_variance / n, 0.0)
    standard_error = math.sqrt(variance_of_mean)
    if standard_error <= 1e-15:
        t_stat = math.inf if mean else 0.0
        p_value = 0.0 if mean else 1.0
    else:
        t_stat = mean / standard_error
        p_value = float(2.0 * student_t.sf(abs(t_stat), df=n - 1))
    return {
        "n": n,
        "lags": used_lags,
        "standard_error": round(standard_error, 8),
        "t_stat": round(t_stat, 4) if math.isfinite(t_stat) else t_stat,
        "p_value": round(p_value, 8),
    }


def moving_block_bootstrap_mean_ci(
    values: Iterable[float],
    confidence: float = 0.95,
    block_size: int | None = None,
    samples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int | list[float] | None]:
    """用循环移动区块 bootstrap 构造均值置信区间，保留局部时间依赖。"""
    xs = [float(x) for x in values if math.isfinite(float(x))]
    n = len(xs)
    if n < 2 or samples < 100:
        return {"block_size": 0, "samples": samples, "confidence": confidence, "ci": None}
    block = block_size or max(2, round(n ** (1 / 3)))
    block = min(max(1, int(block)), n)
    rng = random.Random(seed)
    means: list[float] = []
    blocks_needed = math.ceil(n / block)
    for _ in range(samples):
        draw: list[float] = []
        for _ in range(blocks_needed):
            start = rng.randrange(n)
            draw.extend(xs[(start + offset) % n] for offset in range(block))
        means.append(statistics.mean(draw[:n]))
    means.sort()
    tail = (1.0 - confidence) / 2.0
    low_idx = max(0, min(samples - 1, int(math.floor(tail * samples))))
    high_idx = max(0, min(samples - 1, int(math.ceil((1.0 - tail) * samples)) - 1))
    return {
        "block_size": block,
        "samples": samples,
        "confidence": confidence,
        "ci": [round(means[low_idx], 6), round(means[high_idx], 6)],
    }


def benjamini_hochberg(p_values: dict[str, float], alpha: float = 0.05) -> dict[str, dict[str, float | bool]]:
    """对一组 p 值执行 Benjamini-Hochberg 假发现率校正。"""
    valid = [(name, min(max(float(p), 0.0), 1.0)) for name, p in p_values.items() if math.isfinite(float(p))]
    ordered = sorted(valid, key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[str, float] = {}
    running = 1.0
    for rank, (name, p_value) in reversed(list(enumerate(ordered, 1))):
        running = min(running, p_value * m / rank)
        adjusted[name] = min(running, 1.0)
    return {
        name: {"q_value": round(adjusted[name], 8), "significant": adjusted[name] <= alpha}
        for name, _ in valid
    }


def summarize_signal(values: Iterable[float]) -> dict[str, object]:
    """统一输出 HAC 检验与区块 bootstrap 结果。"""
    xs = [float(x) for x in values if math.isfinite(float(x))]
    return {
        "newey_west": newey_west_mean_test(xs),
        "moving_block_bootstrap": moving_block_bootstrap_mean_ci(xs),
    }
