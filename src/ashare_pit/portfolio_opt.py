#!/usr/bin/env python3
"""仅做多、行业与规模中性的约束组合优化。

用优化权重替代简单的“排序后取 Top-N 等权”：在满仓、仅做多、单票上限、行业中性
和规模中性约束下最大化因子 alpha，并可限制单次换手。

    min_w   −αᵀw + (ridge/2)·wᵀw
    s.t.    Σ w = 1
            Σ_{i∈g} w_i = Σ_{i∈g} b_i     for each industry g   (industry-neutral)
            sᵀw = sᵀb                       (size-neutral; s = standardized log-mcap)
            0 ≤ w_i ≤ cap

其中 b 是基准权重，α 是综合因子分。L2 ridge 是简化的风险代理，用于分散权重并
避免线性目标落到极端角点；基准权重本身是可行解。带换手约束时优先使用 cvxpy，
其余情况使用 scipy SLSQP；求解失败时回退基准权重，防止中途生成不可解释持仓。
"""

from __future__ import annotations

import numpy as np


def _standardize(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else v - v.mean()


def _qp_turnover(
    a: np.ndarray, A: np.ndarray, t: np.ndarray, cap: float,
    wp: np.ndarray, turnover_cap: float, ridge: float,
) -> np.ndarray | None:
    """QP: min −aᵀw + (ridge/2)‖w‖²  s.t. Aw=t, 0≤w≤cap, and ‖w−wp‖₁ ≤ turnover_cap
    when holding a prior book. The ridge (a diagonal risk-model proxy) is what keeps
    the book diversified — a *linear* objective would seek a sparse vertex and, under
    a tight turnover cap, ratchet down to a handful of names over time. The turnover
    cap is the paper's "严格控制单次换仓比例" that lets T+2 survive real costs.

    Solved with CLARABEL (interior-point; faster and far more accurate than OSQP on the
    L1-turnover problem here), OSQP as fallback. If the turnover-capped problem is
    infeasible (drift too large for the budget), it falls back to a FULL neutral rebuild
    (no turnover cap) rather than holding a degenerate book — paying turnover that period
    but staying diversified and neutral. Returns the weight vector, or None on failure."""
    import cvxpy as cp  # local import: cvxpy optional until used

    n = len(a)
    w = cp.Variable(n)
    obj = cp.Minimize(-a @ w + 0.5 * ridge * cp.sum_squares(w))
    base = [A @ w == t, w >= 0, w <= cap]
    cons = base + ([cp.norm1(w - wp) <= turnover_cap] if wp.sum() > 1e-9 else [])
    for constraints in (cons, base):  # try turnover-capped, then full rebuild
        for solver in (cp.CLARABEL, cp.OSQP):
            try:
                cp.Problem(obj, constraints).solve(solver=solver, verbose=False)
            except Exception:
                continue
            if w.value is not None:
                return np.asarray(w.value, float).flatten()
    return None


def optimize_weights(
    codes: list[str],
    alpha: dict[str, float],
    bench_w: dict[str, float],
    industry: dict[str, str] | None = None,
    size: dict[str, float] | None = None,
    w_cap: float = 0.03,
    ridge: float = 1.0,
    neutralize: tuple[str, ...] = ("industry", "size"),
    extra_styles: dict[str, dict[str, float]] | None = None,
    w_prev: dict[str, float] | None = None,
    min_weight: float = 1e-6,
    turnover_cap: float | None = None,
) -> dict[str, float]:
    """Return {code: weight} for the neutral, alpha-maximising long-only portfolio.

    turnover_cap set → the paper's setup: ridge-QP (max expected return − risk proxy)
    with a hard single-period turnover cap ‖w−w_prev‖₁ ≤ turnover_cap, solved by OSQP;
    the ridge keeps it diversified so a tight cap can't ratchet it down to a few names.
    Otherwise a small-ridge QP (SLSQP)."""
    n = len(codes)
    if n == 0:
        return {}
    a = _standardize(np.array([alpha.get(c, 0.0) for c in codes], float))

    b = np.array([max(bench_w.get(c, 0.0), 0.0) for c in codes], float)
    b = b / b.sum() if b.sum() > 0 else np.ones(n) / n
    cap = max(w_cap, float(b.max()) + 1e-9)  # keep b feasible → problem feasible

    # Linear equality rows  A w = t  (all constraints here are equalities).
    rows: list[np.ndarray] = [np.ones(n)]
    targets: list[float] = [1.0]

    if "industry" in neutralize and industry:
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(codes):
            g = industry.get(c)
            if g is not None:
                groups.setdefault(g, []).append(i)
        for idx in groups.values():
            row = np.zeros(n)
            row[idx] = 1.0
            rows.append(row)
            targets.append(float(b[idx].sum()))

    if "size" in neutralize and size:
        s = _standardize(np.array([size.get(c, 0.0) for c in codes], float))
        rows.append(s)
        targets.append(float(s @ b))

    for vec in (extra_styles or {}).values():
        v = _standardize(np.array([vec.get(c, 0.0) for c in codes], float))
        rows.append(v)
        targets.append(float(v @ b))

    A = np.vstack(rows)
    t = np.array(targets, float)

    if turnover_cap is not None:
        wp = np.array([(w_prev or {}).get(c, 0.0) for c in codes], float)
        w = _qp_turnover(a, A, t, cap, wp, turnover_cap, ridge)
        if w is None:  # even the full rebuild failed → hold current book
            w = wp if wp.sum() > 1e-9 else b
        w = np.clip(w, 0.0, cap)
        w = w / w.sum() if w.sum() > 0 else b
        return {c: float(w[i]) for i, c in enumerate(codes) if w[i] > min_weight}

    from scipy.optimize import minimize  # local import: scipy optional until used
    cons = [{
        "type": "eq",
        "fun": lambda w, A=A, t=t: A @ w - t,
        "jac": lambda w, A=A: A,
    }]
    bounds = [(0.0, cap)] * n

    w0 = np.array([(w_prev or {}).get(c, b[i]) for i, c in enumerate(codes)], float)
    w0 = np.clip(w0, 0.0, cap)
    w0 = w0 / w0.sum() if w0.sum() > 0 else b.copy()

    def fun(w: np.ndarray) -> float:
        return float(-a @ w + 0.5 * ridge * (w @ w))

    def jac(w: np.ndarray) -> np.ndarray:
        return -a + ridge * w

    res = minimize(fun, w0, jac=jac, bounds=bounds, constraints=cons,
                   method="SLSQP", options={"maxiter": 300, "ftol": 1e-10})
    w = res.x if res.success else b
    w = np.clip(w, 0.0, cap)
    w = w / w.sum() if w.sum() > 0 else b
    return {c: float(w[i]) for i, c in enumerate(codes) if w[i] > min_weight}


def active_exposures(
    weights: dict[str, float],
    bench_w: dict[str, float],
    industry: dict[str, str] | None = None,
    size: dict[str, float] | None = None,
) -> dict[str, float]:
    """Diagnostics: max |industry active weight| and |size active exposure| (on the
    same standardized log-mcap axis the constraint uses)."""
    codes = list({*weights, *bench_w})
    b = np.array([max(bench_w.get(c, 0.0), 0.0) for c in codes], float)
    b = b / b.sum() if b.sum() > 0 else np.ones(len(codes)) / len(codes)
    w = np.array([weights.get(c, 0.0) for c in codes], float)
    out: dict[str, float] = {"sum_w": float(w.sum())}
    if industry:
        groups: dict[str, list[int]] = {}
        for i, c in enumerate(codes):
            g = industry.get(c)
            if g is not None:
                groups.setdefault(g, []).append(i)
        out["max_abs_industry_active"] = max(
            (abs(float(w[idx].sum() - b[idx].sum())) for idx in groups.values()),
            default=0.0,
        )
    if size:
        s = _standardize(np.array([size.get(c, 0.0) for c in codes], float))
        out["size_active_exposure"] = float(s @ (w - b))
    return out


# ---------------------------------------------------------------------------
# 离线自检
# ---------------------------------------------------------------------------
def _self_test() -> None:
    codes = [f"S{i}" for i in range(8)]
    # Two industries A/B (4 each); size = log-mcap = i. alpha alternates so it is
    # NOT collinear with the size axis (a collinear alpha would be fully absorbed by
    # the size-neutral constraint, leaving zero tilt) — it keeps a component the
    # neutral constraints don't span.
    industry = {c: ("A" if i < 4 else "B") for i, c in enumerate(codes)}
    size = {c: float(i) for i, c in enumerate(codes)}
    alpha = {c: float(i % 2) for i, c in enumerate(codes)}  # 0,1,0,1,... orthogonal-ish to size
    # Cap-weighted benchmark (bigger i → bigger cap).
    raw = {c: 1.0 + i for i, c in enumerate(codes)}
    tot = sum(raw.values())
    bench = {c: raw[c] / tot for c in codes}

    w = optimize_weights(codes, alpha, bench, industry, size,
                         w_cap=0.30, ridge=1.0, neutralize=("industry", "size"))
    assert abs(sum(w.values()) - 1.0) < 1e-6, f"权重和应为1: {sum(w.values())}"
    assert all(-1e-9 <= x <= 0.30 + 1e-6 for x in w.values()), f"应满足 box: {w}"
    exp = active_exposures(w, bench, industry, size)
    assert exp["max_abs_industry_active"] < 1e-4, f"行业主动暴露应≈0: {exp}"
    assert abs(exp["size_active_exposure"]) < 1e-4, f"规模主动暴露应≈0: {exp}"
    print(f"[PASS] 优化器: Σw=1、0≤w≤cap、行业残差 {exp['max_abs_industry_active']:.2e}、"
          f"规模主动暴露 {exp['size_active_exposure']:.2e}")

    # Alpha tilt: within the neutral feasible set, higher-alpha names get more weight
    # than their benchmark share on average (net long-alpha exposure > 0).
    a_arr = _standardize(np.array([alpha[c] for c in codes]))
    b_arr = np.array([bench[c] for c in codes])
    w_arr = np.array([w.get(c, 0.0) for c in codes])
    assert a_arr @ (w_arr - b_arr) > 0, "组合应对 alpha 有正的主动暴露"
    print(f"[PASS] alpha 倾斜: 主动 alpha 暴露 {a_arr @ (w_arr - b_arr):.3f} > 0")

    # Neutralize=() → only Σw=1; pure alpha maximisation still respects cap. Use a
    # strictly monotone alpha so the top name is unique.
    mono = {c: float(i) for i, c in enumerate(codes)}  # S7 unique best
    w2 = optimize_weights(codes, mono, bench, industry, size,
                          w_cap=0.30, ridge=0.1, neutralize=())
    assert abs(sum(w2.values()) - 1.0) < 1e-6
    assert w2.get("S7", 0) >= max(w2.values()) - 1e-6, "无中性时最高alpha应权重最大"
    print(f"[PASS] 无中性: 纯 alpha 最大化, S7 权重最大 ({w2.get('S7', 0):.3f})")

    # Feasibility guard: cap below max benchmark weight is auto-raised, not infeasible.
    w3 = optimize_weights(codes, alpha, bench, industry, size, w_cap=0.001)
    assert abs(sum(w3.values()) - 1.0) < 1e-6, "cap 过小应自动抬高保持可行"
    print("[PASS] 可行性保护: cap<max(b) 时自动抬高, 解仍可行")

    # Turnover-capped ridge-QP (paper setup): first build is unconstrained; a later
    # rebalance from a prior book respects ‖w−w_prev‖₁ ≤ cap and stays neutral. OSQP
    # is a numerical solver, so residuals are ~1e-4 (not exact).
    tol = 5e-3
    w_build = optimize_weights(codes, alpha, bench, industry, size, w_cap=0.30, ridge=1.0,
                               neutralize=("industry", "size"), turnover_cap=0.20, w_prev={})
    assert abs(sum(w_build.values()) - 1.0) < tol, "首建应满仓"
    eb = active_exposures(w_build, bench, industry, size)
    assert eb["max_abs_industry_active"] < tol and abs(eb["size_active_exposure"]) < tol
    # Now step to a fresh alpha with a tight turnover cap; trades must be bounded.
    alpha2 = {c: float((i + 3) % 2) for i, c in enumerate(codes)}
    tau = 0.10
    w_step = optimize_weights(codes, alpha2, bench, industry, size, w_cap=0.30, ridge=1.0,
                              neutralize=("industry", "size"), turnover_cap=tau, w_prev=w_build)
    moved = sum(abs(w_step.get(c, 0.0) - w_build.get(c, 0.0)) for c in codes)
    assert moved <= tau + tol, f"单期换手 {moved:.4f} 应 ≤ {tau}"
    es = active_exposures(w_step, bench, industry, size)
    assert es["max_abs_industry_active"] < tol and abs(es["size_active_exposure"]) < tol
    # Ridge keeps the book diversified (no ratchet-to-a-few-names collapse).
    assert len(w_build) >= 4, f"ridge 应保持分散, 实持仓 {len(w_build)}"
    print(f"[PASS] 换手约束 QP: 首建满仓中性({len(w_build)}只); 单期换手 {moved:.3f} ≤ {tau}, 仍中性")
    print("ALL PASS")


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Industry/style-neutral portfolio optimizer.")
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        _self_test()
    else:
        p.error("Use --self-test.")


if __name__ == "__main__":
    main()
