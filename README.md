# A-Share Point-in-Time Multi-Factor Backtesting Framework

[English](README.md) | [中文](README.zh-CN.md)

[![Quant research checks](https://github.com/Kenny030501/a-share-pit-multifactor/actions/workflows/ci.yml/badge.svg)](https://github.com/Kenny030501/a-share-pit-multifactor/actions/workflows/ci.yml)

An end-to-end research framework for testing cross-sectional factors across the mainland China A-share universe without using financial information before it was legally available.

The project covers data ingestion, point-in-time snapshots, robust factor inference, portfolio construction, turnover and capacity diagnostics, and expanding-window out-of-sample validation.

## What this project demonstrates

- Point-in-time handling of prices, valuation data, and financial statements.
- Cross-sectional IC/IR tests and quantile portfolio diagnostics.
- Industry- and size-neutral portfolio optimization with turnover constraints.
- Walk-forward model selection and explicit transaction costs.
- Newey-West IC inference, moving-block bootstrap intervals, and Benjamini-Hochberg multiple-testing control.
- Walk-forward validation, turnover diagnostics, and optional AUM-aware square-root market-impact estimates.
- Honest reporting of negative results, survivorship bias, and risk-model limitations.

## Research scale

- 5,527 A-share securities.
- Approximately 12.54 million daily-price rows in the local research database.
- About 6,300 lines of Python across the data, statistics, diagnostics, backtest, capacity, optimization, validation, and test modules.
- Reproduction study of short-horizon price-volume factors under a strict T+2 convention.

## Selected findings

- Six traditional factors failed the realistic benchmark test, with approximately -10% to -12% annualized excess return in the tested specification.
- The adaptive walk-forward selector beat the benchmark in only 1 of 3 folds and lost 4.21% annualized versus the benchmark over the combined test window.
- A fixed price-volume reversal reference produced +5.62% annualized excess return and 0.87 Sharpe, but it remains descriptive unless its specification was frozen before the test window.
- The strict T+2 neutralized portfolio produced +1.22% annualized net excess return versus the CSI 500 proxy after turnover controls.

These are historical research results, not expected returns or investment advice. See [the Chinese project report](reports/research/project-report-zh.md) for the full evidence and limitations.

## 60-second evidence audit

The repository includes a data-free audit that rebuilds the key evidence from the committed result files. It checks factor significance, train-to-test degradation, and the effect of turnover control without downloading the 1.6 GB research database.

```bash
make audit
```

![Walk-forward generalization gap](reports/figures/walk-forward-generalization.svg)

![Turnover and explicit trading costs](reports/figures/turnover-cost-control.svg)

![GTJA factor IC](reports/figures/factor-ic-significance.svg)

The full audit is in [Quant research evidence audit](reports/audit/quant-research-audit.md).
For an interview-ready explanation of the design decisions and evidence boundaries, see [Quant interview brief](reports/interview/quant-interview-brief.md).

## Architecture

```text
AkShare market/fundamental data + benchmark series
    -> rebuildable cache and SQLite store
    -> conservative point-in-time snapshots
    -> factor construction and cross-sectional IC
    -> HAC / block-bootstrap / multiple-testing audit
    -> constrained portfolio construction
    -> explicit costs, turnover and AUM-capacity stress
    -> expanding-window out-of-sample validation
    -> committed evidence audit and figures
```

Key implementation choices:

- Factor diagnostics and backtests share the same point-in-time data and scoring primitives.
- Training and unseen test windows reuse the same execution engine; there is no separate optimistic test path.
- Large factor screens report serial-correlation-aware inference and FDR control before a candidate can advance.
- Bulk market data remains reproducible local state under `data/`; compact evidence outputs stay reviewable in Git.

| Path | Purpose |
| --- | --- |
| `src/ashare_pit/` | Reusable data, factor, backtest, statistics, capacity, and validation library |
| `scripts/` | Operational entry points, including SQLite build and GTJA191 research runs |
| `data/raw/` | Local source data; ignored except for the directory placeholder |
| `data/interim/` | Rebuildable cache and SQLite research store; ignored by Git |
| `data/processed/` | Optional derived datasets; ignored by Git |
| `reports/research/` | Methodology and strategy reports |
| `reports/audit/` | Rebuildable evidence audit |
| `reports/figures/` | Committed, color-vision-accessible evidence charts |
| `tests/` | Offline unit and integration checks used by GitHub Actions |
| `results/` | Saved research outputs used in the reports |
| `pyproject.toml` / `uv.lock` | Reproducible package and dependency definition |
| `Makefile` | One-command audit, lint, test, and offline verification |

## Reproduction

```bash
uv sync --dev

uv run ashare-data --demo
uv run ashare-data --self-test
uv run ashare-diagnostics --help
uv run ashare-backtest --help
uv run ashare-walkforward --help
uv run pytest -q
make check
```

To enable AUM-aware capacity diagnostics in a full local run:

```bash
uv run ashare-backtest --full-market --skip-pe --benchmark 000905 \
  --strategy gtja_pv --universe csi500_synth --construction optimize \
  --start 2012-07-01 --end 2017-04-30 --turnover-cap 0.15 --rebalance-freq 2d \
  --required pv_divergence,opening_gap,abnormal_volume,amplitude_divergence \
  --portfolio-aum-millions 100 --output results/capacity_100m.json
```

The capacity layer assumes one-day execution and reports 63-day ADV participation with a volatility-scaled square-root impact estimate. It is a stress model, not a substitute for realized order-level slippage calibration.

The raw cache and 1.6 GB SQLite research database under `data/interim/` are intentionally excluded. A local legacy `.data_cache/` is still detected automatically during migration. The data can be rebuilt from the public adapters, subject to the providers' availability and terms. Saved result files are included so the reported analysis remains inspectable without committing bulk market data.

## Important limitations

- The stock-selection universe is based on currently available A-share identifiers and therefore retains survivorship bias from delisted names.
- Historical index membership is approximated rather than sourced from an official point-in-time constituent history.
- Financial statements are conservatively gated by statutory reporting deadlines; the public data does not provide a verified exchange-announcement timestamp for every observation.
- The risk model uses an L2 ridge proxy rather than a full commercial covariance model.
- Older committed IC summaries predate the robust-inference fields; their audit p-values use an independent-period approximation. Fresh diagnostic runs emit Newey-West, block-bootstrap, and BH-FDR results directly.
- Results depend on free public data whose historical coverage and endpoint behavior may change.

## Data and copyright

No paid research PDF, raw bulk market database, API credential, or proprietary dataset is included. Referenced research is identified for methodological context only. Data users are responsible for complying with each upstream provider's terms.

No license is granted for commercial reuse. This repository is published as a research and recruitment portfolio.
