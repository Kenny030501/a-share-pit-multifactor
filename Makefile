UV ?= uv

.PHONY: audit lint test self-test check

audit:
	$(UV) run ashare-audit --output results/quant_research_audit.json --markdown reports/audit/quant-research-audit.md --assets-dir reports/figures

lint:
	$(UV) run ruff check .

test:
	$(UV) run pytest -q

self-test:
	$(UV) run ashare-backtest --self-test
	$(UV) run ashare-diagnostics --self-test
	$(UV) run ashare-walkforward --self-test
	$(UV) run ashare-portfolio-check --self-test

check: lint test self-test audit
