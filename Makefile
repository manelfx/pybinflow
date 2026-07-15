UV := uv run

# These are commands, not filesystem targets. Declaring them phony prevents
# existing directories such as `tests/` from making Make skip their recipes.
.PHONY: tests format format-check lint typecheck check goldens goldens-checkpoint goldens-promote

# Run the default fast unit-test suite selected by pytest's `testpaths` setting.
tests:
	$(UV) pytest -v

# Rewrite Python source and test files into Ruff's canonical formatting style.
format:
	$(UV) ruff format src tests

# Verify formatting without modifying files; suitable for review and CI.
format-check:
	$(UV) ruff format --check src tests

# Check source and tests for lint issues such as unused imports and bad patterns.
lint:
	$(UV) ruff check src tests

# Statically type-check production code. Tests are intentionally excluded
# because partial fakes and monkeypatching make their type diagnostics noisy.
typecheck:
	$(UV) ty check src

# Run the regular non-golden development checks without changing source files.
check: format-check lint typecheck tests

# Render and compare the complete golden matrix. This can be expensive and
# writes fresh candidate artifacts under `tests/_actual/`.
goldens:
	$(UV) pytest -v tests/goldens/test_cfg_goldens.py

# Render and compare only checkpoint-marked goldens. Set
# `BINGRAPH_GOLDEN_CONFIGS` in the environment to choose configurations.
goldens-checkpoint:
	$(UV) pytest -v -m checkpoint tests/goldens/test_cfg_goldens.py

# Copy complete existing `_actual` trees into `tests/goldens/` without
# rerendering CFGs. Run a full compare-mode golden suite before promoting.
goldens-promote:
	BINGRAPH_GOLDEN_MODE=promote \
		$(UV) pytest -v tests/goldens/test_cfg_goldens.py
