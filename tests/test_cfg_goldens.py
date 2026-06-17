"""
Golden-file regression tests for CFG rendering.

How this module works:

1. Test matrix
   Pytest parametrizes over every row in `playground_functions.csv` and the four
   supported rendering configurations. That means the full suite collects
   `4 x N` tests, where `N` is the number of CSV rows.

2. Fresh render output
   Each test patches `get_settings()` so the app uses the requested test
   configuration, then extracts the real nested `_render_cfg()` function from
   `create_app()` and renders one CFG as raw DOT text.

3. Compare mode
   Default behavior is compare mode (`BINGRAPH_GOLDEN_MODE=compare`, or unset).
   The freshly rendered output is always written under `tests/_actual/...`.
   The test then compares that new output against the committed golden file
   under `tests/goldens/<config-name>/...`.

4. Promote mode
   When `BINGRAPH_GOLDEN_MODE=promote`, the module does not rerender CFGs.
   Instead, it copies the already generated files from `tests/_actual/...`
   into the committed golden directories under `tests/goldens/<config-name>/...`.
   This keeps promotion fast and makes it an explicit "accept what compare mode
   already produced" workflow.

5. Summary files
   At the end of the module run, one `summary.json` file is written per config
   under `tests/_actual/<config-name>/...`. In promote mode, that summary is
   also copied into the golden directory.

6. First-time bootstrap
   To create goldens for the first time:
   - Run compare mode. It will render files into `tests/_actual/...` and fail
     because no committed goldens exist yet.
   - Inspect the generated `_actual` files.
   - Run `BINGRAPH_GOLDEN_MODE=promote` to copy `_actual` into the golden
     directories.

7. Review workflow
   - Run compare mode to detect regressions.
   - Inspect files under `tests/_actual/...` when a mismatch occurs.
   - If the new output is correct, rerun with
     `BINGRAPH_GOLDEN_MODE=promote` and review the resulting git diff.

Useful environment variables:
   - `BINGRAPH_GOLDEN_MODE=compare|promote`
   - `BINGRAPH_GOLDEN_CONFIGS=name1,name2,...` to run only selected configs
   - `BINGRAPH_GOLDEN_LIMIT=<N>` to limit the CSV rows during local smoke tests
"""

from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import asdict, dataclass
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator
from unittest.mock import patch

import pytest

import bingraph.helpers as helpers_module
from bingraph.api import app as app_module
from bingraph.core import project as project_module
from bingraph.core import render as render_module
from bingraph.helpers import settings as settings_module


TESTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_DIR.parent
CSV_PATH = TESTS_DIR / "playground_functions.csv"
PLAYGROUND_ROOT = PROJECT_ROOT / "playground" / "angr-binaries"
SUMMARY_NAME = "summary.json"
ACTUAL_ROOT = TESTS_DIR / "_actual"
GOLDENS_ROOT = TESTS_DIR / "goldens"
MODE_ENV = "BINGRAPH_GOLDEN_MODE"
CONFIGS_ENV = "BINGRAPH_GOLDEN_CONFIGS"
LIMIT_ENV = "BINGRAPH_GOLDEN_LIMIT"


@dataclass(frozen=True)
class GoldenConfig:
    name: str
    cfg_mode: str
    comments: bool = True
    keep_state: bool = False


@dataclass
class StubSettings:
    root: Path
    cfg_mode: str
    comments: bool
    keep_state: bool
    debug: bool = False
    client: Any = None
    server: Any = None
    log_level: str = "ERROR"

    def model_dump(self) -> dict[str, Any]:
        data = asdict(self)
        data["root"] = str(self.root)
        return data


CONFIGS = [
    # Four supported regression configurations requested by the test design.
    GoldenConfig(name="fast_comments_true", cfg_mode="fast", comments=True, keep_state=False),
    GoldenConfig(name="fast_comments_false", cfg_mode="fast", comments=False, keep_state=False),
    GoldenConfig(name="emulated_keep_state_true", cfg_mode="emulated", comments=True, keep_state=True),
    GoldenConfig(name="emulated_keep_state_false", cfg_mode="emulated", comments=True, keep_state=False),
]


@dataclass
class ConfigRunState:
    expected_files: set[Path]
    entries: int = 0
    render_successes: int = 0
    render_failures: int = 0
    golden_matches: int = 0
    golden_mismatches: int = 0
    missing_goldens: int = 0
    render_crashes: int = 0
    render_recoveries: int = 0


def _iter_rows(limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield normalized CSV rows for parametrized golden tests."""

    # Convert CSV rows into the small payload each parametrized test needs.
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            yield {
                "index": index,
                "filepath": row["filepath"],
                "function_name": row["funcname"],
                "function_addr": row["funcaddr"],
            }
            if limit is not None and index >= limit:
                return


def _sanitize_filename(value: str, max_length: int = 80) -> str:
    """Convert arbitrary text into a filesystem-safe path fragment."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not sanitized:
        sanitized = "unknown"
    return sanitized[:max_length]


def _sanitize_flat_artifact_part(value: str, max_length: int = 160) -> str:
    """Convert text into a flat filename fragment while preserving commas."""

    sanitized = re.sub(r"[^A-Za-z0-9,._-]+", "_", value.strip()).strip(".,")
    if not sanitized:
        sanitized = "unknown"
    return sanitized[:max_length]


def _artifact_relative_path(row: dict[str, Any]) -> Path:
    """Build the stable relative artifact path for one CSV entry."""

    # Keep artifacts flat under each config directory so the original CSV path
    # can be inferred directly from the filename, without recreating folders.
    filepath_part = _sanitize_flat_artifact_part(row["filepath"].replace("/", ","))
    funcaddr_part = _sanitize_flat_artifact_part(row["function_addr"])
    funcname_part = _sanitize_flat_artifact_part(row["function_name"])
    return Path(f"{filepath_part},{funcaddr_part},{funcname_part}.dot")


def _summary_payload(
    config: GoldenConfig,
    state: ConfigRunState,
    limit: int | None,
) -> dict[str, Any]:
    """Serialize one configuration run-state into summary.json fields."""

    return {
        "config": asdict(config),
        "csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "root": str(PLAYGROUND_ROOT.relative_to(PROJECT_ROOT)),
        "entries": state.entries,
        "render_successes": state.render_successes,
        "render_failures": state.render_failures,
        "golden_matches": state.golden_matches,
        "golden_mismatches": state.golden_mismatches,
        "missing_goldens": state.missing_goldens,
        "render_crashes": state.render_crashes,
        "render_recoveries": state.render_recoveries,
        "format": "raw",
        "artifact_extension": ".dot",
        "limit": limit,
    }


def _serialize_summary(payload: dict[str, Any]) -> str:
    """Render summary metadata as deterministic JSON text."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _error_artifact(exc: Exception) -> str:
    """Convert a render exception into a persisted pseudo-artifact."""

    return f"# render-error\n{type(exc).__name__}: {exc}\n"


def _is_error_artifact(text: str) -> bool:
    """Return True when an artifact contains a captured render exception."""

    return text.startswith("# render-error\n")


def _normalize_raw_dot(text: str) -> str:
    """Normalize raw DOT output so goldens are stable across line ordering."""

    # Raw DOT edge ordering can vary between renders, especially in emulated mode.
    # Sorting node and edge lines separately makes golden comparisons stable.
    if not text.startswith("digraph G {\n") or not text.rstrip().endswith("}"):
        return text

    lines = text.splitlines()
    if len(lines) < 3:
        return text

    header = [lines[0]]
    footer = [lines[-1]]
    body = lines[1:-1]

    node_lines = sorted(line for line in body if "->" not in line)
    edge_lines = sorted(line for line in body if "->" in line)
    normalized = header + node_lines + edge_lines + footer
    return "\n".join(normalized) + "\n"


def _clear_caches() -> None:
    """Reset cached CFG helpers so each test sees its own patched settings."""

    # The app and CFG helpers are cached; clear them so each test sees the
    # patched settings for its own configuration.
    render_module.render_cfg.cache_clear()
    project_module._get_project.cache_clear()
    project_module._get_fast_cfg.cache_clear()
    project_module._get_emu_cfg.cache_clear()


def _extract_render_cfg() -> Callable[[str, str, str, str], str]:
    """Extract the nested `_render_cfg` callable from the real FastAPI app."""

    # `_render_cfg` is nested inside `create_app()`, so pull it out from the
    # `/api/cfg` route closure instead of duplicating application logic here.
    _clear_caches()

    app = app_module.create_app()
    for route in app.routes:
        if getattr(route, "path", None) != "/api/cfg":
            continue
        for cell in route.endpoint.__closure__ or ():
            candidate = cell.cell_contents
            if callable(candidate) and getattr(candidate, "__name__", "") == "_render_cfg":
                return candidate

    raise AssertionError("Unable to extract _render_cfg from create_app()")


def _existing_files(config_dir: Path) -> set[Path]:
    """Return every file currently present under a generated artifact directory."""

    if not config_dir.exists():
        return set()
    return {path.relative_to(config_dir) for path in config_dir.rglob("*") if path.is_file()}


def _prune_stale_files(config_dir: Path, expected_files: set[Path]) -> None:
    """Delete generated files that are no longer expected for a given run."""

    # Keep generated directories tidy when row limits change or old artifacts disappear.
    for stale_path in sorted(_existing_files(config_dir) - expected_files, reverse=True):
        (config_dir / stale_path).unlink()
    for directory in sorted((path for path in config_dir.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def _env_limit() -> int | None:
    """Read the optional CSV row limit used for small local smoke tests."""

    raw_limit = os.getenv(LIMIT_ENV)
    if raw_limit is None or raw_limit == "":
        return None
    return int(raw_limit)


def _golden_mode() -> str:
    """Return the active golden workflow mode from the environment."""

    # Two explicit workflows:
    # - compare: write `_actual`, compare against committed goldens
    # - promote: copy previously generated `_actual` files into the golden tree
    mode = os.getenv(MODE_ENV, "compare").strip().lower()
    valid_modes = {"compare", "promote"}
    if mode not in valid_modes:
        raise ValueError(f"{MODE_ENV} must be one of: {', '.join(sorted(valid_modes))}")
    return mode


def _selected_configs() -> list[GoldenConfig]:
    """Return the active config subset requested through the environment."""

    raw_configs = os.getenv(CONFIGS_ENV, "").strip()
    if not raw_configs:
        return CONFIGS

    available = {config.name: config for config in CONFIGS}
    selected_names = [name.strip() for name in raw_configs.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in available]
    if unknown:
        raise ValueError(
            f"{CONFIGS_ENV} contains unknown config(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(sorted(available))}"
        )

    # Preserve the order requested by the user while silently deduplicating repeats.
    selected: list[GoldenConfig] = []
    seen: set[str] = set()
    for name in selected_names:
        if name not in seen:
            selected.append(available[name])
            seen.add(name)
    return selected


def _row_id(row: dict[str, Any]) -> str:
    """Build a readable pytest id for one CSV-driven row case."""

    return (
        f"{_sanitize_filename(row['filepath'].replace('/', ','), 32)}-"
        f"{_sanitize_filename(row['function_addr'], 18)}-"
        f"{_sanitize_filename(row['function_name'], 24)}"
    )


def _copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    """Copy one generated `_actual` tree into the committed golden tree."""

    # Copy the generated `_actual` tree into the committed golden tree, replacing
    # files in place and removing stale golden files that no longer exist in `_actual`.
    expected_files = _existing_files(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for relpath in expected_files:
        target = dst_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_dir / relpath, target)
    _prune_stale_files(dst_dir, expected_files)


CURRENT_MODE = _golden_mode()
ACTIVE_CONFIGS = _selected_configs()
ROWS = tuple(_iter_rows(_env_limit())) if CURRENT_MODE == "compare" else ()
RUN_STATE: dict[str, ConfigRunState] = {}


@pytest.fixture(scope="module", autouse=True)
def _manage_summary_files() -> Iterator[None]:
    """Track per-config run stats and write summary files after the module finishes."""

    global RUN_STATE

    mode = CURRENT_MODE
    if mode != "compare":
        # Promote mode is a pure filesystem copy and should not touch run stats.
        yield
        return

    limit = _env_limit()
    # Track per-config aggregate stats as the parametrized row tests execute.
    RUN_STATE = {config.name: ConfigRunState(expected_files={Path(SUMMARY_NAME)}) for config in ACTIVE_CONFIGS}

    yield

    failures: list[str] = []
    for config in ACTIVE_CONFIGS:
        # After all row tests finish, materialize the summary that describes the
        # aggregate status for this config and keep `_actual` pruned to exactly
        # the files produced in the current run.
        state = RUN_STATE[config.name]
        golden_dir = GOLDENS_ROOT / config.name
        actual_dir = ACTUAL_ROOT / config.name
        actual_summary_path = actual_dir / SUMMARY_NAME
        golden_summary_path = golden_dir / SUMMARY_NAME
        summary_text = _serialize_summary(
            _summary_payload(
                config=config,
                state=state,
                limit=limit,
            )
        )

        actual_dir.mkdir(parents=True, exist_ok=True)
        actual_summary_path.write_text(summary_text, encoding="utf-8")
        _prune_stale_files(actual_dir, state.expected_files)

        if limit is not None:
            # Limited smoke runs intentionally do not validate committed summary
            # files because their counts are only a partial view of the corpus.
            continue

        # In full compare runs, the committed summary must match the just-computed
        # summary, and there should be no unexpected files left in the golden tree.
        if not golden_summary_path.exists():
            failures.append(f"{config.name}: missing summary file {golden_summary_path.relative_to(TESTS_DIR)}")
        elif golden_summary_path.read_text(encoding="utf-8") != summary_text:
            failures.append(f"{config.name}: summary mismatch {golden_summary_path.relative_to(TESTS_DIR)}")

        unexpected_files = sorted(_existing_files(golden_dir) - state.expected_files)
        failures.extend(f"{config.name}: unexpected golden file {path}" for path in unexpected_files)

    RUN_STATE = {}

    if failures:
        preview = "\n".join(failures[:20])
        remaining = len(failures) - min(len(failures), 20)
        suffix = f"\n... and {remaining} more failure(s)" if remaining else ""
        pytest.fail(f"Summary validation had {len(failures)} issue(s).\n{preview}{suffix}")


if CURRENT_MODE == "compare":
    @pytest.mark.slow
    @pytest.mark.parametrize("config", ACTIVE_CONFIGS, ids=[config.name for config in ACTIVE_CONFIGS])
    @pytest.mark.parametrize("row", ROWS, ids=_row_id)
    def test_render_cfg_goldens(config: GoldenConfig, row: dict[str, Any]) -> None:
        """Render one CFG case, store `_actual`, and compare it against its golden."""

        golden_dir = GOLDENS_ROOT / config.name
        actual_dir = ACTUAL_ROOT / config.name

        # Build the runtime settings object that the real app code will consult.
        settings = StubSettings(
            root=PLAYGROUND_ROOT,
            cfg_mode=config.cfg_mode,
            comments=config.comments,
            keep_state=config.keep_state,
        )
        artifact_relpath = _artifact_relative_path(row)
        golden_path = golden_dir / artifact_relpath
        actual_path = actual_dir / artifact_relpath
        state = RUN_STATE[config.name]
        state.entries += 1
        state.expected_files.add(artifact_relpath)

        # Patch every module that imports `get_settings()` directly so the render path
        # sees a coherent configuration from app entrypoint down to CFG generation.
        with patch.object(settings_module, "get_settings", return_value=settings), \
                patch.object(helpers_module, "get_settings", return_value=settings), \
                patch.object(app_module, "get_settings", return_value=settings), \
                patch.object(render_module, "get_settings", return_value=settings), \
                patch.object(project_module, "get_settings", return_value=settings):
            render_cfg = _extract_render_cfg()
            try:
                artifact_text = render_cfg(row["filepath"], row["function_addr"], config.cfg_mode, "raw")
                artifact_text = _normalize_raw_dot(artifact_text)
                state.render_successes += 1
            except Exception as exc:  # pragma: no cover - exercised against real corpus
                artifact_text = _error_artifact(exc)
                state.render_failures += 1
            finally:
                _clear_caches()

        # Always keep the newly rendered candidate output on disk for inspection.
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        actual_path.write_text(artifact_text, encoding="utf-8")

        # Compare mode requires an existing committed golden file.
        if not golden_path.exists():
            state.missing_goldens += 1
            pytest.fail(
                f"Missing golden file for {config.name}: {artifact_relpath}\n"
                f"Review {actual_path.relative_to(TESTS_DIR)} and rerun with {MODE_ENV}=promote to create or update goldens."
            )
        expected_text = golden_path.read_text(encoding="utf-8")
        if expected_text != artifact_text:
            state.golden_mismatches += 1
            # Distinguish between three important cases in test output:
            # - the new run crashed
            # - the old golden expected a crash but the run recovered
            # - both are outputs, but the output changed
            if _is_error_artifact(artifact_text):
                state.render_crashes += 1
                pytest.fail(
                    f"Render crashed for {config.name}: {artifact_relpath}\n"
                    f"Inspect {actual_path.relative_to(TESTS_DIR)} for the captured exception.\n"
                    f"Promote with {MODE_ENV}=promote only if this error artifact is the new expected result."
                )
            if _is_error_artifact(expected_text):
                state.render_recoveries += 1
                pytest.fail(
                    f"Render recovered for {config.name}: {artifact_relpath}\n"
                    f"The committed golden currently expects an error artifact, but the new run produced output.\n"
                    f"Inspect {actual_path.relative_to(TESTS_DIR)} and promote with {MODE_ENV}=promote if recovery is expected."
                )
            pytest.fail(
                f"Render output changed for {config.name}: {artifact_relpath}\n"
                f"Inspect {actual_path.relative_to(TESTS_DIR)} and promote with {MODE_ENV}=promote if the new output is correct."
            )
        # Only exact output matches count as successful golden comparisons.
        state.golden_matches += 1


@pytest.mark.slow
def test_promote_actual_to_goldens() -> None:
    """Copy previously generated `_actual` artifacts into the golden directories."""

    if CURRENT_MODE != "promote":
        pytest.skip("Promotion copy step only runs when BINGRAPH_GOLDEN_MODE=promote.")

    # Promote mode assumes compare mode already generated the `_actual` tree.
    assert ACTUAL_ROOT.exists(), (
        f"Missing {ACTUAL_ROOT.relative_to(TESTS_DIR)} directory.\n"
        f"Run compare mode first to generate candidate outputs before promoting."
    )

    for config in ACTIVE_CONFIGS:
        actual_dir = ACTUAL_ROOT / config.name
        golden_dir = GOLDENS_ROOT / config.name
        assert actual_dir.exists(), (
            f"Missing {actual_dir.relative_to(TESTS_DIR)}.\n"
            f"Run compare mode first so there is something to promote."
        )
        assert (actual_dir / SUMMARY_NAME).exists(), (
            f"Missing summary file in {actual_dir.relative_to(TESTS_DIR)}.\n"
            f"Run compare mode first so promotion copies a complete artifact set."
        )
        # Promotion is a pure copy operation: no rerendering, just accept what
        # compare mode already wrote into `_actual`.
        _copy_tree_contents(actual_dir, golden_dir)
