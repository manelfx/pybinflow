from __future__ import annotations

from pathlib import Path

import pytest

from binflow.core.symtab import list_function_symbols
from binflow.core.project import load_project
from binflow.core.render import render_function_cfg_svg


def _looks_like_binary(path: Path) -> bool:
    try:
        data = path.read_bytes()[:4]
    except OSError:
        return False
    return data.startswith(b"\x7fELF") or data.startswith(b"MZ") or data in {
        b"\xfe\xed\xfa\xcf",  # Mach-O 64
        b"\xcf\xfa\xed\xfe",  # Mach-O 64 (reverse)
        b"\xfe\xed\xfa\xce",  # Mach-O 32
        b"\xce\xfa\xed\xfe",  # Mach-O 32 (reverse)
    }


def _find_any_binary(root: Path) -> Path | None:
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        if _looks_like_binary(path):
            return path
    return None


@pytest.mark.slow
def test_render_cfg_svg_from_playground() -> None:
    root = Path("playground/angr-binaries")
    if not root.exists():
        pytest.skip("playground binaries not present")

    binary = _find_any_binary(root)
    if binary is None:
        pytest.skip("no binaries found in playground")

    project = load_project(binary)
    cfg = project.analyses.CFGFast(normalize=True, show_progressbar=False)
    funcs = [func for func in cfg.kb.functions.values() if func.addr != 0]
    if not funcs:
        pytest.skip("no CFG functions found")
    svg = render_function_cfg_svg(project, funcs[0].addr)
    assert svg.strip().startswith("<svg")