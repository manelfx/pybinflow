from __future__ import annotations

from pathlib import Path

import pytest

from binflow.core.utils import demangle, resolve_under_root


def test_demangle_passthrough() -> None:
    name = "plain_function"
    assert demangle(name) == name


def test_resolve_under_root_blocks_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    (root / "bin").write_text("x")
    resolved = resolve_under_root(root, "bin")
    assert resolved == root / "bin"

    with pytest.raises(ValueError):
        resolve_under_root(root, "../etc/passwd")