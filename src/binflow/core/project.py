from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import angr


def _cache_key(path: Path) -> Tuple[str, float]:
    """
    Generate a unique cache key for the given file path.
    
    The cache key is based on the file's string path and its last modification time.

    Args:
        path (Path): The file path for which to generate the key.

    Returns:
        Tuple[str, float]: A tuple of the file's string representation and its modification time.
    """
    stat = path.stat()
    return (str(path), stat.st_mtime)


@lru_cache
def _get_project(spath: str, mtime: float) -> angr.Project:
    """
    Returns an angr project from a (maybe cathed) image.
    
    Args:
        spath (str): The file path of the binary to analyze.
        mtime (float): Binary modification time -- note this is not used but still cached,
            so updated binaries are not incorrectly cached.

    Returns:
        angr.Project: The loaded angr project.
    """
    return angr.Project(spath, auto_load_libs=False)


def load_project(path: Path) -> angr.Project:
    """
    Load an angr project from the specified path.

    Args:
        path (Path): The file path of the binary to analyze.

    Returns:
        angr.Project: The loaded angr project.
    """
    spath, mtime = _cache_key(path)
    return _get_project(spath, mtime)


@lru_cache
def get_fast_cfg(project: angr.Project) -> angr.analyses.CFGFast:
    """
    Build or retrieve the fast control flow graph (CFGFast) for the given project.

    Args:
        project (angr.Project): The angr project for which to build the fast CFG.

    Returns:
        angr.analyses.CFGFast: The fast control flow graph of the project.
    """
    return project.analyses.CFGFast(normalize=True, show_progressbar=False)


@lru_cache
def get_emu_cfg(project: angr.Project, func_addr: int) -> angr.analyses.CFGEmulated:
    """
    Build or retrieve the emulated control flow graph (CFGEmulated) for a given function.

    Args:
        project (angr.Project): The angr project for which to build the emulated CFG.

    Returns:
        angr.analyses.CFGEmulated: The emulated control flow graph of the project.
    """
    return project.analyses.CFGEmulated(starts=[func_addr], call_depth=0, normalize=True)
