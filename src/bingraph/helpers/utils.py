from __future__ import annotations
from functools import lru_cache, wraps
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

from loguru import logger
import cxxfilt


MODULE_NAME = "bingraph"


def time_it(func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)  # Preserves the original function's name and docstring
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start_time = perf_counter()  # Highest resolution clock available

        result = func(*args, **kwargs)   # Execute the actual function

        end_time = perf_counter()
        execution_time = end_time - start_time

        # Log the result
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper


def resolve_under_root(root: Path, relpath: str) -> Path:
    """
    Resolve a file's relative path under the given root directory.

    This function ensures that the provided relative path does not start with a slash,
    resolves the absolute path under the provided root, and verifies that the resolved path
    does not escape the root directory. It raises:
      - ValueError if the path is absolute or escapes the root directory.
      - FileNotFoundError if the resolved path does not exist.

    Args:
        root (Path): The root directory.
        relpath (str): The file path relative to the root.

    Returns:
        Path: The resolved absolute file path under the root.
    """
    if relpath.startswith("/"):
        raise ValueError("filepath must be relative to root")
    candidate = (root / relpath).resolve()
    root_resolved = root.resolve()
    if root_resolved not in candidate.parents and candidate != root_resolved:
        raise ValueError("filepath escapes root directory")
    if not candidate.exists():
        raise FileNotFoundError(f"Binary not found: {candidate}")
    return candidate


@lru_cache(maxsize=512)
def demangle(name: str) -> str:
    """
    Demangle a given C++ mangled name using cxxfilt.

    This function uses an LRU cache to store previous results. If the provided name is empty,
    it returns an empty string. If the mangled name cannot be demangled due to an invalid name,
    it returns the original name.

    Args:
        name (str): The mangled name to be demangled.

    Returns:
        str: The demangled name if successfully demangled, otherwise the original name.
    """
    if not name:
        return ""
    try:
        return cxxfilt.demangle(name)
    except cxxfilt.InvalidName:
        return name
