"""Custom CFG discovery, validation, and localized repair."""

from .anomalies import iter_function_nodes, log_cfg_status
from .repair import build_custom_cfg

__all__ = ["build_custom_cfg", "iter_function_nodes", "log_cfg_status"]
