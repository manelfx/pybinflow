"""Jump-target primitives shared by custom CFG reconstruction."""

from __future__ import annotations

from .models import FunctionBounds


# Static table recovery is deliberately bounded. Larger index domains require a
# stronger range proof than the local VEX matcher currently provides.
MAX_STATIC_JUMPTABLE_ENTRIES = 256


def is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Return whether one direct target remains inside the function range."""

    return target is not None and bounds.addr <= target < bounds.end_addr
