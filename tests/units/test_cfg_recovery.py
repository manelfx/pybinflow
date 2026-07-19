"""Focused tests for Capstone/VEX-assisted block recovery boundaries."""

from types import SimpleNamespace

from bingraph.cfg.models import BlockSpec
from bingraph.cfg.models import FunctionBounds
from bingraph.cfg.recovery import _native_vex_transfer_end
from bingraph.cfg.repair import _block_has_unresolved_indirect_transfer


def _bounds() -> FunctionBounds:
    """Return minimal bounds for native VEX boundary tests."""

    return FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))


def test_native_vex_call_boundary_is_accepted() -> None:
    """Accept an in-bounds native VEX call boundary absent from Capstone groups."""

    project = SimpleNamespace(
        factory=SimpleNamespace(
            block=lambda _addr: SimpleNamespace(
                size=8,
                vex=SimpleNamespace(jumpkind="Ijk_Call"),
            )
        )
    )

    assert _native_vex_transfer_end(project, _bounds(), 0x1000) == 0x1008


def test_native_vex_nontransfer_boundary_is_ignored() -> None:
    """Leave ordinary basic-block discovery to Capstone and existing logic."""

    project = SimpleNamespace(
        factory=SimpleNamespace(
            block=lambda _addr: SimpleNamespace(
                size=8,
                vex=SimpleNamespace(jumpkind="Ijk_Boring"),
            )
        )
    )

    assert _native_vex_transfer_end(project, _bounds(), 0x1000) is None


def test_native_vex_boundary_outside_function_is_ignored() -> None:
    """Reject a native block whose boundary exceeds the selected function."""

    project = SimpleNamespace(
        factory=SimpleNamespace(
            block=lambda _addr: SimpleNamespace(
                size=0x200,
                vex=SimpleNamespace(jumpkind="Ijk_Call"),
            )
        )
    )

    assert _native_vex_transfer_end(project, _bounds(), 0x1000) is None


def test_indirect_call_preserves_its_unresolved_target() -> None:
    """Keep CFGFast's unresolved call target when a replacement has no target."""

    block = BlockSpec(
        addr=0x1000,
        size=4,
        instruction_addrs=(0x1000,),
        jumpkind="Ijk_Call",
        fallthrough_addr=0x1004,
    )

    assert _block_has_unresolved_indirect_transfer(block)
