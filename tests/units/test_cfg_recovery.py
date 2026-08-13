"""Focused tests for Capstone/VEX-assisted block recovery boundaries."""

from types import SimpleNamespace
from typing import cast

from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.models import BlockSpec
from bingraph.cfg.models import FunctionBounds
from bingraph.cfg import decode
from bingraph.cfg import recovery
from bingraph.cfg.recovery import find_shared_instruction_tail
from bingraph.cfg.decode import (
    _native_vex_transfer_end,
    call_fallthrough_addr,
    lift_block_terminator,
)
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


def test_call_fallthrough_accepts_a_known_external_function_entry() -> None:
    """Preserve a call continuation that begins a neighboring function."""

    project = SimpleNamespace(
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(rebased_addr=addr, is_function=True)
                if addr == 0x1100
                else None
            )
        )
    )

    assert call_fallthrough_addr(project, _bounds(), 0x1100) == 0x1100


def test_call_fallthrough_rejects_unknown_external_bytes() -> None:
    """Avoid inventing a continuation beyond the selected function range."""

    project = SimpleNamespace(loader=SimpleNamespace(find_symbol=lambda _addr: None))

    assert call_fallthrough_addr(project, _bounds(), 0x1100) is None


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


def test_delayed_branch_exit_uses_delay_slot_provenance(
    monkeypatch,
) -> None:
    """Keep a MIPS branch target when VEX tags its delay-slot instruction."""

    branch = SimpleNamespace(address=0x1000, size=4)
    delay_slot = SimpleNamespace(address=0x1004, size=4)
    vex = SimpleNamespace(
        jumpkind="Ijk_Boring",
        next=SimpleNamespace(),
        exit_statements=(
            (
                0x1004,
                None,
                SimpleNamespace(
                    jumpkind="Ijk_Boring",
                    dst=SimpleNamespace(value=0x1010),
                ),
            ),
        ),
    )
    project = SimpleNamespace(
        arch=SimpleNamespace(name="MIPS64"),
        factory=SimpleNamespace(
            block=lambda *_args, **_kwargs: SimpleNamespace(vex=vex)
        ),
    )
    semantics = SimpleNamespace(
        is_ret=lambda: False,
        is_call=lambda: False,
        is_conditional_jump=lambda: True,
        direct_target=lambda: 0x1010,
    )
    monkeypatch.setattr(decode, "control_transfer_index", lambda *_args: 0)
    monkeypatch.setattr(decode, "InsnSemantics", lambda _insn: semantics)

    terminator = lift_block_terminator(project, _bounds(), [branch, delay_slot])

    assert terminator.direct_targets == (0x1010,)


def test_shared_instruction_tail_requires_linear_prefixes(
    monkeypatch,
) -> None:
    """Find a shared tail when both instruction streams reach it linearly."""

    class _Insn:
        """Represent one instruction with the fields used by tail detection."""

        def __init__(self, addr: int) -> None:
            """Create an instruction with stable byte identity."""

            self.address = addr
            self.bytes = addr.to_bytes(4, "little")

    first = object()
    second = object()
    decoded = {
        first: (_Insn(0x1000), _Insn(0x1010)),
        second: (_Insn(0x1004), _Insn(0x1010)),
    }
    monkeypatch.setattr(
        decode.DecodedNode,
        "from_node",
        lambda node: decode.DecodedNode(decoded[node]),
    )
    monkeypatch.setattr(
        recovery,
        "control_transfer_index",
        lambda _arch, _insns: None,
    )

    tail = find_shared_instruction_tail(
        "X86", [cast(CFGNode, first), cast(CFGNode, second)]
    )

    assert tail is not None
    assert tail.nodes == (cast(CFGNode, first), cast(CFGNode, second))
    assert tail.start_addr == 0x1010
    assert tail.instruction_addrs == (0x1010,)
