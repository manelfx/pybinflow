"""Unit tests for static jump-table target recovery helpers."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import networkx as nx
import pyvex

from bingraph.cfg.jumps import _jump_table_target_addr, _x86_pc_thunk_base_addr
from bingraph.cfg.models import FunctionBounds, StaticJumpTable


@dataclass(frozen=True)
class _Node:
    """Provide the small hashable CFG-node surface used by jump helpers."""

    addr: int
    size: int
    vex: object = field(compare=False, hash=False)

    @property
    def block(self) -> SimpleNamespace:
        """Expose the node VEX through angr's block-shaped API."""

        return SimpleNamespace(vex=self.vex)


def _table(*, target_displacement: int = 0) -> StaticJumpTable:
    """Create a 32-bit relative table description for focused helper tests."""

    return StaticJumpTable(
        base_register_offset=20,
        base_bits=32,
        table_displacement=0,
        index_register_offset=12,
        index_bits=32,
        entry_size=4,
        endness="Iend_LE",
        signed_entries=False,
        target_displacement=target_displacement,
    )


def test_x86_pc_thunk_proves_the_dispatcher_base_register() -> None:
    """Accept a fallthrough from the matching GCC PC thunk call."""

    call_target = 0x4000
    predecessor = _Node(
        0x1000,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    graph = nx.DiGraph([(predecessor, dispatcher)])
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.bx")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert (
        _x86_pc_thunk_base_addr(project, graph, bounds, dispatcher, _table()) == 0x1005
    )


def test_x86_pc_thunk_rejects_a_call_to_the_wrong_register_thunk() -> None:
    """Reject a direct call whose thunk does not initialize the table register."""

    call_target = 0x4000
    predecessor = _Node(
        0x1000,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    graph = nx.DiGraph([(predecessor, dispatcher)])
    project = SimpleNamespace(
        arch=SimpleNamespace(name="X86", bits=32, register_names={20: "ebx"}),
        loader=SimpleNamespace(
            find_symbol=lambda addr: (
                SimpleNamespace(name="__x86.get_pc_thunk.ax")
                if addr == call_target
                else None
            )
        ),
    )
    bounds = FunctionBounds(0x1000, 0x1100, 0x100, SimpleNamespace(name="f"))

    assert _x86_pc_thunk_base_addr(project, graph, bounds, dispatcher, _table()) is None


def test_relative_table_target_wraps_to_the_architecture_width() -> None:
    """Interpret a raw 32-bit relative entry using machine-width arithmetic."""

    assert (
        _jump_table_target_addr(
            0x80626C1, _table(target_displacement=0x4B9C7), 0xFFFB46A8
        )
        == 0x8062730
    )
