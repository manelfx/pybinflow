"""Unit tests for static jump-table target recovery helpers."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import archinfo
import networkx as nx
import pyvex

from bingraph.cfg.jumps import (
    _jump_table_target_addr,
    _read_static_jump_table_targets,
    _vex_direct_jump_table,
    _vex_guarded_index_upper_bound,
    _vex_relative_jump_table,
    _x86_pc_thunk_base_addr,
    _x86_pc_thunk_guarded_entry_count,
    static_jump_target_rejection_reason,
)
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


def _table(
    *, target_displacement: int = 0, index_register_offset: int = 12
) -> StaticJumpTable:
    """Create a 32-bit relative table description for focused helper tests."""

    return StaticJumpTable(
        base_register_offset=20,
        base_bits=32,
        table_displacement=0,
        index_register_offset=index_register_offset,
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


def test_x86_pc_thunk_prefers_an_immediate_dispatcher_predecessor() -> None:
    """Keep one local PC-thunk proof from conflicting with unrelated PIC setup."""

    call_target = 0x4000
    call_vex = SimpleNamespace(
        jumpkind="Ijk_Call",
        next=pyvex.expr.Const(pyvex.const.U32(call_target)),
        statements=(),
    )
    predecessor = _Node(0x1000, 5, call_vex)
    dispatcher = _Node(0x1005, 4, SimpleNamespace())
    other_thunk_call = _Node(0x1020, 5, call_vex)
    other_fallthrough = _Node(
        0x1025,
        4,
        pyvex.lift(bytes.fromhex("83c320c3"), 0x1025, archinfo.ArchX86()),
    )
    graph = nx.DiGraph(
        [(predecessor, dispatcher), (other_thunk_call, other_fallthrough)]
    )
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


def test_x86_pc_thunk_recovers_the_guarded_table_length() -> None:
    """Carry an unsigned index bound through the thunk's fake-return edge."""

    call_target = 0x4000
    guard = _Node(
        0x1000,
        5,
        pyvex.lift(bytes.fromhex("83f8207200"), 0x1000, archinfo.ArchX86()),
    )
    thunk_call = _Node(
        0x1005,
        5,
        SimpleNamespace(
            jumpkind="Ijk_Call",
            next=pyvex.expr.Const(pyvex.const.U32(call_target)),
            statements=(),
        ),
    )
    dispatcher = _Node(0x100A, 4, SimpleNamespace())
    graph = nx.DiGraph([(guard, thunk_call), (thunk_call, dispatcher)])
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
        _x86_pc_thunk_guarded_entry_count(
            project,
            graph,
            bounds,
            dispatcher,
            _table(index_register_offset=8),
        )
        == 32
    )


def test_relative_table_target_wraps_to_the_architecture_width() -> None:
    """Interpret a raw 32-bit relative entry using machine-width arithmetic."""

    assert (
        _jump_table_target_addr(
            0x80626C1, _table(target_displacement=0x4B9C7), 0xFFFB46A8
        )
        == 0x8062730
    )


def test_direct_jump_table_keeps_absolute_entries() -> None:
    """Recognize a VEX-lifted absolute table dispatch without rebasing entries."""

    # jmp dword ptr [ebx + eax * 4]
    vex = pyvex.lift(bytes.fromhex("ff2483"), 0x1000, archinfo.ArchX86())

    table = _vex_direct_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_register_offset == 20  # ebx
    assert table.index_register_offset == 8  # eax
    assert table.entry_size == 4
    assert not table.entries_are_relative
    assert _jump_table_target_addr(0x1000, table, 0x2000) == 0x2000


def test_relative_jump_table_accepts_a_guarded_full_width_index() -> None:
    """Recognize the common AMD64 signed-relative table dispatch form."""

    # movsxd r9, dword ptr [r11 + r9 * 4]; lea r9, [r11 + r9]; jmp r9
    vex = pyvex.lift(
        bytes.fromhex("4f630c8b4f8d0c0b41ffe1"), 0x1000, archinfo.ArchAMD64()
    )

    table = _vex_relative_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_bits == 64
    assert table.entry_size == 4
    assert table.signed_entries
    assert table.entries_are_relative


def test_relative_jump_table_accepts_a_vex_folded_static_base() -> None:
    """Recognize a RIP-relative table base folded to a VEX constant."""

    # lea r11, [rip + 0x7389d]; movsxd rdx, [r11 + rdx * 4];
    # lea rdx, [r11 + rdx]; jmp rdx
    vex = pyvex.lift(
        bytes.fromhex("4c8d1d9d38070049631493498d1413ffe2"),
        0x42F50C,
        archinfo.ArchAMD64(),
    )

    table = _vex_relative_jump_table(vex, allow_full_width_index=True)

    assert table is not None
    assert table.base_register_offset is None
    assert table.static_base_addr == 0x4A2DB0
    assert table.index_bits == 64


def test_guarded_jump_table_bound_accepts_unsigned_strict_less_than() -> None:
    """Treat an unsigned ``index < limit`` guard as ``limit`` table entries."""

    # cmp rdx, 0x20; jb 0x1006
    vex = pyvex.lift(bytes.fromhex("4883fa207200"), 0x1000, archinfo.ArchAMD64())

    assert _vex_guarded_index_upper_bound(vex, 0x1006, (32, 64)) == 31


def test_guarded_jump_table_bound_tracks_a_same_block_index_assignment() -> None:
    """Use the index value written before the guard rather than only a raw GET."""

    # mov ecx, [esp + 0x10]; cmp ecx, 0x20; jb 0x1009
    vex = pyvex.lift(bytes.fromhex("8b4c241083f9207200"), 0x1000, archinfo.ArchX86())

    assert _vex_guarded_index_upper_bound(vex, 0x1009, (12, 32)) == 31


def test_unreadable_static_jump_table_returns_none() -> None:
    """Distinguish an unreadable table from a readable table without targets."""

    project = SimpleNamespace(
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda *_args: (_ for _ in ()).throw(ValueError("unmapped"))
            )
        )
    )

    assert _read_static_jump_table_targets(project, _table(), 0x1000, 2) is None


@dataclass(frozen=True)
class _Region:
    """Model the executable mapping bit used by static-target validation."""

    is_executable: bool


def _target_project(
    section: _Region | None, *, synthetic: bool = False
) -> SimpleNamespace:
    """Create the minimal loader surface required by target validation."""

    obj = SimpleNamespace(
        find_section_containing=lambda _addr: section,
        find_segment_containing=lambda _addr: None,
    )
    loader = SimpleNamespace(find_object_containing=lambda _addr: obj)
    if synthetic:
        loader.extern_object = obj
    return SimpleNamespace(loader=loader)


def test_static_jump_target_accepts_executable_in_function_code() -> None:
    """Accept an in-function entry backed by an executable section."""

    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True)), 0x1080
        )
        is None
    )


def test_static_jump_target_rejects_non_code_or_synthetic_values() -> None:
    """Keep data, unmapped, and synthetic table values out of CFG recovery."""

    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=False)), 0x1080
        )
        == "non_executable"
    )
    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True)), 0x1200
        )
        is None
    )
    assert (
        static_jump_target_rejection_reason(
            _target_project(_Region(is_executable=True), synthetic=True),
            0x1080,
        )
        == "synthetic"
    )
    unmapped = SimpleNamespace(
        loader=SimpleNamespace(find_object_containing=lambda _addr: None)
    )
    assert static_jump_target_rejection_reason(unmapped, 0x1080) == "unmapped"
