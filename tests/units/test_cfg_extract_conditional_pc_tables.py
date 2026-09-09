"""Regression coverage for VEX-guarded memory loads used as the next PC."""

from __future__ import annotations

from pathlib import Path

from angr import KnowledgeBase

from bingraph.cfg.decode import decode_bounded_block
from bingraph.cfg_extract import build_extracted_cfg
from bingraph.cfg_extract import builder as builder_module
from bingraph.core.project import load_project


def test_extract_recovers_a_guarded_pc_load_table() -> None:
    """Keep both paths of ``ldrls pc, [pc, r3, lsl #2]`` in the CFG."""

    project = load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x446F2C
    ).bounds

    block = decode_bounded_block(
        project,
        bounds,
        0x446F58,
        set(),
        split_unclassified_indirect_vex_transfers=True,
    )

    assert block is not None
    assert block.instruction_addrs == (0x446F58, 0x446F5C, 0x446F60)
    assert block.fallthrough_addr == 0x446F64

    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x446F2C)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x446F58]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {0x446F64, 0x446F78, 0x4471B0}
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.static_jump_target_edges_added == 2
    assert not any(
        node.simprocedure_name == "UnresolvableJumpTarget"
        for node in cfg.graph.nodes()
        if node.is_simprocedure
    )


def test_extract_recovers_a_guarded_arithmetic_pc_dispatch() -> None:
    """Recover a finite ``addls pc, pc, r2, lsl #2`` target range."""

    project = load_project(Path("angr-binaries/tests/armel/libc.so.6"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x47E114)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x47E114]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {
        0x47E140,
        0x47E144,
        0x47E148,
        0x47E14C,
        0x47E150,
        0x47E154,
        0x47E158,
        0x47E15C,
        0x47E160,
    }
    assert cfg.extract_stats.conditional_pc_dispatches_resolved == 1
    assert cfg.extract_stats.conditional_pc_targets_recovered == 8


def test_extract_recovers_a_clz_derived_arithmetic_pc_dispatch() -> None:
    """Recover the bounded predicated dispatcher in ``__aeabi_idiv``."""

    project = load_project(Path("angr-binaries/tests/armel/test_division"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x8670)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x86A0]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {0x86BC, *range(0x86CC, 0x8835, 0xC)}
    assert 0x86C0 not in successors
    assert cfg.extract_stats.conditional_pc_dispatches_resolved == 1
    assert cfg.extract_stats.conditional_pc_targets_recovered == 31


def test_extract_recovers_a_scaled_static_byte_table() -> None:
    """Recover VEX-scaled byte-table targets without an ARM mnemonic rule."""

    project = load_project(
        Path("angr-binaries/tests/armhf/amp_challenge_07.gcc.dyn.unstripped")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x401D29)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x401D31]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {0x401D39, 0x401D59, 0x401D61}
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.static_jump_target_edges_added == 3
