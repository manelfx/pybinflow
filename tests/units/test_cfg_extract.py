"""Tests for the experimental CFG extractor independent of CFGFast."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from angr import KnowledgeBase
import networkx as nx

from bingraph.cfg_extract import build_extracted_cfg
from bingraph.cfg_extract.anomalies import find_extracted_cfg_anomalies
from bingraph.cfg_extract import builder as builder_module
from bingraph.cfg_extract.sweep import recover_executable_components
from bingraph.cfg.models import BlockSpec, FunctionBounds, StaticJumpTable
from bingraph.cfg.models import StaticJumpTablePlan
from bingraph.cfg.decode import decode_bounded_block
from bingraph.cfg_extract.models import ExtractedCFGStats
from bingraph.core import project as project_module


def test_extract_builder_decodes_a_bounded_function_without_cfgfast() -> None:
    """Build normal function blocks without requesting an angr CFG analysis."""

    project = project_module.load_project(
        Path("angr-binaries/tests/samples/ais3_crackme")
    )
    kb = KnowledgeBase(project)

    with patch.object(project.analyses, "CFGFast", side_effect=AssertionError):
        cfg = build_extracted_cfg(project, kb, 0x40043C)

    nodes = [node for node in cfg.graph.nodes() if not node.is_simprocedure]
    assert [node.addr for node in nodes] == [0x40043C, 0x40044C, 0x40044E]
    assert cfg.functions.get(0x40043C) is not None


def test_extract_mode_bypasses_fast_cfg(monkeypatch) -> None:
    """Route the public extract mode directly to independent construction."""

    project = project_module.load_project(
        Path("angr-binaries/tests/samples/ais3_crackme")
    )
    project_module.get_cfg.cache_clear()
    monkeypatch.setattr(
        project_module,
        "_get_fast_cfg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("CFGFast called")),
    )

    cfg = project_module.get_cfg(project, 0x40043C, "extract")

    assert sum(not node.is_simprocedure for node in cfg.graph.nodes()) == 3


def test_extract_preserves_conditional_return_fallthrough() -> None:
    """Keep the non-returning paths of VEX conditional returns recoverable."""

    project = project_module.load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    bxeq_bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x44D480
    ).bounds
    popeq_bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x473818
    ).bounds

    bxeq = decode_bounded_block(
        project,
        bxeq_bounds,
        0x44D480,
        set(),
        preserve_conditional_return_fallthrough=True,
    )
    popeq = decode_bounded_block(
        project,
        popeq_bounds,
        0x473828,
        set(),
        preserve_conditional_return_fallthrough=True,
    )

    assert bxeq is not None
    assert bxeq.jumpkind == "Ijk_Boring"
    assert bxeq.fallthrough_addr == 0x44D488
    assert popeq is not None
    assert popeq.jumpkind == "Ijk_Boring"
    assert popeq.fallthrough_addr == 0x473848


def test_extract_retains_external_call_target() -> None:
    """Keep a resolved direct callee even when it lies outside function bounds."""

    project = project_module.load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x44D480
    )

    block = decode_bounded_block(project, session.bounds, 0x44D534, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x500048,)


def test_extract_retains_unnamed_external_call_target() -> None:
    """Keep a direct callee whose address has no loader symbol."""

    project = project_module.load_project(
        Path("angr-binaries/tests/ppc64el/fauxware_static")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x10028860
    )

    block = decode_bounded_block(project, session.bounds, 0x10028BC0, set())

    assert project.loader.find_symbol(0x10026988) is None
    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x10026988,)


def test_extract_leader_is_not_requeued_after_recovery() -> None:
    """Keep a cycle from repeatedly scheduling an unchanged completed block."""

    session = object.__new__(builder_module._ExtractionSession)
    session.bounds = SimpleNamespace(addr=0x1000, end_addr=0x1100)
    session.leaders = {0x1000}
    session.blocks = {0x1000: SimpleNamespace(size=4)}
    session.pending = []
    session.pending_addrs = set()
    session.stats = SimpleNamespace(block_redecodes=0)

    session._add_leader(0x1000)

    assert session.pending == []


def test_extract_static_table_discovery_discards_stale_snapshot_plans(
    monkeypatch,
) -> None:
    """Rebuild table planning when a recovered target splits a later block."""

    session = object.__new__(builder_module._ExtractionSession)
    session.bounds = SimpleNamespace(addr=0x1000, end_addr=0x1200)
    session.leaders = {0x1000, 0x1100}
    session.pending = deque()
    session.pending_addrs = set()
    session.blocks = {
        0x1000: BlockSpec(0x1000, 0x20, (0x1000,), "Ijk_Boring"),
        0x1100: BlockSpec(0x1100, 4, (0x1100,), "Ijk_Boring"),
    }
    session.static_targets = {}
    session.stats = ExtractedCFGStats()
    session.project = SimpleNamespace()

    dispatcher = object()
    stale_node = object()
    monkeypatch.setattr(
        session,
        "_analysis_graph",
        lambda: (nx.DiGraph(), {0x1100: dispatcher, 0x1000: stale_node}),
    )
    monkeypatch.setattr(
        builder_module,
        "plan_static_jump_table",
        lambda *_args: (
            StaticJumpTablePlan(
                StaticJumpTable(
                    base_register_offset=None,
                    base_bits=32,
                    table_displacement=0,
                    index_register_offset=0,
                    index_bits=32,
                    entry_size=4,
                    endness="Iend_LE",
                    signed_entries=False,
                ),
                0x2000,
                1,
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        builder_module,
        "_read_static_jump_table_targets",
        lambda *_args: (0x1008,),
    )
    monkeypatch.setattr(
        builder_module,
        "static_jump_target_rejection_reason",
        lambda *_args: None,
    )
    monkeypatch.setattr(session, "_decode_all_blocks", lambda: None)

    session._discover_static_jump_targets()

    assert 0x1000 not in session.blocks
    assert session.static_targets == {0x1100: (0x1008,)}


def test_extract_recovers_reconnecting_components_from_one_dispatcher() -> None:
    """Attach only reconnecting components behind one unresolved dispatcher."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x80A7DB0)

    assert cfg.extract_stats.sweep_runs == 1
    assert cfg.extract_stats.sweep_candidate_blocks > 100
    assert cfg.extract_stats.sweep_candidate_components > 0
    assert cfg.extract_stats.sweep_reconnecting_components > 0
    assert cfg.extract_stats.sweep_reconnecting_blocks > 100
    assert cfg.extract_stats.sweep_component_roots_attached > 0
    assert cfg.extract_stats.output_anomalies == 0

    dispatcher = next(
        node
        for node in cfg.graph.nodes()
        if node.addr == 0x80A7E0B and not node.is_simprocedure
    )
    unresolved = next(
        node
        for node in cfg.graph.nodes()
        if node.is_simprocedure and node.name == "UnresolvableJumpTarget"
    )
    successors = set(cfg.graph.successors(dispatcher))
    assert unresolved in successors
    assert len(successors) == cfg.extract_stats.sweep_component_roots_attached + 1
    assert not tuple(cfg.graph.successors(unresolved))


def test_executable_sweep_closes_direct_targets_before_reporting_components() -> None:
    """Audit components retain the normal extractor's exact-leader invariant."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x80A7DB0
    )
    session._decode_all_blocks()
    session._discover_static_jump_targets()

    sweep = recover_executable_components(project, session.bounds, session.blocks)

    assert sweep.audit.candidate_blocks > 100
    assert sweep.audit.decode_failures == 0
    for block in sweep.blocks.values():
        for target in block.direct_targets:
            if session.bounds.addr <= target < session.bounds.end_addr:
                assert target in sweep.blocks


class _Node:
    """Minimal hashable extracted-node stand-in for structural checks."""

    def __init__(self, addr: int, size: int, instruction_addrs: tuple[int, ...]):
        self.addr = addr
        self.size = size
        self.instruction_addrs = instruction_addrs
        self.is_simprocedure = False
        self.function_address = 0x1000


def test_extract_validation_rejects_targets_inside_other_blocks() -> None:
    """Require every direct target to become an exact block leader."""

    source = _Node(0x1000, 4, (0x1000,))
    covering = _Node(0x1002, 4, (0x1002,))
    graph = nx.DiGraph([(source, covering)])
    bounds = FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f"))
    blocks = {
        0x1000: BlockSpec(0x1000, 4, (0x1000,), "Ijk_Boring", (0x1003,)),
        0x1002: BlockSpec(0x1002, 4, (0x1002,), "Ijk_Ret"),
    }

    anomalies = find_extracted_cfg_anomalies(graph, bounds, 0x1000, blocks)

    assert {anomaly.kind for anomaly in anomalies} == {
        "overlapping_blocks",
        "missing_direct_edge",
        "target_inside_block",
    }
