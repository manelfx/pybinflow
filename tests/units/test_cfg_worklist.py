"""Fast unit tests for the custom CFG repair worklist contracts."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import cast

import networkx as nx
import archinfo
import pyvex
import pytest
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg import nodes as nodes_module
from bingraph.cfg import repair as cfg_module
from bingraph.cfg import graph as graph_module
from bingraph.cfg import models as models_module


def _bare_session() -> cfg_module._RepairSession:
    """Allocate only the repair-session state needed by these focused tests."""

    session = object.__new__(cfg_module._RepairSession)
    session.queue = deque()
    session.pending = {}
    session.repaired_nodes = set()
    session.leaders = models_module.BlockLeaderRegistry({})
    session._overlapping_entry_starts = set()
    session._deferred_unresolved_control_edges = {}
    session.mutation_revision = 0
    session._stop_starts_revision = -1
    session._stop_starts = set()
    session.last_requeue_states = {}
    session.stats = models_module.CustomCFGStats()
    session.func_addr = 0x1000
    session._bound_nodes_revision = -1
    session._bound_nodes_snapshot = ()
    session.bounds = cast(
        models_module.FunctionBounds,
        SimpleNamespace(addr=0x1000, end_addr=0x2000),
    )
    session.graph = object()
    session.project = SimpleNamespace(arch=SimpleNamespace(is_thumb=lambda addr: False))
    return session


class _SourceNode:
    """Hashable CFG-node stand-in whose identity remains distinct at one address."""

    def __init__(self, addr: int) -> None:
        """Create a stand-in node at one address."""

        self.addr = addr


def _source_node(addr: int = 0x1100) -> CFGNode:
    """Return a distinct CFG node stand-in at the requested address."""

    return cast(CFGNode, _SourceNode(addr))


class _NodeGraph:
    """Minimal graph surface used to exercise external-target node reuse."""

    def __init__(self) -> None:
        """Create an empty in-memory node collection."""

        self._nodes: list[object] = []

    def nodes(self) -> list[object]:
        """Return the currently stored graph nodes."""

        return list(self._nodes)

    def add_node(self, node: object) -> None:
        """Store one node in the test graph."""

        self._nodes.append(node)


class _BlockNode:
    """Hashable node stand-in carrying a VEX block for fallback-candidate tests."""

    def __init__(self, addr: int, data: bytes) -> None:
        """Lift one bounded x86 block from ``data`` at ``addr``."""

        self.addr = addr
        self.size = len(data)
        self.is_simprocedure = False
        self.block = SimpleNamespace(
            vex=pyvex.lift(data, addr, archinfo.ArchX86(), max_bytes=len(data))
        )


class _MaterializedNode:
    """Minimal ordinary CFG-node stand-in for recovery-boundary tests."""

    def __init__(self, addr: int, size: int = 1) -> None:
        """Create a live materialized node at one address."""

        self.addr = addr
        self.size = size
        self.is_simprocedure = False


def test_queue_merges_edge_claims_by_source_identity() -> None:
    """Merge different source nodes for one target into one pending obligation."""

    session = _bare_session()
    first_source = _source_node()
    second_source = _source_node()
    first = cfg_module.RepairObligation(
        addr=0x1200,
        reason="first_source",
        source_node=first_source,
    )
    second = cfg_module.RepairObligation(
        addr=0x1200,
        reason="second_source",
        source_node=second_source,
        jumpkind="Ijk_Call",
        preserve_exact_addr=True,
    )

    assert first_source is not second_source
    assert first_source.addr == second_source.addr
    assert session._queue_if_needed(first)
    assert not session._queue_if_needed(second)

    pending = session.pending[("recover", 0x1200)]
    assert list(session.queue) == [("recover", 0x1200)]
    assert pending.reasons == {"first_source", "second_source"}
    assert pending.edge_claims == {
        cfg_module.EdgeClaim(first_source, "Ijk_Boring"),
        cfg_module.EdgeClaim(second_source, "Ijk_Call"),
    }
    assert pending.preserve_exact_addr


def test_queue_keeps_recovery_and_reconciliation_separate() -> None:
    """Keep distinct worklist actions separate even when their address matches."""

    session = _bare_session()
    recovery = cfg_module.RepairObligation(
        addr=0x1200,
        reason="recover_block",
    )
    reconciliation = cfg_module.RepairObligation(
        addr=0x1200,
        reason="reconcile_edges",
        action="reconcile",
    )

    assert session._queue_if_needed(recovery)
    assert session._queue_if_needed(reconciliation)
    assert list(session.queue) == [("recover", 0x1200), ("reconcile", 0x1200)]
    assert set(session.pending) == {("recover", 0x1200), ("reconcile", 0x1200)}


def test_live_delayed_direct_cfgfast_target_remains_a_recovery_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a delayed-branch CFGFast target while its source is repaired."""

    session = _bare_session()
    recovered_start = _MaterializedNode(0x1200)
    direct_target = _MaterializedNode(0x1210)
    monkeypatch.setattr(
        session,
        "_bound_nodes",
        lambda: (recovered_start, direct_target),
    )
    monkeypatch.setattr(session, "_is_preservable_seed_node", lambda _node: False)
    monkeypatch.setattr(session, "_node_has_delay_slot", lambda _node: True)
    monkeypatch.setattr(
        session,
        "_preserved_successor_starts",
        lambda node: (
            ((direct_target.addr,), None) if node is recovered_start else ((), None)
        ),
    )
    monkeypatch.setattr(session, "_nodes_at_addr", lambda _addr: [])

    assert 0x1210 in session.current_stop_addrs(0x1200)


def test_static_jump_diagnostics_count_final_unbounded_dispatchers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count a final unresolved dispatcher once by its missing proof."""

    session = _bare_session()
    dispatcher = _source_node(0x1234)
    monkeypatch.setattr(
        session, "_unresolved_indirect_dispatchers", lambda: (dispatcher,)
    )
    monkeypatch.setattr(
        session,
        "_static_jump_table_plan",
        lambda _node: (None, "unbounded_index"),
    )

    session._collect_static_jump_table_diagnostics()

    assert session.stats.static_jump_dispatchers_unresolved == 1
    assert session.stats.static_jump_unbounded_index == 1


def test_static_jump_table_removes_a_stale_unresolved_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Remove the unresolved leaf when every proven table target already exists."""

    session = _bare_session()
    session.project = object()
    source = _source_node(0x1234)
    concrete_target = _source_node(0x1200)
    unresolved_target = _source_node(0x601050)
    unresolved_target.is_simprocedure = True
    unresolved_target.simprocedure_name = "UnresolvableJumpTarget"
    session.graph = nx.DiGraph([(source, concrete_target), (source, unresolved_target)])
    session.resolved_static_table_sources = set()
    monkeypatch.setattr(session, "_bound_nodes", lambda: (source,))
    monkeypatch.setattr(
        session,
        "_static_jump_table_plan",
        lambda _node: (
            SimpleNamespace(table=object(), base_addr=0, entry_count=1),
            None,
        ),
    )
    monkeypatch.setattr(
        cfg_module,
        "_read_static_jump_table_targets",
        lambda *_args: (concrete_target.addr,),
    )
    monkeypatch.setattr(
        cfg_module,
        "_static_jump_target_rejection_reason",
        lambda *_args: None,
    )
    monkeypatch.setattr(
        session,
        "_resolve_successor",
        lambda *_args, **_kwargs: pytest.fail("no target should be materialized"),
    )

    assert session._resolve_static_jump_tables() == 1
    assert not session.graph.has_edge(source, unresolved_target)
    assert session.stats.unresolved_jump_edges_removed == 1


def test_static_jump_table_keeps_unresolved_target_for_invalid_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid partially resolving a table that contains an unsafe entry."""

    session = _bare_session()
    session.project = object()
    source = _source_node(0x1234)
    unresolved_target = _source_node(0x601050)
    unresolved_target.is_simprocedure = True
    unresolved_target.simprocedure_name = "UnresolvableJumpTarget"
    session.graph = nx.DiGraph([(source, unresolved_target)])
    session.resolved_static_table_sources = set()
    monkeypatch.setattr(session, "_bound_nodes", lambda: (source,))
    monkeypatch.setattr(
        session,
        "_static_jump_table_plan",
        lambda _node: (
            SimpleNamespace(table=object(), base_addr=0, entry_count=1),
            None,
        ),
    )
    monkeypatch.setattr(
        cfg_module,
        "_read_static_jump_table_targets",
        lambda *_args: (0x1200,),
    )
    monkeypatch.setattr(
        cfg_module,
        "_static_jump_target_rejection_reason",
        lambda *_args: "non_executable",
    )
    monkeypatch.setattr(
        session,
        "_resolve_successor",
        lambda *_args, **_kwargs: pytest.fail("invalid targets must not materialize"),
    )

    assert session._resolve_static_jump_tables() == 0
    assert session.graph.has_edge(source, unresolved_target)


def test_unresolved_fallback_skips_transparent_padding_candidates() -> None:
    """Attach unknown indirect candidates after no-op alignment padding."""

    session = _bare_session()
    fallback = _source_node(0x601050)
    fallback.is_simprocedure = True
    fallback.simprocedure_name = "UnresolvableJumpTarget"
    padding = _BlockNode(0x1100, bytes.fromhex("89f68d3f"))
    target = _source_node(0x1104)
    session.graph = nx.DiGraph()
    session.graph.add_edge(padding, target, jumpkind="Ijk_Boring")
    session._unresolved_jump_fallback_nodes = lambda: [fallback]
    session._disconnected_function_nodes = lambda: [padding, target]

    assert session._attach_unresolved_jump_fallbacks()
    assert not session.graph.has_edge(fallback, padding)
    assert session.graph.has_edge(fallback, target)


def test_unresolved_fallback_keeps_padding_with_a_known_predecessor() -> None:
    """Retain exact landing padding when an ordinary CFG edge targets it."""

    session = _bare_session()
    fallback = _source_node(0x601050)
    fallback.is_simprocedure = True
    fallback.simprocedure_name = "UnresolvableJumpTarget"
    source = _source_node(0x1000)
    padding = _BlockNode(0x1100, bytes.fromhex("89f68d3f"))
    target = _source_node(0x1104)
    session.graph = nx.DiGraph()
    session.graph.add_edge(source, padding, jumpkind="Ijk_Boring")
    session.graph.add_edge(padding, target, jumpkind="Ijk_Boring")
    session._unresolved_jump_fallback_nodes = lambda: [fallback]
    session._disconnected_function_nodes = lambda: [padding, target]

    assert session._attach_unresolved_jump_fallbacks()
    assert session.graph.has_edge(fallback, padding)


def test_unresolved_fallback_drops_padding_before_a_reachable_successor() -> None:
    """Do not add an unknown edge when padding already falls into live code."""

    session = _bare_session()
    fallback = _source_node(0x601050)
    fallback.is_simprocedure = True
    fallback.simprocedure_name = "UnresolvableJumpTarget"
    entry = _source_node(0x1000)
    padding = _BlockNode(0x1100, bytes.fromhex("89f68d3f"))
    target = _source_node(0x1104)
    entry.size = 1
    target.size = 1
    session.graph = nx.DiGraph()
    session.graph.add_edge(entry, target, jumpkind="Ijk_Boring")
    session.graph.add_edge(padding, target, jumpkind="Ijk_Boring")
    session._unresolved_jump_fallback_nodes = lambda: [fallback]
    session._disconnected_function_nodes = lambda: [padding]

    assert not session._attach_unresolved_jump_fallbacks()
    assert not session.graph.has_edge(fallback, padding)
    assert not session.graph.has_edge(fallback, target)


def test_unresolved_fallback_skips_disconnected_alternate_mode_nodes() -> None:
    """Avoid treating a disconnected alternate decode stream as a jump target."""

    session = _bare_session()
    fallback = _source_node(0x601050)
    fallback.is_simprocedure = True
    fallback.simprocedure_name = "UnresolvableJumpTarget"
    entry = _source_node(0x1000)
    entry.size = 1
    entry.thumb = False
    same_mode = _source_node(0x1100)
    same_mode.size = 1
    same_mode.thumb = False
    alternate_mode = _source_node(0x1201)
    alternate_mode.size = 1
    alternate_mode.thumb = True
    session.graph = nx.DiGraph()
    session.graph.add_node(entry)
    session._unresolved_jump_fallback_nodes = lambda: [fallback]
    session._disconnected_function_nodes = lambda: [same_mode, alternate_mode]

    assert session._attach_unresolved_jump_fallbacks()
    assert session.graph.has_edge(fallback, same_mode)
    assert not session.graph.has_edge(fallback, alternate_mode)


def test_immediate_entry_downgrades_to_queued_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep an unresolved immediate request intact when local recovery fails."""

    session = _bare_session()
    source = _source_node()
    request = cfg_module.RepairObligation(
        addr=0x1200,
        reason="missing_successor",
        source_node=source,
        jumpkind="Ijk_Call",
        preserve_exact_addr=True,
        resolution_policy="immediate",
    )
    placeholder = cast(CFGNode, object())
    queued: list[cfg_module.RepairObligation] = []

    monkeypatch.setattr(
        session,
        "_recovered_coverage_has_live_predecessor",
        lambda addr: False,
    )
    monkeypatch.setattr(session, "_recover_entry_now", lambda obligation: None)
    monkeypatch.setattr(session, "_resolve_covering_entry", lambda obligation: None)
    monkeypatch.setattr(session, "_nodes_at_addr", lambda addr: [])
    monkeypatch.setattr(session, "_claim_placeholder", lambda obligation: placeholder)
    monkeypatch.setattr(session, "_queue_if_needed", queued.append)

    assert session.ensure_block_entry(request) is placeholder
    assert queued == [
        cfg_module.RepairObligation(
            addr=0x1200,
            reason="missing_successor",
            source_node=source,
            jumpkind="Ijk_Call",
            preserve_exact_addr=True,
        )
    ]


def test_immediate_entry_returns_a_recovered_node_without_queuing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid placeholder and queue work when immediate recovery succeeds."""

    session = _bare_session()
    request = cfg_module.RepairObligation(
        addr=0x1200,
        reason="missing_successor",
        resolution_policy="immediate",
    )
    recovered = cast(CFGNode, object())

    monkeypatch.setattr(
        session,
        "_recovered_coverage_has_live_predecessor",
        lambda addr: False,
    )
    monkeypatch.setattr(session, "_recover_entry_now", lambda obligation: recovered)
    monkeypatch.setattr(session, "_resolve_covering_entry", lambda obligation: None)
    monkeypatch.setattr(
        session,
        "_claim_placeholder",
        lambda obligation: pytest.fail("successful recovery created a placeholder"),
    )

    assert session.ensure_block_entry(request) is recovered
    assert not session.pending
    assert not session.queue


def test_immediate_covered_entry_queues_the_split_before_local_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Split a covering block before immediately decoding an interior target."""

    session = _bare_session()
    request = cfg_module.RepairObligation(
        addr=0x1208,
        reason="missing_successor",
        preserve_exact_addr=True,
        resolution_policy="immediate",
    )
    placeholder = cast(CFGNode, object())
    covering_requests: list[cfg_module.RepairObligation] = []

    def resolve_covering(obligation: cfg_module.RepairObligation) -> CFGNode:
        covering_requests.append(obligation)
        return placeholder

    monkeypatch.setattr(
        session,
        "_recovered_coverage_has_live_predecessor",
        lambda addr: True,
    )
    monkeypatch.setattr(session, "_resolve_covering_entry", resolve_covering)
    monkeypatch.setattr(
        session,
        "_recover_entry_now",
        lambda obligation: pytest.fail("covered target bypassed split handling"),
    )

    assert session.ensure_block_entry(request) is placeholder
    assert covering_requests == [
        cfg_module.RepairObligation(
            addr=0x1208,
            reason="missing_successor",
            preserve_exact_addr=True,
        )
    ]


def test_recovered_coverage_requires_a_valid_predecessor_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect a recovered covering block only when an edge must enter it."""

    session = _bare_session()
    predecessor = _source_node(0x1100)
    covering_node = _source_node(0x1200)
    session.graph = nx.DiGraph([(predecessor, covering_node)])
    session.project = SimpleNamespace(arch=SimpleNamespace(max_inst_bytes=16))
    session.recovered_blocks = {
        covering_node: cfg_module.BlockSpec(
            addr=0x1200,
            size=16,
            instruction_addrs=(0x1200,),
            jumpkind="Ijk_Boring",
        )
    }
    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(cfg_module, "_decode_one", lambda *_args: None)
    monkeypatch.setattr(
        session,
        "_preserved_successor_starts",
        lambda node: ((covering_node.addr,), None),
    )

    assert session._recovered_coverage_has_live_predecessor(0x1208)

    monkeypatch.setattr(
        session,
        "_preserved_successor_starts",
        lambda node: ((0x1208,), None),
    )
    assert not session._recovered_coverage_has_live_predecessor(0x1208)


def test_post_prefix_target_does_not_protect_recovered_coverage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Allow a valid branch after a prefix to replace recovered coverage."""

    session = _bare_session()
    session.project = SimpleNamespace(arch=SimpleNamespace(max_inst_bytes=16))
    predecessor = _source_node(0x1100)
    covering_node = _source_node(0x1200)
    session.graph = nx.DiGraph([(predecessor, covering_node)])
    session.recovered_blocks = {
        covering_node: cfg_module.BlockSpec(
            addr=0x1200,
            size=3,
            instruction_addrs=(0x1200,),
            jumpkind="Ijk_Boring",
        )
    }
    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(
        session,
        "_preserved_successor_starts",
        lambda node: ((covering_node.addr,), None),
    )
    monkeypatch.setattr(
        cfg_module,
        "_decode_one",
        lambda *_args: SimpleNamespace(
            address=0x1200,
            size=3,
            prefix=(0xF0, 0, 0, 0),
        ),
    )

    assert not session._recovered_coverage_has_live_predecessor(0x1201)


def test_linear_split_defers_unresolved_call_edge_to_the_call_suffix() -> None:
    """Keep an indirect-call leaf while splitting off a linear prefix block."""

    session = _bare_session()
    unresolved_target = _source_node(0x601050)
    edge = (unresolved_target, "Ijk_Call")
    prefix = cfg_module.BlockSpec(
        addr=0x1100,
        size=4,
        instruction_addrs=(0x1100,),
        jumpkind="Ijk_Fallthrough",
        fallthrough_addr=0x1104,
    )
    suffix = cfg_module.BlockSpec(
        addr=0x1104,
        size=4,
        instruction_addrs=(0x1104,),
        jumpkind="Ijk_Call",
        fallthrough_addr=0x1108,
    )

    assert session._route_unresolved_control_edges(prefix, [edge]) == []
    assert session._deferred_unresolved_control_edges == {0x1104: [edge]}
    assert session._route_unresolved_control_edges(suffix, []) == [edge]
    assert not session._deferred_unresolved_control_edges


def test_covered_entry_queues_covering_repair_before_split_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the ordering required to expose a block entry through a node."""

    session = _bare_session()
    source = _source_node()
    covering_node = cast(CFGNode, SimpleNamespace(addr=0x1100, size=16))
    request = cfg_module.RepairObligation(
        addr=0x1108,
        reason="direct_target",
        source_node=source,
        preserve_exact_addr=True,
    )
    placeholder = cast(CFGNode, object())
    queued: list[cfg_module.RepairObligation] = []

    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(
        cfg_module,
        "_addr_is_mid_instruction_start",
        lambda *args: False,
    )
    monkeypatch.setattr(session, "_is_preservable_seed_node", lambda node: True)
    monkeypatch.setattr(session, "_claim_placeholder", lambda obligation: placeholder)
    monkeypatch.setattr(session, "_queue_if_needed", queued.append)

    assert session._resolve_covering_entry(request) is placeholder
    assert session.leaders.starts_with_reason("explicit_split") == {0x1108}
    assert session.mutation_revision == 1
    assert queued == [
        cfg_module.RepairObligation(addr=0x1100, reason="split_for_0x1108"),
        request,
    ]


def test_nonexact_mid_instruction_entry_preserves_a_valid_covering_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not create or repair a synthetic leader inside a valid instruction."""

    session = _bare_session()
    covering_node = cast(CFGNode, SimpleNamespace(addr=0x1100, size=16))
    request = cfg_module.RepairObligation(
        addr=0x1108,
        reason="invalid_mid_instruction_target",
    )
    queued: list[cfg_module.RepairObligation] = []

    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(
        cfg_module,
        "_addr_is_mid_instruction_start",
        lambda *args: True,
    )
    monkeypatch.setattr(session, "_is_preservable_seed_node", lambda node: True)
    monkeypatch.setattr(session, "_queue_if_needed", queued.append)
    monkeypatch.setattr(
        session,
        "_claim_placeholder",
        lambda obligation: pytest.fail("mid-instruction target created a placeholder"),
    )

    assert session._resolve_covering_entry(request) is covering_node
    assert session.leaders.starts_with_reason("explicit_split") == set()
    assert queued == []


def test_exact_thumb_mid_instruction_entry_queues_an_overlapping_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep a VEX-derived direct target as an alternate instruction stream."""

    session = _bare_session()
    source = _source_node()
    covering_node = cast(CFGNode, SimpleNamespace(addr=0x1100, size=16))
    request = cfg_module.RepairObligation(
        addr=0x1108,
        reason="direct_target",
        source_node=source,
        preserve_exact_addr=True,
    )
    placeholder = cast(CFGNode, object())
    queued: list[cfg_module.RepairObligation] = []

    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(cfg_module, "_addr_is_mid_instruction_start", lambda *_: True)
    monkeypatch.setattr(session, "_is_thumb_entry", lambda addr: True)
    monkeypatch.setattr(session, "_is_preservable_seed_node", lambda node: True)
    monkeypatch.setattr(session, "_claim_placeholder", lambda obligation: placeholder)
    monkeypatch.setattr(session, "_queue_if_needed", queued.append)

    assert session._resolve_covering_entry(request) is placeholder
    assert session._overlapping_entry_starts == {0x1108}
    assert session.leaders.starts_with_reason("explicit_split") == set()
    assert queued == [request]


def test_exact_nonthumb_mid_instruction_entry_preserves_covering_node(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a malformed non-Thumb direct target inside one instruction."""

    session = _bare_session()
    covering_node = cast(CFGNode, SimpleNamespace(addr=0x1100, size=16))
    request = cfg_module.RepairObligation(
        addr=0x1108,
        reason="direct_target",
        preserve_exact_addr=True,
    )
    queued: list[cfg_module.RepairObligation] = []

    monkeypatch.setattr(session, "_covering_nodes", lambda addr: [covering_node])
    monkeypatch.setattr(cfg_module, "_addr_is_mid_instruction_start", lambda *_: True)
    monkeypatch.setattr(session, "_is_preservable_seed_node", lambda node: True)
    monkeypatch.setattr(session, "_queue_if_needed", queued.append)
    monkeypatch.setattr(
        session,
        "_claim_placeholder",
        lambda obligation: pytest.fail("non-Thumb target created a placeholder"),
    )

    assert session._resolve_covering_entry(request) is covering_node
    assert session._overlapping_entry_starts == set()
    assert queued == []


def test_stalled_requeue_fails_when_pending_state_is_unchanged() -> None:
    """Reject a requeued obligation that cannot produce new repair work."""

    session = _bare_session()
    request = cfg_module.RepairObligation(addr=0x1200, reason="stalled")
    key = ("recover", 0x1200)
    session._queue_if_needed(request)
    obligation = session.pending[key]

    session._record_obligation_progress(key, obligation)

    with pytest.raises(
        RuntimeError, match="graph and pending repair state did not change"
    ):
        session._record_obligation_progress(key, obligation)


def test_stalled_requeue_ignores_new_diagnostic_reasons() -> None:
    """Reject a retry when only diagnostic text changed in the pending work."""

    session = _bare_session()
    key = ("recover", 0x1200)
    session._queue_if_needed(cfg_module.RepairObligation(addr=0x1200, reason="first"))
    obligation = session.pending[key]

    session._record_obligation_progress(key, obligation)
    obligation.merge(cfg_module.RepairObligation(addr=0x1200, reason="second"))

    with pytest.raises(
        RuntimeError, match="graph and pending repair state did not change"
    ):
        session._record_obligation_progress(key, obligation)


def test_stalled_requeue_allows_a_new_edge_claim() -> None:
    """Allow another attempt when merging a claim changes pending repair state."""

    session = _bare_session()
    first_source = _source_node()
    second_source = _source_node()
    key = ("recover", 0x1200)
    session._queue_if_needed(
        cfg_module.RepairObligation(
            addr=0x1200,
            reason="first_source",
            source_node=first_source,
        )
    )
    obligation = session.pending[key]

    session._record_obligation_progress(key, obligation)
    obligation.merge(
        cfg_module.RepairObligation(
            addr=0x1200,
            reason="second_source",
            source_node=second_source,
        )
    )
    session._record_obligation_progress(key, obligation)

    with pytest.raises(
        RuntimeError, match="graph and pending repair state did not change"
    ):
        session._record_obligation_progress(key, obligation)


def test_stalled_requeue_allows_a_graph_mutation() -> None:
    """Allow another attempt when another repair action changed the live graph."""

    session = _bare_session()
    key = ("recover", 0x1200)
    session._queue_if_needed(
        cfg_module.RepairObligation(addr=0x1200, reason="first_attempt")
    )
    obligation = session.pending[key]

    session._record_obligation_progress(key, obligation)
    session.mutation_revision += 1
    session._record_obligation_progress(key, obligation)

    with pytest.raises(
        RuntimeError, match="graph and pending repair state did not change"
    ):
        session._record_obligation_progress(key, obligation)


def test_external_target_node_is_created_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one synthetic leaf when multiple edges target the same external addr."""

    graph = _NodeGraph()
    seed_cfg = cast(CFGBase, object())
    external_node = SimpleNamespace(addr=0x4000, is_simprocedure=True)
    monkeypatch.setattr(
        nodes_module,
        "make_external_target_node",
        lambda *args: external_node,
    )

    first, first_created = nodes_module.ensure_external_target_node(
        seed_cfg,
        cast(graph_module.CFGGraph, graph),
        0x1000,
        0x4000,
    )
    second, second_created = nodes_module.ensure_external_target_node(
        seed_cfg,
        cast(graph_module.CFGGraph, graph),
        0x1000,
        0x4000,
    )

    assert first is external_node
    assert second is external_node
    assert first_created
    assert not second_created
    assert graph.nodes() == [external_node]


def test_undecodable_target_node_is_created_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reuse one terminal leaf when multiple branches hit undecodable bytes."""

    graph = _NodeGraph()
    seed_cfg = cast(CFGBase, object())
    undecodable_node = SimpleNamespace(
        addr=0x1200,
        is_simprocedure=True,
        simprocedure_name="UndecodableInstructionTarget",
    )
    monkeypatch.setattr(
        nodes_module,
        "make_undecodable_target_node",
        lambda *args: undecodable_node,
    )

    first, first_created = nodes_module.ensure_undecodable_target_node(
        seed_cfg,
        cast(graph_module.CFGGraph, graph),
        0x1000,
        0x1200,
    )
    second, second_created = nodes_module.ensure_undecodable_target_node(
        seed_cfg,
        cast(graph_module.CFGGraph, graph),
        0x1000,
        0x1200,
    )

    assert first is undecodable_node
    assert second is undecodable_node
    assert first_created
    assert not second_created
    assert graph.nodes() == [undecodable_node]


def test_requeue_preserves_merged_claims_and_reconciliation_action() -> None:
    """Requeue all claims without losing the repair action or exact-start flag."""

    session = _bare_session()
    first_source = _source_node()
    second_source = _source_node()
    obligation = cfg_module.PendingObligation(
        addr=0x1200,
        action="reconcile",
        reasons={"first", "second"},
        edge_claims={
            cfg_module.EdgeClaim(first_source, "Ijk_Boring"),
            cfg_module.EdgeClaim(second_source, "Ijk_Call"),
        },
        preserve_exact_addr=True,
    )

    session._requeue_pending(obligation)

    pending = session.pending[("reconcile", 0x1200)]
    assert list(session.queue) == [("reconcile", 0x1200)]
    assert pending.reasons == {"first, second"}
    assert pending.edge_claims == obligation.edge_claims
    assert pending.preserve_exact_addr


def test_splice_drops_removed_alternate_mode_stream_edge() -> None:
    """Do not rewire a stale Thumb stream into an overlapping ARM block."""

    session = _bare_session()
    thumb_source = cast(CFGNode, SimpleNamespace(thumb=True))
    thumb_target = cast(CFGNode, SimpleNamespace(thumb=True))
    arm_source = cast(CFGNode, SimpleNamespace(thumb=False))
    session.graph = SimpleNamespace(predecessors=lambda _node: (thumb_source,))

    assert not session._rewire_incoming_overlap_edge(thumb_source, thumb_target, False)
    assert session._rewire_incoming_overlap_edge(arm_source, thumb_target, False)

    session.graph = SimpleNamespace(
        predecessors=lambda _node: (thumb_source, arm_source)
    )
    assert session._rewire_incoming_overlap_edge(thumb_source, thumb_target, False)

    unresolved = cast(
        CFGNode,
        SimpleNamespace(
            is_simprocedure=True,
            simprocedure_name="UnresolvableJumpTarget",
        ),
    )
    session.graph = SimpleNamespace(
        predecessors=lambda _node: (thumb_source, unresolved)
    )
    assert session._rewire_incoming_overlap_edge(thumb_source, thumb_target, False)
