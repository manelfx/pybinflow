"""Fast unit tests for the custom CFG repair worklist contracts."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import cast

import pytest
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg import repair as cfg_module


def _bare_session() -> cfg_module._RepairSession:
    """Allocate only the repair-session state needed by these focused tests."""

    session = object.__new__(cfg_module._RepairSession)
    session.queue = deque()
    session.pending = {}
    session.leaders = cfg_module.BlockLeaderRegistry({})
    session.mutation_revision = 0
    session.last_requeue_states = {}
    session.stats = cfg_module.CustomCFGStats()
    session.bounds = cast(
        cfg_module.FunctionBounds,
        SimpleNamespace(addr=0x1000, end_addr=0x2000),
    )
    session.graph = object()
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

    monkeypatch.setattr(session, "_recover_entry_now", lambda obligation: recovered)
    monkeypatch.setattr(
        session,
        "_claim_placeholder",
        lambda obligation: pytest.fail("successful recovery created a placeholder"),
    )

    assert session.ensure_block_entry(request) is recovered
    assert not session.pending
    assert not session.queue


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


def test_mid_instruction_entry_repairs_the_covering_node_without_a_split(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not preserve a synthetic leader in the middle of a valid instruction."""

    session = _bare_session()
    covering_node = cast(CFGNode, SimpleNamespace(addr=0x1100, size=16))
    request = cfg_module.RepairObligation(
        addr=0x1108,
        reason="invalid_mid_instruction_target",
        preserve_exact_addr=True,
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
    assert queued == [
        cfg_module.RepairObligation(
            addr=0x1100,
            reason="covering_node_for_0x1108",
        )
    ]


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
        cfg_module,
        "_make_external_target_node",
        lambda *args: external_node,
    )

    first, first_created = cfg_module._ensure_external_target_node(
        seed_cfg,
        cast(cfg_module.CFGGraph, graph),
        0x1000,
        0x4000,
    )
    second, second_created = cfg_module._ensure_external_target_node(
        seed_cfg,
        cast(cfg_module.CFGGraph, graph),
        0x1000,
        0x4000,
    )

    assert first is external_node
    assert second is external_node
    assert first_created
    assert not second_created
    assert graph.nodes() == [external_node]


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
