"""
Custom CFG discovery and repair helpers.

This module implements the `cfg_mode="custom"` path as a localized repair pass
over a bounded CFGFast graph. The intent is to preserve angr metadata and any
already-correct CFGFast structure while patching only the malformed parts of the
graph.

Terminology used throughout the module:

- seed CFG:
  The initial bounded CFGFast graph for one function.
- anomaly:
  A block shape we do not trust, such as malformed byte coverage, a missing
  jump successor, or a truncated leaf.
- repair:
  Replacing stale seed nodes with newly decoded blocks while preserving good
  incoming/outgoing structure around them.
- obligation:
  A queued worklist item for either recovering a block entry or reconciling a
  live block's local invariants. Requests for the same action and address are
  merged, retaining all their edge claims, split requirements, and reasons.
- edge claim:
  A required outgoing edge from one exact source node to an obligation's
  address. Identity matters because distinct CFG nodes can share an address.
- resolution policy:
  Immediate resolution attempts local recovery for a missing successor of a
  live node. If it cannot recover that target, the request becomes ordinary
  queued work; all other requests begin queued.
- leader:
  An address that must begin a recovered block. The repair session derives
  leaders from the function entry, direct targets, fallthroughs, and explicit
  splits through stale nodes, then uses them to bound local decoding.
- reconciliation:
  The local worklist action that restores required direct successors for a live
  node and queues recovery again only if the node still violates an invariant.
- terminator:
  The control-transfer summary of a recovered block: return, call, direct jump,
  or plain fallthrough, plus any direct targets or fallthrough address.
- placeholder:
  A temporary zero-sized target node that records an unresolved block entry.
  It may receive incoming edge claims, but never supplies control-flow
  semantics itself; recovery replaces it with a decoded block or cleanup
  removes it.
- unresolved-jump fallback:
  An `UnresolvableJumpTarget` simprocedure keeps an indirect dispatch visible
  when static recovery cannot prove its targets. It is connected to otherwise
  disconnected in-function blocks so cleanup preserves those CFGFast-discovered
  regions without inventing direct case edges from the original dispatch.

High-level algorithm:

1. Build a bounded CFGFast graph for the target function.
2. If the seed graph shows no known anomalies, return it unchanged.
3. Otherwise, seed a worklist with one recovery obligation per anomalous
   address. The queue merges requests for the same action and address.
4. Dispatch each obligation:
   - recovery resolves an existing entry, a covered entry, or a placeholder;
     it decodes and splices a bounded replacement block when required,
   - reconciliation restores direct successor edges and requeues recovery only
     for nodes that still violate an invariant,
   - both actions may add edge claims, leaders, and neighboring reconciliation
     work as the live graph changes.
5. Reject a requeued obligation when neither the graph revision nor its merged
   repair state changed. A separate iteration limit protects against a graph
   that continues changing without converging.
6. When the worklist is empty, connect any remaining unresolved indirect-jump
   placeholder to disconnected seed blocks. A placeholder with exactly one
   indirect source is flattened into explicitly marked unresolved candidate
   edges, preserving uncertainty without retaining a synthetic intermediary.
   Then remove genuinely unreachable stale nodes and temporary placeholders.
   These preserving edges deliberately do not claim new basic-block leaders.
   Cleanup can expose a new local anomaly, such as a linear split whose extra
   predecessors were stale. If cleanup changed the graph, queue those newly
   visible anomalies for another worklist pass. If cleanup made no change,
   report any remaining anomalies instead of retrying them indefinitely.
7. Expose the repaired graph through a small CFG-like wrapper.

The implementation is intentionally conservative. It prefers localized repairs
over whole-function reconstruction so already-correct arch-specific behavior
from CFGFast remains intact.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from types import SimpleNamespace

from angr import KnowledgeBase, Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger

from .graph import (
    add_successor_edge as _add_successor_edge,
    cfg_graph as _cfg_graph,
    cleanup_unreachable_function_nodes as _cleanup_unreachable_function_nodes,
    is_unresolvable_control_target as _is_unresolvable_control_target,
    is_unresolvable_jump_target as _is_unresolvable_jump_target,
    iter_graph_bound_nodes as _iter_graph_bound_nodes,
    node_ends_in_indirect_jump as _node_ends_in_indirect_jump,
    node_intersects_bounds as _node_intersects_bounds,
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
    node_is_placeholder as _node_is_placeholder,
    node_range_end as _node_range_end,
    node_vex as _node_vex,
    ranges_overlap as _ranges_overlap,
    remove_nodes as _remove_nodes,
)
from .jumps import (
    is_direct_target_valid as _is_direct_target_valid,
)
from .decode import (
    DecodedNode,
    clear_decoded_node_cache as _clear_decoded_node_cache,
    decode_one as _decode_one,
)
from .models import (
    BlockLeaderRegistry,
    BlockSpec,
    CustomCFG,
    CustomCFGStats,
    EdgeClaim,
    EdgeJumpKind,
    EntryResolutionPolicy,
    PendingObligation,
    RepairObligation,
)
from .anomalies import (
    CFGAnomalyDetector,
    _iter_seed_function_nodes,
    _lookup_function_bounds,
    node_has_linear_merge_successor,
)
from .jumps import (
    _constant_register_from_predecessors,
    _guarded_jump_table_entry_count,
    _in_function_jump_table_entry_count,
    _read_static_jump_table_targets,
    _seed_graph_direct_targets,
    _seed_node_expected_successors,
    _unique_static_register_value,
    _vex_relative_jump_table,
    _x86_pc_thunk_base_addr,
)
from .nodes import (
    ensure_external_target_node as _ensure_external_target_node,
    ensure_undecodable_target_node as _ensure_undecodable_target_node,
    make_cfg_node as _make_cfg_node,
    make_placeholder_node as _make_placeholder_node,
    prune_orphan_simprocedures as _prune_orphan_simprocedures,
    prune_placeholders as _prune_placeholders,
)
from .recovery import (
    find_shared_instruction_tail as _find_shared_instruction_tail,
    recover_block as _recover_block,
)


# Last-resort protection for repair loops that keep mutating the graph without
# converging. Stable requeues are diagnosed earlier by PendingObligation state.
MAX_CUSTOM_CFG_WORKLIST_ITERATIONS = 5_000


def _custom_model_marker() -> SimpleNamespace:
    """Return the minimal model metadata currently needed by callers."""

    return SimpleNamespace(ident="CFGFastCustom")


def _node_has_forced_split(node, forced_block_starts: set[int]) -> bool:
    """Return True when a known block start falls inside this node's range."""

    node_end = _node_range_end(node)
    return any(node.addr < addr < node_end for addr in forced_block_starts)


def _addr_is_mid_instruction_start(node, addr: int) -> bool:
    """
    Return True when `addr` falls inside one decoded instruction of `node`.

    This is stricter than merely checking whether `addr` is covered by the
    node's nominal byte range. Malformed CFGFast nodes often advertise a stale
    size that extends beyond the last decoded instruction. Those trailing bytes
    may still be legitimate new block leaders and must not be suppressed.
    """

    try:
        return DecodedNode.from_node(node).contains_mid_instruction_addr(addr)
    except Exception:
        return False


def _block_has_unresolved_indirect_transfer(block: BlockSpec) -> bool:
    """Return whether a recovered block needs a preserved unresolved leaf."""

    if block.direct_targets:
        return False
    return (
        block.jumpkind == "Ijk_Boring" and block.fallthrough_addr is None
    ) or block.jumpkind == "Ijk_Call"


class _RepairSession:
    """Mutable state and helpers for one worklist-driven CFG repair run."""

    def __init__(self, project: Project, seed_cfg: CFGBase, func_addr: int):
        self.project = project
        self.seed_cfg = seed_cfg
        self.func_addr = func_addr
        seed_function = seed_cfg.kb.functions.get(func_addr)
        seed_name = getattr(seed_function, "name", None)
        self.bounds = _lookup_function_bounds(
            project,
            func_addr,
            display_name=seed_name
            if isinstance(seed_name, str) and seed_name
            else None,
        )
        # Mutate the live SpillingCFG wrapper in place, but stay on its public
        # API. Its private backing graph stores tuple keys that the renderer
        # cannot consume directly.
        self.graph = _cfg_graph(seed_cfg)
        self.anomalies = CFGAnomalyDetector(project, self.graph, self.bounds, func_addr)
        self.mutation_revision = 0
        self._bound_nodes_revision = -1
        self._bound_nodes_snapshot: tuple[CFGNode, ...] = ()
        self.stats = CustomCFGStats()
        self.stats.input_blocks, self.stats.input_edges = self._graph_shape()
        self.queue: deque[tuple[str, int]] = deque()
        self.pending: dict[tuple[str, int], PendingObligation] = {}
        self.repaired_nodes: set[CFGNode] = set()
        self.recovered_blocks: dict[CFGNode, BlockSpec] = {}
        self.leaders = BlockLeaderRegistry({func_addr: {"function_entry"}})
        self.processed_counts: dict[int, int] = {}
        self.last_requeue_states: dict[
            tuple[str, int],
            tuple[int, tuple[bool, tuple[tuple[int, EdgeJumpKind], ...]]],
        ] = {}
        self.resolved_static_table_sources: set[int] = set()
        self.iterations = 0
        self._stop_starts_revision = -1
        self._stop_starts: set[int] = set()

    def _graph_shape(self) -> tuple[int, int]:
        """Return the number of in-bounds blocks and their outgoing CFG edges."""

        blocks = {
            node for node in self._bound_nodes() if _node_is_materialized_cfg_node(node)
        }
        edges = sum(
            1 for source, _, _ in self.graph.edges(data=True) if source in blocks
        )
        return len(blocks), edges

    def _bound_nodes(self) -> tuple[CFGNode, ...]:
        """Return a revision-scoped snapshot of live in-bounds CFG nodes."""

        if self._bound_nodes_revision != self.mutation_revision:
            self._bound_nodes_snapshot = tuple(
                _iter_graph_bound_nodes(self.graph, self.bounds)
            )
            self._bound_nodes_revision = self.mutation_revision
        return self._bound_nodes_snapshot

    def _nodes_at_addr(self, addr: int) -> list[CFGNode]:
        """Return cached in-bounds nodes beginning at ``addr``."""

        return [node for node in self._bound_nodes() if node.addr == addr]

    def _covering_nodes(self, addr: int) -> list[CFGNode]:
        """Return cached in-bounds nodes whose byte range covers ``addr``."""

        return [
            node
            for node in self._bound_nodes()
            if node.addr <= addr < _node_range_end(node)
        ]

    def log_stats(self) -> None:
        """Capture the final graph shape and emit one custom-repair summary."""

        try:
            self.stats.output_blocks, self.stats.output_edges = self._graph_shape()
            self.stats.output_anomalies = len(self._anomalous_addrs())
        except Exception as exc:
            logger.warning(
                f"Custom CFG could not finish collecting stats for {self.func_addr:#x}: "
                f"{exc}"
            )
        logger.info(
            f"Custom CFG stats for function {self.func_addr:#x}: {self.stats.as_dict()}"
        )

    def _is_preservable_seed_node(self, node) -> bool:
        """
        Return True when an unrepaired seed node may still preserve CFG structure.

        Malformed seed fragments should not freeze extra block leaders or
        successor expectations. Once repaired, the replacement node will drive
        discovery with its own recovered semantics.
        """

        return node in self.repaired_nodes or self.anomalies.node_is_acceptable(
            self._explicit_split_starts(),
            node,
        )

    def _is_required_prefixed_instruction_entry(
        self,
        node: CFGNode,
        block: BlockSpec,
    ) -> bool:
        """Return whether ``node`` is a direct target after an instruction prefix.

        Some binaries intentionally branch after a prefix byte. For example,
        x86 may target the instruction following a LOCK prefix while another
        path executes the prefixed form. Preserve that alternate, valid entry
        only when the seed graph proves that it is a direct branch target.
        A branch into any other part of a decoded instruction is a malformed
        CFGFast edge, not an alternative instruction stream.
        """

        if not self._is_preservable_seed_node(node):
            return False

        has_direct_predecessor = any(
            node.addr in _seed_graph_direct_targets(self.graph, predecessor)
            for predecessor in self.graph.predecessors(node)
            if self._is_preservable_seed_node(predecessor)
        )
        if not has_direct_predecessor:
            return False

        max_inst_bytes = getattr(self.project.arch, "max_inst_bytes", 16)
        insn = _decode_one(self.project, block.addr, max_inst_bytes)
        if insn is None:
            return False

        # Capstone exposes leading instruction prefixes separately from the
        # opcode. Entering just after all of them is an intentional alternate
        # stream; other interior byte offsets are not.
        prefix_size = sum(1 for prefix in getattr(insn, "prefix", ()) if prefix)
        return (
            prefix_size > 0
            and node.addr == insn.address + prefix_size
            and node.addr < insn.address + insn.size
        )

    def _note_mutation(self) -> None:
        """Advance the revision after a live CFG graph mutation."""

        self.mutation_revision += 1

    def _add_edge(
        self,
        src: CFGNode,
        dst: CFGNode,
        jumpkind: EdgeJumpKind,
        *,
        unresolved_indirect: bool = False,
    ) -> bool:
        """Add an edge and record whether it changed the live graph."""

        changed = _add_successor_edge(
            self.graph,
            src,
            dst,
            jumpkind,
            unresolved_indirect=unresolved_indirect,
        )
        if changed:
            self.stats.edges_added += 1
            self._note_mutation()
        return changed

    def _canonicalize_function_ownership(self) -> None:
        """Assign all in-bounds custom-CFG blocks to the requested function."""

        reassigned = 0
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue
            if node.function_address == self.func_addr:
                continue
            # CFGFast can create provisional functions for disconnected code
            # regions. Once custom repair includes those in the symbol-bounded
            # graph, renderers must not hide them based on that stale owner.
            node.function_address = self.func_addr
            reassigned += 1
        if reassigned:
            self.stats.function_owners_canonicalized += reassigned
            logger.info(
                f"Assigned {reassigned} in-bounds CFG block(s) to function "
                f"{self.func_addr:#x}"
            )

    def _resolve_static_jump_tables(self) -> int:
        """Recover high-confidence relative table targets from unresolved jumps."""

        resolved_sources = 0
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue

            vex = _node_vex(node)
            if vex is None:
                continue
            table = _vex_relative_jump_table(vex)
            pic_base_addr = None
            if (
                table is None
                and self.project.arch.name == "X86"
                and self.project.arch.bits == 32
            ):
                pic_table = _vex_relative_jump_table(vex, allow_full_width_index=True)
                if pic_table is not None:
                    pic_base_addr = _x86_pc_thunk_base_addr(
                        self.project,
                        self.graph,
                        self.bounds,
                        node,
                        pic_table,
                    )
                    if pic_base_addr is not None:
                        table = pic_table
            if table is None or table.base_bits != self.project.arch.bits:
                continue
            entry_count = _guarded_jump_table_entry_count(
                self.graph,
                self.bounds,
                node,
                table,
            )
            base_addr = pic_base_addr
            if base_addr is None:
                base_addr = _constant_register_from_predecessors(
                    self.graph,
                    self.bounds,
                    node,
                    table.base_register_offset,
                )
            if base_addr is None:
                # A disconnected table dispatcher may not have a complete
                # predecessor path back to its base definition. Scan the
                # bounded VEX blocks instead, but accept a value only when all
                # static definitions for this register agree.
                base_addr = _unique_static_register_value(
                    self.graph,
                    self.bounds,
                    table.base_register_offset,
                )
            if base_addr is None:
                continue
            if entry_count is None and pic_base_addr is not None:
                entry_count = _in_function_jump_table_entry_count(
                    self.project,
                    self.bounds,
                    table,
                    base_addr,
                )
            if entry_count is None:
                continue
            targets = _read_static_jump_table_targets(
                self.project,
                table,
                base_addr,
                entry_count,
            )
            if not targets:
                continue

            existing_target_addrs = {
                successor.addr for successor in self.graph.successors(node)
            }
            missing_targets = [
                target for target in targets if target not in existing_target_addrs
            ]
            if not missing_targets:
                unresolved_targets = []
            else:
                unresolved_targets = [
                    successor
                    for successor in self.graph.successors(node)
                    if _is_unresolvable_jump_target(successor)
                ]

            if not missing_targets and not unresolved_targets:
                continue

            logger.info(
                f"Resolved static jump table at {node.addr:#x} with "
                f"{len(missing_targets)} missing target(s)"
            )
            for target_addr in missing_targets:
                self._resolve_successor(
                    node,
                    target_addr,
                    "Ijk_Boring",
                    reason=f"static_jump_table_of_{node.addr:#x}",
                    preserve_exact_addr=True,
                    materialize_external=True,
                )

            # The guarded VEX form and every table entry have now been proven,
            # so the original indirect-jump placeholder is no longer needed.
            for unresolved_target in unresolved_targets:
                self.graph.remove_edge(node, unresolved_target)
                self._note_mutation()
            self.stats.static_jump_targets_added += len(missing_targets)
            self.stats.unresolved_jump_edges_removed += len(unresolved_targets)
            self.resolved_static_table_sources.add(node.addr)
            self.stats.static_jump_tables_resolved = len(
                self.resolved_static_table_sources
            )
            resolved_sources += 1

        return resolved_sources

    def _unresolved_jump_fallback_nodes(self) -> list[CFGNode]:
        """Return synthetic leaves still reached from unresolved indirect jumps."""

        return sorted(
            {
                successor
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node)
                for successor in self.graph.successors(node)
                if _is_unresolvable_jump_target(successor)
            },
            key=lambda node: node.addr,
        )

    def _reachable_from_entry(self) -> set[CFGNode]:
        """Return all graph nodes reachable through the current entry edges."""

        reachable: set[CFGNode] = set()
        queue: deque[CFGNode] = deque(self._nodes_at_addr(self.func_addr))
        while queue:
            node = queue.popleft()
            if node in reachable:
                continue
            reachable.add(node)
            queue.extend(self.graph.successors(node))
        return reachable

    def _disconnected_function_nodes(self) -> list[CFGNode]:
        """Return every in-function node disconnected from the entry graph."""

        reachable = self._reachable_from_entry()
        return sorted(
            (
                node
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node) and node not in reachable
            ),
            key=lambda node: node.addr,
        )

    def _attach_unresolved_jump_fallbacks(self) -> bool:
        """Keep disconnected seed regions reachable through unresolved jump leaves."""

        fallback_nodes = self._unresolved_jump_fallback_nodes()
        if not fallback_nodes:
            return False
        disconnected_nodes = self._disconnected_function_nodes()
        changed = False
        for fallback in fallback_nodes:
            for node in disconnected_nodes:
                if self._add_edge(fallback, node, "Ijk_Boring"):
                    self.stats.unresolved_fallback_edges_added += 1
                    changed = True
        if changed:
            logger.info(
                f"Connected {len(disconnected_nodes)} disconnected function block(s) through "
                f"{len(fallback_nodes)} unresolved indirect-jump target(s)"
            )
        return changed

    def _flatten_single_source_unresolved_fallback(self) -> bool:
        """Replace one unambiguous unresolved-jump placeholder with candidate edges.

        A single indirect-jump source and a single
        ``UnresolvableJumpTarget`` form a synthetic intermediary rather than a
        meaningful CFG block. Once its in-function candidate targets have been
        attached, move those edges to the dispatcher itself while retaining
        their ``unresolved_indirect`` marker for rendering. Multiple fallback
        nodes, extra dispatcher successors, or non-local targets are left
        untouched because the intermediary still carries useful structure.
        """

        fallbacks = self._unresolved_jump_fallback_nodes()
        if len(fallbacks) != 1:
            return False

        fallback = fallbacks[0]
        predecessors = list(self.graph.predecessors(fallback))
        targets = list(self.graph.successors(fallback))
        if len(predecessors) != 1 or not targets:
            return False

        source = predecessors[0]
        if (
            not _node_is_materialized_cfg_node(source)
            or not _node_intersects_bounds(source, self.bounds)
            or not _node_ends_in_indirect_jump(source)
            or list(self.graph.successors(source)) != [fallback]
            or any(
                not _node_is_materialized_cfg_node(target)
                or not _node_intersects_bounds(target, self.bounds)
                for target in targets
            )
        ):
            return False

        for target in targets:
            edge_data = self.graph.get_edge_data(fallback, target) or {}
            self._add_edge(
                source,
                target,
                edge_data.get("jumpkind", "Ijk_Boring"),
                unresolved_indirect=True,
            )

        self.graph.remove_edge(source, fallback)
        self.graph.remove_node(fallback)
        self.stats.unresolved_fallbacks_flattened += 1
        self.stats.unresolved_candidate_edges_flattened += len(targets)
        self._note_mutation()
        logger.info(
            f"Flattened unresolved indirect jump at {source.addr:#x} to "
            f"{len(targets)} in-function candidate target(s)"
        )
        return True

    def _record_obligation_progress(
        self,
        key: tuple[str, int],
        obligation: PendingObligation,
    ) -> None:
        """Reject a requeued obligation whose graph and repair state are unchanged."""

        pending = self.pending.get(key)
        if pending is None:
            self.last_requeue_states.pop(key, None)
            return

        state = self.mutation_revision, pending.fingerprint()
        previous_state = self.last_requeue_states.get(key)
        if state != previous_state:
            self.last_requeue_states[key] = state
            return

        reasons = ", ".join(sorted(pending.reasons))
        raise RuntimeError(
            f"Custom CFG stalled while {obligation.action} at {obligation.addr:#x}: "
            "the graph and pending repair state did not change "
            f"(reasons: {reasons})"
        )

    def _preserved_successor_starts(self, node) -> tuple[tuple[int, ...], int | None]:
        """
        Return the starts that `node` is allowed to preserve in the live graph.

        Repaired nodes preserve both direct targets and true fallthroughs. Seed
        nodes only preserve direct targets already materialized in the graph,
        since CFGFast fallthrough splits are often the very artifacts we are
        trying to erase.
        """

        recovered = self.recovered_blocks.get(node)
        if recovered is not None:
            return recovered.direct_targets, recovered.fallthrough_addr

        return _seed_graph_direct_targets(self.graph, node), None

    def _explicit_split_starts(self) -> set[int]:
        """Return leaders created by a requested split through an old node."""

        return self.leaders.starts_with_reason("explicit_split")

    def ensure_block_entry(
        self,
        obligation: RepairObligation,
    ) -> CFGNode | None:
        """
        Ensure one obligation is represented in the graph and queued for repair.

        If the obligation address is currently covered by a larger node, this
        method also decides whether we should re-run repair at the covering
        node's start or at the requested split address itself.
        """

        addr = obligation.addr
        if not (self.bounds.addr <= addr < self.bounds.end_addr):
            return None

        if obligation.resolution_policy == "immediate":
            recovered = self._recover_entry_now(obligation)
            if recovered is not None:
                return recovered
            # An immediate local attempt is a convergence aid, not a separate
            # execution path. Once it cannot recover the entry, hand the same
            # request to normal worklist processing.
            obligation = replace(obligation, resolution_policy="queued")

        covering_entry = self._resolve_covering_entry(obligation)
        if covering_entry is not None:
            return covering_entry

        for node in self._nodes_at_addr(addr):
            if self._is_preservable_seed_node(node):
                self._connect_source_to_node(obligation, node)
                return node

        placeholder = self._claim_placeholder(obligation)
        self._queue_if_needed(obligation)
        return placeholder

    def _resolve_covering_entry(
        self,
        obligation: RepairObligation,
    ) -> CFGNode | None:
        """Resolve an entry covered by an existing node, or leave it deferred."""

        addr = obligation.addr
        for node in self._covering_nodes(addr):
            if node.addr == addr or _node_is_placeholder(node):
                continue

            if _addr_is_mid_instruction_start(node, addr):
                allow_exact_split = (
                    obligation.preserve_exact_addr
                    and not self._is_preservable_seed_node(node)
                )
                if allow_exact_split:
                    continue
                # The requested address falls inside a decoded instruction of
                # a covering node. Unless this is an explicit direct-branch
                # target punching through a stale covering node, do not freeze
                # the byte offset as a synthetic block leader: it only causes
                # placeholder ping-pong around invalid starts such as 0x4358ff
                # / 0x43597e in __strstr_avx512. Legitimate taken targets like
                # 0x42367a / 0x445a5e still get through when the covering node
                # is itself stale.
                if not self._is_preservable_seed_node(node):
                    self._queue_if_needed(
                        RepairObligation(
                            addr=node.addr,
                            reason=f"covering_node_for_{addr:#x}",
                        )
                    )
                return node

            if self.leaders.add(addr, "explicit_split"):
                self.stats.explicit_splits += 1
                self._note_mutation()
            placeholder = self._claim_placeholder(obligation)

            repair_addr = node.addr
            repair_reason = f"split_for_{addr:#x}"
            if not self._is_preservable_seed_node(node):
                repair_addr = addr
                repair_reason = obligation.reason

            self._queue_if_needed(
                RepairObligation(
                    addr=repair_addr,
                    reason=repair_reason,
                    source_node=obligation.source_node if repair_addr == addr else None,
                    jumpkind=obligation.jumpkind,
                )
            )
            # Keep the requested split point alive as its own obligation. The
            # covering block must be repaired first, but we still need a later
            # pass to materialize the block that starts exactly at `addr`.
            self._queue_if_needed(obligation)
            return placeholder

        return None

    def _claim_placeholder(
        self,
        obligation: RepairObligation,
    ) -> CFGNode:
        """Get the target placeholder for an entry and attach its edge claims."""

        placeholder = next(
            (
                node
                for node in self._nodes_at_addr(obligation.addr)
                if _node_is_placeholder(node)
            ),
            None,
        )
        if placeholder is None:
            placeholder = _make_placeholder_node(
                self.seed_cfg,
                self.func_addr,
                obligation.addr,
            )
            self.graph.add_node(placeholder)
            self.stats.placeholders_created += 1
            self._note_mutation()
        self._connect_source_to_node(obligation, placeholder)
        return placeholder

    def _materialize_undecodable_target(
        self, obligation: RepairObligation | PendingObligation
    ) -> CFGNode:
        """Replace a failed in-bounds decode with one terminal synthetic target."""

        target, created = _ensure_undecodable_target_node(
            self.seed_cfg,
            self.graph,
            self.func_addr,
            obligation.addr,
        )
        if created:
            self.stats.undecodable_targets_created += 1
            self._note_mutation()
            logger.warning(
                f"Custom CFG represents undecodable in-function target "
                f"{obligation.addr:#x} as a terminal leaf"
            )

        placeholders = [
            node
            for node in self._nodes_at_addr(obligation.addr)
            if _node_is_placeholder(node)
        ]
        for placeholder in placeholders:
            for predecessor in list(self.graph.predecessors(placeholder)):
                edge_data = self.graph.get_edge_data(predecessor, placeholder) or {}
                self._add_edge(
                    predecessor,
                    target,
                    edge_data.get("jumpkind", "Ijk_Boring"),
                    unresolved_indirect=edge_data.get("unresolved_indirect", False),
                )
            self.graph.remove_node(placeholder)
            self._note_mutation()

        self._connect_source_to_node(obligation, target)
        return target

    def _recover_entry_now(self, obligation: RepairObligation) -> CFGNode | None:
        """Resolve an existing entry or decode one immediately for local repair."""

        exact_nodes = [
            node
            for node in self._nodes_at_addr(obligation.addr)
            if _node_is_materialized_cfg_node(node)
        ]
        acceptable_node = self._first_acceptable_entry(exact_nodes)
        if acceptable_node is not None:
            return acceptable_node

        block = _recover_block(
            self.project,
            self.bounds,
            obligation.addr,
            self.current_stop_addrs(obligation.addr),
        )
        if block is None:
            return self._materialize_undecodable_target(obligation)
        if block.addr != obligation.addr:
            return None

        recovered_node = self.splice_block(block)
        return recovered_node

    def _first_acceptable_entry(
        self,
        nodes: Iterable[CFGNode],
    ) -> CFGNode | None:
        """Return the first live node that can satisfy an entry request unchanged."""

        return next(
            (
                node
                for node in nodes
                if self.anomalies.node_is_acceptable(
                    self._explicit_split_starts(),
                    node,
                )
            ),
            None,
        )

    def _connect_source_to_node(
        self,
        obligation: RepairObligation | PendingObligation,
        node: CFGNode,
    ) -> None:
        """Materialize every available source-edge claim into ``node``."""

        if isinstance(obligation, RepairObligation):
            claims = (
                {EdgeClaim(obligation.source_node, obligation.jumpkind)}
                if obligation.source_node is not None
                else set()
            )
        else:
            claims = obligation.edge_claims

        for claim in claims:
            if not any(source is claim.source_node for source in self.graph.nodes()):
                continue
            self._add_edge(claim.source_node, node, claim.jumpkind)

    def _queue_if_needed(self, request: RepairObligation) -> bool:
        """Merge a request into the pending work item for its action and address."""

        if request.resolution_policy != "queued":
            raise ValueError("Only queued obligations may enter the worklist")

        key = (request.action, request.addr)
        pending = self.pending.get(key)
        if pending is not None:
            pending.merge(request)
            return False

        self.pending[key] = PendingObligation.from_request(request)
        self.queue.append(key)
        return True

    def _requeue_pending(
        self,
        obligation: PendingObligation,
    ) -> None:
        """Turn merged work back into one or more ordinary queue requests."""

        reason = ", ".join(sorted(obligation.reasons))
        if not obligation.edge_claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=obligation.addr,
                    reason=reason,
                    action=obligation.action,
                    preserve_exact_addr=obligation.preserve_exact_addr,
                )
            )
            return

        for claim in obligation.edge_claims:
            self._queue_if_needed(
                RepairObligation(
                    addr=obligation.addr,
                    reason=reason,
                    action=obligation.action,
                    source_node=claim.source_node,
                    jumpkind=claim.jumpkind,
                    preserve_exact_addr=obligation.preserve_exact_addr,
                )
            )

    def _queue_reconciliation(self, node: CFGNode, reason: str) -> None:
        """Queue a local invariant check for one materialized node."""

        if not _node_is_materialized_cfg_node(node):
            return
        if not _node_intersects_bounds(node, self.bounds):
            return
        self._queue_if_needed(
            RepairObligation(addr=node.addr, reason=reason, action="reconcile")
        )

    def _queue_reconciliation_neighborhood(self, node: CFGNode) -> None:
        """Recheck the nodes whose local invariants a splice may have changed."""

        neighbors = [node, *self.graph.predecessors(node), *self.graph.successors(node)]
        for neighbor in neighbors:
            self._queue_reconciliation(neighbor, f"neighbor_of_{node.addr:#x}")

    def _reconcile_node(self, node: CFGNode) -> None:
        """Satisfy local edge invariants before scheduling a full block recovery."""

        for expectation in self.anomalies.missing_jump_successors(node):
            self._resolve_successor(
                node,
                expectation.addr,
                expectation.jumpkind,
                reason=f"missing_successor_of_{node.addr:#x}",
                preserve_exact_addr=expectation.preserve_exact_addr,
                resolution_policy="immediate",
                materialize_external=True,
            )

        if self.anomalies.node_needs_repair(node):
            self._queue_if_needed(
                RepairObligation(
                    addr=node.addr,
                    reason=f"remaining_anomaly_at_{node.addr:#x}",
                )
            )

    def _reconcile_addr(self, addr: int) -> None:
        """Reconcile every materialized node currently starting at ``addr``."""

        for node in self._nodes_at_addr(addr):
            if _node_is_materialized_cfg_node(node):
                self._reconcile_node(node)

    def _materialize_external_successor(
        self,
        src: CFGNode,
        target: int,
        jumpkind: EdgeJumpKind,
    ) -> bool:
        """
        Attach one out-of-function successor through a shared synthetic leaf.

        This keeps the policy for external targets centralized regardless of
        whether the source block is conditional, unconditional, or call-like.
        """

        if _is_direct_target_valid(self.bounds, target):
            return False

        leaf, created = _ensure_external_target_node(
            self.seed_cfg,
            self.graph,
            self.func_addr,
            target,
        )
        if created:
            self.stats.external_targets_created += 1
            self._note_mutation()
        self._add_edge(src, leaf, jumpkind)
        return True

    def _resolve_successor(
        self,
        src: CFGNode,
        target: int,
        jumpkind: EdgeJumpKind,
        *,
        reason: str,
        preserve_exact_addr: bool,
        resolution_policy: EntryResolutionPolicy = "queued",
        materialize_external: bool = False,
    ) -> None:
        """
        Resolve one successor target and connect it from `src`.

        Direct targets may materialize an external leaf. In-function targets
        enter through `ensure_block_entry()`, which owns the
        preserve/split/placeholder/recovery decision. Callers choose immediate
        resolution only when a live node is missing a required successor.
        """

        if materialize_external and self._materialize_external_successor(
            src, target, jumpkind
        ):
            return

        node = self.ensure_block_entry(
            RepairObligation(
                addr=target,
                reason=reason,
                source_node=src,
                jumpkind=jumpkind,
                preserve_exact_addr=preserve_exact_addr,
                resolution_policy=resolution_policy,
            ),
        )
        if node is not None:
            self._add_edge(src, node, jumpkind)

    def ensure_expected_successors(self, src: CFGNode) -> None:
        """
        Recreate the successor edges implied by one preserved predecessor node.

        This keeps good CFGFast predecessors intact while still discovering
        repaired targets or fresh split points around them.
        """

        direct_targets, fallthrough_addr = _seed_node_expected_successors(src)
        for target in direct_targets:
            self._resolve_successor(
                src,
                target,
                "Ijk_Boring",
                reason=f"expected_successor_of_{src.addr:#x}",
                preserve_exact_addr=True,
                materialize_external=True,
            )

        if fallthrough_addr is not None:
            self._resolve_successor(
                src,
                fallthrough_addr,
                "Ijk_Boring",
                reason=f"expected_fallthrough_of_{src.addr:#x}",
                preserve_exact_addr=False,
            )

    def current_stop_addrs(self, addr: int) -> set[int]:
        """Return the hard stop addresses used for bounded recovery at `addr`."""

        if self._stop_starts_revision != self.mutation_revision:
            self._stop_starts = self._current_stop_starts()
            self._stop_starts_revision = self.mutation_revision

        # A linear tail may be absorbed only while its own predecessor is
        # being recovered. Other blocks can legitimately fall through to that
        # same address, so it must remain a leader for their recovery.
        linear_tails = {
            successor.addr
            for node in self._nodes_at_addr(addr)
            if self.anomalies.check_linear_merge_successor(node)
            for successor in self.graph.successors(node)
        }
        return {
            target
            for target in self._stop_starts
            if addr < target < self.bounds.end_addr and target not in linear_tails
        }

    def _current_stop_starts(self) -> set[int]:
        """
        Return the live leader starts that must bound recovered blocks.

        The result is invalidated by ``mutation_revision`` whenever the graph
        or leader set changes. `current_stop_addrs()` applies the narrower
        exception for a linear tail while recovering its direct predecessor.
        """

        starts = self.leaders.starts()
        for node in self._bound_nodes():
            if not _node_is_materialized_cfg_node(node):
                continue

            if self._is_preservable_seed_node(node) or node in self.repaired_nodes:
                direct_targets, fallthrough_addr = self._preserved_successor_starts(
                    node
                )
                for target in direct_targets:
                    if self.bounds.addr <= target < self.bounds.end_addr:
                        starts.add(target)
                # Only recovered blocks have trustworthy fallthrough
                # semantics. CFGFast seed fallthrough edges may be the split
                # artifacts this repair pass is meant to remove.
                if (
                    node in self.repaired_nodes
                    and fallthrough_addr is not None
                    and self.bounds.addr <= fallthrough_addr < self.bounds.end_addr
                ):
                    starts.add(fallthrough_addr)

        return starts

    def splice_block(self, block: BlockSpec) -> CFGNode:
        """
        Replace every stale overlapping node with one recovered block.

        Incoming edges from preserved predecessors are rewired after stale nodes
        are removed. Preserved predecessors may also enqueue additional sibling
        successors implied by their own semantics.
        """

        recovered_start = block.addr
        recovered_end = block.addr + block.size

        removed_nodes = []
        for node in self._bound_nodes():
            replaces_same_start = node.addr == recovered_start
            overlaps_recovered_range = _ranges_overlap(
                node.addr, _node_range_end(node), recovered_start, recovered_end
            )
            if not replaces_same_start and not overlaps_recovered_range:
                continue

            if not replaces_same_start and self._is_required_prefixed_instruction_entry(
                node, block
            ):
                continue
            removed_nodes.append(node)
        removed_set = set(removed_nodes)
        removed_blocks = [
            node for node in removed_nodes if _node_is_materialized_cfg_node(node)
        ]
        linear_merges = sum(
            1
            for node in removed_blocks
            if node_has_linear_merge_successor(self.graph, node)
            and any(
                successor in removed_set for successor in self.graph.successors(node)
            )
        )

        incoming_edges = [
            (src, dst, dict(data))
            for src, dst, data in list(self.graph.edges(data=True))
            if dst in removed_set and src not in removed_set
        ]
        unresolved_jump_edges = [
            (dst, data.get("jumpkind", "Ijk_Boring"))
            for src, dst, data in list(self.graph.edges(data=True))
            if src in removed_set and _is_unresolvable_control_target(dst)
        ]

        _remove_nodes(self.graph, removed_nodes)
        for node in removed_nodes:
            self.recovered_blocks.pop(node, None)
        self.stats.blocks_redecoded += 1
        self.stats.blocks_replaced += len(removed_blocks)
        self.stats.linear_block_merges += linear_merges

        recovered_node = _make_cfg_node(
            self.seed_cfg, self.func_addr, self.bounds, block
        )
        self.graph.add_node(recovered_node)
        self._note_mutation()
        self.repaired_nodes.add(recovered_node)
        self.recovered_blocks[recovered_node] = block

        for pred, _, data in incoming_edges:
            self._add_edge(
                pred,
                recovered_node,
                data.get("jumpkind", "Ijk_Boring"),
            )

            if pred in self.repaired_nodes:
                continue
            if not self.anomalies.node_is_acceptable(
                self._explicit_split_starts(),
                pred,
            ):
                continue

            self.ensure_expected_successors(pred)

        # A recovered indirect transfer has no concrete BlockSpec target.
        # Retain CFGFast's unresolved placeholder until static table recovery
        # proves real jump targets, or another analysis resolves the call.
        if _block_has_unresolved_indirect_transfer(block):
            for unresolved_target, jumpkind in unresolved_jump_edges:
                self._add_edge(recovered_node, unresolved_target, jumpkind)

        for target in block.direct_targets:
            edge_jumpkind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
            self._resolve_successor(
                recovered_node,
                target,
                edge_jumpkind,
                reason=f"direct_target_of_{block.addr:#x}",
                preserve_exact_addr=True,
                materialize_external=True,
            )

        if block.fallthrough_addr is not None:
            self._resolve_successor(
                recovered_node,
                block.fallthrough_addr,
                "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
                reason=f"fallthrough_of_{block.addr:#x}",
                preserve_exact_addr=False,
            )

        self._queue_reconciliation_neighborhood(recovered_node)

        return recovered_node

    def _factor_shared_instruction_tail(self) -> bool:
        """Replace overlapping block tails with one shared instruction node.

        A direct branch can intentionally enter immediately after an x86
        instruction prefix, producing two distinct instruction streams that
        later execute the same instructions. Re-decode both linear prefixes
        and their common tail independently, then replace all participating
        nodes in one graph mutation. Doing this atomically avoids routing a
        predecessor into the wrong overlapping stream while an intermediate
        replacement exists.
        """

        nodes = [
            node for node in self._bound_nodes() if _node_is_materialized_cfg_node(node)
        ]
        tail = _find_shared_instruction_tail(self.project.arch.name, nodes)
        if tail is None:
            return False

        if any(
            source in tail.nodes and target in tail.nodes
            for source, target, _ in self.graph.edges(data=True)
        ):
            return False

        prefix_blocks: dict[CFGNode, BlockSpec] = {}
        for node in tail.nodes:
            prefix = _recover_block(
                self.project,
                self.bounds,
                node.addr,
                self.current_stop_addrs(node.addr) | {tail.start_addr},
            )
            if (
                prefix is None
                or prefix.jumpkind != "Ijk_Fallthrough"
                or prefix.fallthrough_addr != tail.start_addr
            ):
                return False
            prefix_blocks[node] = prefix

        tail_block = _recover_block(
            self.project,
            self.bounds,
            tail.start_addr,
            self.current_stop_addrs(tail.start_addr),
        )
        if tail_block is None or tail_block.instruction_addrs != tail.instruction_addrs:
            return False

        self.leaders.add(tail.start_addr, "shared_instruction_tail")
        incoming_edges = [
            (source, target, dict(data))
            for source, target, data in self.graph.edges(data=True)
            if target in tail.nodes and source not in tail.nodes
        ]
        _remove_nodes(self.graph, tail.nodes)
        for node in tail.nodes:
            self.repaired_nodes.discard(node)
            self.recovered_blocks.pop(node, None)

        replacements: dict[CFGNode, CFGNode] = {}
        for node, block in prefix_blocks.items():
            replacement = _make_cfg_node(
                self.seed_cfg, self.func_addr, self.bounds, block
            )
            self.graph.add_node(replacement)
            self.repaired_nodes.add(replacement)
            self.recovered_blocks[replacement] = block
            replacements[node] = replacement

        tail_node = _make_cfg_node(
            self.seed_cfg, self.func_addr, self.bounds, tail_block
        )
        self.graph.add_node(tail_node)
        self.repaired_nodes.add(tail_node)
        self.recovered_blocks[tail_node] = tail_block
        self.stats.blocks_redecoded += len(prefix_blocks) + 1
        self.stats.blocks_replaced += len(tail.nodes)
        self.stats.shared_instruction_tails_factored += 1
        self._note_mutation()

        for source, target, data in incoming_edges:
            self._add_edge(
                source,
                replacements[target],
                data.get("jumpkind", "Ijk_Boring"),
            )
        for prefix in replacements.values():
            self._add_edge(prefix, tail_node, "Ijk_Boring")

        for target in tail_block.direct_targets:
            edge_jumpkind = (
                "Ijk_Call" if tail_block.jumpkind == "Ijk_Call" else "Ijk_Boring"
            )
            self._resolve_successor(
                tail_node,
                target,
                edge_jumpkind,
                reason=f"shared_tail_target_of_{tail_block.addr:#x}",
                preserve_exact_addr=True,
                materialize_external=True,
            )
        if tail_block.fallthrough_addr is not None:
            self._resolve_successor(
                tail_node,
                tail_block.fallthrough_addr,
                "Ijk_FakeRet" if tail_block.jumpkind == "Ijk_Call" else "Ijk_Boring",
                reason=f"shared_tail_fallthrough_of_{tail_block.addr:#x}",
                preserve_exact_addr=False,
            )

        self._queue_reconciliation_neighborhood(tail_node)
        return True

    def _cleanup(self) -> bool:
        """Prune stale nodes and report whether cleanup changed the live graph."""

        self.stats.cleanup_rounds += 1
        changed = self._attach_unresolved_jump_fallbacks()
        changed |= self._flatten_single_source_unresolved_fallback()
        unreachable_removed = _cleanup_unreachable_function_nodes(
            self.graph, self.bounds, self.func_addr
        )
        placeholders_pruned = _prune_placeholders(self.graph)
        orphan_simprocedures_pruned = _prune_orphan_simprocedures(self.graph)
        self.stats.unreachable_blocks_removed += unreachable_removed
        self.stats.placeholders_pruned += placeholders_pruned
        self.stats.orphan_simprocedures_pruned += orphan_simprocedures_pruned
        changed |= bool(
            unreachable_removed or placeholders_pruned or orphan_simprocedures_pruned
        )
        if changed:
            self._note_mutation()
        return changed

    def _anomalous_addrs(self) -> list[int]:
        """Return the current in-bounds materialized block starts needing repair."""

        return sorted(
            {
                node.addr
                for node in self._bound_nodes()
                if _node_is_materialized_cfg_node(node)
                and self.anomalies.node_needs_repair(node)
            }
        )

    def _initial_anomalous_addrs(self) -> list[int]:
        """Return seed anomalies plus in-bounds nodes mis-owned by CFGFast."""

        seed_anomalies = {
            node.addr
            for node in _iter_seed_function_nodes(self.seed_cfg, self.func_addr)
            if self.anomalies.node_needs_repair(node)
        }
        ownership_boundaries = {
            node.addr
            for node in self._bound_nodes()
            if self.anomalies.check_foreign_function_owner(node)
        }
        return sorted(seed_anomalies | ownership_boundaries)

    def _queue_recoveries(self, addrs: Iterable[int], reason: str) -> None:
        """Seed ordinary recovery work for a group of anomalous block starts."""

        for addr in addrs:
            # These addresses were already classified as anomalous. Queue them
            # directly instead of treating a preserved seed node as acceptable.
            self._queue_if_needed(RepairObligation(addr=addr, reason=reason))

    def _drain_worklist(self) -> None:
        """Process queued work while enforcing the global repair iteration limit."""

        while self.queue:
            self.iterations += 1
            if self.iterations > MAX_CUSTOM_CFG_WORKLIST_ITERATIONS:
                raise RuntimeError(
                    "Custom CFG worklist exceeded "
                    f"{MAX_CUSTOM_CFG_WORKLIST_ITERATIONS} iterations for {self.func_addr:#x}; "
                    f"top counts: {self.processed_counts}"
                )

            key = self.queue.popleft()
            obligation = self.pending.pop(key)
            self.stats.worklist_obligations += 1
            addr = obligation.addr
            self.processed_counts[addr] = self.processed_counts.get(addr, 0) + 1
            if self.processed_counts[addr] <= 5:
                logger.debug(
                    f"Custom CFG processing {addr:#x} for function {self.func_addr:#x} "
                    f"(visit {self.processed_counts[addr]})"
                )

            self._process_obligation(obligation)
            self._record_obligation_progress(key, obligation)

    def _process_reconciliation(self, obligation: PendingObligation) -> None:
        """Reconcile every live node at one queued address."""

        self._reconcile_addr(obligation.addr)

    def _process_recovery(self, obligation: PendingObligation) -> None:
        """Recover a queued address or requeue the work needed to expose it."""

        addr = obligation.addr
        current_nodes = self._nodes_at_addr(addr)
        covering_nodes = [
            node
            for node in self._covering_nodes(addr)
            if node.addr != addr and not _node_is_placeholder(node)
        ]
        if covering_nodes:
            if addr in self._explicit_split_starts():
                # Another node still covers this forced split point. Requeue
                # both the covering node and the split address so the target
                # is revisited after the prefix block gets truncated.
                for node in covering_nodes:
                    self._queue_if_needed(
                        RepairObligation(
                            addr=node.addr,
                            reason=f"split_for_{addr:#x}",
                        )
                    )
                self._requeue_pending(obligation)
            return

        acceptable_node = self._first_acceptable_entry(current_nodes)
        if acceptable_node is not None:
            self._connect_source_to_node(obligation, acceptable_node)
            return

        block = _recover_block(
            self.project, self.bounds, addr, self.current_stop_addrs(addr)
        )
        if block is None:
            self._materialize_undecodable_target(obligation)
            return

        recovered_node = self.splice_block(block)
        self._connect_source_to_node(obligation, recovered_node)

    def _process_obligation(self, obligation: PendingObligation) -> None:
        """Dispatch one in-bounds worklist item to its action-specific handler."""

        if not (self.bounds.addr <= obligation.addr < self.bounds.end_addr):
            return

        if obligation.action == "reconcile":
            self._process_reconciliation(obligation)
            return

        self._process_recovery(obligation)

    def run(self) -> CFGBase | CustomCFG:
        """Execute the repair worklist and return the repaired CFG wrapper."""

        # Capture seed anomalies before any static table recovery mutates the
        # input graph. The normal initial classification below remains in its
        # original order so the repair behavior itself does not change.
        initial_bad_addrs = self._initial_anomalous_addrs()
        self.stats.input_anomalies = len(initial_bad_addrs)
        resolved_tables = self._resolve_static_jump_tables()
        if resolved_tables:
            # Table resolution changes the seed graph, so only then does a
            # second classification reflect the graph we are about to repair.
            initial_bad_addrs = self._initial_anomalous_addrs()
        attempted_anomaly_addrs = set(initial_bad_addrs)
        if initial_bad_addrs:
            logger.info(
                f"Repairing seed CFG for function {self.func_addr:#x} with "
                f"{len(initial_bad_addrs)} anomalous block start(s)"
            )
        elif resolved_tables:
            logger.info(
                f"Repairing seed CFG for function {self.func_addr:#x} after resolving "
                f"{resolved_tables} static jump table(s)"
            )

        self._queue_recoveries(initial_bad_addrs, "seed_anomaly")

        while True:
            self._drain_worklist()

            # Factor valid overlapping instruction streams only after ordinary
            # recovery has stabilized their local predecessors and successors.
            # The rewrite can queue new local reconciliation, so drain that
            # work before proceeding to jump-table recovery or cleanup.
            while self._factor_shared_instruction_tail():
                self._drain_worklist()

            # Worklist recovery can replace an indirect-dispatch source and
            # therefore discard table edges found before repair. Re-scan the
            # live nodes so a recovered source receives its proven targets.
            resolved_tables += self._resolve_static_jump_tables()

            cleanup_changed = self._cleanup()
            remaining_bad_addrs = self._anomalous_addrs()
            if not remaining_bad_addrs:
                break
            if not cleanup_changed:
                newly_exposed_addrs = sorted(
                    set(remaining_bad_addrs) - attempted_anomaly_addrs
                )
                if newly_exposed_addrs:
                    logger.info(
                        f"Repair discovered {len(newly_exposed_addrs)} new anomaly "
                        f"start(s) for function {self.func_addr:#x}; continuing repair"
                    )
                    attempted_anomaly_addrs.update(newly_exposed_addrs)
                    self._queue_recoveries(newly_exposed_addrs, "post_repair_anomaly")
                    continue
                logger.warning(
                    f"Custom CFG repair for {self.func_addr:#x} stopped with "
                    f"{len(remaining_bad_addrs)} unresolved anomaly start(s): "
                    f"{', '.join(hex(addr) for addr in remaining_bad_addrs)}"
                )
                break

            logger.info(
                f"Cleanup exposed {len(remaining_bad_addrs)} anomaly start(s) for "
                f"function {self.func_addr:#x}; continuing repair"
            )
            attempted_anomaly_addrs.update(remaining_bad_addrs)
            self._queue_recoveries(remaining_bad_addrs, "post_cleanup_anomaly")
            if not self.queue:
                logger.warning(
                    f"Custom CFG repair for {self.func_addr:#x} could not queue "
                    "cleanup-exposed anomalies"
                )
                break

        self._canonicalize_function_ownership()
        result = CustomCFG(
            graph=self.graph,
            model=_custom_model_marker(),
            functions=self.seed_cfg.functions,
            kb=self.seed_cfg.kb,
        )
        return result


def build_custom_cfg(
    project: Project,
    kb: KnowledgeBase,
    func_addr: int,
    seed_cfg: CFGBase,
) -> CFGBase | CustomCFG:
    """
    Build a custom repaired CFG for one function starting from CFGFast output.

    The explicit KB parameter mirrors the higher-level CFG plumbing even though
    the current repair pass operates directly on the provided seed CFG graph.
    `_RepairSession.run()` owns both initial anomaly discovery and repair, so
    the custom path performs one coherent classification before it mutates the
    seed graph.
    """

    logger.info(f"Building custom CFG for function {func_addr:#x}")
    _clear_decoded_node_cache()
    session = _RepairSession(project, seed_cfg, func_addr)
    try:
        return session.run()
    finally:
        # Keep transformation counters available even when custom repair fails.
        session.log_stats()
