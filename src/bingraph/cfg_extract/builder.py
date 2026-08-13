"""Bounded CFG extraction without CFGFast.

The extractor starts at one function symbol and grows only through addresses
proven by decoded direct transfers.  A shared leader set keeps recovered blocks
non-overlapping: whenever a newly discovered target falls inside an existing
block, that block is re-decoded with the target as a stop address. VEX-proven
static jump tables contribute additional leaders through the shared resolver;
remaining indirect transfers stay explicit synthetic leaves. The extractor
never reads CFGFast's discovered regions.
"""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace
from typing import cast

from angr import KnowledgeBase, Project
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from loguru import logger
import networkx as nx

from bingraph.cfg.anomalies import _lookup_function_bounds
from bingraph.cfg.graph import CFGGraph, add_successor_edge
from bingraph.cfg.jumps import (
    _read_static_jump_table_targets,
    plan_static_jump_table,
    static_jump_target_rejection_reason,
)
from bingraph.cfg.models import BlockSpec, FunctionBounds
from bingraph.cfg.decode import decode_bounded_block

from .anomalies import find_extracted_cfg_anomalies
from .models import ExtractedCFG, ExtractedCFGStats
from .sweep import recover_executable_components, select_reconnecting_components


def _thumb_mode(project: Project, addr: int) -> bool:
    """Return the execution mode encoded by an ARM/Thumb address."""

    try:
        return bool(project.arch.is_thumb(addr))
    except AttributeError:
        return False


def _node_name(bounds: FunctionBounds, addr: int) -> str:
    """Return the stable display name used for a recovered block."""

    if addr == bounds.addr:
        return bounds.name
    return f"{bounds.name}+0x{addr - bounds.addr:x}"


def _make_block_node(
    model: CFGModel,
    project: Project,
    func_addr: int,
    bounds: FunctionBounds,
    block: BlockSpec,
) -> CFGNode:
    """Materialize one recovered normal CFG node."""

    return CFGNode(
        block.addr,
        block.size,
        cfg=model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        thumb=_thumb_mode(project, block.addr),
        name=_node_name(bounds, block.addr),
    )


def _external_target_name(project: Project, addr: int) -> str:
    """Return the loader symbol name for an external target when known."""

    symbol = project.loader.find_symbol(addr)
    name = getattr(symbol, "name", None)
    return name if isinstance(name, str) and name else f"ExternalTarget_{addr:#x}"


def _make_leaf_node(
    model: CFGModel,
    func_addr: int,
    addr: int,
    name: str,
) -> CFGNode:
    """Materialize a synthetic CFG leaf without making it a function block."""

    return CFGNode(
        addr,
        0,
        cfg=model,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        simprocedure_name=name,
        name=name,
    )


class _ExtractionSession:
    """Own the leader worklist and graph materialization for one function."""

    def __init__(self, project: Project, kb: KnowledgeBase, func_addr: int) -> None:
        self.project = project
        self.kb = kb
        self.func_addr = func_addr
        self.bounds = _lookup_function_bounds(project, func_addr)
        manager = SimpleNamespace(_kb=kb)
        self.model = CFGModel("CFGExtract", cfg_manager=manager)
        self.graph = cast(CFGGraph, self.model.graph)
        self.stats = ExtractedCFGStats()
        self.leaders = {func_addr}
        self.pending = deque([func_addr])
        self.pending_addrs = {func_addr}
        self.blocks: dict[int, BlockSpec] = {}
        self.leaf_nodes: dict[tuple[int, str], CFGNode] = {}
        self.static_targets: dict[int, tuple[int, ...]] = {}
        self.sweep_dispatcher_addr: int | None = None
        self.sweep_component_roots: frozenset[int] = frozenset()

    def _queue(self, addr: int) -> None:
        """Schedule one in-bounds block leader only once per pending round."""

        if not self.bounds.addr <= addr < self.bounds.end_addr:
            return
        if addr in self.pending_addrs:
            return
        self.pending.append(addr)
        self.pending_addrs.add(addr)

    def _add_leader(self, addr: int) -> None:
        """Record a discovered in-bounds target and re-split covering blocks."""

        if not self.bounds.addr <= addr < self.bounds.end_addr:
            return
        is_new_leader = addr not in self.leaders
        self.leaders.add(addr)
        if is_new_leader:
            self.stats.leaders_discovered += 1
            for start, block in tuple(self.blocks.items()):
                if start < addr < start + block.size:
                    del self.blocks[start]
                    self.stats.block_redecodes += 1
                    self.stats.leaders_split_existing_block += 1
                    self._queue(start)
        if is_new_leader or addr not in self.blocks:
            self._queue(addr)

    def _decode_all_blocks(self) -> None:
        """Drain discovered leaders until their block boundaries stabilize."""

        while self.pending:
            addr = self.pending.popleft()
            self.pending_addrs.remove(addr)
            block = decode_bounded_block(
                self.project,
                self.bounds,
                addr,
                self.leaders - {addr},
                preserve_conditional_return_fallthrough=True,
            )
            if block is None:
                self.stats.decode_failures += 1
                logger.warning(
                    f"Extract CFG could not decode block at {addr:#x} for "
                    f"function {self.func_addr:#x}"
                )
                continue
            self.blocks[addr] = block
            self.stats.blocks_decoded += 1
            for target in (*block.direct_targets, block.fallthrough_addr):
                if target is not None:
                    self._add_leader(target)

    def _analysis_graph(self) -> tuple[CFGGraph, dict[int, CFGNode]]:
        """Build a temporary direct-edge graph for shared table planning."""

        graph = cast(CFGGraph, nx.DiGraph())
        nodes = {
            addr: _make_block_node(
                self.model, self.project, self.func_addr, self.bounds, block
            )
            for addr, block in self.blocks.items()
        }
        for node in nodes.values():
            graph.add_node(node)
        for addr, block in self.blocks.items():
            source = nodes[addr]
            for target in (*block.direct_targets, block.fallthrough_addr):
                if target is None or target not in nodes:
                    continue
                add_successor_edge(
                    graph,
                    source,
                    nodes[target],
                    "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
                )
        return graph, nodes

    def _discover_static_jump_targets(self) -> None:
        """Use shared VEX table proofs to add further in-function leaders."""

        while True:
            graph, nodes = self._analysis_graph()
            discovered = False
            plans: dict[int, tuple[int, ...]] = {}
            for addr, node in nodes.items():
                # Adding a target can split a later block from this snapshot.
                # Skip its now-stale node; the next round analyzes its decode.
                block = self.blocks.get(addr)
                if block is None:
                    continue
                if (
                    block.jumpkind != "Ijk_Boring"
                    or block.direct_targets
                    or block.fallthrough_addr is not None
                ):
                    continue
                plan, reason = plan_static_jump_table(
                    self.project, graph, self.bounds, node
                )
                if plan is None:
                    self.stats.static_jump_dispatchers_unresolved += 1
                    if reason is not None:
                        field = f"static_jump_{reason}"
                        setattr(self.stats, field, getattr(self.stats, field) + 1)
                    continue
                targets = _read_static_jump_table_targets(
                    self.project, plan.table, plan.base_addr, plan.entry_count
                )
                if targets is None:
                    self.stats.static_jump_table_unreadable += 1
                    continue
                rejected = [
                    static_jump_target_rejection_reason(self.project, target)
                    for target in targets
                ]
                if any(rejected):
                    self.stats.static_jump_table_rejected_targets += 1
                    continue
                plans[addr] = targets
                self.stats.static_jump_tables_resolved += 1
                self.stats.static_jump_targets_read += len(targets)
                for target in targets:
                    if self.bounds.addr <= target < self.bounds.end_addr:
                        before = target in self.blocks or target in self.pending_addrs
                        self._add_leader(target)
                        discovered |= not before

            if discovered:
                # Block splits invalidate plans built from this graph snapshot.
                # Decode and rebuild the analysis graph before retaining any.
                self._decode_all_blocks()
                continue
            self.static_targets.update(plans)
            return

    def _leaf(self, addr: int, name: str) -> CFGNode:
        """Return a unique synthetic leaf for one address/name pair."""

        key = (addr, name)
        node = self.leaf_nodes.get(key)
        if node is None:
            node = _make_leaf_node(self.model, self.func_addr, addr, name)
            self.graph.add_node(node)
            self.leaf_nodes[key] = node
            self.stats.synthetic_leaves_created += 1
        return node

    def _target_node(self, addr: int) -> CFGNode:
        """Return a recovered destination or a precise synthetic leaf."""

        node = self.nodes.get(addr)
        if node is not None:
            return node
        if self.bounds.addr <= addr < self.bounds.end_addr:
            self.stats.undecodable_targets += 1
            return self._leaf(addr, "UndecodableInstructionTarget")
        self.stats.external_targets += 1
        return self._leaf(addr, _external_target_name(self.project, addr))

    def _fallthrough_target_node(self, addr: int) -> CFGNode | None:
        """Return a valid continuation without inventing one past bad bytes."""

        node = self.nodes.get(addr)
        if node is not None:
            return node
        if self.bounds.addr <= addr < self.bounds.end_addr:
            # A known branch target may be useful as an explicit undecodable
            # leaf, but normal execution must not fall through to bytes that
            # failed bounded decoding. This matches the custom renderer's
            # existing treatment of invalid sequential continuations.
            return None
        return self._target_node(addr)

    def _materialize_edges(self) -> None:
        """Create normal nodes and their decoded direct control-flow edges."""

        self.nodes = {
            addr: _make_block_node(
                self.model, self.project, self.func_addr, self.bounds, block
            )
            for addr, block in sorted(self.blocks.items())
        }
        for node in self.nodes.values():
            self.graph.add_node(node)

        for addr, block in sorted(self.blocks.items()):
            source = self.nodes[addr]
            for target in block.direct_targets:
                destination = self._target_node(target)
                jumpkind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                if add_successor_edge(self.graph, source, destination, jumpkind):
                    self.stats.direct_edges += 1

            for target in self.static_targets.get(addr, ()):
                destination = self._target_node(target)
                if add_successor_edge(self.graph, source, destination, "Ijk_Boring"):
                    self.stats.static_jump_targets_added += 1

            if block.fallthrough_addr is not None:
                destination = self._fallthrough_target_node(block.fallthrough_addr)
                if destination is not None:
                    jumpkind = (
                        "Ijk_FakeRet" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                    )
                    if add_successor_edge(self.graph, source, destination, jumpkind):
                        self.stats.fallthrough_edges += 1

            if (
                block.jumpkind == "Ijk_Boring"
                and not block.direct_targets
                and block.fallthrough_addr is None
                and not self.static_targets.get(addr)
            ):
                unresolved = self._leaf(0xFFFFFFFFFFFFFFF0, "UnresolvableJumpTarget")
                if add_successor_edge(
                    self.graph,
                    source,
                    unresolved,
                    "Ijk_Boring",
                    unresolved_indirect=True,
                ):
                    self.stats.unresolved_indirect_targets += 1

    def _recover_reconnecting_components(self) -> None:
        """Attach leader-closed components behind one unresolved dispatcher."""

        dispatchers = [
            addr
            for addr, block in self.blocks.items()
            if (
                block.jumpkind == "Ijk_Boring"
                and not block.direct_targets
                and block.fallthrough_addr is None
                and not self.static_targets.get(addr)
            )
        ]
        if len(dispatchers) != 1:
            return

        recovered_addrs = set(self.blocks)
        sweep = recover_executable_components(self.project, self.bounds, self.blocks)
        audit = sweep.audit
        self.stats.sweep_runs += 1
        self.stats.sweep_candidate_blocks += audit.candidate_blocks
        self.stats.sweep_candidate_instructions += audit.candidate_instructions
        self.stats.sweep_candidate_components += audit.candidate_components
        self.stats.sweep_decode_failures += audit.decode_failures
        self.stats.sweep_non_executable_bytes += audit.non_executable_bytes
        selected = select_reconnecting_components(sweep, recovered_addrs)
        if not selected.blocks:
            return

        self.blocks = dict(selected.blocks)
        self.leaders = set(self.blocks)
        self.sweep_dispatcher_addr = dispatchers[0]
        self.sweep_component_roots = selected.roots
        self.stats.sweep_reconnecting_components += selected.component_count
        reconnecting_block_count = len(selected.blocks) - len(sweep.reachable_addrs)
        self.stats.sweep_reconnecting_blocks += reconnecting_block_count
        logger.info(
            f"Extract CFG recovery for {self.func_addr:#x}: selected "
            f"{reconnecting_block_count} block(s) from "
            f"{selected.component_count} "
            f"reconnecting component(s) behind {dispatchers[0]:#x}"
        )

    def _attach_reconnecting_components(self) -> None:
        """Connect selected roots directly while retaining the unknown leaf.

        One dispatcher makes the source of every selected component
        unambiguous, so draw the candidate edges directly from it. The
        ``UnresolvableJumpTarget`` edge remains as an explicit catch-all:
        executable-range recovery cannot prove that these are every possible
        target, including targets outside the bounded function region.
        """

        if self.sweep_dispatcher_addr is None:
            return
        dispatcher = self.nodes[self.sweep_dispatcher_addr]
        for addr in self.sweep_component_roots:
            if add_successor_edge(
                self.graph,
                dispatcher,
                self.nodes[addr],
                "Ijk_Boring",
                unresolved_indirect=True,
            ):
                self.stats.sweep_component_roots_attached += 1

    def build(self) -> ExtractedCFG:
        """Recover the bounded function graph and expose it to rendering."""

        self._decode_all_blocks()
        self._discover_static_jump_targets()
        self._recover_reconnecting_components()
        self._materialize_edges()
        self._attach_reconnecting_components()
        self.kb.functions.function(self.func_addr, name=self.bounds.name, create=True)
        for block in self.blocks.values():
            if block.jumpkind == "Ijk_Call":
                self.stats.calls += 1
            elif block.jumpkind == "Ijk_Ret":
                self.stats.returns += 1
            elif block.jumpkind == "Ijk_Terminal":
                self.stats.terminal_blocks += 1
            elif block.direct_targets:
                self.stats.direct_branches += 1
                self.stats.conditional_branches += block.fallthrough_addr is not None

        anomalies = find_extracted_cfg_anomalies(
            self.graph, self.bounds, self.func_addr, self.blocks
        )
        self.stats.output_anomalies = len(anomalies)
        if anomalies:
            for anomaly in anomalies:
                logger.warning(anomaly.message)
        else:
            logger.info(
                f"Extracted CFG for {self.func_addr:#x} passed structural validation"
            )
        return ExtractedCFG(
            graph=self.graph,
            model=self.model,
            functions=self.kb.functions,
            kb=self.kb,
            extract_stats=self.stats,
        )


def build_extracted_cfg(
    project: Project, kb: KnowledgeBase, func_addr: int
) -> ExtractedCFG:
    """Build one experimental function CFG without invoking CFGFast."""

    logger.info(f"Extracting CFG for function {func_addr:#x} without CFGFast")
    cfg = _ExtractionSession(project, kb, func_addr).build()
    logger.info(
        f"Extracted CFG for {func_addr:#x}: "
        f"{len(tuple(cfg.graph.nodes()))} nodes, "
        f"{len(tuple(cfg.graph.edges()))} edges, "
        f"stats={cfg.extract_stats.as_dict()}"
    )
    return cfg
