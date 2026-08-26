"""Data models exposed by the experimental independent CFG extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

from angr import KnowledgeBase
from angr.knowledge_plugins.cfg import CFGModel


@dataclass
class ExtractedCFGStats:
    """Counters describing one bounded independent CFG extraction."""

    leaders_discovered: int = 0
    leaders_split_existing_block: int = 0
    blocks_decoded: int = 0
    block_redecodes: int = 0
    decode_failures: int = 0
    data_leaders_rejected: int = 0
    call_fallthroughs_suppressed: int = 0
    calls: int = 0
    syscalls: int = 0
    direct_branches: int = 0
    conditional_branches: int = 0
    returns: int = 0
    terminal_blocks: int = 0
    direct_edges: int = 0
    fallthrough_edges: int = 0
    unresolved_indirect_targets: int = 0
    unresolved_call_targets: int = 0
    external_targets: int = 0
    undecodable_targets: int = 0
    synthetic_leaves_created: int = 0
    static_jump_tables_resolved: int = 0
    static_jump_targets_read: int = 0
    static_jump_targets_added: int = 0
    static_jump_dispatchers_unresolved: int = 0
    static_jump_no_vex: int = 0
    static_jump_no_table_shape: int = 0
    static_jump_unknown_base: int = 0
    static_jump_unbounded_index: int = 0
    static_jump_table_unreadable: int = 0
    static_jump_table_rejected_targets: int = 0
    sweep_runs: int = 0
    sweep_candidate_blocks: int = 0
    sweep_candidate_instructions: int = 0
    sweep_candidate_components: int = 0
    sweep_decode_failures: int = 0
    sweep_non_executable_bytes: int = 0
    sweep_reconnecting_components: int = 0
    sweep_reconnecting_blocks: int = 0
    sweep_component_roots_attached: int = 0
    output_anomalies: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return stable log-friendly extraction counters."""

        return asdict(self)


class ExtractedCFG(SimpleNamespace):
    """Small CFGBase-compatible surface consumed by bingraph rendering."""

    graph: Any
    model: CFGModel
    functions: Any
    kb: KnowledgeBase
    extract_stats: ExtractedCFGStats
