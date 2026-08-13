"""CFGFast-specific helpers for repairing overlapping seed blocks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

from angr.knowledge_plugins.cfg import CFGNode
from capstone import CsInsn

from bingraph.helpers.capstone import control_transfer_index

from .decode import DecodedNode


@dataclass(frozen=True)
class SharedInstructionTail:
    """One common instruction suffix owned by overlapping CFG nodes."""

    nodes: tuple[CFGNode, ...]
    start_addr: int
    instruction_addrs: tuple[int, ...]


def _common_instruction_suffix(
    left: tuple[CsInsn, ...], right: tuple[CsInsn, ...]
) -> tuple[CsInsn, ...]:
    """Return the longest exact suffix shared by two decoded instruction lists."""

    suffix: list[CsInsn] = []
    for left_insn, right_insn in zip(reversed(left), reversed(right), strict=False):
        if (
            left_insn.address != right_insn.address
            or left_insn.bytes != right_insn.bytes
        ):
            break
        suffix.append(left_insn)
    return tuple(reversed(suffix))


def find_shared_instruction_tail(
    arch_name: str, nodes: Sequence[CFGNode]
) -> SharedInstructionTail | None:
    """Find one safely factorable common instruction suffix.

    Variable-length ISAs can have two valid instruction streams that overlap in
    bytes. We only factor a suffix when every participating stream reaches it
    linearly. The caller independently re-decodes the prefixes and tail before
    changing the graph, which verifies the suffix owns one unambiguous
    continuation even when its control transfer is in a later block.
    """

    decoded_nodes = [
        (node, decoded.insns)
        for node in nodes
        if (decoded := DecodedNode.from_node(node)).insns
    ]
    candidates: list[SharedInstructionTail] = []
    for (_, left), (_, right) in combinations(decoded_nodes, 2):
        assert left is not None
        assert right is not None
        suffix = _common_instruction_suffix(left, right)
        if not suffix or len(suffix) == len(left) or len(suffix) == len(right):
            continue

        suffix_addrs = tuple(insn.address for insn in suffix)
        shared_nodes = tuple(
            node
            for node, insns in decoded_nodes
            if insns is not None
            and len(insns) > len(suffix)
            and tuple(insn.address for insn in insns[-len(suffix) :]) == suffix_addrs
            and control_transfer_index(arch_name, list(insns[: -len(suffix)])) is None
        )
        if len(shared_nodes) < 2:
            continue
        candidates.append(
            SharedInstructionTail(
                nodes=shared_nodes,
                start_addr=suffix_addrs[0],
                instruction_addrs=suffix_addrs,
            )
        )

    if not candidates:
        return None
    return max(
        candidates,
        key=lambda candidate: (
            len(candidate.instruction_addrs),
            len(candidate.nodes),
            -candidate.start_addr,
        ),
    )
