"""Capstone-backed node inspection used by custom CFG reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from angr import Project
from capstone import CsInsn


# CFGNode equality is address/block-ID based, so a recovered replacement can
# compare equal to the stale node it supersedes. Keep the object alive in each
# entry and key by identity so replacement nodes never reuse stale decoding.
_DECODED_NODE_CACHE: dict[int, tuple[object, DecodedNode]] = {}


def clear_decoded_node_cache() -> None:
    """Discard decoded-node views from the preceding custom CFG build."""

    _DECODED_NODE_CACHE.clear()


def decode_raw_capstone_insns(
    project: Project,
    addr: int,
    size: int,
    *,
    count: int = 0,
) -> tuple[CsInsn, ...]:
    """Decode bytes directly through the project's architecture Capstone engine."""

    try:
        data = project.loader.memory.load(addr, size)
        return tuple(project.arch.capstone.disasm(data, addr, count=count))
    except Exception:
        return ()


def decode_one(project: Project, addr: int, size: int) -> CsInsn | None:
    """Decode one instruction with mode-aware Capstone and a raw fallback."""

    try:
        block = project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        )
        capstone_insns = block.capstone.insns
        if capstone_insns:
            return capstone_insns[0].insn
    except Exception:
        pass

    fallback_insns = decode_raw_capstone_insns(project, addr, size, count=1)
    return fallback_insns[0] if fallback_insns else None


def lift_instruction_vex(project: Project, insn: CsInsn):
    """Lift one instruction without inheriting CFG-node block boundaries."""

    try:
        return project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return None


@dataclass(frozen=True)
class DecodedNode:
    """The Capstone instruction view of one CFG node, when it is available."""

    insns: tuple[CsInsn, ...] | None
    inspection_error: Exception | None = None

    @classmethod
    def from_node(cls, node) -> DecodedNode:
        """Read and cache Capstone instructions for one live CFG node.

        Anomaly checks inspect the same CFGFast nodes repeatedly while the
        worklist repairs nearby blocks. ``node.block.capstone`` constructs a
        fresh angr Block each time, so retaining this immutable view avoids
        repeatedly disassembling unchanged node bytes. The cache is scoped to
        object identity because angr CFG nodes compare by block identity; a
        recovered replacement can otherwise collide with the stale node it
        replaced at the same address.
        """

        cache_key = id(node)
        cached = _DECODED_NODE_CACHE.get(cache_key)
        if cached is not None and cached[0] is node:
            return cached[1]

        try:
            block = node.block
            insns = tuple(item.insn for item in block.capstone.insns)
            if insns:
                decoded = cls(insns)
            else:
                project = block._project
                decoded = cls(decode_raw_capstone_insns(project, node.addr, node.size))

        except Exception as exc:
            decoded = cls(None, exc)

        _DECODED_NODE_CACHE[cache_key] = node, decoded
        return decoded

    @property
    def is_empty(self) -> bool:
        """Return whether Capstone found no instructions in the node."""

        return not self.insns

    @property
    def last(self) -> CsInsn | None:
        """Return the final decoded instruction, if one exists."""

        return self.insns[-1] if self.insns else None

    def has_exact_coverage(self, node) -> bool:
        """Return whether instructions exactly cover the node's declared range."""

        if node.size == 0 or self.insns is None:
            return False
        expected_addr = node.addr
        for insn in self.insns:
            if insn.address != expected_addr:
                return False
            expected_addr += insn.size
        return expected_addr == node.addr + node.size

    def contains_mid_instruction_addr(self, addr: int) -> bool:
        """Return whether ``addr`` falls strictly inside a decoded instruction."""

        if self.insns is None:
            return False
        return any(
            insn.address < addr < insn.address + insn.size for insn in self.insns
        )
