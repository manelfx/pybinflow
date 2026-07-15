"""Capstone-backed node inspection used by custom CFG reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from capstone import CsInsn


@dataclass(frozen=True)
class DecodedNode:
    """The Capstone instruction view of one CFG node, when it is available."""

    insns: tuple[CsInsn, ...] | None
    inspection_error: Exception | None = None

    @classmethod
    def from_node(cls, node) -> DecodedNode:
        """Read a node's Capstone instructions with the usual fallback."""

        try:
            return cls(tuple(item.insn for item in node.block.capstone.insns))
        except (AttributeError, KeyError) as exc:
            return cls(None, exc)

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
