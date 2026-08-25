"""Capstone-backed node inspection used by custom CFG reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from angr import Project
from capstone import CsInsn
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
)

from .models import BlockSpec, FunctionBounds, TerminatorInfo


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
        arch = project.arch
        try:
            is_thumb = arch.is_thumb(addr)
        except AttributeError:
            is_thumb = False

        # ARM/Thumb stores the execution mode in address bit 0. Read bytes at
        # the physical address, while preserving the tagged address in output.
        # ``capstone_thumb`` is an ARM-specific extension not declared on the
        # base archinfo ``Arch`` type.
        thumb_capstone = getattr(arch, "capstone_thumb", None)

        if is_thumb and thumb_capstone is not None:
            capstone = thumb_capstone
            memory_addr = addr & ~1
        else:
            capstone = arch.capstone
            memory_addr = addr
        data = project.loader.memory.load(memory_addr, size)
        return tuple(capstone.disasm(data, addr, count=count))
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


def is_post_prefix_instruction_entry(insn: CsInsn, addr: int) -> bool:
    """Return whether ``addr`` enters immediately after all instruction prefixes.

    Some x86 binaries deliberately branch after an instruction prefix, such as
    ``LOCK``. The remaining opcode bytes form a valid alternate instruction
    stream, unlike arbitrary entries into the middle of an instruction.
    """

    prefix_size = sum(1 for prefix in getattr(insn, "prefix", ()) if prefix)
    return (
        prefix_size > 0
        and addr == insn.address + prefix_size
        and addr < insn.address + insn.size
    )


def _block_insns(project: Project, block: BlockSpec) -> tuple[CsInsn, ...]:
    """Decode a recovered block with Capstone without constructing an angr Block."""

    return decode_raw_capstone_insns(project, block.addr, block.size)


def _containing_block_insn(
    project: Project,
    block: BlockSpec,
    addr: int,
) -> CsInsn | None:
    """Return the recovered instruction that strictly contains ``addr``."""

    return next(
        (
            insn
            for insn in _block_insns(project, block)
            if insn.address < addr < insn.address + insn.size
        ),
        None,
    )


def _max_instruction_bytes(project: Project) -> int:
    """Return the architecture's maximum instruction width with a safe default."""

    try:
        return project.arch.max_inst_bytes
    except AttributeError:
        return 16


def alternate_block_entry_rejoin_addr(
    project: Project,
    block: BlockSpec,
    addr: int,
) -> int | None:
    """Return the shared tail of a bounded alternate instruction stream.

    An entry inside an instruction is only accepted when decoding one
    instruction at that address reaches the original instruction's end. This
    covers x86 post-prefix streams and valid Thumb halfword alternate streams,
    while rejecting arbitrary mid-instruction targets.
    """

    insn = _containing_block_insn(project, block, addr)
    if insn is None:
        return None

    alternate = decode_raw_capstone_insns(
        project,
        addr,
        _max_instruction_bytes(project),
        count=1,
    )
    if len(alternate) != 1:
        return None

    rejoin_addr = insn.address + insn.size
    return (
        rejoin_addr if alternate[0].address + alternate[0].size == rejoin_addr else None
    )


def is_valid_block_entry(project: Project, block: BlockSpec, addr: int) -> bool:
    """Return whether ``addr`` is a normal or supported alternate leader."""

    return addr in block.instruction_addrs or (
        alternate_block_entry_rejoin_addr(project, block, addr) is not None
    )


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


def vex_jumpkind_is_terminal(jumpkind: str) -> bool:
    """Return whether VEX marks a block as a return or synchronous trap."""

    return jumpkind == "Ijk_Ret" or jumpkind.startswith("Ijk_Sig")


def vex_jumpkind_is_syscall(jumpkind: str) -> bool:
    """Return whether VEX marks a transfer into an operating-system service."""

    return jumpkind.startswith("Ijk_Sys_")


def call_fallthrough_addr(
    project: Project, bounds: FunctionBounds, next_addr: int
) -> int | None:
    """Return a call continuation inside this function or at a known next one."""

    if bounds.addr <= next_addr < bounds.end_addr:
        return next_addr

    symbol = project.loader.find_symbol(next_addr)
    if symbol is not None and symbol.rebased_addr == next_addr and symbol.is_function:
        return next_addr

    return None


def _exceptional_instruction_vex_jumpkind(project: Project, insn: CsInsn) -> str | None:
    """Return a terminal or syscall jumpkind for one exceptional instruction."""

    semantic = InsnSemantics(insn)
    if not semantic.may_have_nonfallthrough_vex_semantics():
        return None

    try:
        vex = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"CFG decoder could not lift exceptional instruction at "
            f"{insn.address:#x}: {exc}"
        )
        return None

    if vex_jumpkind_is_terminal(vex.jumpkind) or vex_jumpkind_is_syscall(vex.jumpkind):
        return vex.jumpkind
    return None


def _instruction_has_unclassified_vex_transfer(project: Project, insn: CsInsn) -> bool:
    """Return whether VEX identifies an executable direct target as control flow."""

    semantic = InsnSemantics(insn)
    if semantic.is_control_transfer():
        return False

    target = semantic.direct_target()
    if target is None:
        return False

    obj = project.loader.find_object_containing(target)
    if obj is None:
        return False
    section = obj.find_section_containing(target)
    if section is None or not section.is_executable:
        return False

    lift_size = insn.size
    if arch_has_delay_slot(project.arch.name):
        # VEX needs the executed delay-slot instruction to classify MIPS BAL
        # and similar branch-and-link instructions as calls.
        delay_insn = decode_one(
            project,
            insn.address + insn.size,
            getattr(project.arch, "max_inst_bytes", 16),
        )
        if delay_insn is not None:
            lift_size += delay_insn.size

    try:
        vex = project.factory.block(
            insn.address,
            size=lift_size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"CFG decoder could not lift possible control transfer at "
            f"{insn.address:#x}: {exc}"
        )
        return False

    return vex.jumpkind == "Ijk_Call" or vex_jumpkind_is_terminal(vex.jumpkind)


def _native_vex_transfer_end(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
    *,
    split_syscall_blocks: bool = False,
) -> int | None:
    """Return a native VEX call or terminal boundary absent from Capstone groups."""

    try:
        block = project.factory.block(start_addr)
    except Exception:
        return None

    size = getattr(block, "size", None)
    jumpkind = getattr(block.vex, "jumpkind", None)
    if not isinstance(size, int) or not isinstance(jumpkind, str):
        return None

    end_addr = start_addr + size
    if not start_addr < end_addr <= bounds.end_addr:
        return None
    if (
        jumpkind == "Ijk_Call"
        or vex_jumpkind_is_terminal(jumpkind)
        or (split_syscall_blocks and vex_jumpkind_is_syscall(jumpkind))
    ):
        return end_addr
    return None


def lift_block_terminator(
    project: Project,
    bounds: FunctionBounds,
    block_insns: list[CsInsn],
    has_nonfallthrough_vex_terminator: bool = False,
    unclassified_vex_terminator_addr: int | None = None,
    preserve_conditional_return_fallthrough: bool = False,
    split_syscall_blocks: bool = False,
) -> TerminatorInfo:
    """Lift decoded block bytes and derive their control-flow shape."""

    # Jump-table support consumes the basic decoding utilities in this module.
    # Delay the reverse dependency until terminator classification to avoid an
    # import cycle while still sharing its target-validity policy.
    from .jumps import is_direct_target_valid

    block_end_addr = block_insns[-1].address + block_insns[-1].size

    term_idx = control_transfer_index(project.arch.name, block_insns)
    if term_idx is None and unclassified_vex_terminator_addr is not None:
        term_idx = next(
            (
                index
                for index, insn in enumerate(block_insns)
                if insn.address == unclassified_vex_terminator_addr
            ),
            None,
        )

    if term_idx is None:
        if has_nonfallthrough_vex_terminator:
            return TerminatorInfo(jumpkind="Ijk_Terminal")

        next_addr = block_end_addr
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr
        )

    tail_insns = block_insns[term_idx:]
    last = tail_insns[0]
    semantic = InsnSemantics(last)
    next_addr = block_end_addr

    tail_addr = tail_insns[0].address
    tail_size = sum(insn.size for insn in tail_insns)
    block_addr = block_insns[0].address
    block_size = sum(insn.size for insn in block_insns)

    def _lift(addr: int, size: int):
        return project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex

    try:
        vex = _lift(tail_addr, tail_size)
    except Exception:
        try:
            vex = _lift(block_addr, block_size)
        except Exception as exc:
            raise RuntimeError(
                f"CFG decoder failed lifting control-transfer block at {block_addr:#x} "
                f"(terminator {last.address:#x}: {last.mnemonic} {last.op_str})"
            ) from exc

    terminator_addrs = {last.address}
    if arch_has_delay_slot(project.arch.name) and len(tail_insns) > 1:
        terminator_addrs.add(tail_insns[1].address)

    exit_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr not in terminator_addrs:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            exit_targets.append(target)

    default_target: int | None = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            default_target = target

    # A terminal VEX return can still carry an explicit ordinary Exit to the
    # next instruction. This represents the not-taken path of a conditional
    # return; the taken return target is caller-dependent and is not drawn.
    if (
        preserve_conditional_return_fallthrough
        and vex.jumpkind == "Ijk_Ret"
        and next_addr in exit_targets
    ):
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(jumpkind="Ijk_Boring", fallthrough_addr=fallthrough_addr)

    if semantic.is_ret() or vex.jumpkind == "Ijk_Ret":
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if split_syscall_blocks and vex_jumpkind_is_syscall(vex.jumpkind):
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Syscall",
            fallthrough_addr=fallthrough_addr,
            syscall_jumpkind=vex.jumpkind,
        )

    if vex_jumpkind_is_terminal(vex.jumpkind):
        return TerminatorInfo(jumpkind="Ijk_Terminal")

    if semantic.is_call() or vex.jumpkind == "Ijk_Call":
        direct_targets: tuple[int, ...] = ()
        if isinstance(default_target, int):
            # A constant VEX call target is precise even when it has no loader
            # symbol. The extractor materializes an ExternalTarget leaf for
            # such unnamed callees, allowing later render policy to decide
            # whether it should be visible.
            direct_targets = (default_target,)
        fallthrough_addr = call_fallthrough_addr(project, bounds, next_addr)
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
        )

    if exit_targets or semantic.is_conditional_jump():
        if last.address in exit_targets:
            direct_target = semantic.direct_target()
            if direct_target is None:
                direct_target = (
                    default_target
                    if isinstance(default_target, int)
                    and is_direct_target_valid(bounds, default_target)
                    else None
                )
            fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
            return TerminatorInfo(
                jumpkind="Ijk_Boring",
                direct_targets=(direct_target,) if direct_target is not None else (),
                fallthrough_addr=fallthrough_addr,
            )

        all_targets: list[int] = list(exit_targets)
        if isinstance(default_target, int) and default_target not in all_targets:
            all_targets.append(default_target)
        fallthrough_addr = (
            next_addr
            if next_addr in all_targets and next_addr < bounds.end_addr
            else None
        )
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(
                target for target in all_targets if target != fallthrough_addr
            ),
            fallthrough_addr=fallthrough_addr,
        )

    if (
        semantic.is_jump()
        and isinstance(default_target, int)
        and is_direct_target_valid(bounds, default_target)
    ):
        return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=(default_target,))

    direct_target = semantic.direct_target()
    if semantic.is_jump() and direct_target is not None:
        return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=(direct_target,))

    direct_targets: tuple[int, ...] = ()
    if isinstance(default_target, int) and is_direct_target_valid(
        bounds, default_target
    ):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def decode_bounded_block(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
    stop_addrs: set[int],
    *,
    preserve_conditional_return_fallthrough: bool = False,
    split_syscall_blocks: bool = False,
) -> BlockSpec | None:
    """Decode one bounded block until control flow or a known leader stops it."""

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    cur = start_addr
    insns: list[CsInsn] = []
    has_delay_slot = arch_has_delay_slot(project.arch.name)
    has_nonfallthrough_vex_terminator = False
    unclassified_vex_terminator_addr: int | None = None
    # Capstone does not consistently group trap instructions as control flow.
    # Let VEX provide a native boundary on every architecture. Recognized
    # delay-slot branches still take the explicit delay-slot path below.
    native_vex_transfer_end = _native_vex_transfer_end(
        project,
        bounds,
        start_addr,
        split_syscall_blocks=split_syscall_blocks,
    )

    while bounds.addr <= cur < bounds.end_addr:
        if insns and cur in stop_addrs:
            break

        insn = decode_one(project, cur, max_inst_bytes)
        if insn is None:
            logger.warning(f"CFG decoder could not decode instruction at {cur:#x}")
            break

        insns.append(insn)
        semantic = InsnSemantics(insn)
        next_addr = insn.address + insn.size

        if semantic.is_control_transfer():
            if has_delay_slot and bounds.addr <= next_addr < bounds.end_addr:
                delay_insn = decode_one(project, next_addr, max_inst_bytes)
                if delay_insn is not None:
                    insns.append(delay_insn)
            break

        if semantic.is_undefined_instruction_trap():
            # VEX reports x86 UD2 as Ijk_NoDecode. It is nevertheless an
            # intentional synchronous trap, never a linear fallthrough.
            has_nonfallthrough_vex_terminator = True
            break

        exceptional_jumpkind = _exceptional_instruction_vex_jumpkind(project, insn)
        if exceptional_jumpkind is not None:
            if split_syscall_blocks and vex_jumpkind_is_syscall(exceptional_jumpkind):
                unclassified_vex_terminator_addr = insn.address
                break
            if not vex_jumpkind_is_syscall(exceptional_jumpkind):
                has_nonfallthrough_vex_terminator = True
                break

        if _instruction_has_unclassified_vex_transfer(project, insn):
            unclassified_vex_terminator_addr = insn.address
            if has_delay_slot and bounds.addr <= next_addr < bounds.end_addr:
                delay_insn = decode_one(project, next_addr, max_inst_bytes)
                if delay_insn is not None:
                    insns.append(delay_insn)
            break

        if next_addr == native_vex_transfer_end:
            unclassified_vex_terminator_addr = insn.address
            break

        cur = next_addr

    if not insns:
        return None

    terminator = lift_block_terminator(
        project,
        bounds,
        insns,
        has_nonfallthrough_vex_terminator=has_nonfallthrough_vex_terminator,
        unclassified_vex_terminator_addr=unclassified_vex_terminator_addr,
        preserve_conditional_return_fallthrough=preserve_conditional_return_fallthrough,
        split_syscall_blocks=split_syscall_blocks,
    )
    block = BlockSpec(
        addr=insns[0].address,
        size=sum(obj.size for obj in insns),
        instruction_addrs=tuple(obj.address for obj in insns),
        jumpkind=terminator.jumpkind,
        direct_targets=terminator.direct_targets,
        fallthrough_addr=terminator.fallthrough_addr,
        syscall_jumpkind=terminator.syscall_jumpkind,
    )

    block_end = block.addr + block.size
    internal_targets = sorted(
        target
        for target in block.direct_targets
        if block.addr < target < block_end and target not in stop_addrs
    )
    if internal_targets:
        return decode_bounded_block(
            project,
            bounds,
            start_addr,
            stop_addrs | {internal_targets[0]},
            preserve_conditional_return_fallthrough=preserve_conditional_return_fallthrough,
            split_syscall_blocks=split_syscall_blocks,
        )

    return block


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
