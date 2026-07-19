"""Bounded Capstone/VEX block recovery for localized CFG repair."""

from __future__ import annotations

from angr import Project
from capstone import CsInsn
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
)

from .decode import decode_one
from .jumps import is_direct_target_valid
from .models import BlockSpec, FunctionBounds, TerminatorInfo


def vex_jumpkind_is_terminal(jumpkind: str) -> bool:
    """Return whether VEX marks a block as a return or a synchronous trap."""

    return jumpkind == "Ijk_Ret" or jumpkind.startswith("Ijk_Sig")


def _instruction_has_nonfallthrough_vex_semantics(
    project: Project, insn: CsInsn
) -> bool:
    """Return whether VEX models one exceptional instruction as terminal."""

    semantic = InsnSemantics(insn)
    if not semantic.may_have_nonfallthrough_vex_semantics():
        return False

    try:
        vex = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"Custom CFG could not lift exceptional instruction at {insn.address:#x}: {exc}"
        )
        return False

    return vex_jumpkind_is_terminal(vex.jumpkind)


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
            f"Custom CFG could not lift possible control transfer at "
            f"{insn.address:#x}: {exc}"
        )
        return False

    return vex.jumpkind == "Ijk_Call" or vex_jumpkind_is_terminal(vex.jumpkind)


def _native_vex_transfer_end(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
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
    if jumpkind == "Ijk_Call" or vex_jumpkind_is_terminal(jumpkind):
        return end_addr
    return None


def lift_block_terminator(
    project: Project,
    bounds: FunctionBounds,
    block_insns: list[CsInsn],
    has_nonfallthrough_vex_terminator: bool = False,
    unclassified_vex_terminator_addr: int | None = None,
) -> TerminatorInfo:
    """Lift a recovered block with VEX and derive its control-flow shape."""

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
                f"Custom CFG failed lifting control-transfer block at {block_addr:#x} "
                f"(terminator {last.address:#x}: {last.mnemonic} {last.op_str})"
            ) from exc

    exit_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr != last.address:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            exit_targets.append(target)

    default_target: int | None = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            default_target = target

    # Some compact return encodings are only recognized by the lifter. For
    # example, Capstone classifies RISC-V ``c.jr ra`` as a generic jump while
    # VEX correctly lifts its instruction tail as ``Ijk_Ret``.
    if semantic.is_ret() or vex.jumpkind == "Ijk_Ret":
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if vex_jumpkind_is_terminal(vex.jumpkind):
        return TerminatorInfo(jumpkind="Ijk_Terminal")

    if semantic.is_call() or vex.jumpkind == "Ijk_Call":
        direct_targets: tuple[int, ...] = ()
        if isinstance(default_target, int) and is_direct_target_valid(
            bounds, default_target
        ):
            direct_targets = (default_target,)
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
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
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(default_target,),
        )

    direct_target = semantic.direct_target()
    if semantic.is_jump() and direct_target is not None:
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=(direct_target,),
        )

    direct_targets: tuple[int, ...] = ()
    if isinstance(default_target, int) and is_direct_target_valid(
        bounds, default_target
    ):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def recover_block(
    project: Project, bounds: FunctionBounds, start_addr: int, stop_addrs: set[int]
) -> BlockSpec | None:
    """Decode one block until control flow or a known block start bounds it."""

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    cur = start_addr
    insns: list[CsInsn] = []
    has_delay_slot = arch_has_delay_slot(project.arch.name)
    has_nonfallthrough_vex_terminator = False
    unclassified_vex_terminator_addr: int | None = None
    native_vex_transfer_end = None
    if not has_delay_slot:
        # Capstone omits control-flow groups for a few encodings, including
        # S390 `basr`. One native block lift recovers a trustworthy call or
        # terminal boundary without paying to lift every decoded instruction.
        native_vex_transfer_end = _native_vex_transfer_end(project, bounds, start_addr)

    while bounds.addr <= cur < bounds.end_addr:
        if insns and cur in stop_addrs:
            break

        insn = decode_one(project, cur, max_inst_bytes)
        if insn is None:
            logger.warning(f"Custom CFG could not decode instruction at {cur:#x}")
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

        if _instruction_has_nonfallthrough_vex_semantics(project, insn):
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
    )
    block = BlockSpec(
        addr=insns[0].address,
        size=sum(obj.size for obj in insns),
        instruction_addrs=tuple(obj.address for obj in insns),
        jumpkind=terminator.jumpkind,
        direct_targets=terminator.direct_targets,
        fallthrough_addr=terminator.fallthrough_addr,
    )

    block_end = block.addr + block.size
    internal_targets = sorted(
        target
        for target in block.direct_targets
        if block.addr < target < block_end and target not in stop_addrs
    )
    if internal_targets:
        return recover_block(
            project, bounds, start_addr, stop_addrs | {internal_targets[0]}
        )

    return block
