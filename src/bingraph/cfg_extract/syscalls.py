"""Conservative SimOS-backed syscall recovery for extracted blocks."""

from __future__ import annotations

from dataclasses import dataclass

from angr import Project, options
from angr.simos.userland import SimUserland

from bingraph.cfg.models import BlockSpec


_UNRESOLVABLE_SYSCALL_ADDR = 0xFFFFFFFFFFFFFFE0


@dataclass(frozen=True)
class ResolvedSyscall:
    """One concrete syscall target recovered from a terminating block."""

    addr: int
    name: str
    no_return: bool


def unknown_syscall_target(project: Project) -> ResolvedSyscall:
    """Return angr's stable synthetic identity for an unresolved syscall."""

    try:
        simos = project.simos
        if not isinstance(simos, SimUserland):
            raise TypeError("SimOS does not expose userland syscalls")
        number = simos.unknown_syscall_number
        if not isinstance(number, int):
            raise TypeError("missing unknown syscall number")
        procedure = simos.syscall_from_number(number)
        addr = getattr(procedure, "addr", None)
        name = getattr(procedure, "display_name", None)
        if isinstance(addr, int) and isinstance(name, str) and name:
            return ResolvedSyscall(addr, name, no_return=False)
    except (AttributeError, TypeError, ValueError):
        pass
    return ResolvedSyscall(
        _UNRESOLVABLE_SYSCALL_ADDR, "UnresolvableSyscallTarget", no_return=False
    )


def resolve_static_syscall(
    project: Project, block: BlockSpec
) -> ResolvedSyscall | None:
    """Resolve a syscall when its number is concrete before its terminator.

    SimOS owns the architecture and ABI-specific syscall-number convention. The
    extractor executes only the linear prefix before the already-bounded
    syscall instruction from a blank symbolic state. Any symbolic, ambiguous,
    or unsupported result remains unresolved.
    """

    if block.jumpkind != "Ijk_Syscall" or block.size <= 0:
        return None

    try:
        simos = project.simos
        if not isinstance(simos, SimUserland):
            return None
        state = project.factory.blank_state(
            addr=block.addr,
            add_options={
                options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
            },
        )
        syscall_addr = block.instruction_addrs[-1]
        prefix_size = syscall_addr - block.addr
        if prefix_size > 0:
            successors = project.factory.successors(
                state, addr=block.addr, size=prefix_size
            )
            if (
                len(successors.flat_successors) != 1
                or len(successors.all_successors) != 1
            ):
                return None
            state = successors.flat_successors[0]
            if state.addr != syscall_addr:
                return None
        abi = simos.syscall_abi(state)
        number = state.solver.eval_one(simos.syscall_cc(state).syscall_num(state))
        if not isinstance(number, int) or number == simos.unknown_syscall_number:
            return None
        procedure = simos.syscall_from_number(number, abi=abi)
        if procedure is None:
            return None
        addr = getattr(procedure, "addr", None)
        name = getattr(procedure, "display_name", None)
        if not isinstance(addr, int) or not isinstance(name, str) or not name:
            return None
        return ResolvedSyscall(addr, name, no_return=bool(procedure.NO_RET))
    except Exception:
        # Local symbolic execution is an optional proof. Preserve the generic
        # unknown syscall leaf for engines or ABI states it cannot model.
        return None
