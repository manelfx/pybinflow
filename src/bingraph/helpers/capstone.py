"""Small Capstone helpers shared by CFG discovery and rendering."""

from capstone import (
    CS_GRP_CALL,
    CS_GRP_INT,
    CS_GRP_IRET,
    CS_GRP_JUMP,
    CS_GRP_PRIVILEGE,
    CS_GRP_RET,
    CS_OP_IMM,
    CsInsn,
)
from capstone.arm import ARM_CC_AL, ARM_CC_INVALID, ARM_INS_IT
from capstone.systemz import SYSZ_INS_BC
from capstone.x86 import X86_INS_JMP, X86_INS_LJMP, X86_INS_UD2
from capstone.x86_const import X86_GRP_AVX512


class InsnSemantics:
    """Expose the Capstone control-flow properties used by bingraph."""

    def __init__(self, insn: CsInsn):
        self.insn = insn

    @property
    def address(self) -> int:
        """Return the instruction address."""

        return self.insn.address

    @property
    def size(self) -> int:
        """Return the encoded instruction size."""

        return self.insn.size

    def is_ret(self) -> bool:
        """Return whether Capstone classifies this instruction as a return."""

        return CS_GRP_RET in self.insn.groups

    def is_call(self) -> bool:
        """Return whether Capstone classifies this instruction as a call."""

        return CS_GRP_CALL in self.insn.groups

    def is_jump(self) -> bool:
        """Return whether Capstone classifies this instruction as a jump."""

        if CS_GRP_JUMP not in self.insn.groups:
            return False

        # SystemZ's BC instruction uses its first immediate operand as a
        # condition mask. A zero mask is an architectural no-op, even though
        # Capstone places it in the generic jump group.
        if self.insn.id != SYSZ_INS_BC:
            return True
        first_operand = self.insn.operands[0]
        return not (
            first_operand.type == CS_OP_IMM
            and isinstance(first_operand.imm, int)
            and first_operand.imm == 0
        )

    def is_control_transfer(self) -> bool:
        """Return whether this instruction changes normal control flow."""

        return self.is_ret() or self.is_call() or self.is_jump()

    def is_avx512(self) -> bool:
        """Return whether Capstone classifies this x86 instruction as AVX-512."""

        return X86_GRP_AVX512 in self.insn.groups

    def is_undefined_instruction_trap(self) -> bool:
        """Return whether this is x86's intentional undefined-instruction trap."""

        return self.insn.id == X86_INS_UD2

    def may_have_nonfallthrough_vex_semantics(self) -> bool:
        """Return whether this system instruction needs a narrow VEX check."""

        return bool(
            {CS_GRP_INT, CS_GRP_IRET, CS_GRP_PRIVILEGE}.intersection(self.insn.groups)
        )

    def is_conditional_jump(self) -> bool:
        """Return whether this direct jump has an architecture-level condition."""

        if not self.is_jump() or self.direct_target() is None:
            return False

        if len(self.insn.operands) > 1:
            return True

        arm_cc = getattr(self.insn, "cc", ARM_CC_INVALID)
        if arm_cc == ARM_CC_AL:
            return False
        if arm_cc not in {ARM_CC_INVALID, ARM_CC_AL}:
            return True

        insn_id = getattr(self.insn, "id", None)
        if insn_id in {X86_INS_JMP, X86_INS_LJMP}:
            return False

        return True

    def has_explicit_branch_condition(self) -> bool:
        """Return whether a direct branch carries its condition as an operand."""

        return (
            self.is_jump()
            and self.direct_target() is not None
            and len(self.insn.operands) > 1
        )

    def direct_target(self) -> int | None:
        """Return the final immediate operand when it is a branch target."""

        # Capstone places a direct target in the final immediate operand for
        # instructions such as Thumb `cbz r2, #target` and S390 `cije`.
        for operand in reversed(self.insn.operands):
            if getattr(operand, "type", None) != CS_OP_IMM:
                continue

            imm = getattr(operand, "imm", None)
            if isinstance(imm, int):
                return imm

        return None

    def direct_target_for_arch(self, arch_name: str) -> int | None:
        """Return the direct target in the address form used by ``arch_name``.

        Capstone exposes RISC-V direct branch operands as signed PC-relative
        displacements, unlike its absolute targets on the architectures used
        by most callers. Keep ``direct_target()`` as the raw Capstone value
        and normalize only consumers that compare it with CFG node addresses.
        """

        target = self.direct_target()
        if target is None:
            return None
        if arch_name in {"RISCV32", "RISCV64"} and self.is_jump():
            return self.address + target
        return target


def proven_unconditional_direct_target(
    arch_name: str, insns: list[CsInsn], terminator_index: int
) -> int | None:
    """Return a direct target only when Capstone proves it is unconditional.

    Thumb's ``IT`` instruction predicates a bounded sequence of later
    instructions without changing their individual Capstone condition codes.
    Leave a branch inside that sequence to VEX, but do not let an expired
    ``IT`` predicate hide a later unconditional branch.
    """

    try:
        terminator = InsnSemantics(insns[terminator_index])
        is_unconditional_jump = (
            terminator.is_jump()
            and not terminator.is_call()
            and not terminator.is_conditional_jump()
        )
    except AttributeError:
        return None
    if not is_unconditional_jump:
        return None
    if arch_name.startswith("ARM") and _thumb_it_predicates_instruction(
        insns, terminator_index
    ):
        return None
    return terminator.direct_target_for_arch(arch_name)


def _thumb_it_predicates_instruction(insns: list[CsInsn], index: int) -> bool:
    """Return whether a preceding Thumb IT encoding predicates ``insns[index]``.

    Capstone exposes the IT opcode and raw encoding, but not the number of
    instructions selected by its mask. The least-significant set mask bit
    encodes that count, so inspect the instruction bytes instead of parsing
    presentation mnemonics such as ``itt`` or ``ittt``.
    """

    for it_index, insn in enumerate(insns[:index]):
        if insn.id != ARM_INS_IT:
            continue
        try:
            mask = insn.bytes[0] & 0xF
        except (AttributeError, IndexError):
            # Without the encoding, keep the conservative behavior: this may
            # be an IT-predicated branch and VEX remains the safer authority.
            return True
        if mask == 0:
            return True
        predicate_count = 5 - (mask & -mask).bit_length()
        if index <= it_index + predicate_count:
            return True
    return False


def arch_has_delay_slot(arch_name: str) -> bool:
    """Return whether a named architecture consumes one delay-slot instruction."""

    return arch_name in {"MIPS32", "MIPS64"}


def control_transfer_index(
    arch_name: str, insns: list[CsInsn], *, strict: bool = True
) -> int | None:
    """Return the effective control-transfer index, accounting for delay slots.

    ``strict=False`` permits inspection of CFGFast seed blocks where an
    architecture-specific PC setup instruction appears before later linear
    instructions. Recovery keeps the default because rebuilt blocks must end
    at their first real control-transfer instruction.
    """

    has_delay_slot = arch_has_delay_slot(arch_name)
    for index in range(len(insns) - 1, -1, -1):
        if not InsnSemantics(insns[index]).is_control_transfer():
            continue

        if index != len(insns) - 1 and not has_delay_slot and strict:
            raise RuntimeError(
                f"Recovered non-delay block at {insns[0].address:#x} has trailing "
                f"instructions after control transfer {insns[index].address:#x}"
            )
        return index

    return None
