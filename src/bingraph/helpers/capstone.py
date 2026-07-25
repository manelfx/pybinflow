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
from capstone.arm import ARM_CC_AL, ARM_CC_INVALID
from capstone.systemz import SYSZ_INS_BC
from capstone.x86 import X86_INS_JMP, X86_INS_LJMP
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

        # Some architectures use a direct branch or call to the immediately
        # following instruction as a PC-relative register setup idiom. It has
        # side effects, but all executable paths remain linear in the CFG.
        if self.direct_target() == self.address + self.size:
            return False
        return self.is_ret() or self.is_call() or self.is_jump()

    def is_avx512(self) -> bool:
        """Return whether Capstone classifies this x86 instruction as AVX-512."""

        return X86_GRP_AVX512 in self.insn.groups

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


def arch_has_delay_slot(arch_name: str) -> bool:
    """Return whether a named architecture consumes one delay-slot instruction."""

    return arch_name in {"MIPS32", "MIPS64"}


def control_transfer_index(arch_name: str, insns: list[CsInsn]) -> int | None:
    """Return the effective control-transfer index, accounting for delay slots."""

    has_delay_slot = arch_has_delay_slot(arch_name)
    for index in range(len(insns) - 1, -1, -1):
        if not InsnSemantics(insns[index]).is_control_transfer():
            continue

        if index != len(insns) - 1 and not has_delay_slot:
            raise RuntimeError(
                f"Recovered non-delay block at {insns[0].address:#x} has trailing "
                f"instructions after control transfer {insns[index].address:#x}"
            )
        return index

    return None
