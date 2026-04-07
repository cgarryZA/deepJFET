"""Intel 4004 instruction set definition and decoder.

Decodes 4004 machine code into structured instruction objects,
identifying which hardware resources each instruction touches.
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class Resource(Enum):
    """Hardware resources that an instruction can touch."""
    ACC = auto()          # Accumulator
    CY = auto()           # Carry flag
    REG = auto()          # Single index register (RRRR)
    REG_PAIR = auto()     # Register pair (RRR)
    STACK = auto()        # Subroutine stack (push/pop)
    PC = auto()           # Program counter
    RAM = auto()          # RAM main memory
    RAM_STATUS = auto()   # RAM status characters
    RAM_PORT = auto()     # RAM output port
    ROM_PORT = auto()     # ROM I/O port
    CMD_REG = auto()      # Command register (bank select)


@dataclass
class Instruction:
    """A decoded 4004 instruction."""
    address: int
    opcode: int           # Full first byte
    mnemonic: str
    size: int             # 1 or 2 bytes
    operand: int = None   # Second byte if 2-word instruction

    # Resource usage
    reads: set = field(default_factory=set)    # Resources read
    writes: set = field(default_factory=set)   # Resources written

    # Specific register/pair accessed (None if N/A)
    register: int = None       # 0-15 for RRRR
    register_pair: int = None  # 0-7 for RRR (pair number, not register number)

    # Stack effect
    stack_push: bool = False
    stack_pop: bool = False

    # Branch target (None if not a branch)
    branch_target: int = None
    is_conditional: bool = False

    # RAM/ROM status character index (0-3)
    status_char: int = None

    @property
    def register_pair_regs(self) -> tuple:
        """Return the two register numbers in a pair."""
        if self.register_pair is not None:
            return (self.register_pair * 2, self.register_pair * 2 + 1)
        return None


def decode(data: bytes, base_address: int = 0) -> list:
    """Decode a stream of 4004 machine code bytes into Instructions.

    Args:
        data: raw bytes of 4004 machine code
        base_address: address of first byte in ROM

    Returns:
        List of Instruction objects in order.
    """
    instructions = []
    i = 0

    while i < len(data):
        addr = base_address + i
        b0 = data[i]
        opr = (b0 >> 4) & 0xF  # High nibble
        opa = b0 & 0xF          # Low nibble

        inst = Instruction(address=addr, opcode=b0, mnemonic="???", size=1)

        if b0 == 0x00:
            # NOP
            inst.mnemonic = "NOP"

        elif opr == 0x1:
            # JCN — Jump conditional (2 bytes)
            inst.mnemonic = "JCN"
            inst.size = 2
            if i + 1 < len(data):
                inst.operand = data[i + 1]
                # Branch within same page
                page = addr & 0xF00
                inst.branch_target = page | inst.operand
            inst.is_conditional = True
            inst.reads = {Resource.ACC, Resource.CY, Resource.PC}
            inst.writes = {Resource.PC}

        elif opr == 0x2 and (opa & 1) == 0:
            # FIM — Fetch immediate (2 bytes)
            inst.mnemonic = "FIM"
            inst.size = 2
            inst.register_pair = (opa >> 1)
            if i + 1 < len(data):
                inst.operand = data[i + 1]
            inst.writes = {Resource.REG_PAIR}

        elif opr == 0x2 and (opa & 1) == 1:
            # SRC — Send register control
            inst.mnemonic = "SRC"
            inst.register_pair = (opa >> 1)
            inst.reads = {Resource.REG_PAIR}
            # SRC sets up the RAM/ROM address — it enables subsequent RAM/ROM ops

        elif opr == 0x3 and (opa & 1) == 0:
            # FIN — Fetch indirect from ROM
            inst.mnemonic = "FIN"
            inst.register_pair = (opa >> 1)
            # Always reads register pair 0 for the address
            inst.reads = {Resource.REG_PAIR}  # reads pair 0
            inst.writes = {Resource.REG_PAIR}  # writes designated pair

        elif opr == 0x3 and (opa & 1) == 1:
            # JIN — Jump indirect
            inst.mnemonic = "JIN"
            inst.register_pair = (opa >> 1)
            inst.reads = {Resource.REG_PAIR, Resource.PC}
            inst.writes = {Resource.PC}

        elif opr == 0x4:
            # JUN — Jump unconditional (2 bytes)
            inst.mnemonic = "JUN"
            inst.size = 2
            if i + 1 < len(data):
                inst.operand = data[i + 1]
                inst.branch_target = (opa << 8) | inst.operand
            inst.writes = {Resource.PC}

        elif opr == 0x5:
            # JMS — Jump to subroutine (2 bytes)
            inst.mnemonic = "JMS"
            inst.size = 2
            inst.stack_push = True
            if i + 1 < len(data):
                inst.operand = data[i + 1]
                inst.branch_target = (opa << 8) | inst.operand
            inst.reads = {Resource.PC}
            inst.writes = {Resource.PC, Resource.STACK}

        elif opr == 0x6:
            # INC — Increment register
            inst.mnemonic = "INC"
            inst.register = opa
            inst.reads = {Resource.REG}
            inst.writes = {Resource.REG}

        elif opr == 0x7:
            # ISZ — Increment and skip if zero (2 bytes)
            inst.mnemonic = "ISZ"
            inst.size = 2
            inst.register = opa
            inst.is_conditional = True
            if i + 1 < len(data):
                inst.operand = data[i + 1]
                page = addr & 0xF00
                inst.branch_target = page | inst.operand
            inst.reads = {Resource.REG, Resource.PC}
            inst.writes = {Resource.REG, Resource.PC}

        elif opr == 0x8:
            # ADD — Add register to accumulator
            inst.mnemonic = "ADD"
            inst.register = opa
            inst.reads = {Resource.REG, Resource.ACC, Resource.CY}
            inst.writes = {Resource.ACC, Resource.CY}

        elif opr == 0x9:
            # SUB — Subtract register from accumulator
            inst.mnemonic = "SUB"
            inst.register = opa
            inst.reads = {Resource.REG, Resource.ACC, Resource.CY}
            inst.writes = {Resource.ACC, Resource.CY}

        elif opr == 0xA:
            # LD — Load register to accumulator
            inst.mnemonic = "LD"
            inst.register = opa
            inst.reads = {Resource.REG}
            inst.writes = {Resource.ACC}

        elif opr == 0xB:
            # XCH — Exchange register and accumulator
            inst.mnemonic = "XCH"
            inst.register = opa
            inst.reads = {Resource.REG, Resource.ACC}
            inst.writes = {Resource.REG, Resource.ACC}

        elif opr == 0xC:
            # BBL — Branch back and load
            inst.mnemonic = "BBL"
            inst.stack_pop = True
            inst.reads = {Resource.STACK}
            inst.writes = {Resource.ACC, Resource.PC}

        elif opr == 0xD:
            # LDM — Load immediate to accumulator
            inst.mnemonic = "LDM"
            inst.writes = {Resource.ACC}

        elif opr == 0xE:
            # I/O and RAM instructions
            _decode_io(inst, opa)

        elif opr == 0xF:
            # Accumulator group
            _decode_acc_group(inst, opa)

        instructions.append(inst)
        i += inst.size

    return instructions


def _decode_io(inst: Instruction, opa: int):
    """Decode 0xE_ I/O and RAM instructions."""
    io_table = {
        0x0: ("WRM",  {Resource.ACC},        {Resource.RAM}),
        0x1: ("WMP",  {Resource.ACC},        {Resource.RAM_PORT}),
        0x2: ("WRR",  {Resource.ACC},        {Resource.ROM_PORT}),
        0x4: ("WR0",  {Resource.ACC},        {Resource.RAM_STATUS}),
        0x5: ("WR1",  {Resource.ACC},        {Resource.RAM_STATUS}),
        0x6: ("WR2",  {Resource.ACC},        {Resource.RAM_STATUS}),
        0x7: ("WR3",  {Resource.ACC},        {Resource.RAM_STATUS}),
        0x8: ("SBM",  {Resource.RAM, Resource.ACC, Resource.CY}, {Resource.ACC, Resource.CY}),
        0x9: ("RDM",  {Resource.RAM},        {Resource.ACC}),
        0xA: ("RDR",  {Resource.ROM_PORT},   {Resource.ACC}),
        0xB: ("ADM",  {Resource.RAM, Resource.ACC, Resource.CY}, {Resource.ACC, Resource.CY}),
        0xC: ("RD0",  {Resource.RAM_STATUS}, {Resource.ACC}),
        0xD: ("RD1",  {Resource.RAM_STATUS}, {Resource.ACC}),
        0xE: ("RD2",  {Resource.RAM_STATUS}, {Resource.ACC}),
        0xF: ("RD3",  {Resource.RAM_STATUS}, {Resource.ACC}),
    }

    if opa in io_table:
        inst.mnemonic, inst.reads, inst.writes = io_table[opa]
        # Track which status character
        if opa in (0x4, 0xC):
            inst.status_char = 0
        elif opa in (0x5, 0xD):
            inst.status_char = 1
        elif opa in (0x6, 0xE):
            inst.status_char = 2
        elif opa in (0x7, 0xF):
            inst.status_char = 3


def _decode_acc_group(inst: Instruction, opa: int):
    """Decode 0xF_ accumulator group instructions."""
    acc_table = {
        0x0: ("CLB", {}, {Resource.ACC, Resource.CY}),
        0x1: ("CLC", {}, {Resource.CY}),
        0x2: ("IAC", {Resource.ACC}, {Resource.ACC, Resource.CY}),
        0x3: ("CMC", {Resource.CY}, {Resource.CY}),
        0x4: ("CMA", {Resource.ACC}, {Resource.ACC}),
        0x5: ("RAL", {Resource.ACC, Resource.CY}, {Resource.ACC, Resource.CY}),
        0x6: ("RAR", {Resource.ACC, Resource.CY}, {Resource.ACC, Resource.CY}),
        0x7: ("TCC", {Resource.CY}, {Resource.ACC, Resource.CY}),
        0x8: ("DAC", {Resource.ACC}, {Resource.ACC, Resource.CY}),
        0x9: ("TCS", {Resource.CY}, {Resource.ACC, Resource.CY}),
        0xA: ("STC", {}, {Resource.CY}),
        0xB: ("DAA", {Resource.ACC, Resource.CY}, {Resource.ACC, Resource.CY}),
        0xC: ("KBP", {Resource.ACC}, {Resource.ACC}),
        0xD: ("DCL", {Resource.ACC}, {Resource.CMD_REG}),
    }

    if opa in acc_table:
        inst.mnemonic, inst.reads, inst.writes = acc_table[opa]


def disassemble(instructions: list) -> str:
    """Pretty-print a list of decoded instructions."""
    lines = []
    for inst in instructions:
        if inst.size == 2:
            hex_str = f"{inst.opcode:02X} {inst.operand:02X}"
        else:
            hex_str = f"{inst.opcode:02X}   "

        extra = ""
        if inst.register is not None:
            extra += f" R{inst.register}"
        if inst.register_pair is not None:
            extra += f" RP{inst.register_pair}(R{inst.register_pair*2},R{inst.register_pair*2+1})"
        if inst.branch_target is not None:
            extra += f" -> 0x{inst.branch_target:03X}"

        lines.append(f"  {inst.address:03X}: {hex_str}  {inst.mnemonic:<5s}{extra}")

    return "\n".join(lines)
