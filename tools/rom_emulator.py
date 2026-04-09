#!/usr/bin/env python3
"""ROM emulator for the 4004 CPU.

Instead of simulating 16K+ ROM bit cells in LTSpice, this tool:
1. Assembles a 4004 program to machine code
2. Simulates the CPU's program counter progression (handling branches/calls)
3. Generates PWL files for D0In-D3In that present instruction data at the
   correct micro-phase timing

The output PWL files can be used directly in LTSpice to drive the CPU's
data input bus, replacing the ROM entirely.

Timing model (from controls.asc clock definitions):
    CLK period:     10us (100 kHz)
    ROMCLK:         CLK shifted by -2us
    One micro-phase: 10us (one CLK period)

Fetch cycle (per instruction):
    Micro1-3: CPU outputs 12-bit PC address onto D0-D3 (we ignore these)
    Micro4:   ROM should present OPR (high nibble) on D0In-D3In
    Micro5:   ROM should present OPA (low nibble) on D0In-D3In
    Micro6-8: Execute (1-word instruction resets at Micro8)

    For 2-word instructions (JCN, FIM, JUN, JMS, ISZ):
    Micro9-11:  CPU outputs address for 2nd byte
    Micro12:    ROM presents 2nd byte high nibble
    Micro13:    ROM presents 2nd byte low nibble
    Micro14-16: Execute 2nd part

Usage:
    python tools/rom_emulator.py cpus/4004/programs/Load5/Load5.asm --cycles 20
    python tools/rom_emulator.py program.asm --output ltspice/output/
"""

import argparse
import os
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
PROJECT_ROOT = os.path.abspath(_root)
sys.path.insert(0, _root)

# Logic voltage levels (from controls.asc)
V_HIGH = -0.8   # Logic 1
V_LOW = -3.6    # Logic 0

# Timing
CLK_PERIOD = 10e-6       # 10 microseconds
# Per-instruction clock cycle counts.
# 1-word instructions do: micro 1-2-3-4-5 then execute phases.
# 2-word instructions do: micro 1-2-3-4-5 then 6-7 then 1-2-3-4-5 again for byte 2.
# The number here is total clock cycles for the full instruction.
INST_PHASES = {
    # 6 cycles
    0xFC: 6,   # KBP

    # 7 cycles
    0xD0: 7,   # LDM
    0xF0: 7,   # CLB
    0xF1: 7,   # CLC
    0xF3: 7,   # CMC
    0xF4: 7,   # CMA
    0xF9: 7,   # TCS
    0xFA: 7,   # STC
    0xFD: 7,   # DCL

    # 8 cycles
    0x00: 8,   # NOP
    0xA0: 8,   # LD
    0xC0: 8,   # BBL
    0xF2: 8,   # IAC
    0xF5: 8,   # RAL
    0xF6: 8,   # RAR
    0xF7: 8,   # TCC
    0xF8: 8,   # DAC
    0xFB: 8,   # DAA

    # 9 cycles
    0x60: 9,   # INC
    0x80: 9,   # ADD
    0x90: 9,   # SUB
    0xE8: 9,   # SBM
    0xEB: 9,   # ADM

    # 10 cycles
    0xB0: 10,  # XCH

    # 15 cycles (2-word: 7 + 8 fetch phases)
    0x10: 15,  # JCN

    # 16 cycles (2-word: 7 + 9 or similar)
    0x20: 16,  # FIM (when even OPA)
    0x30: 16,  # FIN (when even OPA), JIN (when odd OPA)
    0x40: 16,  # JUN

    # 18 cycles (2-word)
    0x50: 18,  # JMS
    0x70: 18,  # ISZ

    # I/O: WR/RD variants = 8, SRC = 9
    0xE0: 8,   # WRM
    0xE1: 8,   # WMP
    0xE2: 8,   # WRR
    0xE4: 8,   # WR0
    0xE5: 8,   # WR1
    0xE6: 8,   # WR2
    0xE7: 8,   # WR3
    0xE9: 8,   # RDM
    0xEA: 8,   # RDR
    0xEC: 8,   # RD0
    0xED: 8,   # RD1
    0xEE: 8,   # RD2
    0xEF: 8,   # RD3
}

DEFAULT_PHASES = 8
RISE_TIME = 100e-9        # 100ns rise/fall for PWL transitions

# Phase offsets within an instruction cycle (relative to cycle start)
# Micro1 starts at t=0 of the cycle, each phase is CLK_PERIOD long
# ROM data must be stable BEFORE the CPU reads it
# ROMCLK leads CLK by 2us, so ROM data should be ready 2us before the phase
ROMCLK_LEAD = 5e-6       # Must be enough for JFET+OR gate propagation (~3-5us)

# Which micro-phases expect ROM data on the bus
# From reference Load5.raw analysis:
#   Bus sees OPR during Micro5, OPA during Micro6.
#   But our PWL data needs to arrive ~1 CLK earlier to propagate through
#   the JFET+OR gate chain. So we target the START of Micro5 and Micro6,
#   meaning we set data at the end of Micro4 and Micro5.
#
# After two iterations of comparison, the correct offsets are:
#   Phase 5 (0-indexed) = Micro6 timing for OPR
#   Phase 6 (0-indexed) = Micro7 timing for OPA
# Byte 1 fetch: IR1 loads during Micro4, IR2 loads during Micro5.
# The data must propagate through the JFET+OR gate chain (~5us) before
# the IR latches it. So we present data one phase early:
#   OPR data at phase 4 (0-indexed) -> settles during Micro4->5 transition
#   OPA data at phase 5 (0-indexed) -> settles during Micro5->6 transition
# This was verified working at 94% match on Load5.
FETCH_PHASE_OPR = 5       # Data arrives during Micro5->Micro6 boundary
FETCH_PHASE_OPA = 6       # Data arrives during Micro6->Micro7 boundary
# Byte 2 fetch (2-word instructions): after Micro6-7, counter resets,
# then Micro1-3 (address), Micro4 loads IR3, Micro5 loads IR4.
# = 7 phases from instruction start + 5,6 = phases 12,13
FETCH_PHASE_OPR2 = 12     # IR3 (second byte OPR)
FETCH_PHASE_OPA2 = 13     # IR4 (second byte OPA)


# ── Assembler (simplified, reuses logic from Assembler.py) ──────────────

OPCODES_1WORD = {
    'NOP': 0x00,
    'WRM': 0xE0, 'WMP': 0xE1, 'WRR': 0xE2, 'WPM': 0xE3,
    'WR0': 0xE4, 'WR1': 0xE5, 'WR2': 0xE6, 'WR3': 0xE7,
    'SBM': 0xE8, 'RDM': 0xE9, 'RDR': 0xEA, 'ADM': 0xEB,
    'RD0': 0xEC, 'RD1': 0xED, 'RD2': 0xEE, 'RD3': 0xEF,
    'CLB': 0xF0, 'CLC': 0xF1, 'IAC': 0xF2, 'CMC': 0xF3,
    'CMA': 0xF4, 'RAL': 0xF5, 'RAR': 0xF6, 'TCC': 0xF7,
    'DAC': 0xF8, 'TCS': 0xF9, 'STC': 0xFA, 'DAA': 0xFB,
    'KBP': 0xFC, 'DCL': 0xFD,
}

OPCODES_1WORD_REG = {
    'INC': 0x60, 'ADD': 0x80, 'SUB': 0x90, 'LD': 0xA0,
    'XCH': 0xB0, 'BBL': 0xC0, 'LDM': 0xD0,
}

OPCODES_1WORD_PAIR = {
    'SRC': 0x21, 'FIN': 0x30, 'JIN': 0x31,
}

OPCODES_2WORD = {
    'JCN': 0x10, 'FIM': 0x20, 'JUN': 0x40, 'JMS': 0x50, 'ISZ': 0x70,
}

TWO_WORD_MNEMONICS = set(OPCODES_2WORD.keys())


def assemble(asm_text: str) -> list:
    """Assemble 4004 assembly text into a list of bytes (ROM image).

    Returns list of ints (0-255), one per byte.
    """
    lines = [l.strip() for l in asm_text.strip().split('\n') if l.strip()]
    rom = []

    # First pass: find labels
    labels = {}
    addr = 0
    for line in lines:
        parts = line.split()
        mnemonic = parts[0]
        if mnemonic not in OPCODES_1WORD and mnemonic not in OPCODES_1WORD_REG \
           and mnemonic not in OPCODES_1WORD_PAIR and mnemonic not in OPCODES_2WORD:
            # It's a label
            labels[mnemonic] = addr
            parts = parts[1:]
            mnemonic = parts[0] if parts else None
            if mnemonic is None:
                continue
        if mnemonic in TWO_WORD_MNEMONICS:
            addr += 2
        else:
            addr += 1

    # Second pass: generate bytes
    for line in lines:
        parts = line.split()
        mnemonic = parts[0]
        if mnemonic in labels and len(parts) > 1:
            parts = parts[1:]
            mnemonic = parts[0]

        if mnemonic == 'NOP':
            rom.append(0x00)

        elif mnemonic in OPCODES_1WORD:
            rom.append(OPCODES_1WORD[mnemonic])

        elif mnemonic in OPCODES_1WORD_REG:
            reg = int(parts[1])
            rom.append(OPCODES_1WORD_REG[mnemonic] | (reg & 0xF))

        elif mnemonic in OPCODES_1WORD_PAIR:
            pair = int(parts[1])
            rom.append(OPCODES_1WORD_PAIR[mnemonic] | ((pair & 0x7) << 1))

        elif mnemonic == 'JCN':
            cond = int(parts[1])
            if parts[2] in labels:
                addr12 = labels[parts[2]]
            else:
                addr12 = (int(parts[2]) << 8) | (int(parts[3]) << 4) | int(parts[4])
            rom.append(0x10 | (cond & 0xF))
            rom.append(addr12 & 0xFF)

        elif mnemonic == 'FIM':
            pair = int(parts[1])
            # Two data nibbles
            d_high = int(parts[2]) & 0xF
            d_low = int(parts[3]) & 0xF
            rom.append(0x20 | ((pair & 0x7) << 1))
            rom.append((d_high << 4) | d_low)

        elif mnemonic == 'JUN':
            if parts[1] in labels:
                addr12 = labels[parts[1]]
            else:
                addr12 = (int(parts[1]) << 8) | (int(parts[2]) << 4) | int(parts[3])
            rom.append(0x40 | ((addr12 >> 8) & 0xF))
            rom.append(addr12 & 0xFF)

        elif mnemonic == 'JMS':
            if parts[1] in labels:
                addr12 = labels[parts[1]]
            else:
                addr12 = (int(parts[1]) << 8) | (int(parts[2]) << 4) | int(parts[3])
            rom.append(0x50 | ((addr12 >> 8) & 0xF))
            rom.append(addr12 & 0xFF)

        elif mnemonic == 'ISZ':
            reg = int(parts[1])
            if parts[2] in labels:
                addr12 = labels[parts[2]]
            else:
                addr12 = (int(parts[2]) << 8) | (int(parts[3]) << 4) | int(parts[4])
            rom.append(0x70 | (reg & 0xF))
            rom.append(addr12 & 0xFF)

    return rom


# ── CPU Simulator (tracks PC, ACC, CY, registers for branch prediction) ──

class CPU4004Sim:
    """Minimal 4004 CPU simulator for tracking program counter progression."""

    def __init__(self, rom: list):
        self.rom = rom
        self.pc = 0
        self.acc = 0
        self.cy = 0
        self.regs = [0] * 16
        self.stack = [0, 0, 0]
        self.stack_ptr = 0

    def fetch_byte(self) -> int:
        if self.pc < len(self.rom):
            b = self.rom[self.pc]
        else:
            b = 0x00  # NOP for unmapped ROM
        self.pc = (self.pc + 1) & 0xFFF
        return b

    def is_two_word(self, opcode: int) -> bool:
        opr = (opcode >> 4) & 0xF
        return opr in (0x1, 0x2, 0x4, 0x5, 0x7) and not ((opr == 0x2 and opcode & 1) or (opr == 0x3))

    def execute_one(self) -> tuple:
        """Execute one instruction.

        Returns (pc_before_fetch, byte1, byte2_or_None, is_two_word).
        """
        pc_start = self.pc
        b1 = self.fetch_byte()
        opr = (b1 >> 4) & 0xF
        opa = b1 & 0xF

        b2 = None
        two_word = False

        if opr == 0x0:
            pass  # NOP

        elif opr == 0x1:
            # JCN
            b2 = self.fetch_byte()
            two_word = True
            c1 = (opa >> 3) & 1
            c2 = (opa >> 2) & 1
            c3 = (opa >> 1) & 1
            c4 = opa & 1
            cond = (c2 and self.acc == 0) or (c3 and self.cy == 1) or c4  # simplified
            if c1:
                cond = not cond
            if cond:
                page = pc_start & 0xF00
                self.pc = page | b2

        elif opr == 0x2 and (opa & 1) == 0:
            # FIM
            b2 = self.fetch_byte()
            two_word = True
            pair = (opa >> 1) & 0x7
            self.regs[pair * 2] = (b2 >> 4) & 0xF
            self.regs[pair * 2 + 1] = b2 & 0xF

        elif opr == 0x2 and (opa & 1) == 1:
            pass  # SRC — sends register pair to address bus (no PC change)

        elif opr == 0x3 and (opa & 1) == 0:
            pass  # FIN — indirect fetch (complex, skip for now)

        elif opr == 0x3 and (opa & 1) == 1:
            # JIN
            pair = (opa >> 1) & 0x7
            page = self.pc & 0xF00
            self.pc = page | (self.regs[pair*2] << 4) | self.regs[pair*2+1]

        elif opr == 0x4:
            # JUN
            b2 = self.fetch_byte()
            two_word = True
            self.pc = (opa << 8) | b2

        elif opr == 0x5:
            # JMS
            b2 = self.fetch_byte()
            two_word = True
            self.stack[self.stack_ptr] = self.pc
            self.stack_ptr = (self.stack_ptr + 1) % 3
            self.pc = (opa << 8) | b2

        elif opr == 0x6:
            # INC
            self.regs[opa] = (self.regs[opa] + 1) & 0xF

        elif opr == 0x7:
            # ISZ
            b2 = self.fetch_byte()
            two_word = True
            self.regs[opa] = (self.regs[opa] + 1) & 0xF
            if self.regs[opa] != 0:
                page = pc_start & 0xF00
                self.pc = page | b2

        elif opr == 0x8:
            # ADD
            result = self.acc + self.regs[opa] + self.cy
            self.acc = result & 0xF
            self.cy = 1 if result > 0xF else 0

        elif opr == 0x9:
            # SUB
            result = self.acc + (~self.regs[opa] & 0xF) + self.cy
            self.acc = result & 0xF
            self.cy = 1 if result > 0xF else 0

        elif opr == 0xA:
            # LD
            self.acc = self.regs[opa]

        elif opr == 0xB:
            # XCH
            self.acc, self.regs[opa] = self.regs[opa], self.acc

        elif opr == 0xC:
            # BBL
            self.stack_ptr = (self.stack_ptr - 1) % 3
            self.pc = self.stack[self.stack_ptr]
            self.acc = opa

        elif opr == 0xD:
            # LDM
            self.acc = opa

        elif opr == 0xE:
            # I/O and RAM — handle accumulator effects
            if opa == 0x0:   pass  # WRM
            elif opa == 0x2: pass  # WRR
            elif opa == 0x8:       # SBM
                result = self.acc + 0xF + (1 - self.cy)  # complement + borrow
                self.acc = result & 0xF
                self.cy = 1 if result > 0xF else 0
            elif opa == 0x9: pass  # RDM — would load from RAM
            elif opa == 0xB:       # ADM
                result = self.acc + self.cy  # would add RAM content
                self.acc = result & 0xF
                self.cy = 1 if result > 0xF else 0

        elif opr == 0xF:
            # Accumulator group
            if opa == 0x0:    # CLB
                self.acc = 0; self.cy = 0
            elif opa == 0x1:  # CLC
                self.cy = 0
            elif opa == 0x2:  # IAC
                result = self.acc + 1
                self.acc = result & 0xF
                self.cy = 1 if result > 0xF else 0
            elif opa == 0x3:  # CMC
                self.cy ^= 1
            elif opa == 0x4:  # CMA
                self.acc = (~self.acc) & 0xF
            elif opa == 0x5:  # RAL
                val = (self.acc << 1) | self.cy
                self.acc = val & 0xF
                self.cy = (val >> 4) & 1
            elif opa == 0x6:  # RAR
                val = (self.cy << 4) | self.acc
                self.acc = (val >> 1) & 0xF
                self.cy = val & 1
            elif opa == 0x7:  # TCC
                self.acc = self.cy
                self.cy = 0
            elif opa == 0x8:  # DAC
                result = self.acc + 0xF  # subtract 1 = add 15
                self.acc = result & 0xF
                self.cy = 1 if result > 0xF else 0
            elif opa == 0x9:  # TCS
                self.acc = 10 if self.cy else 9
                self.cy = 0
            elif opa == 0xA:  # STC
                self.cy = 1
            elif opa == 0xB:  # DAA
                if self.acc > 9 or self.cy:
                    result = self.acc + 6
                    self.acc = result & 0xF
                    if result > 0xF:
                        self.cy = 1
            elif opa == 0xC:  # KBP
                kbp = {0:0, 1:1, 2:2, 4:3, 8:4}
                self.acc = kbp.get(self.acc, 0xF)
            elif opa == 0xD:  # DCL
                pass

        return pc_start, b1, b2, two_word


# ── PWL Generation ──────────────────────────────────────────────────────

def get_phase_count(opcode: int) -> int:
    """Get the number of CLK cycles for an instruction."""
    opa = opcode & 0xF

    # Check exact match first (for accumulator/IO group)
    if opcode in INST_PHASES:
        return INST_PHASES[opcode]

    # Check by OPR (high nibble) for register-addressed instructions
    base = opcode & 0xF0
    if base in INST_PHASES:
        return INST_PHASES[base]

    # Special: SRC = 0x2R1 (odd OPA) = 9 cycles, FIM = 0x2R0 (even OPA) = 16 cycles
    opr = (opcode >> 4) & 0xF
    if opr == 0x2:
        return 9 if (opa & 1) else 16  # SRC=9, FIM=16

    # Special: JIN = 0x3R1 (odd OPA) = 16 cycles, FIN = 0x3R0 (even OPA) = 16 cycles
    if opr == 0x3:
        return 16

    return DEFAULT_PHASES


def bit_voltage(byte_val: int, bit: int) -> float:
    """Get PWL voltage for a specific bit of a byte value."""
    return V_HIGH if (byte_val >> bit) & 1 else V_LOW


def generate_pwl(rom: list, num_cycles: int, startup_delay: float = 2e-6,
                 start_address: int = 0) -> dict:
    """Generate PWL waveforms for D0In-D3In by simulating program execution.

    PWL files use proper step transitions: each value change is a pair of
    points — hold at old value, then near-instant transition to new value.
    This avoids LTSpice's linear interpolation creating ramp/sawtooth artifacts.

    Args:
        rom: assembled ROM image (list of bytes)
        num_cycles: number of instruction cycles to simulate
        startup_delay: seconds before first instruction cycle starts.
            The CPU's step counter reaches Micro1 at ~13us after power-on.
            Default 2us means first ROM data appears at ~50us, matching
            the real CPU timing observed in Load5.raw.

    Returns:
        dict with keys 'd0in', 'd1in', 'd2in', 'd3in',
        each mapping to a list of (time, voltage) tuples.
    """
    cpu = CPU4004Sim(rom)

    # Track current voltage per line so we can emit proper step transitions
    current_v = {f'd{i}in': V_LOW for i in range(4)}
    pwl = {f'd{i}in': [(0, V_LOW)] for i in range(4)}

    def set_bus(t, nibble):
        """Set all 4 data lines to a nibble value at time t with step transitions."""
        for bit in range(4):
            name = f'd{bit}in'
            v = bit_voltage(nibble, bit)
            if v != current_v[name]:
                # Hold old value right before transition
                pwl[name].append((t - RISE_TIME, current_v[name]))
                # Step to new value
                pwl[name].append((t, v))
                current_v[name] = v

    t = startup_delay

    trace = []

    for cycle in range(num_cycles):
        pc, b1, b2, two_word = cpu.execute_one()
        trace.append((cycle, pc, b1, b2))

        n_phases = get_phase_count(b1)

        opr = (b1 >> 4) & 0xF
        opa = b1 & 0xF

        # Byte 1: OPR during Micro4, OPA during Micro5
        t_opr = t + (FETCH_PHASE_OPR * CLK_PERIOD) - ROMCLK_LEAD
        t_opa = t + (FETCH_PHASE_OPA * CLK_PERIOD) - ROMCLK_LEAD
        t_clear1 = t + ((FETCH_PHASE_OPA + 1) * CLK_PERIOD) - ROMCLK_LEAD

        set_bus(t_opr, opr)
        set_bus(t_opa, opa)
        set_bus(t_clear1, 0)  # Clear bus (all low)

        # Byte 2 for two-word instructions
        if two_word and b2 is not None:
            opr2 = (b2 >> 4) & 0xF
            opa2 = b2 & 0xF

            t_opr2 = t + (FETCH_PHASE_OPR2 * CLK_PERIOD) - ROMCLK_LEAD
            t_opa2 = t + (FETCH_PHASE_OPA2 * CLK_PERIOD) - ROMCLK_LEAD
            t_clear2 = t + ((FETCH_PHASE_OPA2 + 1) * CLK_PERIOD) - ROMCLK_LEAD

            set_bus(t_opr2, opr2)
            set_bus(t_opa2, opa2)
            set_bus(t_clear2, 0)

        t += n_phases * CLK_PERIOD

    # Add a final hold point at the end
    for bit in range(4):
        name = f'd{bit}in'
        pwl[name].append((t, current_v[name]))

    return pwl, trace


def write_pwl_files(pwl: dict, output_dir: str):
    """Write PWL waveforms to files, ensuring strictly monotonic timestamps.

    Removes redundant points (same voltage as previous) and nudges any
    remaining duplicate timestamps so LTSpice sees monotonic data.
    """
    os.makedirs(output_dir, exist_ok=True)
    MIN_DT = 10e-9  # 10ns minimum gap between points

    for name, points in pwl.items():
        # First pass: remove redundant consecutive same-voltage points
        # (keep first and last of a run, drop middle)
        cleaned = [points[0]]
        for i in range(1, len(points)):
            t, v = points[i]
            if v != cleaned[-1][1] or i == len(points) - 1:
                cleaned.append((t, v))
            elif i + 1 < len(points) and points[i + 1][1] != v:
                # Keep this point — it's the last before a transition
                cleaned.append((t, v))

        # Second pass: enforce strictly increasing timestamps
        final = [cleaned[0]]
        for i in range(1, len(cleaned)):
            t, v = cleaned[i]
            if t <= final[-1][0]:
                t = final[-1][0] + MIN_DT
            final.append((t, v))

        path = os.path.join(output_dir, f"{name}.pwl")
        with open(path, "w") as f:
            for time, voltage in final:
                f.write(f"{time:.9e} {voltage:.4f}\n")
        print(f"  {name}.pwl: {len(final)} points, {final[-1][0]*1e6:.1f}us")


def setup_program(cpu_name: str, program_name: str, asm_text: str,
                  num_cycles: int = 20, startup_delay: float = 2e-6,
                  verbose: bool = False):
    """Full workflow: create program folder, assemble, generate PWLs, build CPU .asc.

    Creates:
        cpus/<cpu>/programs/<program>/<program>.asm  (copy of source)
        cpus/<cpu>/programs/<program>/d0in.pwl ... d3in.pwl
        cpus/<cpu>/d0in.pwl ... d3in.pwl  (symlinks/copies for LTSpice)
        cpus/<cpu>/<cpu>.asc  (combined CPU schematic)

    After LTSpice simulation, copy the .raw file:
        cpus/<cpu>/<cpu>.raw  ->  cpus/<cpu>/programs/<program>/<program>.raw
    """
    cpu_dir = os.path.join(PROJECT_ROOT, "cpus", cpu_name)
    prog_dir = os.path.join(cpu_dir, "programs", program_name)
    os.makedirs(prog_dir, exist_ok=True)

    # Save .asm to program folder
    asm_path = os.path.join(prog_dir, f"{program_name}.asm")
    with open(asm_path, "w") as f:
        f.write(asm_text)
    print(f"Program: {asm_path}")

    # Assemble
    rom = assemble(asm_text)
    print(f"Assembled {len(rom)} bytes: {' '.join(f'{b:02X}' for b in rom)}")

    # Generate PWL
    pwl, trace = generate_pwl(rom, num_cycles, startup_delay)

    if verbose:
        print(f"\nExecution trace ({len(trace)} cycles):")
        for cycle, pc, b1, b2 in trace:
            if b2 is not None:
                print(f"  [{cycle:3d}] PC={pc:03X}: {b1:02X} {b2:02X}")
            else:
                print(f"  [{cycle:3d}] PC={pc:03X}: {b1:02X}")

    # Write PWL to program folder
    print(f"\nWriting PWL files to {prog_dir}/")
    write_pwl_files(pwl, prog_dir)

    # Copy PWL files to CPU root (where the .asc expects them)
    for name in ['d0in', 'd1in', 'd2in', 'd3in']:
        src = os.path.join(prog_dir, f"{name}.pwl")
        dst = os.path.join(cpu_dir, f"{name}.pwl")
        import shutil
        shutil.copy2(src, dst)
    print(f"Copied PWL files to {cpu_dir}/")

    # Build the CPU .asc
    print()
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
    from build_cpu import build
    build(cpu_name, output_name=f"{cpu_name}.asc")

    # Patch .tran in the built .asc to match program length
    total_time = pwl['d0in'][-1][0]
    tran_us = int(total_time * 1e6) + 100  # add 100us margin
    asc_path = os.path.join(cpu_dir, f"{cpu_name}.asc")
    with open(asc_path, "r") as f:
        content = f.read()
    import re
    content = re.sub(
        r'(\.tran\s+\S+\s+)\S+(us\s+)',
        rf'\g<1>{tran_us}\g<2>',
        content
    )
    with open(asc_path, "w") as f:
        f.write(content)
    print(f"Patched .tran to {tran_us}us")

    print(f"\nReady to simulate:")
    print(f"  Open: cpus/{cpu_name}/{cpu_name}.asc")
    print(f"  .tran covers: {total_time*1e6:.0f}us")
    print(f"\nAfter simulation, copy the .raw:")
    print(f"  cpus/{cpu_name}/{cpu_name}.raw -> cpus/{cpu_name}/programs/{program_name}/{program_name}.raw")

    return prog_dir, trace


def main():
    parser = argparse.ArgumentParser(description="4004 ROM emulator — generate PWL files")
    parser.add_argument("program", help="Program name or path to .asm file")
    parser.add_argument("--cpu", "-c", default="4004",
                        help="CPU project name (default: 4004)")
    parser.add_argument("--cycles", "-n", type=int, default=20,
                        help="Number of instruction cycles to simulate (default: 20)")
    parser.add_argument("--startup-delay", type=float, default=2e-6,
                        help="Seconds before first instruction cycle (default: 2e-6)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print execution trace")

    args = parser.parse_args()

    # Determine program name and load asm text
    if os.path.isfile(args.program):
        # Given a file path
        asm_path = args.program
        program_name = os.path.splitext(os.path.basename(asm_path))[0]
        with open(asm_path, "r") as f:
            asm_text = f.read()
    else:
        # Given a program name — look in programs folder
        cpu_dir = os.path.join(PROJECT_ROOT, "cpus", args.cpu)
        asm_path = os.path.join(cpu_dir, "programs", args.program, f"{args.program}.asm")
        if not os.path.isfile(asm_path):
            # Try as a bare name in programs/
            for root, dirs, files in os.walk(os.path.join(cpu_dir, "programs")):
                for f in files:
                    if f.lower() == f"{args.program.lower()}.asm":
                        asm_path = os.path.join(root, f)
                        break
        if not os.path.isfile(asm_path):
            print(f"ERROR: Cannot find program '{args.program}'")
            sys.exit(1)
        program_name = args.program
        with open(asm_path, "r") as f:
            asm_text = f.read()

    setup_program(args.cpu, program_name, asm_text,
                  args.cycles, args.startup_delay, args.verbose)


if __name__ == "__main__":
    main()
