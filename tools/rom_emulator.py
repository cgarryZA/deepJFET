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
sys.path.insert(0, _root)

# Logic voltage levels (from controls.asc)
V_HIGH = -0.8   # Logic 1
V_LOW = -3.6    # Logic 0

# Timing
CLK_PERIOD = 10e-6       # 10 microseconds
# Per-instruction phase counts (from micro_instructions.asc analysis)
# Total CLK periods per instruction cycle, including M1-M5 fetch phases.
INST_PHASES = {
    # 6 phases (60us)
    0xFC: 6,   # KBP

    # 7 phases (70us)
    0xD0: 7,   # LDM (0xD0-0xDF all LDM)
    0xF0: 7,   # CLB
    0xF1: 7,   # CLC
    0xF3: 7,   # CMC
    0xF4: 7,   # CMA
    0xF9: 7,   # TCS
    0xFA: 7,   # STC
    0xFD: 7,   # DCL

    # 8 phases (80us)
    0x00: 8,   # NOP
    0xA0: 8,   # LD (0xA0-0xAF)
    0xC0: 8,   # BBL (0xC0-0xCF)
    0xF2: 8,   # IAC
    0xF5: 8,   # RAL
    0xF6: 8,   # RAR
    0xF7: 8,   # TCC
    0xF8: 8,   # DAC
    0xFB: 8,   # DAA

    # 9 phases (90us)
    0x60: 9,   # INC (0x60-0x6F)
    0x90: 9,   # SUB (0x90-0x9F)
    0xE8: 9,   # SBM
    0xEB: 9,   # ADM
    # SRC, FIN, JIN also 9 but they're special (0x2R1, 0x3R0, 0x3R1)

    # 10 phases (100us)
    0xB0: 10,  # XCH (0xB0-0xBF)

    # 11 phases (110us)
    0x50: 11,  # JMS (0x50-0x5F) — 2-word

    # 12 phases (120us)
    0x80: 12,  # ADD (0x80-0x8F)
    0x70: 12,  # ISZ (0x70-0x7F) — 2-word
}

# 2-word instructions (need second byte fetch)
# JCN=0x1x, FIM=0x2x0, JUN=0x4x, JMS=0x5x, ISZ=0x7x
# Their phase count already includes the second fetch cycle.

# Default for I/O instructions not listed: 9 phases
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
FETCH_PHASE_OPR = 5       # Data arrives during Micro5->Micro6 boundary
FETCH_PHASE_OPA = 6       # Data arrives during Micro6->Micro7 boundary
FETCH_PHASE_OPR2 = 13     # Second byte high nibble
FETCH_PHASE_OPA2 = 14     # Second byte low nibble


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
    """Get the number of CLK phases for an instruction."""
    opr = (opcode >> 4) & 0xF

    # Check exact match first (for accumulator group instructions)
    if opcode in INST_PHASES:
        return INST_PHASES[opcode]

    # Check by OPR (high nibble) for register-addressed instructions
    base = opcode & 0xF0
    if base in INST_PHASES:
        return INST_PHASES[base]

    # Special cases by OPR
    if opr == 0x1:  return 8   # JCN — 8 phases per word, but 2 words
    if opr == 0x2:  return 9   # FIM/SRC
    if opr == 0x3:  return 9   # FIN/JIN
    if opr == 0x4:  return 9   # JUN — 2-word but 9 phases total

    # I/O instructions (0xE0-0xEF) not already matched
    if opr == 0xE:  return 9

    return DEFAULT_PHASES


def bit_voltage(byte_val: int, bit: int) -> float:
    """Get PWL voltage for a specific bit of a byte value."""
    return V_HIGH if (byte_val >> bit) & 1 else V_LOW


def generate_pwl(rom: list, num_cycles: int, startup_delay: float = 2e-6) -> dict:
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
    """Write PWL waveforms to files."""
    os.makedirs(output_dir, exist_ok=True)
    for name, points in pwl.items():
        path = os.path.join(output_dir, f"{name}.pwl")
        with open(path, "w") as f:
            for time, voltage in points:
                f.write(f"{time:.9e} {voltage:.4f}\n")
        print(f"  {name}.pwl: {len(points)} points, {points[-1][0]*1e6:.1f}us")


def main():
    parser = argparse.ArgumentParser(description="4004 ROM emulator — generate PWL files")
    parser.add_argument("asm_file", help="Path to .asm file")
    parser.add_argument("--cycles", "-n", type=int, default=20,
                        help="Number of instruction cycles to simulate (default: 20)")
    parser.add_argument("--startup-delay", type=float, default=2e-6,
                        help="Seconds before first instruction cycle (default: 2e-6)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output directory (default: same as .asm file)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print execution trace")

    args = parser.parse_args()

    # Read and assemble
    with open(args.asm_file, "r") as f:
        asm_text = f.read()

    rom = assemble(asm_text)
    print(f"Assembled {len(rom)} bytes from {args.asm_file}")
    print(f"ROM: {' '.join(f'{b:02X}' for b in rom)}")

    # Generate PWL
    pwl, trace = generate_pwl(rom, args.cycles, args.startup_delay)

    if args.verbose:
        print(f"\nExecution trace ({len(trace)} cycles):")
        for cycle, pc, b1, b2 in trace:
            if b2 is not None:
                print(f"  [{cycle:3d}] PC={pc:03X}: {b1:02X} {b2:02X}")
            else:
                print(f"  [{cycle:3d}] PC={pc:03X}: {b1:02X}")

    # Write PWL files
    output_dir = args.output or os.path.dirname(args.asm_file)
    print(f"\nWriting PWL files to {output_dir}/")
    write_pwl_files(pwl, output_dir)

    total_time = pwl['d0in'][-1][0]
    print(f"\nTotal simulation time: {total_time*1e6:.1f}us")
    print(f"Suggested .tran: {total_time:.6e}")


if __name__ == "__main__":
    main()
