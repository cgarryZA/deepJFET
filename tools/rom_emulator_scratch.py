#!/usr/bin/env python3
"""ROM emulator for the Scratch Build CPU.

This CPU has a simpler architecture than the 4004:

  - Every instruction is 16 bits (2 bytes). PC always advances by 2
    (or to the jump target).
  - The ROM is direct-wired to the IR. At every Micro1, the 16-bit
    word at PC is presented on the 16 IR_In nets (IR1_3..IR4_0,
    MSB-first ordering: IR1_3 = bit 15 of word, IR4_0 = bit 0).
  - The PC is direct-wired to the ROM address input (the emulator
    just walks the simulated CPU's PC and looks up bytes at that
    address — no bus protocol).
  - WR / RD are memory ops with the 8-bit memory address embedded
    in byte 2 of the instruction (= IR3_3..IR4_0 = bits 7..0 of word).
    On RD, the emulator drives Bus3In..Bus0In with the stored 4-bit
    value at Micro3.

Outputs:
  cpus/Scratch Build/ir{1..4}_{3..0}_in.pwl   (16 IR data lines)
  cpus/Scratch Build/bus{3..0}in.pwl           (4 RD-return lines)
  cpus/Scratch Build/programs/<name>/         (same PWLs, archived)

Fixed at 100 kHz (CLK_PERIOD = 10 us). Frequency sweeping comes later
once the basic architecture is confirmed working.

Usage:
    python tools/rom_emulator_scratch.py <program.asm> --cycles 200
    python tools/rom_emulator_scratch.py simple_test  # by program name
"""

import argparse
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_ROOT = PROJECT_ROOT / 'cpus' / 'Scratch Build'


# ── Logic levels and timing ──────────────────────────────────────────────

V_HIGH = -0.8   # logic 1
V_LOW  = -3.6   # logic 0

CLK_PERIOD    = 10e-6   # one Micro phase
STARTUP_DELAY = 2e-6    # quiet period before first Micro1
RISE_TIME     = 100e-9  # PWL step-transition width


# ── Instruction cycle counts (Micro phases per instruction) ──────────────

INST_CYCLES = {
    'NOP': 2,
    'WRM': 4, 'WR':  4,   # alias
    'RDM': 4, 'RD':  4,
    'CLB': 3, 'CLC': 3,
    'IAC': 4, 'CMC': 3, 'CMA': 4,
    'RAL': 4, 'RAR': 4,
    'TCC': 4, 'DAC': 4,
    'TCS': 3, 'STC': 3,
    'JCN': 3, 'FIM': 5, 'FIN': 5, 'JIN': 4,
    'JUN': 3, 'JMS': 5,
    'INC': 5, 'ISZ': 6,
    'ADD': 6, 'SUB': 6,
    'LD':  4, 'XCH': 6,
    'BBL': 4, 'LDM': 3,
}


# ── Assembler ────────────────────────────────────────────────────────────
#
# Every instruction is a 16-bit word (2 bytes). Byte 1 carries the
# opcode (with possible embedded operand bits in the low nibble); byte 2
# carries an 8-bit operand or 0x00 if unused.
#
# Encoding table (from user spec):
#
#   NOP                   00 00
#   WRM addr     ("WR")   01 AA   (8-bit address)
#   RDM addr     ("RD")   02 AA
#   CLB                   03 00
#   CLC                   04 00
#   IAC                   05 00
#   CMC                   06 00
#   CMA                   07 00
#   RAL                   08 00
#   RAR                   09 00
#   TCC                   0A 00
#   DAC                   0B 00
#   TCS                   0C 00
#   STC                   0D 00
#   JCN cond,addr         1C AA   (low nibble of byte1 = condition)
#   FIM pair,data         2P 0D   (P in bits 3:1 of byte1, LSB=0)
#   FIN pair              3P 00   (P in bits 2:0, bit 3 = 0)
#   JIN pair              3P 00   (P in bits 2:0, bit 3 = 1)
#   JUN addr12            4A AA   (12-bit address: A_high in byte1 low nibble)
#   JMS addr12            5A AA
#   INC reg               6R 00
#   ISZ reg,addr          7R AA
#   ADD reg               8R 00
#   SUB reg               9R 00
#   LD reg                AR 00
#   XCH reg               BR 00
#   BBL data              CD 00
#   LDM data              DD 00

NO_OPERAND = {
    'NOP': 0x00, 'CLB': 0x03, 'CLC': 0x04, 'IAC': 0x05, 'CMC': 0x06,
    'CMA': 0x07, 'RAL': 0x08, 'RAR': 0x09, 'TCC': 0x0A, 'DAC': 0x0B,
    'TCS': 0x0C, 'STC': 0x0D,
}
MEM_OPS = {'WRM': 0x01, 'WR': 0x01, 'RDM': 0x02, 'RD': 0x02}

REG_OPS  = {'INC': 0x60, 'ADD': 0x80, 'SUB': 0x90, 'LD': 0xA0, 'XCH': 0xB0}
DATA_OPS = {'BBL': 0xC0, 'LDM': 0xD0}
PAIR_OPS = {'FIN': 0x30, 'JIN': 0x38}
JMP12    = {'JUN': 0x40, 'JMS': 0x50}

INSTRUCTIONS = (set(NO_OPERAND) | set(MEM_OPS) | set(REG_OPS) | set(DATA_OPS)
                | set(PAIR_OPS) | set(JMP12) | {'JCN', 'FIM', 'ISZ'})


def assemble(asm_text):
    """Two-pass assemble of new-ISA assembly into a list of bytes.

    Each instruction emits 2 bytes (16 bits). The first word at byte 0..1,
    second word at byte 2..3, etc. Labels resolve to byte addresses (so
    a label pointing at instruction N lives at byte 2N).
    """
    lines = [l.strip() for l in asm_text.split('\n') if l.strip()
             and not l.strip().startswith(';')]

    # Strip inline comments (anything after ';')
    lines = [re.sub(r';.*$', '', l).strip() for l in lines]
    lines = [l for l in lines if l]

    # PASS 1: collect labels. Anything in the first token that isn't an
    # instruction mnemonic is treated as a label for the following inst.
    labels = {}
    addr = 0
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].upper() not in INSTRUCTIONS:
            labels[parts[0]] = addr
            parts = parts[1:]
            if not parts:
                continue
        addr += 2   # every instruction = 2 bytes

    # PASS 2: emit bytes
    rom = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0].upper() not in INSTRUCTIONS:
            parts = parts[1:]
            if not parts:
                continue
        mnem = parts[0].upper()
        args = parts[1:]

        def parse_num(s):
            """Accept decimal, hex (0x..), or a known label."""
            if s in labels:
                return labels[s]
            return int(s, 0)

        if mnem in NO_OPERAND:
            rom += [NO_OPERAND[mnem], 0x00]
        elif mnem in MEM_OPS:
            b2 = parse_num(args[0]) & 0xFF if args else 0
            rom += [MEM_OPS[mnem], b2]
        elif mnem in REG_OPS:
            r = parse_num(args[0]) & 0xF
            rom += [REG_OPS[mnem] | r, 0x00]
        elif mnem in DATA_OPS:
            d = parse_num(args[0]) & 0xF
            rom += [DATA_OPS[mnem] | d, 0x00]
        elif mnem in PAIR_OPS:
            p = parse_num(args[0]) & 0x7
            rom += [PAIR_OPS[mnem] | p, 0x00]
        elif mnem in JMP12:
            tgt = parse_num(args[0]) & 0xFFF
            rom += [JMP12[mnem] | ((tgt >> 8) & 0xF), tgt & 0xFF]
        elif mnem == 'JCN':
            cond = parse_num(args[0]) & 0xF
            tgt = parse_num(args[1]) & 0xFF
            rom += [0x10 | cond, tgt]
        elif mnem == 'FIM':
            pair = parse_num(args[0]) & 0x7
            data = parse_num(args[1]) & 0xFF
            rom += [0x20 | (pair << 1), data]
        elif mnem == 'ISZ':
            reg = parse_num(args[0]) & 0xF
            tgt = parse_num(args[1]) & 0xFF
            rom += [0x70 | reg, tgt]
        else:
            raise ValueError(f'Unknown mnemonic: {mnem}')
    return rom


def mnemonic_of(byte1):
    """Reverse-lookup the mnemonic from byte 1 (for cycle-count tables and
    for debug printing)."""
    if byte1 == 0x00: return 'NOP'
    if byte1 == 0x01: return 'WRM'
    if byte1 == 0x02: return 'RDM'
    if byte1 == 0x03: return 'CLB'
    if byte1 == 0x04: return 'CLC'
    if byte1 == 0x05: return 'IAC'
    if byte1 == 0x06: return 'CMC'
    if byte1 == 0x07: return 'CMA'
    if byte1 == 0x08: return 'RAL'
    if byte1 == 0x09: return 'RAR'
    if byte1 == 0x0A: return 'TCC'
    if byte1 == 0x0B: return 'DAC'
    if byte1 == 0x0C: return 'TCS'
    if byte1 == 0x0D: return 'STC'
    nib = byte1 & 0xF0
    if nib == 0x10: return 'JCN'
    if nib == 0x20: return 'FIM'   # technically also SRC if low bit set
    if nib == 0x30:
        return 'JIN' if (byte1 & 0x08) else 'FIN'
    if nib == 0x40: return 'JUN'
    if nib == 0x50: return 'JMS'
    if nib == 0x60: return 'INC'
    if nib == 0x70: return 'ISZ'
    if nib == 0x80: return 'ADD'
    if nib == 0x90: return 'SUB'
    if nib == 0xA0: return 'LD'
    if nib == 0xB0: return 'XCH'
    if nib == 0xC0: return 'BBL'
    if nib == 0xD0: return 'LDM'
    return 'UNK'


# ── CPU simulator ────────────────────────────────────────────────────────

class ScratchBuildCPU:
    """Minimal simulator of the Scratch Build CPU.

    Tracks PC (12-bit byte-pointer), ACC (4-bit), CY, the 16 index
    registers (each 4-bit), a 3-level subroutine stack, and 256
    addresses of 4-bit memory. Power-up state is all 0xF for registers
    and ACC (to match LTspice DFF default), 0 for memory.
    """

    def __init__(self, rom: list):
        self.rom = list(rom)
        self.pc = 0
        self.acc = 0xF
        self.cy = 0
        self.regs = [0xF] * 16
        self.stack = [0, 0, 0]
        self.sp = 0
        self.memory = [0] * 256

    def fetch_word(self):
        """Read the 2-byte word at current PC. Returns (byte1, byte2)."""
        b1 = self.rom[self.pc] if self.pc < len(self.rom) else 0
        b2 = self.rom[self.pc + 1] if (self.pc + 1) < len(self.rom) else 0
        return b1, b2

    def execute_one(self):
        """Execute one instruction. Returns:
            (pc_start, byte1, byte2, n_cycles, mnemonic, rd_value)
        rd_value is None except for RD: the 4-bit memory value the
        emulator must drive on Bus3In..Bus0In at Micro3 of this cycle.
        """
        pc_start = self.pc
        b1, b2 = self.fetch_word()
        mnem = mnemonic_of(b1)
        n_cyc = INST_CYCLES.get(mnem, 4)

        next_pc = (self.pc + 2) & 0xFFF   # default: advance one word
        rd_value = None

        nib = b1 & 0xF0
        opa = b1 & 0x0F

        if mnem == 'NOP':
            pass

        elif mnem == 'WRM':
            # Write ACC to memory[byte2]
            self.memory[b2 & 0xFF] = self.acc & 0xF

        elif mnem == 'RDM':
            v = self.memory[b2 & 0xFF] & 0xF
            self.acc = v
            rd_value = v

        elif mnem == 'CLB':
            self.acc = 0; self.cy = 0
        elif mnem == 'CLC':
            self.cy = 0
        elif mnem == 'IAC':
            s = self.acc + 1
            self.acc = s & 0xF
            self.cy = 1 if s > 0xF else 0
        elif mnem == 'CMC':
            self.cy ^= 1
        elif mnem == 'CMA':
            self.acc = (~self.acc) & 0xF
        elif mnem == 'RAL':
            v = (self.acc << 1) | self.cy
            self.acc = v & 0xF
            self.cy = (v >> 4) & 1
        elif mnem == 'RAR':
            v = (self.cy << 4) | self.acc
            self.acc = (v >> 1) & 0xF
            self.cy = v & 1
        elif mnem == 'TCC':
            self.acc = self.cy
            self.cy = 0
        elif mnem == 'DAC':
            s = self.acc + 0xF
            self.acc = s & 0xF
            self.cy = 1 if s > 0xF else 0
        elif mnem == 'TCS':
            self.acc = 10 if self.cy else 9
            self.cy = 0
        elif mnem == 'STC':
            self.cy = 1

        elif mnem == 'JCN':
            cond_bits = opa
            c1 = (cond_bits >> 3) & 1
            c2 = (cond_bits >> 2) & 1
            c3 = (cond_bits >> 1) & 1
            c4 = cond_bits & 1
            taken = (c2 and self.acc == 0) or (c3 and self.cy == 1) or c4
            if c1:
                taken = not taken
            if taken:
                page = pc_start & 0xF00
                next_pc = page | b2

        elif mnem == 'FIM':
            pair = (opa >> 1) & 0x7
            self.regs[pair * 2] = (b2 >> 4) & 0xF
            self.regs[pair * 2 + 1] = b2 & 0xF

        elif mnem == 'FIN':
            # Fetch indirect: read ROM at (PC_page, regs[0..1]) into a pair
            pair = opa & 0x7
            addr = (pc_start & 0xF00) | (self.regs[0] << 4) | self.regs[1]
            high = self.rom[addr] if addr < len(self.rom) else 0
            self.regs[pair * 2] = (high >> 4) & 0xF
            self.regs[pair * 2 + 1] = high & 0xF

        elif mnem == 'JIN':
            pair = opa & 0x7
            next_pc = (pc_start & 0xF00) | (self.regs[pair*2] << 4) | self.regs[pair*2+1]

        elif mnem == 'JUN':
            next_pc = ((opa & 0xF) << 8) | b2

        elif mnem == 'JMS':
            self.stack[self.sp] = next_pc
            self.sp = (self.sp + 1) % 3
            next_pc = ((opa & 0xF) << 8) | b2

        elif mnem == 'INC':
            self.regs[opa] = (self.regs[opa] + 1) & 0xF

        elif mnem == 'ISZ':
            self.regs[opa] = (self.regs[opa] + 1) & 0xF
            if self.regs[opa] != 0:
                next_pc = (pc_start & 0xF00) | b2

        elif mnem == 'ADD':
            s = self.acc + self.regs[opa] + self.cy
            self.acc = s & 0xF
            self.cy = 1 if s > 0xF else 0

        elif mnem == 'SUB':
            s = self.acc + ((~self.regs[opa]) & 0xF) + self.cy
            self.acc = s & 0xF
            self.cy = 1 if s > 0xF else 0

        elif mnem == 'LD':
            self.acc = self.regs[opa]
        elif mnem == 'XCH':
            self.acc, self.regs[opa] = self.regs[opa], self.acc
        elif mnem == 'BBL':
            self.sp = (self.sp - 1) % 3
            next_pc = self.stack[self.sp]
            self.acc = opa
        elif mnem == 'LDM':
            self.acc = opa

        self.pc = next_pc & 0xFFF
        return pc_start, b1, b2, n_cyc, mnem, rd_value


# ── PWL output ───────────────────────────────────────────────────────────

# Wire labels (matches controls.asc FLAG names exactly).
# Order: bit position within the 16-bit fetched word, MSB first.
# IR1_3 = bit 15 (MSB), IR4_0 = bit 0 (LSB).
IR_BIT_NAMES = []
for nibble in range(1, 5):
    for bit in range(3, -1, -1):
        IR_BIT_NAMES.append(f'ir{nibble}_{bit}_in')

BUS_BIT_NAMES = [f'bus{b}in' for b in range(3, -1, -1)]   # bus3in..bus0in

PWL_NAMES = IR_BIT_NAMES + BUS_BIT_NAMES


def bits_of_word(byte1: int, byte2: int) -> dict:
    """Map a 2-byte instruction word to a dict {pwl_name -> 0|1}."""
    out = {}
    nibbles = [(byte1 >> 4) & 0xF, byte1 & 0xF,
               (byte2 >> 4) & 0xF, byte2 & 0xF]
    for n_idx, nibble in enumerate(nibbles):
        for bit in range(3, -1, -1):
            name = f'ir{n_idx + 1}_{bit}_in'
            out[name] = (nibble >> bit) & 1
    return out


def bus_bits_of(value4: int) -> dict:
    """Map a 4-bit value to the 4 Bus_In PWL channels."""
    out = {}
    for bit in range(3, -1, -1):
        out[f'bus{bit}in'] = (value4 >> bit) & 1
    return out


def generate_pwls(rom: list, max_cycles: int = 100, verbose: bool = False):
    """Walk the simulated program for up to max_cycles instructions.
    Returns a dict {pwl_name -> list_of_(t,V)_tuples}."""
    cpu = ScratchBuildCPU(rom)

    # PWL streams start at logic LOW
    current_v = {name: V_LOW for name in PWL_NAMES}
    pwl = {name: [(0.0, V_LOW)] for name in PWL_NAMES}

    def set_lines(t, name_to_bit: dict):
        """Schedule transitions: for each entry in name_to_bit, if it
        differs from the current voltage on that line, emit a hold point
        right before t and a step at t."""
        for name, bit in name_to_bit.items():
            new_v = V_HIGH if bit else V_LOW
            if new_v != current_v[name]:
                pwl[name].append((max(0.0, t - RISE_TIME), current_v[name]))
                pwl[name].append((t, new_v))
                current_v[name] = new_v

    t = STARTUP_DELAY
    trace = []

    for cycle in range(max_cycles):
        pc, b1, b2, n_cyc, mnem, rd_val = cpu.execute_one()
        if verbose:
            print(f'  [{cycle:3d}] PC={pc:03X}: {b1:02X} {b2:02X}  '
                  f'{mnem:<4}  next_PC={cpu.pc:03X}  cycles={n_cyc}'
                  + (f'  rd={rd_val:X}' if rd_val is not None else ''))
        trace.append((cycle, pc, b1, b2, mnem, n_cyc))

        # At Micro1 of this instruction (= time t), drive 16 IR_In bits
        set_lines(t, bits_of_word(b1, b2))

        # At Micro3 of RD instructions, drive 4 Bus_In bits with the
        # stored memory value
        if mnem == 'RDM' and rd_val is not None:
            t_micro3 = t + 2 * CLK_PERIOD
            set_lines(t_micro3, bus_bits_of(rd_val))

        t += n_cyc * CLK_PERIOD

    # Final hold point at t (end of last instruction's cycles)
    for name in PWL_NAMES:
        pwl[name].append((t, current_v[name]))

    return pwl, trace, t


def write_pwl_files(pwl: dict, out_dir: Path):
    """Write each PWL stream to disk with monotonic, deduplicated points."""
    out_dir.mkdir(parents=True, exist_ok=True)
    MIN_DT = 10e-9
    for name, points in pwl.items():
        # Dedupe consecutive same-V points (keep the boundary points)
        cleaned = [points[0]]
        for i in range(1, len(points)):
            t, v = points[i]
            if v != cleaned[-1][1] or i == len(points) - 1:
                cleaned.append((t, v))
            elif i + 1 < len(points) and points[i + 1][1] != v:
                cleaned.append((t, v))
        # Enforce strict monotonic time
        final = [cleaned[0]]
        for i in range(1, len(cleaned)):
            t, v = cleaned[i]
            if t <= final[-1][0]:
                t = final[-1][0] + MIN_DT
            final.append((t, v))
        path = out_dir / f'{name}.pwl'
        with open(path, 'w') as fh:
            for t, v in final:
                fh.write(f'{t:.9e} {v:.4f}\n')


# ── .tran patcher ────────────────────────────────────────────────────────

def patch_tran(asc_path: Path, end_time_s: float, margin_us: float = 100.0):
    """Rewrite the .tran directive in CPU.asc so it covers the program."""
    tran_us = int(end_time_s * 1e6) + int(margin_us)
    text = asc_path.read_text()
    # Match ".tran 0 <stop> 1ps 0.01" — replace just the stop time
    # (group 2 captures the unit suffix and the rest of the args so we
    # don't double up the "us").
    new_text, n = re.subn(
        r'(\.tran\s+\S+\s+)\S+(us\s+\S+\s+\S+)',
        rf'\g<1>{tran_us}\g<2>',
        text)
    if n == 0:
        # No existing .tran directive — append one (LTspice TEXT block)
        new_text = text.rstrip() + (
            f'\nTEXT 0 -2000 Left 2 !.tran 0 {tran_us}us 1ps 0.01\n')
    asc_path.write_text(new_text)
    return tran_us


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('program', help='Path to .asm file or program name')
    p.add_argument('--cycles', type=int, default=100,
                   help='Number of instructions to simulate (default 100)')
    p.add_argument('--verbose', '-v', action='store_true',
                   help='Print per-instruction execution trace')
    p.add_argument('--no-patch-tran', action='store_true',
                   help="Don't rewrite the .tran directive in CPU.asc")
    args = p.parse_args()

    # Resolve program path
    if os.path.isfile(args.program):
        asm_path = Path(args.program)
        program_name = asm_path.stem
    else:
        program_name = args.program
        asm_path = CPU_ROOT / 'programs' / program_name / f'{program_name}.asm'
        if not asm_path.exists():
            print(f'ERROR: cannot find {asm_path}', file=sys.stderr)
            sys.exit(1)

    asm_text = asm_path.read_text()
    rom = assemble(asm_text)
    print(f'Assembled {asm_path.name}: {len(rom)} bytes = '
          f'{len(rom)//2} instructions')
    print('  ROM dump (first 32 bytes): ' +
          ' '.join(f'{b:02X}' for b in rom[:32]))

    # Generate PWLs
    pwl, trace, t_end = generate_pwls(rom, max_cycles=args.cycles,
                                       verbose=args.verbose)
    print(f'\nSimulated {len(trace)} instructions, final t = {t_end*1e6:.2f}us')

    # Write into the CPU root (where CPU.asc references them)
    write_pwl_files(pwl, CPU_ROOT)
    # Also archive a copy in the program folder
    prog_dir = CPU_ROOT / 'programs' / program_name
    write_pwl_files(pwl, prog_dir)
    # Save the source .asm alongside if it wasn't already there
    prog_asm = prog_dir / f'{program_name}.asm'
    if not prog_asm.exists() or prog_asm.resolve() != asm_path.resolve():
        prog_asm.write_text(asm_text)

    # Print PWL stats
    print(f'\nPWL files written to {CPU_ROOT}/ and {prog_dir}/:')
    for name in PWL_NAMES:
        pts = pwl[name]
        print(f'  {name}.pwl: {len(pts)} points')

    # Patch the .tran on the built CPU.asc
    asc_path = CPU_ROOT / 'CPU.asc'
    if asc_path.exists() and not args.no_patch_tran:
        tran_us = patch_tran(asc_path, t_end, margin_us=200.0)
        # 1 CLK period at 100 kHz = 10 us
        print(f'\nPatched .tran in {asc_path.name}: {tran_us}us covers '
              f'~{tran_us/10:.0f} CLK periods at 100 kHz')
    elif args.no_patch_tran:
        print('\n--no-patch-tran: skipping .tran update')
    else:
        print(f'\nWARNING: {asc_path} not built yet — run '
              f'tools/build_scratch_build.py first.')


if __name__ == '__main__':
    main()
