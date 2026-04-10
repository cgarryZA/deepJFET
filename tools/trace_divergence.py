#!/usr/bin/env python3
"""Trace FloatingPoint execution: compare Python simulator vs LTSpice .raw file.

Finds the exact instruction cycle where the CPU's internal state diverges
from the expected Python simulation. Reads micro1 edges from the .raw file
to locate instruction boundaries, then compares ACC, CY, and all 16
scratchpad registers.

Usage:
    python tools/trace_divergence.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os
import sys
import struct
import argparse
import numpy as np

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from tools.rom_emulator import (
    assemble, CPU4004Sim, get_phase_count, CLK_PERIOD,
    FETCH_PHASE_OPR, FETCH_PHASE_OPA
)


# ── .raw file reader (from reg_viewer.py) ────────────────────────────────

class RawReader:
    """Reads LTSpice .raw files with mixed precision (time=double, rest=float)."""

    def __init__(self, path):
        self.path = path
        self._parse_header()

    def _parse_header(self):
        with open(self.path, "rb") as f:
            header = f.read(2_000_000)

        text = header.decode("utf-16-le", errors="replace")

        self.n_vars = 0
        self.n_points = 0
        self.var_names = {}
        self.var_list = []

        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("No. Variables"):
                self.n_vars = int(s.split(":")[1].strip())
            elif s.startswith("No. Points"):
                self.n_points = int(s.split(":")[1].strip())
            elif "\t" in s:
                parts = s.split("\t")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        self.var_names[name.lower()] = idx
                        while len(self.var_list) <= idx:
                            self.var_list.append("")
                        self.var_list[idx] = name
                    except ValueError:
                        pass
            elif s == "Binary:":
                break

        idx = text.find("Binary:")
        self.data_start = (idx + len("Binary:") + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4

    def _load_time_array(self):
        self.times = np.zeros(self.n_points)
        with open(self.path, "rb") as f:
            for i in range(self.n_points):
                f.seek(self.data_start + i * self.row_size)
                self.times[i] = struct.unpack_from("d", f.read(8), 0)[0]

    def read_point(self, pt_idx):
        with open(self.path, "rb") as f:
            f.seek(self.data_start + pt_idx * self.row_size)
            row = f.read(self.row_size)
        t = struct.unpack_from("d", row, 0)[0]
        values = {}
        for name, idx in self.var_names.items():
            if idx == 0:
                values[name] = t
            else:
                values[name] = struct.unpack_from("f", row, 8 + (idx - 1) * 4)[0]
        return t, values

    def read_signal_at_point(self, pt_idx, signal_name):
        idx = self.var_names.get(signal_name.lower())
        if idx is None:
            return None
        with open(self.path, "rb") as f:
            if idx == 0:
                f.seek(self.data_start + pt_idx * self.row_size)
                return struct.unpack_from("d", f.read(8), 0)[0]
            else:
                f.seek(self.data_start + pt_idx * self.row_size + 8 + (idx - 1) * 4)
                return struct.unpack_from("f", f.read(4), 0)[0]

    def get_nibble(self, values, prefix, n_bits=4, thresh=-2.5):
        val = 0
        for i in range(n_bits):
            key = f"v({prefix}{i})"
            if key in values and values[key] > thresh:
                val |= (1 << i)
        return val

    def find_micro1_edges(self):
        """Find all Micro1 rising edges (instruction boundaries)."""
        edges = []
        m1_key = "v(micro1)"
        m1_idx = self.var_names.get(m1_key)
        if m1_idx is None:
            print(f"WARNING: signal 'micro1' not found in .raw file")
            print(f"Available signals containing 'micro': ", end="")
            print([n for n in self.var_names if 'micro' in n])
            return edges

        prev_low = True
        with open(self.path, "rb") as f:
            for pt in range(self.n_points):
                f.seek(self.data_start + pt * self.row_size + 8 + (m1_idx - 1) * 4)
                val = struct.unpack_from("f", f.read(4), 0)[0]
                is_high = val > -2.5
                if is_high and prev_low:
                    edges.append(pt)
                prev_low = not is_high
        return edges


def extract_cpu_state(raw, pt_idx):
    """Extract full CPU state at a time point."""
    t, vals = raw.read_point(pt_idx)
    nib = raw.get_nibble

    state = {
        'time': t,
        'acc': nib(vals, "acc"),
        'ir1': nib(vals, "ir1_"),
        'ir2': nib(vals, "ir2_"),
        'pc1': nib(vals, "pc1"),
        'pc2': nib(vals, "pc2"),
        'pc3': nib(vals, "pc3"),
        'bus': nib(vals, "bus"),
    }

    # Carry flag
    cy_val = vals.get("v(cf0)", -5)
    state['cy'] = 1 if cy_val > -2.5 else 0

    # Scratchpad registers
    state['regs'] = []
    for r in range(16):
        state['regs'].append(nib(vals, f"scratch{r}_"))

    state['pc'] = (state['pc3'] << 8) | (state['pc2'] << 4) | state['pc1']
    state['ir'] = (state['ir1'] << 4) | state['ir2']

    return state


MNEMONICS = {
    0x0: 'NOP', 0x1: 'JCN', 0x2: 'FIM', 0x3: 'FIN/JIN',
    0x4: 'JUN', 0x5: 'JMS', 0x6: 'INC', 0x7: 'ISZ',
    0x8: 'ADD', 0x9: 'SUB', 0xA: 'LD', 0xB: 'XCH',
    0xC: 'BBL', 0xD: 'LDM',
}
ACC_MNEMONICS = {
    0xF0: 'CLB', 0xF1: 'CLC', 0xF2: 'IAC', 0xF3: 'CMC', 0xF4: 'CMA',
    0xF5: 'RAL', 0xF6: 'RAR', 0xF7: 'TCC', 0xF8: 'DAC', 0xF9: 'TCS',
    0xFA: 'STC', 0xFB: 'DAA', 0xFC: 'KBP', 0xFD: 'DCL',
}


def decode_mnemonic(opcode):
    if opcode in ACC_MNEMONICS:
        return ACC_MNEMONICS[opcode]
    opr = (opcode >> 4) & 0xF
    opa = opcode & 0xF
    base = MNEMONICS.get(opr, '???')
    if opr in (0x6, 0x8, 0x9, 0xA, 0xB):
        return f"{base} R{opa}"
    elif opr == 0xD:
        return f"LDM {opa}"
    elif opr == 0xC:
        return f"BBL {opa}"
    return f"{base} {opa:X}"


def main():
    parser = argparse.ArgumentParser(description="Trace FloatingPoint divergence")
    parser.add_argument("raw_file", help="Path to extracted .raw file")
    parser.add_argument("--asm", default=None, help="Path to .asm file (default: auto-detect)")
    parser.add_argument("--max-cycles", type=int, default=500, help="Max cycles to trace")
    parser.add_argument("--startup-delay", type=float, default=2e-6)
    parser.add_argument("--sample-offset", type=int, default=5,
                        help="Sample N points after micro1 edge (for signal settling)")
    args = parser.parse_args()

    # Load assembly
    if args.asm:
        asm_path = args.asm
    else:
        raw_dir = os.path.dirname(args.raw_file)
        candidates = [f for f in os.listdir(raw_dir) if f.endswith('.asm')]
        if candidates:
            asm_path = os.path.join(raw_dir, candidates[0])
        else:
            print("ERROR: No .asm file found. Use --asm to specify.")
            sys.exit(1)

    print(f"Assembly: {asm_path}")
    with open(asm_path) as f:
        asm_text = f.read()
    rom = assemble(asm_text)
    print(f"ROM: {len(rom)} bytes")

    # Load .raw file
    print(f"Loading .raw: {args.raw_file}")
    raw = RawReader(args.raw_file)
    print(f"  {raw.n_vars} signals, {raw.n_points} points")

    # Check available signals
    print("\nChecking signal availability...")
    test_signals = ['v(acc0)', 'v(cf0)', 'v(ir1_0)', 'v(scratch0_0)', 'v(micro1)']
    for sig in test_signals:
        found = sig in raw.var_names
        print(f"  {sig}: {'FOUND' if found else 'MISSING'}")

    # Load time array for time-based sampling
    print("\nLoading time array...")
    raw._load_time_array()
    print(f"  Time range: {raw.times[0]*1e6:.1f}us to {raw.times[-1]*1e6:.1f}us")

    # Initialize Python simulator with JFET power-up state (all regs = 0xF)
    cpu = CPU4004Sim(rom)
    cpu.regs = [0xF] * 16
    cpu.acc = 0xF
    cpu.cy = 0

    print(f"\n{'='*80}")
    print(f"Tracing instruction-by-instruction (up to {args.max_cycles} cycles)")
    print(f"Python sim initialized: ACC=0xF, CY=0, all regs=0xF (JFET power-up)")
    print(f"Using time-based sampling (startup={args.startup_delay*1e6:.1f}us, CLK={CLK_PERIOD*1e6:.1f}us)")
    print(f"{'='*80}")

    first_divergence = None
    divergence_count = 0

    # Time-based: track cumulative time from start
    t_current = args.startup_delay  # time at start of instruction 0

    cycle = 0
    while cycle < args.max_cycles:
        # Save state BEFORE execution
        pre_acc = cpu.acc
        pre_cy = cpu.cy
        pre_regs = list(cpu.regs)

        # Python sim: execute one instruction
        pc_start, b1, b2, two_word = cpu.execute_one()

        # Expected state AFTER this instruction
        exp_acc = cpu.acc
        exp_cy = cpu.cy
        exp_regs = list(cpu.regs)

        mnem = decode_mnemonic(b1)
        if two_word and b2 is not None:
            mnem += f" 0x{b2:02X}"

        # Advance time by this instruction's phase count
        n_phases = get_phase_count(b1)
        t_current += n_phases * CLK_PERIOD

        # Sample at the END of the current instruction (minus 1 CLK for margin)
        # This avoids reading state that's already been modified by the next instruction
        t_sample = t_current - 1 * CLK_PERIOD

        # Find nearest point in .raw
        pt_idx = np.searchsorted(raw.times, t_sample)
        if pt_idx >= raw.n_points:
            print(f"\nReached end of .raw file at cycle {cycle}")
            break

        hw = extract_cpu_state(raw, pt_idx)

        # Compare
        mismatches = []
        if hw['acc'] != exp_acc:
            mismatches.append(f"ACC: hw=0x{hw['acc']:X} exp=0x{exp_acc:X}")
        if hw['cy'] != exp_cy:
            mismatches.append(f"CY: hw={hw['cy']} exp={exp_cy}")
        for r in range(16):
            if hw['regs'][r] != exp_regs[r]:
                mismatches.append(f"R{r}: hw=0x{hw['regs'][r]:X} exp=0x{exp_regs[r]:X}")

        # Find what changed
        reg_changes = []
        for r in range(16):
            if exp_regs[r] != pre_regs[r]:
                reg_changes.append(f"R{r}:{pre_regs[r]:X}->{exp_regs[r]:X}")

        if mismatches:
            divergence_count += 1
            marker = " <<<< FIRST" if first_divergence is None else ""
            if first_divergence is None:
                first_divergence = cycle
            print(f"\n[{cycle:3d}] PC=0x{pc_start:03X} {b1:02X}{f' {b2:02X}' if b2 is not None else '   '}"
                  f"  {mnem:12s}  t={hw['time']*1e3:.4f}ms{marker}")
            if reg_changes:
                print(f"       Changes: ACC:{pre_acc:X}->{exp_acc:X} CY:{pre_cy}->{exp_cy} {' '.join(reg_changes)}")
            for m in mismatches:
                print(f"       MISMATCH: {m}")

            # Print full state for first few divergences
            if divergence_count <= 10:
                print(f"       HW regs:  [{','.join(f'{r:X}' for r in hw['regs'])}]")
                print(f"       Exp regs: [{','.join(f'{r:X}' for r in exp_regs)}]")

            # If this is an XCH and it diverged, sync Python to hardware
            # to isolate future errors
        else:
            # Print all OK cycles during init, then periodic
            if cycle < 50 or cycle % 50 == 0:
                chg = f"  ({' '.join(reg_changes)})" if reg_changes else ""
                print(f"[{cycle:3d}] PC=0x{pc_start:03X} {b1:02X}"
                      f"{'  ' + format(b2, '02X') if b2 is not None else '   '}"
                      f"  {mnem:12s}  OK  ACC=0x{exp_acc:X} CY={exp_cy}{chg}")

        cycle += 1

    print(f"\n{'='*80}")
    print(f"Trace complete: {cycle} cycles, {divergence_count} divergences")
    if first_divergence is not None:
        print(f"First divergence at cycle {first_divergence}")
    else:
        print("No divergences found!")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
