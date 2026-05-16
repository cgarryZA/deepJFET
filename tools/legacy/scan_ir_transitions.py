#!/usr/bin/env python3
"""Scan the .raw file for IR transitions and dump CPU state at each.

Instead of assuming timing, this script:
1. Reads IR1 and IR2 at every time point
2. Detects when IR changes (new instruction loaded)
3. Reads full CPU state at each transition
4. Compares against Python simulator

This gives ground truth about what the hardware actually does.

Usage:
    python tools/scan_ir_transitions.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os
import sys
import struct
import argparse
import numpy as np

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from tools.rom_emulator import assemble, CPU4004Sim


class FastRawReader:
    """Optimized .raw reader that bulk-reads specific signals."""

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

    def bulk_read_signals(self, signal_names):
        """Read specific signals across ALL time points. Returns dict of arrays."""
        indices = {}
        for name in signal_names:
            key = name.lower()
            if key in self.var_names:
                indices[name] = self.var_names[key]

        result = {name: np.zeros(self.n_points, dtype=np.float32) for name in indices}
        times = np.zeros(self.n_points, dtype=np.float64)

        with open(self.path, "rb") as f:
            for pt in range(self.n_points):
                base = self.data_start + pt * self.row_size
                f.seek(base)
                times[pt] = struct.unpack_from("d", f.read(8), 0)[0]

                for name, idx in indices.items():
                    if idx == 0:
                        result[name][pt] = times[pt]
                    else:
                        f.seek(base + 8 + (idx - 1) * 4)
                        result[name][pt] = struct.unpack_from("f", f.read(4), 0)[0]

        return times, result

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

    def get_nibble(self, values, prefix, n_bits=4, thresh=-2.5):
        val = 0
        for i in range(n_bits):
            key = f"v({prefix}{i})"
            if key in values and values[key] > thresh:
                val |= (1 << i)
        return val


MNEMONICS = {
    0x0: 'NOP', 0x1: 'JCN', 0x2: 'FIM/SRC', 0x3: 'FIN/JIN',
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
    parser = argparse.ArgumentParser(description="Scan IR transitions in .raw file")
    parser.add_argument("raw_file", help="Path to .raw file")
    parser.add_argument("--asm", default=None)
    parser.add_argument("--max-transitions", type=int, default=100)
    parser.add_argument("--settle-points", type=int, default=20,
                        help="Points to skip after IR change for settling")
    args = parser.parse_args()

    # Load assembly for Python sim comparison
    if args.asm:
        asm_path = args.asm
    else:
        raw_dir = os.path.dirname(args.raw_file)
        candidates = [f for f in os.listdir(raw_dir) if f.endswith('.asm')]
        asm_path = os.path.join(raw_dir, candidates[0]) if candidates else None

    rom = None
    cpu = None
    if asm_path:
        with open(asm_path) as f:
            asm_text = f.read()
        rom = assemble(asm_text)
        cpu = CPU4004Sim(rom)
        cpu.regs = [0xF] * 16
        cpu.acc = 0xF
        cpu.cy = 0
        print(f"Assembly: {asm_path} ({len(rom)} bytes)")

    print(f"Loading .raw: {args.raw_file}")
    raw = FastRawReader(args.raw_file)
    print(f"  {raw.n_vars} signals, {raw.n_points} points")

    # Bulk read IR signals for fast scanning
    print("Bulk reading IR signals...")
    ir_signals = ['v(ir1_0)', 'v(ir1_1)', 'v(ir1_2)', 'v(ir1_3)',
                  'v(ir2_0)', 'v(ir2_1)', 'v(ir2_2)', 'v(ir2_3)']
    times, data = raw.bulk_read_signals(ir_signals)
    print(f"  Time range: {times[0]*1e6:.1f}us to {times[-1]*1e6:.1f}us")

    # Convert to nibble values
    thresh = -2.5
    ir1_vals = np.zeros(raw.n_points, dtype=np.int32)
    ir2_vals = np.zeros(raw.n_points, dtype=np.int32)
    for bit in range(4):
        ir1_vals |= ((data[f'v(ir1_{bit})'] > thresh).astype(np.int32) << bit)
        ir2_vals |= ((data[f'v(ir2_{bit})'] > thresh).astype(np.int32) << bit)

    ir_combined = (ir1_vals << 4) | ir2_vals

    # Find IR transitions
    transitions = []
    prev_ir = ir_combined[0]
    for pt in range(1, raw.n_points):
        if ir_combined[pt] != prev_ir:
            transitions.append(pt)
            prev_ir = ir_combined[pt]

    print(f"Found {len(transitions)} IR transitions")

    # Filter for STABLE IR values (persisting > min_stable_us)
    min_stable_pts = 30  # Minimum points an IR value must persist
    stable_irs = []  # (start_pt, end_pt, ir_value)

    i = 0
    while i < len(transitions):
        start_pt = transitions[i]
        ir_val = ir_combined[start_pt]

        # Find how long this IR value persists
        end_pt = transitions[i + 1] if i + 1 < len(transitions) else raw.n_points - 1
        duration_pts = end_pt - start_pt

        if duration_pts >= min_stable_pts:
            stable_irs.append((start_pt, end_pt, ir_val))
        i += 1

    print(f"Stable IR values (>{min_stable_pts} points): {len(stable_irs)}")

    # Also generate expected ROM sequence
    if cpu:
        expected_seq = []
        sim = CPU4004Sim(rom)
        for _ in range(500):
            pc = sim.pc
            b1 = sim.fetch_byte()
            opr = (b1 >> 4) & 0xF
            # Check if 2-word
            two_word = opr in (0x1, 0x2, 0x4, 0x5, 0x7) and not (
                (opr == 0x2 and b1 & 1) or (opr == 0x3))
            if two_word:
                b2 = sim.fetch_byte()
            else:
                b2 = None
            expected_seq.append((pc, b1, b2))
            # Minimal execution for PC tracking
            sim2 = CPU4004Sim(rom)
            sim2.pc = 0
            sim2.regs = [0] * 16
        break  # Just get the first instruction for now

    # Print stable IR sequence with comparison to expected ROM
    print(f"\n{'='*100}")
    print(f"{'#':>4} {'Time':>10} {'Dur':>6} {'IR':>4} {'Mnem':>12} {'Exp':>4} {'ExpMnem':>12} {'Match':>6}"
          f" {'ACC':>4} {'CY':>3} {'Regs':>60}")
    print(f"{'='*100}")

    # Re-simulate for expected sequence
    sim = CPU4004Sim(rom)
    prev_regs = [0xF] * 16

    for i, (start_pt, end_pt, ir_val) in enumerate(stable_irs[:args.max_transitions]):
        # Read state at midpoint of this stable IR (for best settling)
        mid_pt = (start_pt + end_pt) // 2
        mid_pt = min(mid_pt, raw.n_points - 1)
        t, vals = raw.read_point(mid_pt)

        dur_us = (times[min(end_pt, raw.n_points-1)] - times[start_pt]) * 1e6

        ir1 = (ir_val >> 4) & 0xF
        ir2 = ir_val & 0xF
        mnem = decode_mnemonic(ir_val)

        acc = raw.get_nibble(vals, "acc")
        cy_val = vals.get("v(cf0)", -5)
        cy = 1 if cy_val > thresh else 0

        regs = []
        for r in range(16):
            regs.append(raw.get_nibble(vals, f"scratch{r}_"))

        changes = []
        for r in range(16):
            if regs[r] != prev_regs[r]:
                changes.append(f"R{r}:{prev_regs[r]:X}->{regs[r]:X}")

        # Expected from Python sim
        if i < len(expected_seq):
            exp_pc, exp_b1, exp_b2 = expected_seq[i]
            exp_mnem = decode_mnemonic(exp_b1)
            match = "OK" if exp_b1 == ir_val else "FAIL"
        else:
            exp_b1 = 0
            exp_mnem = "?"
            match = "?"

        regs_str = ','.join(f'{r:X}' for r in regs)
        change_str = f"  [{', '.join(changes)}]" if changes else ""

        print(f"{i:4d} {t*1e3:10.4f}ms {dur_us:5.0f}us {ir_val:02X} {mnem:>12s} "
              f"{exp_b1:02X} {exp_mnem:>12s} {match:>6} "
              f"ACC={acc:X} CY={cy} [{regs_str}]{change_str}")

        prev_regs = list(regs)

    print(f"\n{'='*100}")
    print(f"Shown {min(len(stable_irs), args.max_transitions)} stable IR values")


if __name__ == "__main__":
    main()
