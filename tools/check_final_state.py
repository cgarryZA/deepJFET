#!/usr/bin/env python3
"""Check final register state of FloatingPoint program.

Reads the end of the .raw file and compares against expected results.
Also dumps stable IR values (sampled during execute phase, not fetch).

Usage:
    python tools/check_final_state.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os
import sys
import struct
import argparse
import numpy as np

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from tools.rom_emulator import assemble, CPU4004Sim


class RawReader:
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
                    except ValueError:
                        pass
            elif s == "Binary:":
                break
        idx = text.find("Binary:")
        self.data_start = (idx + len("Binary:") + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4

    def load_times(self):
        self.times = np.zeros(self.n_points, dtype=np.float64)
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

    def get_nibble(self, values, prefix, n_bits=4, thresh=-2.5):
        val = 0
        for i in range(n_bits):
            key = f"v({prefix}{i})"
            if key in values and values[key] > thresh:
                val |= (1 << i)
        return val


def extract_state(raw, pt_idx):
    t, vals = raw.read_point(pt_idx)
    nib = raw.get_nibble
    return {
        'time': t,
        'acc': nib(vals, "acc"),
        'ir1': nib(vals, "ir1_"),
        'ir2': nib(vals, "ir2_"),
        'cy': 1 if vals.get("v(cf0)", -5) > -2.5 else 0,
        'bus': nib(vals, "bus"),
        'regs': [nib(vals, f"scratch{r}_") for r in range(16)],
        'pc1': nib(vals, "pc1"),
        'pc2': nib(vals, "pc2"),
        'pc3': nib(vals, "pc3"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("--asm", default=None)
    args = parser.parse_args()

    # Load .raw
    raw = RawReader(args.raw_file)
    print(f"Loaded: {raw.n_vars} signals, {raw.n_points} points")

    # Read state near the end
    for offset in [100, 50, 10, 1]:
        pt = raw.n_points - offset
        s = extract_state(raw, pt)
        ir = (s['ir1'] << 4) | s['ir2']
        pc = (s['pc3'] << 8) | (s['pc2'] << 4) | s['pc1']
        print(f"\nt={s['time']*1e3:.3f}ms (pt {pt}): IR=0x{ir:02X} PC=0x{pc:03X}"
              f" ACC=0x{s['acc']:X} CY={s['cy']}")
        print(f"  Regs: [{', '.join(f'R{i}=0x{r:X}' for i, r in enumerate(s['regs']))}]")

    # Expected results from Python sim
    if args.asm:
        asm_path = args.asm
    else:
        raw_dir = os.path.dirname(args.raw_file)
        candidates = [f for f in os.listdir(raw_dir) if f.endswith('.asm')]
        asm_path = os.path.join(raw_dir, candidates[0]) if candidates else None

    if asm_path:
        with open(asm_path) as f:
            asm_text = f.read()
        rom = assemble(asm_text)
        cpu = CPU4004Sim(rom)
        # Execute until NOP at end (PC wraps or hits NOP)
        for i in range(1000):
            pc_before = cpu.pc
            pc_start, b1, b2, two_word = cpu.execute_one()
            if b1 == 0x00 and pc_before > 0x080:  # NOP after main code
                break

        print(f"\n{'='*70}")
        print(f"Python sim final state (after {i+1} instructions):")
        print(f"  ACC=0x{cpu.acc:X} CY={cpu.cy} PC=0x{cpu.pc:03X}")
        print(f"  Regs: [{', '.join(f'R{i}=0x{r:X}' for i, r in enumerate(cpu.regs))}]")

        # Compare
        final = extract_state(raw, raw.n_points - 50)
        print(f"\n{'='*70}")
        print(f"COMPARISON (HW vs Expected):")
        print(f"{'='*70}")

        all_ok = True
        for r in range(16):
            hw_val = final['regs'][r]
            exp_val = cpu.regs[r]
            status = "OK" if hw_val == exp_val else "FAIL"
            if status == "FAIL":
                all_ok = False
            print(f"  R{r:2d}: HW=0x{hw_val:X}  Expected=0x{exp_val:X}  {status}")

        hw_acc = final['acc']
        hw_cy = final['cy']
        print(f"  ACC: HW=0x{hw_acc:X}  Expected=0x{cpu.acc:X}  "
              f"{'OK' if hw_acc == cpu.acc else 'FAIL'}")
        print(f"  CY:  HW={hw_cy}    Expected={cpu.cy}    "
              f"{'OK' if hw_cy == cpu.cy else 'FAIL'}")

        if all_ok:
            print(f"\nAll registers match! Computation is correct.")
        else:
            # Interpret the product
            # FloatingPoint: 244 * 2560 = 624640
            # R12:R13 = mantissa high, R14:R15 = exponent/mantissa low
            print(f"\n--- Interpretation ---")
            print(f"FloatingPoint multiply: 244 (0xF4) * 2560 (0xA00)")
            print(f"  Expected product: R12:R13 = 0x{cpu.regs[12]:X}{cpu.regs[13]:X}")
            print(f"  Hardware product: R12:R13 = 0x{final['regs'][12]:X}{final['regs'][13]:X}")
            print(f"  Expected exp:    R14:R15 = 0x{cpu.regs[14]:X}{cpu.regs[15]:X}")
            print(f"  Hardware exp:    R14:R15 = 0x{final['regs'][14]:X}{final['regs'][15]:X}")


if __name__ == "__main__":
    main()
