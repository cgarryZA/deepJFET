#!/usr/bin/env python3
"""Check Index register values during XCH instructions.

Reads the .raw file, finds XCH instructions by IR value,
and checks what Index0-Index3 values are during execution.
This verifies the scratchpad address decode.

Usage:
    python tools/check_index_decode.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os, sys, struct, argparse
import numpy as np

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)


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

    def bulk_read_signal(self, signal_name):
        idx = self.var_names.get(signal_name.lower())
        if idx is None:
            return None
        arr = np.zeros(self.n_points, dtype=np.float32)
        with open(self.path, "rb") as f:
            for pt in range(self.n_points):
                if idx == 0:
                    f.seek(self.data_start + pt * self.row_size)
                    arr[pt] = struct.unpack_from("d", f.read(8), 0)[0]
                else:
                    f.seek(self.data_start + pt * self.row_size + 8 + (idx - 1) * 4)
                    arr[pt] = struct.unpack_from("f", f.read(4), 0)[0]
        return arr

    def load_times(self):
        self.times = np.zeros(self.n_points, dtype=np.float64)
        with open(self.path, "rb") as f:
            for i in range(self.n_points):
                f.seek(self.data_start + i * self.row_size)
                self.times[i] = struct.unpack_from("d", f.read(8), 0)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    args = parser.parse_args()

    raw = RawReader(args.raw_file)
    print(f"Loaded: {raw.n_vars} signals, {raw.n_points} points")

    print("Loading signals...")
    raw.load_times()

    thresh = -2.5

    # Load all needed signals
    signals = {}
    for name in ['v(ir1_0)', 'v(ir1_1)', 'v(ir1_2)', 'v(ir1_3)',
                  'v(ir2_0)', 'v(ir2_1)', 'v(ir2_2)', 'v(ir2_3)',
                  'v(index0)', 'v(index1)', 'v(index2)', 'v(index3)',
                  'v(indexreg_loading)', 'v(indexreg_load)',
                  'v(scratch000)', 'v(scratch001)', 'v(scratch010)', 'v(scratch011)',
                  'v(scratch100)', 'v(scratch101)', 'v(scratch110)', 'v(scratch111)',
                  'v(micro1)']:
        d = raw.bulk_read_signal(name)
        if d is not None:
            signals[name] = d
            print(f"  {name}: loaded")
        else:
            print(f"  {name}: NOT FOUND")

    # Compute IR value at each point
    ir1 = np.zeros(raw.n_points, dtype=np.int32)
    ir2 = np.zeros(raw.n_points, dtype=np.int32)
    for bit in range(4):
        ir1 |= ((signals[f'v(ir1_{bit})'] > thresh).astype(np.int32) << bit)
        ir2 |= ((signals[f'v(ir2_{bit})'] > thresh).astype(np.int32) << bit)
    ir = (ir1 << 4) | ir2

    # Compute Index value
    index = np.zeros(raw.n_points, dtype=np.int32)
    for bit in range(4):
        index |= ((signals[f'v(index{bit})'] > thresh).astype(np.int32) << bit)

    # Find micro1 edges
    m1 = signals['v(micro1)'] > thresh
    m1_edges = np.where(m1[1:] & ~m1[:-1])[0] + 1

    # For each micro1 edge, read IR and Index midway through the instruction
    # (sample at the micro1 edge + 50% of the way to the next edge)
    print(f"\n{'='*90}")
    print(f"{'Edge':>5} {'Time':>10} {'IR':>4} {'OPR':>4} {'OPA':>4} {'Index':>6} "
          f"{'Idx3210':>8} {'Pair':>6} {'ExpPair':>8} {'Match':>6}")
    print(f"{'='*90}")

    n_shown = 0
    for i in range(len(m1_edges) - 1):
        if n_shown >= 200:
            break

        edge = m1_edges[i]
        next_edge = m1_edges[i + 1]

        # Sample at 95% through the micro-phase (near end, after settling)
        sample_pt = edge + int(0.95 * (next_edge - edge))
        if sample_pt >= raw.n_points:
            break

        ir_val = ir[sample_pt]
        opr = (ir_val >> 4) & 0xF
        opa = ir_val & 0xF
        idx_val = index[sample_pt]

        # Only show XCH, LD, ADD, SUB instructions (which use register addressing)
        if opr in (0x8, 0x9, 0xA, 0xB):  # ADD, SUB, LD, XCH
            t = raw.times[sample_pt]
            idx_bits = f"{(idx_val>>3)&1}{(idx_val>>2)&1}{(idx_val>>1)&1}{idx_val&1}"

            # Expected pair from OPA: pair = OPA >> 1 (bits 3:1)
            exp_pair = (opa >> 1) & 0x7

            # Actual pair from Index: Index3:Index2:Index1
            act_pair = (idx_val >> 1) & 0x7

            match = "OK" if exp_pair == act_pair else "FAIL"

            mnemonics = {0x8: 'ADD', 0x9: 'SUB', 0xA: 'LD', 0xB: 'XCH'}
            mnem = f"{mnemonics[opr]} R{opa}"

            # Also check Index0 vs OPA bit 0
            exp_idx0 = opa & 1
            act_idx0 = idx_val & 1
            idx0_match = "ok" if exp_idx0 == act_idx0 else "FAIL"

            print(f"{i:5d} {t*1e3:10.4f}ms {ir_val:02X} {mnem:>8s} OPA={opa:X} "
                  f"Idx={idx_val:X} [{idx_bits}] "
                  f"pair={act_pair} exp={exp_pair} {match:>6s} "
                  f"idx0={act_idx0} exp0={exp_idx0} {idx0_match}")
            n_shown += 1

    print(f"\nShown {n_shown} register-addressed instructions")


if __name__ == "__main__":
    main()
