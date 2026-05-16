#!/usr/bin/env python3
"""Trace FloatingPoint execution starting from a sync point.

Instead of tracing from the beginning (where timing alignment is hard),
this script:
1. Reads actual hardware state at a user-specified time
2. Initializes the Python sim to match
3. Traces forward, syncing on IR transitions

Usage:
    python tools/trace_from_sync.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
    python tools/trace_from_sync.py ... --sync-time 4.0  # Start at 4.0ms
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
        return self.times

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

    def bulk_read_signal(self, signal_name):
        """Read one signal across all time points."""
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

    def find_time_idx(self, t_target):
        """Find point index nearest to target time."""
        return int(np.searchsorted(self.times, t_target))


def extract_state(raw, pt_idx):
    t, vals = raw.read_point(pt_idx)
    nib = raw.get_nibble
    state = {
        'time': t,
        'acc': nib(vals, "acc"),
        'ir1': nib(vals, "ir1_"),
        'ir2': nib(vals, "ir2_"),
        'cy': 1 if vals.get("v(cf0)", -5) > -2.5 else 0,
        'bus': nib(vals, "bus"),
        'regs': [nib(vals, f"scratch{r}_") for r in range(16)],
    }
    state['ir'] = (state['ir1'] << 4) | state['ir2']
    state['pc1'] = nib(vals, "pc1")
    state['pc2'] = nib(vals, "pc2")
    state['pc3'] = nib(vals, "pc3")
    state['pc'] = (state['pc3'] << 8) | (state['pc2'] << 4) | state['pc1']
    return state


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


def find_stable_ir_at(raw, times, ir1_data, ir2_data, pt_idx, window=50):
    """Find the stable IR value near a time point."""
    # Look at a window around the point and find the most common IR value
    start = max(0, pt_idx - window)
    end = min(raw.n_points, pt_idx + window)
    ir_vals = (ir1_data[start:end].astype(int) << 4) | ir2_data[start:end].astype(int)
    # Return most common value
    vals, counts = np.unique(ir_vals, return_counts=True)
    return int(vals[np.argmax(counts)])


def main():
    parser = argparse.ArgumentParser(description="Trace from sync point")
    parser.add_argument("raw_file")
    parser.add_argument("--asm", default=None)
    parser.add_argument("--sync-time", type=float, default=4.3,
                        help="Time in ms to sync state (default: 4.3ms, after init + data setup)")
    parser.add_argument("--max-cycles", type=int, default=200)
    args = parser.parse_args()

    # Load assembly
    if args.asm:
        asm_path = args.asm
    else:
        raw_dir = os.path.dirname(args.raw_file)
        candidates = [f for f in os.listdir(raw_dir) if f.endswith('.asm')]
        asm_path = os.path.join(raw_dir, candidates[0]) if candidates else None

    with open(asm_path) as f:
        asm_text = f.read()
    rom = assemble(asm_text)
    print(f"ROM: {len(rom)} bytes")

    # Load .raw
    print(f"Loading .raw: {args.raw_file}")
    raw = RawReader(args.raw_file)
    print(f"  {raw.n_vars} signals, {raw.n_points} points")

    print("Loading time array...")
    times = raw.load_times()
    print(f"  {times[0]*1e6:.0f}us to {times[-1]*1e6:.0f}us")

    # Bulk read IR for transition detection
    print("Reading IR signals...")
    thresh = -2.5
    ir1_bits = np.zeros((4, raw.n_points), dtype=np.float32)
    ir2_bits = np.zeros((4, raw.n_points), dtype=np.float32)
    for bit in range(4):
        d = raw.bulk_read_signal(f"v(ir1_{bit})")
        if d is not None:
            ir1_bits[bit] = d
        d = raw.bulk_read_signal(f"v(ir2_{bit})")
        if d is not None:
            ir2_bits[bit] = d

    ir1_vals = np.zeros(raw.n_points, dtype=np.int32)
    ir2_vals = np.zeros(raw.n_points, dtype=np.int32)
    for bit in range(4):
        ir1_vals |= ((ir1_bits[bit] > thresh).astype(np.int32) << bit)
        ir2_vals |= ((ir2_bits[bit] > thresh).astype(np.int32) << bit)
    ir_combined = (ir1_vals << 4) | ir2_vals

    # Find micro1 edges for instruction boundaries
    print("Reading micro1 signal...")
    m1_data = raw.bulk_read_signal("v(micro1)")
    m1_high = m1_data > thresh
    m1_edges = np.where(m1_high[1:] & ~m1_high[:-1])[0] + 1
    print(f"  {len(m1_edges)} micro1 rising edges")

    # Find sync point
    sync_pt = raw.find_time_idx(args.sync_time * 1e-3)
    print(f"\nSync point: t={times[sync_pt]*1e3:.4f}ms (pt {sync_pt})")

    # Read hardware state at sync point
    hw_sync = extract_state(raw, sync_pt)
    print(f"  HW state: ACC=0x{hw_sync['acc']:X} CY={hw_sync['cy']} "
          f"IR=0x{hw_sync['ir']:02X} PC=0x{hw_sync['pc']:03X}")
    print(f"  Regs: [{','.join(f'{r:X}' for r in hw_sync['regs'])}]")

    # Run Python sim to the same point
    # We need to execute instructions until the PC matches
    cpu = CPU4004Sim(rom)
    n_exec = 0
    while n_exec < 1000:
        pc_before = cpu.pc
        cpu.execute_one()
        n_exec += 1
        # Check if we've reached the instruction at sync time
        # We'll match on the IR value at the sync point
        if cpu.pc == hw_sync['pc'] and n_exec > 30:  # past init
            break

    print(f"\n  Python sim after {n_exec} instructions:")
    print(f"  ACC=0x{cpu.acc:X} CY={cpu.cy} PC=0x{cpu.pc:03X}")
    print(f"  Regs: [{','.join(f'{r:X}' for r in cpu.regs)}]")

    # Compare
    diffs = []
    if cpu.acc != hw_sync['acc']:
        diffs.append(f"ACC: py={cpu.acc:X} hw={hw_sync['acc']:X}")
    if cpu.cy != hw_sync['cy']:
        diffs.append(f"CY: py={cpu.cy} hw={hw_sync['cy']}")
    for r in range(16):
        if cpu.regs[r] != hw_sync['regs'][r]:
            diffs.append(f"R{r}: py={cpu.regs[r]:X} hw={hw_sync['regs'][r]:X}")

    if diffs:
        print(f"\n  DIFFERENCES at sync point:")
        for d in diffs:
            print(f"    {d}")
        print(f"\n  Forcing Python sim to match hardware state...")
        cpu.acc = hw_sync['acc']
        cpu.cy = hw_sync['cy']
        cpu.regs = list(hw_sync['regs'])
        # Don't set PC - it should already match
    else:
        print(f"\n  States match at sync point!")

    # Now find micro1 edges AFTER sync point and trace forward
    # Each micro1 edge = start of a new micro-phase cycle
    # For 1-word instructions: 1 micro1 edge per instruction
    # For 2-word instructions: 2 micro1 edges per instruction

    # Find micro1 edges after sync
    post_sync_edges = m1_edges[m1_edges > sync_pt]
    print(f"\n  {len(post_sync_edges)} micro1 edges after sync")

    # Strategy: at each micro1 edge, read the IR. When IR changes to a NEW
    # stable value, that's a new instruction. Read state at the PREVIOUS
    # instruction's last micro1 edge (before this new IR appeared).

    print(f"\n{'='*100}")
    print(f"Tracing from sync point (t={args.sync_time}ms)")
    print(f"{'='*100}")

    prev_ir = ir_combined[sync_pt]
    prev_state_pt = sync_pt
    cycle = 0
    divergence_count = 0

    # Group consecutive micro1 edges with the same stable IR value
    i = 0
    while i < len(post_sync_edges) and cycle < args.max_cycles:
        edge_pt = post_sync_edges[i]

        # Read IR a few points after the edge (for settling)
        sample_pt = min(edge_pt + 10, raw.n_points - 1)
        current_ir = ir_combined[sample_pt]

        # Check if IR has changed from previous
        if current_ir != prev_ir and current_ir != 0x00:
            # New instruction loaded!
            # Read hardware state JUST BEFORE this edge (end of previous instruction)
            read_pt = max(edge_pt - 5, 0)
            hw = extract_state(raw, read_pt)

            mnem = decode_mnemonic(prev_ir)

            # Compare against Python sim
            # The Python sim should have just executed the previous instruction
            mismatches = []
            if hw['acc'] != cpu.acc:
                mismatches.append(f"ACC: hw={hw['acc']:X} py={cpu.acc:X}")
            if hw['cy'] != cpu.cy:
                mismatches.append(f"CY: hw={hw['cy']} py={cpu.cy}")
            for r in range(16):
                if hw['regs'][r] != cpu.regs[r]:
                    mismatches.append(f"R{r}: hw={hw['regs'][r]:X} py={cpu.regs[r]:X}")

            if mismatches:
                divergence_count += 1
                print(f"\n[{cycle:3d}] t={hw['time']*1e3:.4f}ms IR={prev_ir:02X} {mnem:12s}"
                      f"  ACC={hw['acc']:X} CY={hw['cy']}")
                for m in mismatches:
                    print(f"       MISMATCH: {m}")
                if divergence_count <= 10:
                    print(f"       HW regs:  [{','.join(f'{r:X}' for r in hw['regs'])}]")
                    print(f"       PY regs:  [{','.join(f'{r:X}' for r in cpu.regs)}]")

                # Sync Python to hardware to isolate future errors
                if divergence_count <= 3:
                    print(f"       (Syncing Python to HW to isolate next divergence)")
                    cpu.acc = hw['acc']
                    cpu.cy = hw['cy']
                    cpu.regs = list(hw['regs'])
            else:
                if cycle < 20 or cycle % 20 == 0:
                    regs_str = ','.join(f'{r:X}' for r in hw['regs'])
                    print(f"[{cycle:3d}] t={hw['time']*1e3:.4f}ms IR={prev_ir:02X} {mnem:12s}"
                          f"  OK  ACC={hw['acc']:X} CY={hw['cy']} [{regs_str}]")

            # Now execute this instruction in Python sim
            # The new IR tells us what instruction to execute
            # But we need to execute from current PC
            pc_before = cpu.pc
            cpu.execute_one()

            prev_ir = current_ir
            cycle += 1

        i += 1

    print(f"\n{'='*100}")
    print(f"Trace: {cycle} instructions, {divergence_count} divergences")
    print(f"{'='*100}")

    # Print final state
    print(f"\nFinal hardware state:")
    final = extract_state(raw, raw.n_points - 100)
    print(f"  t={final['time']*1e3:.4f}ms ACC={final['acc']:X} CY={final['cy']}")
    print(f"  Regs: [{','.join(f'{r:X}' for r in final['regs'])}]")

    print(f"\nFinal Python sim state:")
    print(f"  ACC={cpu.acc:X} CY={cpu.cy} PC={cpu.pc:03X}")
    print(f"  Regs: [{','.join(f'{r:X}' for r in cpu.regs)}]")


if __name__ == "__main__":
    main()
