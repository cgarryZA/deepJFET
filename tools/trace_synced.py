#!/usr/bin/env python3
"""Trace FloatingPoint with proper late-phase sampling.

Samples at 95% through each micro1 phase (after Index has settled).
Syncs Python sim at a known good point after init, then traces forward
comparing ACC, CY, and all registers at each instruction boundary.

Usage:
    python tools/trace_synced.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os, sys, struct, argparse
import numpy as np

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from tools.rom_emulator import assemble, CPU4004Sim, get_phase_count


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
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("--asm", default=None)
    parser.add_argument("--max-cycles", type=int, default=500)
    args = parser.parse_args()

    if args.asm:
        asm_path = args.asm
    else:
        raw_dir = os.path.dirname(args.raw_file)
        candidates = [f for f in os.listdir(raw_dir) if f.endswith('.asm')]
        asm_path = os.path.join(raw_dir, candidates[0])

    with open(asm_path) as f:
        asm_text = f.read()
    rom = assemble(asm_text)
    print(f"ROM: {len(rom)} bytes")

    raw = RawReader(args.raw_file)
    print(f"Loaded: {raw.n_vars} signals, {raw.n_points} points")

    print("Loading time array and signals...")
    raw.load_times()

    thresh = -2.5

    # Bulk read key signals
    sigs = {}
    for name in (['v(micro1)'] +
                 [f'v(ir1_{i})' for i in range(4)] +
                 [f'v(ir2_{i})' for i in range(4)] +
                 [f'v(acc{i})' for i in range(4)] +
                 ['v(cf0)'] +
                 [f'v(scratch{r}_{b})' for r in range(16) for b in range(4)]):
        d = raw.bulk_read_signal(name)
        if d is not None:
            sigs[name] = d

    print(f"  Loaded {len(sigs)} signals")

    # Compute nibble arrays
    def nibble_array(prefix, n_bits=4):
        arr = np.zeros(raw.n_points, dtype=np.int32)
        for b in range(n_bits):
            key = f'v({prefix}{b})'
            if key in sigs:
                arr |= ((sigs[key] > thresh).astype(np.int32) << b)
        return arr

    ir1 = nibble_array("ir1_")
    ir2 = nibble_array("ir2_")
    ir = (ir1 << 4) | ir2
    acc = nibble_array("acc")
    cy = (sigs['v(cf0)'] > thresh).astype(np.int32)
    scratch = [nibble_array(f"scratch{r}_") for r in range(16)]

    # Find micro1 edges
    m1 = sigs['v(micro1)'] > thresh
    m1_edges = np.where(m1[1:] & ~m1[:-1])[0] + 1
    print(f"  {len(m1_edges)} micro1 edges")

    # For each micro1 edge, sample at 95% (near end of phase, after settling)
    # and get IR, ACC, CY, and all scratchpad values

    # Build list of (time, ir_val, acc_val, cy_val, regs[16]) at each edge
    print("Sampling at 95% through each micro1 phase...")
    edge_data = []
    for i in range(len(m1_edges) - 1):
        e = m1_edges[i]
        ne = m1_edges[i + 1]
        pt = e + int(0.95 * (ne - e))
        if pt >= raw.n_points:
            break
        t = raw.times[pt]
        edge_data.append({
            'time': t,
            'ir': int(ir[pt]),
            'acc': int(acc[pt]),
            'cy': int(cy[pt]),
            'regs': [int(scratch[r][pt]) for r in range(16)],
        })

    print(f"  {len(edge_data)} sampled edges")

    # Each micro1 edge = one instruction phase.
    # 2-word instructions (OPR=1,2,4,5,7 with conditions) use 2 edges.
    # Map each edge to an instruction.
    print("Building instruction list from micro1 edges...")

    instructions = []
    i = 0
    while i < len(edge_data):
        ir_val = edge_data[i]['ir']
        opr = (ir_val >> 4) & 0xF
        opa = ir_val & 0xF

        # Determine if this is a 2-word instruction
        two_word = (opr in (0x1, 0x4, 0x5, 0x7) or
                    (opr == 0x2 and (opa & 1) == 0))  # FIM (even OPA)

        if two_word and i + 1 < len(edge_data):
            # Use state from the second edge (end of 2-word instruction)
            instructions.append({
                'ir': ir_val,
                'state': edge_data[i + 1],
                'two_word': True,
            })
            i += 2
        else:
            instructions.append({
                'ir': ir_val,
                'state': edge_data[i],
                'two_word': False,
            })
            i += 1

    print(f"  {len(instructions)} instructions from {len(edge_data)} micro1 edges")

    # Run Python sim with JFET power-up state
    cpu = CPU4004Sim(rom)
    cpu.regs = [0xF] * 16
    cpu.acc = 0xF

    # Find a sync point: after init+data setup, at JMS MULT
    # The init is: JUN + 32 instr (16 LDM/XCH pairs) + 16 instr (8 data setup pairs) + JMS = ~50
    # Find by looking for t > 3ms where HW regs match expected data setup values
    sync_idx = None
    expected_regs = [0, 0, 0, 0, 0xF, 4, 4, 7, 0xA, 0, 4, 0xB, 0, 0, 0, 0]
    for idx in range(30, min(70, len(instructions))):
        hw = instructions[idx]['state']
        if hw['regs'] == expected_regs and hw['time'] > 3e-3:
            sync_idx = idx
            break

    if sync_idx is None:
        # Fallback: sync at ~4ms
        for idx in range(len(instructions)):
            if instructions[idx]['state']['time'] > 4e-3:
                sync_idx = idx
                break

    # Run Python sim to the same count
    for _ in range(sync_idx):
        cpu.execute_one()

    hw_state = instructions[sync_idx]['state']
    print(f"\nSync at instruction {sync_idx}: IR=0x{instructions[sync_idx]['ir']:02X}"
          f" t={hw_state['time']*1e3:.3f}ms")
    print(f"  HW:  ACC={hw_state['acc']:X} CY={hw_state['cy']} "
          f"regs=[{','.join(f'{r:X}' for r in hw_state['regs'])}]")
    print(f"  PY:  ACC={cpu.acc:X} CY={cpu.cy} "
          f"regs=[{','.join(f'{r:X}' for r in cpu.regs)}]")

    # Check if states match
    diffs = []
    if cpu.acc != hw_state['acc']: diffs.append(f"ACC")
    if cpu.cy != hw_state['cy']: diffs.append(f"CY")
    for r in range(16):
        if cpu.regs[r] != hw_state['regs'][r]: diffs.append(f"R{r}")
    if diffs:
        print(f"  DIFFS at sync: {', '.join(diffs)}")
        print(f"  Forcing PY = HW")
        cpu.acc = hw_state['acc']
        cpu.cy = hw_state['cy']
        cpu.regs = list(hw_state['regs'])
        cpu.pc = cpu.pc  # Keep Python PC (should be correct)
    else:
        print(f"  States MATCH at sync!")

    # Trace forward
    print(f"\n{'='*100}")
    print(f"Tracing from instruction {sync_idx + 1}")
    print(f"{'='*100}")

    divergence_count = 0
    first_div = None

    for idx in range(sync_idx + 1, min(len(instructions), sync_idx + 1 + args.max_cycles)):
        inst = instructions[idx]
        hw = inst['state']

        # Execute in Python sim
        cpu.execute_one()

        mnem = decode_mnemonic(inst['ir'])

        # Compare
        mismatches = []
        if hw['acc'] != cpu.acc:
            mismatches.append(f"ACC:hw={hw['acc']:X},py={cpu.acc:X}")
        if hw['cy'] != cpu.cy:
            mismatches.append(f"CY:hw={hw['cy']},py={cpu.cy}")
        for r in range(16):
            if hw['regs'][r] != cpu.regs[r]:
                mismatches.append(f"R{r}:hw={hw['regs'][r]:X},py={cpu.regs[r]:X}")

        n = idx - sync_idx
        if mismatches:
            divergence_count += 1
            marker = " <<<< FIRST" if first_div is None else ""
            if first_div is None:
                first_div = idx
            print(f"\n[{n:3d}] t={hw['time']*1e3:.3f}ms IR={inst['ir']:02X} {mnem:12s}{marker}")
            for m in mismatches:
                print(f"       {m}")
            if divergence_count <= 10:
                print(f"       HW: ACC={hw['acc']:X} CY={hw['cy']} [{','.join(f'{r:X}' for r in hw['regs'])}]")
                print(f"       PY: ACC={cpu.acc:X} CY={cpu.cy} [{','.join(f'{r:X}' for r in cpu.regs)}]")
                # Sync to isolate next error
                print(f"       (Syncing PY->HW)")
                cpu.acc = hw['acc']
                cpu.cy = hw['cy']
                cpu.regs = list(hw['regs'])
        else:
            if n <= 50 or n % 50 == 0:
                print(f"[{n:3d}] t={hw['time']*1e3:.3f}ms IR={inst['ir']:02X} {mnem:12s}"
                      f"  OK  ACC={hw['acc']:X} CY={hw['cy']}")

    print(f"\n{'='*100}")
    print(f"Traced {min(len(instructions), sync_idx + 1 + args.max_cycles) - sync_idx - 1} instructions")
    print(f"Divergences: {divergence_count}")
    if first_div is not None:
        print(f"First divergence at instruction {first_div} (offset {first_div - sync_idx})")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
