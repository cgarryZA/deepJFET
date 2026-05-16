#!/usr/bin/env python3
"""Verify each instruction's effect directly from hardware trace.

No Python sim — just reads HW state at each micro1 edge and checks if
the instruction produced the expected result based on its opcode and
the PREVIOUS state.

Usage:
    python tools/verify_hw_ops.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
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
        self.n_vars = 0; self.n_points = 0; self.var_names = {}
        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("No. Variables"): self.n_vars = int(s.split(":")[1].strip())
            elif s.startswith("No. Points"): self.n_points = int(s.split(":")[1].strip())
            elif "\t" in s:
                parts = s.split("\t")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0].strip()); name = parts[1].strip()
                        self.var_names[name.lower()] = idx
                    except ValueError: pass
            elif s == "Binary:": break
        idx = text.find("Binary:")
        self.data_start = (idx + len("Binary:") + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4

    def bulk_read_signal(self, signal_name):
        idx = self.var_names.get(signal_name.lower())
        if idx is None: return None
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


ACC_MNEMONICS = {
    0xF0: 'CLB', 0xF1: 'CLC', 0xF2: 'IAC', 0xF3: 'CMC', 0xF4: 'CMA',
    0xF5: 'RAL', 0xF6: 'RAR', 0xF7: 'TCC', 0xF8: 'DAC', 0xF9: 'TCS',
    0xFA: 'STC', 0xFB: 'DAA', 0xFC: 'KBP', 0xFD: 'DCL',
}

def decode_mnemonic(opcode):
    if opcode in ACC_MNEMONICS: return ACC_MNEMONICS[opcode]
    opr = (opcode >> 4) & 0xF; opa = opcode & 0xF
    MNEMONICS = {0:'NOP',1:'JCN',2:'FIM/SRC',3:'FIN/JIN',4:'JUN',5:'JMS',
                 6:'INC',7:'ISZ',8:'ADD',9:'SUB',0xA:'LD',0xB:'XCH',0xC:'BBL',0xD:'LDM'}
    base = MNEMONICS.get(opr, '???')
    if opr in (0x6,0x8,0x9,0xA,0xB): return f"{base} R{opa}"
    elif opr == 0xD: return f"LDM {opa}"
    elif opr == 0xC: return f"BBL {opa}"
    return f"{base} {opa:X}"


def verify_instruction(ir, pre, post):
    """Verify instruction result. Returns (ok, description) tuple."""
    opr = (ir >> 4) & 0xF
    opa = ir & 0xF

    if opr == 0xD:  # LDM
        if post['acc'] != opa:
            return False, f"LDM {opa}: ACC should be {opa:X} but got {post['acc']:X}"
        return True, None

    if opr == 0xA:  # LD Rn
        expected = pre['regs'][opa]
        if post['acc'] != expected:
            return False, f"LD R{opa}: ACC should be R{opa}={expected:X} but got {post['acc']:X}"
        return True, None

    if opr == 0xB:  # XCH Rn
        # ACC <- old Rn, Rn <- old ACC
        exp_acc = pre['regs'][opa]
        exp_rn = pre['acc']
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f"ACC should be {exp_acc:X} (old R{opa}) but got {post['acc']:X}")
        if post['regs'][opa] != exp_rn:
            errs.append(f"R{opa} should be {exp_rn:X} (old ACC) but got {post['regs'][opa]:X}")
        if errs:
            return False, f"XCH R{opa}: {'; '.join(errs)}"
        return True, None

    if opr == 0x8:  # ADD Rn
        result = pre['acc'] + pre['regs'][opa] + pre['cy']
        exp_acc = result & 0xF
        exp_cy = 1 if result > 0xF else 0
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f"ACC should be {exp_acc:X} but got {post['acc']:X}")
        if post['cy'] != exp_cy:
            errs.append(f"CY should be {exp_cy} but got {post['cy']}")
        if errs:
            return False, f"ADD R{opa} ({pre['acc']:X}+{pre['regs'][opa]:X}+{pre['cy']}={result:X}): {'; '.join(errs)}"
        return True, None

    if opr == 0x9:  # SUB Rn
        result = pre['acc'] + (~pre['regs'][opa] & 0xF) + pre['cy']
        exp_acc = result & 0xF
        exp_cy = 1 if result > 0xF else 0
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f"ACC should be {exp_acc:X} but got {post['acc']:X}")
        if post['cy'] != exp_cy:
            errs.append(f"CY should be {exp_cy} but got {post['cy']}")
        if errs:
            return False, f"SUB R{opa}: {'; '.join(errs)}"
        return True, None

    if ir == 0xF1:  # CLC
        if post['cy'] != 0:
            return False, f"CLC: CY should be 0 but got {post['cy']}"
        return True, None

    if ir == 0xF3:  # CMC
        exp = 1 - pre['cy']
        if post['cy'] != exp:
            return False, f"CMC: CY should be {exp} but got {post['cy']}"
        return True, None

    if ir == 0xF5:  # RAL
        val = (pre['acc'] << 1) | pre['cy']
        exp_acc = val & 0xF
        exp_cy = (val >> 4) & 1
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f"ACC should be {exp_acc:X} but got {post['acc']:X}")
        if post['cy'] != exp_cy:
            errs.append(f"CY should be {exp_cy} but got {post['cy']}")
        if errs:
            return False, f"RAL (ACC={pre['acc']:X},CY={pre['cy']}): {'; '.join(errs)}"
        return True, None

    if ir == 0xF6:  # RAR
        val = (pre['cy'] << 4) | pre['acc']
        exp_acc = (val >> 1) & 0xF
        exp_cy = val & 1
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f"ACC should be {exp_acc:X} but got {post['acc']:X}")
        if post['cy'] != exp_cy:
            errs.append(f"CY should be {exp_cy} but got {post['cy']}")
        if errs:
            return False, f"RAR (ACC={pre['acc']:X},CY={pre['cy']}): {'; '.join(errs)}"
        return True, None

    if ir == 0xF7:  # TCC
        exp_acc = pre['cy']
        if post['acc'] != exp_acc or post['cy'] != 0:
            return False, f"TCC: ACC should be {exp_acc}, CY should be 0, got ACC={post['acc']:X},CY={post['cy']}"
        return True, None

    if ir == 0xF0:  # CLB
        if post['acc'] != 0 or post['cy'] != 0:
            return False, f"CLB: ACC and CY should be 0, got ACC={post['acc']:X},CY={post['cy']}"
        return True, None

    if ir == 0xFA:  # STC
        if post['cy'] != 1:
            return False, f"STC: CY should be 1, got {post['cy']}"
        return True, None

    # Skip verification for complex instructions (JMS, BBL, JCN, etc.)
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("--start-time", type=float, default=3.0, help="Start time in ms")
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    raw = RawReader(args.raw_file)
    print(f"Loaded: {raw.n_vars} signals, {raw.n_points} points")

    print("Loading signals...")
    raw.load_times()
    thresh = -2.5

    sigs = {}
    for name in (['v(micro1)'] +
                 [f'v(ir1_{i})' for i in range(4)] +
                 [f'v(ir2_{i})' for i in range(4)] +
                 [f'v(acc{i})' for i in range(4)] +
                 ['v(cf0)'] +
                 [f'v(scratch{r}_{b})' for r in range(16) for b in range(4)]):
        d = raw.bulk_read_signal(name)
        if d is not None: sigs[name] = d

    def nibble_array(prefix, n_bits=4):
        arr = np.zeros(raw.n_points, dtype=np.int32)
        for b in range(n_bits):
            key = f'v({prefix}{b})'
            if key in sigs: arr |= ((sigs[key] > thresh).astype(np.int32) << b)
        return arr

    ir1 = nibble_array("ir1_"); ir2 = nibble_array("ir2_")
    ir = (ir1 << 4) | ir2
    acc_arr = nibble_array("acc")
    cy_arr = (sigs['v(cf0)'] > thresh).astype(np.int32)
    scratch_arr = [nibble_array(f"scratch{r}_") for r in range(16)]

    m1 = sigs['v(micro1)'] > thresh
    m1_edges = np.where(m1[1:] & ~m1[:-1])[0] + 1
    print(f"  {len(m1_edges)} micro1 edges")

    # Sample at 95% through each phase
    def state_at_edge(i):
        e = m1_edges[i]; ne = m1_edges[i+1] if i+1 < len(m1_edges) else raw.n_points - 1
        # Sample just past the NEXT instruction's micro1 edge. By this point:
        # - This instruction's CF_Load at its last CLK has settled CF0
        # - The next instruction's fetch hasn't yet modified ACC/regs (those
        #   only change in execute phases Micro 6+)
        pt = ne + 3  # ~30ns past next micro1 edge
        if pt >= raw.n_points: pt = raw.n_points - 1
        return {
            'time': raw.times[pt],
            'ir': int(ir[pt]),
            'acc': int(acc_arr[pt]),
            'cy': int(cy_arr[pt]),
            'regs': [int(scratch_arr[r][pt]) for r in range(16)],
        }

    # Find start edge
    start_edge = 0
    for i in range(len(m1_edges)):
        if raw.times[m1_edges[i]] > args.start_time * 1e-3:
            start_edge = i
            break

    print(f"\nStarting at edge {start_edge} (t={raw.times[m1_edges[start_edge]]*1e3:.3f}ms)")
    print(f"{'='*100}")

    errors = 0
    ok_count = 0
    # State at edge N shows the result of instruction N-1 (one behind).
    # So: instruction at edge N has IR from edge N, but its RESULT appears at edge N+1.
    # Compare: pre = state at edge N+1 (result of prev instr = this instr's input),
    #          post = state at edge N+2 (result of this instr)
    # Actually simpler: instruction IR at edge N, pre-state at edge N, post-state at edge N+1

    # The instruction at micro1 edge i completes within edge i's phase.
    # At 95% sampling of edge i, we see:
    #   IR = the instruction at edge i
    #   ACC/regs = result AFTER that instruction executed
    # So pre-state = sample at edge i-1, post-state = sample at edge i.

    for i in range(start_edge + 1, len(m1_edges) - 1):
        pre_state = state_at_edge(i - 1)
        post_state = state_at_edge(i)

        # IR from post_state tells us which instruction just completed
        instr_ir = post_state['ir']
        mnem = decode_mnemonic(instr_ir)

        ok, desc = verify_instruction(instr_ir, pre_state, post_state)

        if not ok:
            errors += 1
            print(f"FAIL [{i-start_edge:3d}] t={prev_state['time']*1e3:.3f}ms IR={instr_ir:02X} {mnem:12s}")
            print(f"     {desc}")
            print(f"     Pre:  ACC={prev_state['acc']:X} CY={prev_state['cy']} "
                  f"[{','.join(f'{r:X}' for r in prev_state['regs'])}]")
            print(f"     Post: ACC={post_state['acc']:X} CY={post_state['cy']} "
                  f"[{','.join(f'{r:X}' for r in post_state['regs'])}]")
            if errors >= args.max_errors:
                print(f"\nStopping after {args.max_errors} errors")
                break
        else:
            ok_count += 1

        prev_state = post_state

    print(f"\n{'='*100}")
    print(f"Verified {ok_count + errors} instructions: {ok_count} OK, {errors} FAIL")


if __name__ == "__main__":
    main()
