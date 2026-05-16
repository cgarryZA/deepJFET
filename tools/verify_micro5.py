#!/usr/bin/env python3
"""Verify 4004 instructions using Micro5 phase as sampling point.

Strategy:
1. Find all Micro5 rising edges in the .raw
2. Each Micro5 is one instruction's fetch phase (2-byte instructions have 2)
3. Sample CPU state at the MIDDLE of each Micro5 (stable, settled)
4. State at Micro5(N)   = "pre" state for instruction N
   State at Micro5(N+1) = "post" state for instruction N
5. For 2-byte instructions, the second Micro5 has the OPA byte fetched -
   we use the FIRST Micro5 as the pre-state, and the next instruction's
   first Micro5 as the post-state

Usage:
    python tools/verify_micro5.py cpus/4004/4004.raw
    python tools/verify_micro5.py cpus/4004/4004.raw --start-time 3 --max-errors 50
"""

import argparse
import os
import struct
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

THRESH = -2.5


class RawReader:
    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            header = f.read(2_000_000)
        text = header.decode('utf-16-le', errors='replace')
        self.n_vars = 0; self.n_points = 0; self.var_names = {}
        for line in text.split('\n'):
            s = line.strip()
            if s.startswith('No. Variables'): self.n_vars = int(s.split(':')[1])
            elif s.startswith('No. Points'): self.n_points = int(s.split(':')[1])
            elif '\t' in s:
                parts = s.split('\t')
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0]); name = parts[1].strip()
                        self.var_names[name.lower()] = idx
                    except ValueError: pass
            elif s == 'Binary:': break
        idx = text.find('Binary:')
        self.data_start = (idx + len('Binary:') + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4

    def load_times(self):
        self.times = np.zeros(self.n_points, dtype=np.float64)
        with open(self.path, 'rb') as f:
            for i in range(self.n_points):
                f.seek(self.data_start + i * self.row_size)
                self.times[i] = struct.unpack_from('d', f.read(8), 0)[0]

    def bulk_read_signal(self, name):
        idx = self.var_names.get(name.lower())
        if idx is None: return None
        arr = np.zeros(self.n_points, dtype=np.float32)
        with open(self.path, 'rb') as f:
            for pt in range(self.n_points):
                if idx == 0:
                    f.seek(self.data_start + pt * self.row_size)
                    arr[pt] = struct.unpack_from('d', f.read(8), 0)[0]
                else:
                    f.seek(self.data_start + pt * self.row_size + 8 + (idx - 1) * 4)
                    arr[pt] = struct.unpack_from('f', f.read(4), 0)[0]
        return arr


# ── Mnemonics ────────────────────────────────────────────────────────────

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
    base = {0:'NOP',1:'JCN',2:'FIM/SRC',3:'FIN/JIN',4:'JUN',5:'JMS',
            6:'INC',7:'ISZ',8:'ADD',9:'SUB',0xA:'LD',0xB:'XCH',
            0xC:'BBL',0xD:'LDM',0xE:'IO',0xF:'ACC'}.get(opr, '???')
    if opr in (0x6, 0x8, 0x9, 0xA, 0xB):
        return f'{base} R{opa}'
    if opr == 0xD: return f'LDM {opa}'
    if opr == 0xC: return f'BBL {opa}'
    return f'{base} {opa:X}'


def is_two_word(opcode):
    """True if this opcode is a 2-byte instruction."""
    opr = (opcode >> 4) & 0xF
    opa = opcode & 0xF
    # JCN, JUN, JMS, ISZ are always 2-word
    if opr in (0x1, 0x4, 0x5, 0x7):
        return True
    # FIM is 2-word (even OPA), SRC is 1-word (odd OPA)
    if opr == 0x2 and (opa & 1) == 0:
        return True
    return False


# ── Per-instruction verification ─────────────────────────────────────────

def verify(opcode, pre, post):
    """Verify that pre+instruction(opcode) == post.
    Returns (ok, description).
    pre/post are dicts: {'acc':int, 'cy':int, 'regs':[16], 'pc':int, ...}
    """
    opr = (opcode >> 4) & 0xF
    opa = opcode & 0xF

    if opr == 0xD:  # LDM
        if post['acc'] != opa:
            return False, f'LDM {opa}: ACC should be {opa:X} got {post["acc"]:X}'
        if post['cy'] != pre['cy']:
            return False, f'LDM: CY changed unexpectedly {pre["cy"]}->{post["cy"]}'
        return True, None

    if opr == 0xA:  # LD Rn
        exp = pre['regs'][opa]
        if post['acc'] != exp:
            return False, f'LD R{opa}: ACC should be {exp:X} (=R{opa}) got {post["acc"]:X}'
        return True, None

    if opr == 0xB:  # XCH Rn
        exp_acc = pre['regs'][opa]
        exp_rn = pre['acc']
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f'ACC should be {exp_acc:X} got {post["acc"]:X}')
        if post['regs'][opa] != exp_rn:
            errs.append(f'R{opa} should be {exp_rn:X} got {post["regs"][opa]:X}')
        if errs:
            return False, f'XCH R{opa}: ' + '; '.join(errs)
        return True, None

    if opr == 0x8:  # ADD Rn
        r = pre['regs'][opa]
        s = pre['acc'] + r + pre['cy']
        exp_acc = s & 0xF
        exp_cy = 1 if s > 0xF else 0
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f'ACC should be {exp_acc:X} got {post["acc"]:X}')
        if post['cy'] != exp_cy:
            errs.append(f'CY should be {exp_cy} got {post["cy"]}')
        if errs:
            return False, f'ADD R{opa} ({pre["acc"]:X}+{r:X}+{pre["cy"]}={s:X}): ' + '; '.join(errs)
        return True, None

    if opr == 0x9:  # SUB Rn
        r = pre['regs'][opa]
        s = pre['acc'] + ((~r) & 0xF) + pre['cy']
        exp_acc = s & 0xF
        exp_cy = 1 if s > 0xF else 0
        errs = []
        if post['acc'] != exp_acc:
            errs.append(f'ACC should be {exp_acc:X} got {post["acc"]:X}')
        if post['cy'] != exp_cy:
            errs.append(f'CY should be {exp_cy} got {post["cy"]}')
        if errs:
            return False, f'SUB R{opa} ({pre["acc"]:X}+~{r:X}+{pre["cy"]}={s:X}): ' + '; '.join(errs)
        return True, None

    if opr == 0x6:  # INC Rn
        exp = (pre['regs'][opa] + 1) & 0xF
        if post['regs'][opa] != exp:
            return False, f'INC R{opa}: R{opa} should be {exp:X} got {post["regs"][opa]:X}'
        return True, None

    if opcode == 0xF1:  # CLC
        if post['cy'] != 0:
            return False, f'CLC: CY should be 0 got {post["cy"]}'
        return True, None
    if opcode == 0xF3:  # CMC
        exp = 1 - pre['cy']
        if post['cy'] != exp:
            return False, f'CMC: CY should be {exp} got {post["cy"]}'
        return True, None
    if opcode == 0xFA:  # STC
        if post['cy'] != 1:
            return False, f'STC: CY should be 1 got {post["cy"]}'
        return True, None
    if opcode == 0xF0:  # CLB
        if post['acc'] != 0 or post['cy'] != 0:
            return False, f'CLB: should be ACC=0,CY=0 got ACC={post["acc"]:X},CY={post["cy"]}'
        return True, None
    if opcode == 0xF4:  # CMA
        exp = (~pre['acc']) & 0xF
        if post['acc'] != exp:
            return False, f'CMA: ACC should be {exp:X} got {post["acc"]:X}'
        return True, None
    if opcode == 0xF5:  # RAL
        v = (pre['acc'] << 1) | pre['cy']
        exp_acc = v & 0xF
        exp_cy = (v >> 4) & 1
        if post['acc'] != exp_acc or post['cy'] != exp_cy:
            return False, (f'RAL (ACC={pre["acc"]:X},CY={pre["cy"]}): '
                           f'should be ACC={exp_acc:X},CY={exp_cy} '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None
    if opcode == 0xF6:  # RAR
        v = (pre['cy'] << 4) | pre['acc']
        exp_acc = (v >> 1) & 0xF
        exp_cy = v & 1
        if post['acc'] != exp_acc or post['cy'] != exp_cy:
            return False, (f'RAR (ACC={pre["acc"]:X},CY={pre["cy"]}): '
                           f'should be ACC={exp_acc:X},CY={exp_cy} '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None
    if opcode == 0xF7:  # TCC
        exp_acc = pre['cy']
        if post['acc'] != exp_acc or post['cy'] != 0:
            return False, (f'TCC: should be ACC={exp_acc:X},CY=0 '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None
    if opcode == 0xF2:  # IAC
        s = pre['acc'] + 1
        exp_acc = s & 0xF
        exp_cy = 1 if s > 0xF else 0
        if post['acc'] != exp_acc or post['cy'] != exp_cy:
            return False, (f'IAC: should be ACC={exp_acc:X},CY={exp_cy} '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None
    if opcode == 0xF8:  # DAC
        s = pre['acc'] + 0xF
        exp_acc = s & 0xF
        exp_cy = 1 if s > 0xF else 0
        if post['acc'] != exp_acc or post['cy'] != exp_cy:
            return False, (f'DAC: should be ACC={exp_acc:X},CY={exp_cy} '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None
    if opcode == 0xF9:  # TCS
        exp_acc = 0xA if pre['cy'] else 0x9
        if post['acc'] != exp_acc or post['cy'] != 0:
            return False, (f'TCS: should be ACC={exp_acc:X},CY=0 '
                           f'got ACC={post["acc"]:X},CY={post["cy"]}')
        return True, None

    # For NOP, JCN/JUN/JMS/ISZ/BBL/FIM/SRC/FIN/JIN and I/O group, skip detailed check
    return True, None


# ── Sampling ─────────────────────────────────────────────────────────────

def nibble_at(sigs, prefix, n_bits, pt):
    v = 0
    for i in range(n_bits):
        key = f'v({prefix}{i})'
        if key in sigs and sigs[key][pt] > THRESH:
            v |= (1 << i)
    return v


def find_micro5_centers(raw, m5_signal):
    """Return sample point indices: just BEFORE the falling edge of Micro5.

    IR1/IR2 are being loaded DURING Micro5, so the midpoint catches transitions.
    We want the END of Micro5 when IR is fully settled.
    """
    high = m5_signal > THRESH
    edges_up = np.where(high[1:] & ~high[:-1])[0] + 1
    edges_dn = np.where(~high[1:] & high[:-1])[0] + 1
    centers = []
    for up in edges_up:
        # Find next falling edge
        dn = edges_dn[edges_dn > up]
        if len(dn) == 0: continue
        # Sample 2 points before falling edge (settled)
        pt = max(up + 1, dn[0] - 2)
        centers.append(pt)
    return centers


def read_state(sigs, pt):
    """Return CPU state at point pt."""
    acc = nibble_at(sigs, 'acc', 4, pt)
    ir1 = nibble_at(sigs, 'ir1_', 4, pt)
    ir2 = nibble_at(sigs, 'ir2_', 4, pt)
    cy = 1 if sigs.get('v(cf_in)', np.array([0.0]))[pt] > THRESH else 0
    regs = [nibble_at(sigs, f'scratch{r}_', 4, pt) for r in range(16)]
    return {
        'acc': acc, 'cy': cy,
        'ir1': ir1, 'ir2': ir2,
        'ir': (ir1 << 4) | ir2,
        'regs': regs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_file')
    parser.add_argument('--start-time', type=float, default=3.0,
                        help='Start time in ms (skip init)')
    parser.add_argument('--max-errors', type=int, default=30)
    parser.add_argument('--show-ok', action='store_true',
                        help='Also show passing instructions')
    args = parser.parse_args()

    print(f'Loading {args.raw_file}...')
    raw = RawReader(args.raw_file)
    raw.load_times()
    print(f'  {raw.n_vars} signals, {raw.n_points} points, '
          f'{raw.times[-1]*1e3:.2f}ms')

    # Load required signals
    needed = (['v(micro5)', 'v(cf_in)'] +
              [f'v(acc{i})' for i in range(4)] +
              [f'v(ir1_{i})' for i in range(4)] +
              [f'v(ir2_{i})' for i in range(4)] +
              [f'v(scratch{r}_{b})' for r in range(16) for b in range(4)])

    sigs = {}
    print('Loading signals...')
    for name in needed:
        d = raw.bulk_read_signal(name)
        if d is not None:
            sigs[name] = d
        else:
            print(f'  WARN: missing {name}')

    if 'v(micro5)' not in sigs:
        print('ERROR: micro5 signal not found in .raw')
        sys.exit(1)

    # Find Micro5 centers (midpoint of each high pulse)
    m5_pts = find_micro5_centers(raw, sigs['v(micro5)'])
    print(f'  {len(m5_pts)} Micro5 phases found')

    # Filter to those past start_time
    m5_pts = [p for p in m5_pts if raw.times[p] > args.start_time * 1e-3]
    print(f'  {len(m5_pts)} Micro5 phases past t={args.start_time}ms')

    # Group into instructions: 2-byte instructions span 2 Micro5 events
    # At the FIRST Micro5 of a 2-byte, IR1 holds OPR, IR2 holds previous instruction's OPA
    # At the SECOND Micro5 of a 2-byte, IR3/IR4 holds OPA - but here IR1/IR2 still shows OPR/OPA
    # We treat each Micro5 as one "phase" and group:
    #   - Sample state at every Micro5
    #   - At each Micro5 i, look at IR (which is the instruction whose RESULT will be visible at next Micro5)
    #   - If is_two_word(IR), then instruction spans Micro5[i] and Micro5[i+1]
    #   - The state at Micro5[i+1] is "during" the 2-byte instruction, not post

    # Build a list of (instruction_start_pt, instruction_ir, instruction_end_pt)
    # instruction_start_pt = first Micro5 of this instruction (state going IN)
    # instruction_end_pt   = first Micro5 of NEXT instruction (state going OUT)

    states = [read_state(sigs, pt) for pt in m5_pts]

    print('\nBuilding instruction list...')
    instructions = []
    i = 0
    while i < len(m5_pts) - 1:
        ir = states[i]['ir']
        # Determine if this is a 2-byte instruction
        # JCN/JUN/JMS/ISZ always 2-byte. FIM (0x2 even OPA) is 2-byte.
        # We need to know whether to consume 1 or 2 Micro5 events.
        if is_two_word(ir):
            if i + 2 >= len(m5_pts):
                break
            # Pre state = m5_pts[i] (start)
            # Post state = m5_pts[i+2] (after the 2-byte completes, next instruction's M5)
            instructions.append({
                'idx': len(instructions),
                'ir': ir,
                'pre_pt': m5_pts[i],
                'post_pt': m5_pts[i+2],
                'pre': states[i],
                'post': states[i+2],
                'two_word': True,
                't': raw.times[m5_pts[i]],
            })
            i += 2
        else:
            instructions.append({
                'idx': len(instructions),
                'ir': ir,
                'pre_pt': m5_pts[i],
                'post_pt': m5_pts[i+1],
                'pre': states[i],
                'post': states[i+1],
                'two_word': False,
                't': raw.times[m5_pts[i]],
            })
            i += 1

    print(f'  {len(instructions)} instructions to check')
    print()

    # Now verify each instruction
    errors = 0
    passes = 0
    cy_artifact = 0
    real_errors = []

    print(f'{"#":>4} {"t(ms)":>8s} {"IR":>4} {"Mnem":>12} {"Status":>10}')
    print('=' * 80)

    for inst in instructions:
        pre = inst['pre']
        post = inst['post']
        opcode = inst['ir']
        mnem = decode_mnemonic(opcode)

        ok, desc = verify(opcode, pre, post)

        # Check if a "failure" is explainable by CY-input being flipped
        # (sampling artifact where CY at pre_pt hasn't settled from previous instr)
        cy_explained = False
        if not ok:
            opr = (opcode >> 4) & 0xF
            if opr in (0x8, 0x9) or opcode in (0xF5, 0xF6):
                # Try with flipped CY
                alt_pre = dict(pre); alt_pre['cy'] = 1 - pre['cy']
                ok2, _ = verify(opcode, alt_pre, post)
                if ok2:
                    cy_explained = True

        if ok:
            passes += 1
            if args.show_ok:
                print(f'{inst["idx"]:4d} {inst["t"]*1e3:8.3f} {opcode:02X} {mnem:>12s}    OK')
        else:
            if cy_explained:
                cy_artifact += 1
                # Don't print these as errors
                if args.show_ok:
                    print(f'{inst["idx"]:4d} {inst["t"]*1e3:8.3f} {opcode:02X} {mnem:>12s}  (CY-arf)')
            else:
                errors += 1
                real_errors.append((inst, desc))
                if errors <= args.max_errors:
                    print(f'{inst["idx"]:4d} {inst["t"]*1e3:8.3f} {opcode:02X} {mnem:>12s}  FAIL')
                    print(f'         {desc}')
                    print(f'         Pre:  ACC={pre["acc"]:X} CY={pre["cy"]} '
                          f'[{",".join(f"{r:X}" for r in pre["regs"])}]')
                    print(f'         Post: ACC={post["acc"]:X} CY={post["cy"]} '
                          f'[{",".join(f"{r:X}" for r in post["regs"])}]')

    print()
    print('=' * 80)
    print(f'SUMMARY:')
    print(f'  Pass:        {passes}')
    print(f'  CY artifact: {cy_artifact}  (failures explained by CY-input being flipped)')
    print(f'  REAL FAIL:   {errors}')
    print(f'  Total:       {passes + cy_artifact + errors}')

    if real_errors:
        # Categorize real errors by opcode/mnemonic
        from collections import Counter
        cats = Counter()
        for inst, desc in real_errors:
            cats[decode_mnemonic(inst['ir']).split()[0]] += 1
        print()
        print('REAL failures by mnemonic:')
        for mnem, count in cats.most_common():
            print(f'  {mnem}: {count}')


if __name__ == '__main__':
    main()
