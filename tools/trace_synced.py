#!/usr/bin/env python3
"""Register-level co-validation of an LTspice 4004 trace vs Python reference.

Samples each micro1 phase at 95% (after Index/IR have settled), syncs
the Python emulator to the LTspice CPU at a known good point, then
walks forward comparing ACC, CY, and all sixteen scratch-pad registers
at every instruction boundary.

Two CLI styles, both supported:

  1) Legacy positional .raw path -- behaviour unchanged from before:
       python tools/trace_synced.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
     Hardcoded FloatingPoint sync values are used. Output is stdout only.

  2) Program-aware mode (new): emits paper-ready tables:
       python tools/trace_synced.py --program BitwiseAND
     Looks under cpus/<cpu>/programs/<name>/ for the .raw and
     expected_state.yaml. Writes per-program CSV/TeX/macros under
     paper/data/.
"""

import argparse
import csv
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = Path(__file__).resolve().parent

# Load rom_emulator.py directly so this script works regardless of cwd or
# the presence of a `tools/__init__.py`.
_spec = importlib.util.spec_from_file_location(
    "_rom_emulator", str(TOOLS_DIR / "rom_emulator.py"))
_rom_emulator = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rom_emulator)
assemble = _rom_emulator.assemble
CPU4004Sim = _rom_emulator.CPU4004Sim

# ---------------------------------------------------------------- constants

THRESH_V = -2.5  # logic threshold for SiC JFET output (volts)
POWERUP_REGS = [0xF] * 16
POWERUP_ACC = 0xF
POWERUP_CY = 0

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


def decode_mnemonic(opcode: int) -> str:
    if opcode in ACC_MNEMONICS:
        return ACC_MNEMONICS[opcode]
    opr = (opcode >> 4) & 0xF
    opa = opcode & 0xF
    base = MNEMONICS.get(opr, '???')
    if opr in (0x6, 0x8, 0x9, 0xA, 0xB):
        return f"{base} R{opa}"
    if opr == 0xD:
        return f"LDM {opa}"
    if opr == 0xC:
        return f"BBL {opa}"
    return f"{base} {opa:X}"


# ---------------------------------------------------------------- LTspice raw

class RawReader:
    """Minimal binary LTspice .raw reader. Supports both single and
    double precision sample formats; signals are returned as float32."""

    def __init__(self, path: Path):
        self.path = Path(path)
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

    def bulk_read_signal(self, signal_name: str):
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


# ---------------------------------------------------------------- core trace

def build_instructions(raw: RawReader):
    """Sample each micro1 phase at 95% and pack registers into nibbles."""
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

    def nibble_array(prefix, n_bits=4):
        arr = np.zeros(raw.n_points, dtype=np.int32)
        for b in range(n_bits):
            key = f'v({prefix}{b})'
            if key in sigs:
                arr |= ((sigs[key] > THRESH_V).astype(np.int32) << b)
        return arr

    ir1 = nibble_array("ir1_")
    ir2 = nibble_array("ir2_")
    ir = (ir1 << 4) | ir2
    acc = nibble_array("acc")
    cy = (sigs['v(cf0)'] > THRESH_V).astype(np.int32)
    scratch = [nibble_array(f"scratch{r}_") for r in range(16)]

    m1 = sigs['v(micro1)'] > THRESH_V
    m1_edges = np.where(m1[1:] & ~m1[:-1])[0] + 1

    edges = []
    for i in range(len(m1_edges) - 1):
        e = m1_edges[i]
        ne = m1_edges[i + 1]
        pt = e + int(0.95 * (ne - e))
        if pt >= raw.n_points:
            break
        edges.append({
            'time':  raw.times[pt],
            'ir':    int(ir[pt]),
            'acc':   int(acc[pt]),
            'cy':    int(cy[pt]),
            'regs':  [int(scratch[r][pt]) for r in range(16)],
        })

    # Merge 2-word instructions into single boundaries (state from 2nd edge).
    instructions = []
    i = 0
    while i < len(edges):
        ir_val = edges[i]['ir']
        opr = (ir_val >> 4) & 0xF
        opa = ir_val & 0xF
        two_word = (opr in (0x1, 0x4, 0x5, 0x7) or
                    (opr == 0x2 and (opa & 1) == 0))
        if two_word and i + 1 < len(edges):
            instructions.append({'ir': ir_val, 'state': edges[i + 1], 'two_word': True})
            i += 2
        else:
            instructions.append({'ir': ir_val, 'state': edges[i], 'two_word': False})
            i += 1
    return instructions


def find_sync(instructions, expected_regs, t_min):
    """Locate the first instruction boundary whose state matches the
    expected sync-point register vector and is past the settle time."""
    for idx in range(len(instructions)):
        hw = instructions[idx]['state']
        if hw['regs'] == expected_regs and hw['time'] > t_min:
            return idx
    # Fallback: first instruction past t_min.
    for idx in range(len(instructions)):
        if instructions[idx]['state']['time'] > t_min:
            return idx
    return None


def trace_forward(cpu, instructions, sync_idx, max_cycles, *,
                  resync="never", verbose=True):
    """Run CPU sim in lockstep with LTspice from sync_idx forward.

    resync:
        "never"          -- the headline validation mode. After a
                            mismatch, the Python emulator continues
                            from its own state, so any cascading
                            divergence is exposed in the diff.
        "on-divergence"  -- diagnostic mode: after each mismatch, force
                            the emulator to the LTspice state so the
                            next mismatch is independent of the first.

    Returns a list of per-instruction dicts:
        {cycle, t_ms, ir, mnemonic, hw_acc, py_acc, hw_cy, py_cy,
         hw_regs, py_regs, match, mismatches}
    """
    rows = []
    end_idx = min(len(instructions), sync_idx + 1 + max_cycles)
    for idx in range(sync_idx + 1, end_idx):
        inst = instructions[idx]
        hw = inst['state']
        cpu.execute_one()
        mnem = decode_mnemonic(inst['ir'])

        mismatches = []
        if hw['acc'] != cpu.acc:
            mismatches.append(f"ACC:hw={hw['acc']:X},py={cpu.acc:X}")
        if hw['cy'] != cpu.cy:
            mismatches.append(f"CY:hw={hw['cy']},py={cpu.cy}")
        for r in range(16):
            if hw['regs'][r] != cpu.regs[r]:
                mismatches.append(f"R{r}:hw={hw['regs'][r]:X},py={cpu.regs[r]:X}")

        rows.append({
            'cycle':       idx - sync_idx,
            't_ms':        hw['time'] * 1e3,
            'ir':          inst['ir'],
            'mnemonic':    mnem,
            'hw_acc':      hw['acc'],
            'py_acc':      cpu.acc,
            'hw_cy':       hw['cy'],
            'py_cy':       cpu.cy,
            'hw_regs':     list(hw['regs']),
            'py_regs':     list(cpu.regs),
            'match':       not mismatches,
            'mismatches':  mismatches,
        })

        if mismatches and verbose:
            print(f"\n[{idx - sync_idx:3d}] t={hw['time']*1e3:.3f}ms "
                  f"IR={inst['ir']:02X} {mnem:12s}")
            for m in mismatches:
                print(f"       {m}")

        if mismatches and resync == "on-divergence":
            # Diagnostic: force emulator to LTspice state so the next
            # mismatch is independent of this one. NOT used for the
            # headline pass/fail counts reported in the paper.
            cpu.acc = hw['acc']
            cpu.cy = hw['cy']
            cpu.regs = list(hw['regs'])

    return rows


# ---------------------------------------------------------------- output


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'cycle', 't_ms', 'ir_hex', 'mnemonic',
        'acc_hw', 'acc_py', 'cy_hw', 'cy_py',
    ] + [f'r{r}_hw' for r in range(16)] + [f'r{r}_py' for r in range(16)] + [
        'match', 'mismatches',
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            row = [
                r['cycle'], f"{r['t_ms']:.3f}", f"{r['ir']:02X}", r['mnemonic'],
                r['hw_acc'], r['py_acc'], r['hw_cy'], r['py_cy'],
            ] + r['hw_regs'] + r['py_regs'] + [
                'Y' if r['match'] else 'N', ';'.join(r['mismatches']),
            ]
            w.writerow(row)


def write_tex(rows, program: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_traced = len(rows)
    n_div = sum(1 for r in rows if not r['match'])
    first_div = next((r['cycle'] for r in rows if not r['match']), None)
    if n_div == 0:
        status = r"\textbf{matched}"
        first_div_str = "--"
    else:
        status = rf"\textbf{{{n_div} divergence{'s' if n_div != 1 else ''}}}"
        first_div_str = str(first_div)
    out = [
        f"% Auto-generated by tools/trace_synced.py --program {program}",
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Register-level co-simulation of the LTspice CPU "
        rf"and the Python reference emulator for the {program} program. "
        rf"Every instruction boundary between the documented sync point "
        rf"and program halt is compared in (ACC, CY, R$_0\ldots$R$_{{15}}$).}}",
        rf"\label{{tab:trace-diff-{program.lower()}}}",
        r"\begin{tabular}{@{}lrrl@{}}",
        r"\toprule",
        r"Program & Instructions traced & First divergence & End state \\",
        r"\midrule",
        rf"{program} & {n_traced} & {first_div_str} & {status} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def write_macros(rows, program: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_traced = len(rows)
    n_div = sum(1 for r in rows if not r['match'])
    out = [
        f"% Auto-generated by tools/trace_synced.py --program {program}",
        rf"\newcommand{{\TraceCycles{program}}}{{{n_traced}}}",
        rf"\newcommand{{\TraceDivergences{program}}}{{{n_div}}}",
    ]
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- YAML loader

def parse_yaml_simple(text: str) -> dict:
    """Tiny YAML subset just for expected_state.yaml.

    Supports: top-level scalars, nested scalars (one level deep),
    inline lists `[a, b, c]`, comments, blank lines. No string quoting
    is preserved -- everything is interpreted as int when possible,
    else returned as the raw string.
    """
    out = {}
    current = out
    parent_stack = [out]
    indent_stack = [-1]

    def coerce(s: str):
        s = s.strip()
        if s.startswith("[") and s.endswith("]"):
            return [coerce(x) for x in s[1:-1].split(",") if x.strip()]
        try:
            return int(s, 0)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s.strip()

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        # Pop back if dedented
        while indent <= indent_stack[-1]:
            parent_stack.pop()
            indent_stack.pop()
        current = parent_stack[-1]
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, _, val = stripped.partition(":")
        key = key.strip()
        val = val.strip()
        if not val:
            current[key] = {}
            parent_stack.append(current[key])
            indent_stack.append(indent)
        else:
            current[key] = coerce(val)
    return out


# ---------------------------------------------------------------- program mode

def resolve_program(cpu: str, program: str) -> dict:
    """Locate program directory, .raw file, and expected_state.yaml."""
    program_dir = PROJECT_ROOT / "cpus" / cpu / "programs" / program
    if not program_dir.is_dir():
        sys.exit(f"ERROR: {program_dir} does not exist")

    raws = sorted(program_dir.glob("*.raw"))
    if not raws:
        sys.exit(f"ERROR: no .raw file in {program_dir}.\n"
                 f"       Run LTspice transient on cpus/{cpu}/4004.asc "
                 f"with this program loaded, then save the .raw here.")
    # Prefer an _extracted.raw if present (smaller), else the largest.
    extracted = [r for r in raws if "_extracted" in r.stem and "smoothed" not in r.stem]
    raw_path = extracted[0] if extracted else max(raws, key=lambda p: p.stat().st_size)

    asm_path = next(iter(program_dir.glob("*.asm")), None)
    if asm_path is None:
        sys.exit(f"ERROR: no .asm in {program_dir}")

    yaml_path = program_dir / "expected_state.yaml"
    if yaml_path.is_file():
        yaml_data = parse_yaml_simple(yaml_path.read_text(encoding="utf-8"))
    else:
        yaml_data = None

    return {
        'program':   program,
        'cpu':       cpu,
        'dir':       program_dir,
        'raw':       raw_path,
        'asm':       asm_path,
        'yaml':      yaml_data,
    }


# ---------------------------------------------------------------- legacy defaults

# Hardcoded FloatingPoint sync info, used when only a positional .raw
# path is given (no --program flag). Lifted from the pre-refactor
# version of this script so existing CLI invocations keep working.
FP_LEGACY_SYNC = {
    'after_time_ms': 3.0,
    'expected_regs': [0, 0, 0, 0, 0xF, 4, 4, 7, 0xA, 0, 4, 0xB, 0, 0, 0, 0],
}


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("raw_file", nargs="?", default=None,
                   help="path to LTspice .raw file (legacy positional arg)")
    p.add_argument("--cpu", default="4004",
                   help="CPU project name under cpus/ (program-mode only)")
    p.add_argument("--program", default=None,
                   help="program name under cpus/<cpu>/programs/")
    p.add_argument("--asm", default=None,
                   help="explicit asm path (legacy override)")
    p.add_argument("--max-cycles", type=int, default=20000)
    p.add_argument("--csv", default=None,
                   help="CSV output path (default: paper/data/trace_diff_<program>.csv)")
    p.add_argument("--tex", default=None,
                   help="LaTeX summary output path "
                        "(default: paper/data/trace_diff_<program>.tex)")
    p.add_argument("--macros", default=None,
                   help="macros file output path "
                        "(default: paper/data/trace_diff_<program>_macros.tex)")
    p.add_argument("--resync", choices=("never", "on-divergence"),
                   default="never",
                   help="emulator resync policy after a mismatch "
                        "(default: never -- headline validation mode; "
                        "on-divergence is diagnostic only)")
    p.add_argument("--no-paper", action="store_true",
                   help="don't write CSV/TeX/macros (stdout summary only)")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-divergence stdout chatter")
    args = p.parse_args()

    if not args.program and not args.raw_file:
        p.error("either --program or a positional raw_file is required")

    if args.program:
        info = resolve_program(args.cpu, args.program)
        raw_path = info['raw']
        asm_path = info['asm']
        yaml = info['yaml']
        program = info['program']
    else:
        raw_path = Path(args.raw_file)
        if args.asm:
            asm_path = Path(args.asm)
        else:
            raw_dir = raw_path.parent
            cands = list(raw_dir.glob("*.asm"))
            if not cands:
                sys.exit(f"ERROR: no .asm in {raw_dir}; use --asm")
            asm_path = cands[0]
        yaml = None
        program = raw_path.stem

    print(f"# trace_synced: program={program}")
    print(f"#   .asm:  {asm_path}")
    print(f"#   .raw:  {raw_path}")

    asm_text = asm_path.read_text(encoding="utf-8")
    rom = assemble(asm_text)
    print(f"#   ROM:   {len(rom)} bytes")

    raw = RawReader(raw_path)
    print(f"#   signals={raw.n_vars}, points={raw.n_points}")
    raw.load_times()

    instructions = build_instructions(raw)
    print(f"#   instructions={len(instructions)}")

    # Sync info
    if yaml and 'sync' in yaml:
        sync_regs = yaml['sync'].get('regs', POWERUP_REGS)
        # When sync uses cycle (program-mode YAML), look for an
        # instruction whose state matches AFTER any reasonable t_min.
        # 0.5 ms is a safe default; the YAML may override.
        sync_t_min = yaml['sync'].get('after_time_ms', 0.5) * 1e-3
    else:
        sync_regs = FP_LEGACY_SYNC['expected_regs']
        sync_t_min = FP_LEGACY_SYNC['after_time_ms'] * 1e-3

    sync_idx = find_sync(instructions, sync_regs, sync_t_min)
    if sync_idx is None:
        sys.exit("ERROR: could not find sync point matching expected regs")

    hw_sync = instructions[sync_idx]['state']
    print(f"# sync at instruction {sync_idx}: t={hw_sync['time']*1e3:.3f} ms"
          f"  ACC={hw_sync['acc']:X} CY={hw_sync['cy']}"
          f"  regs=[{','.join(f'{r:X}' for r in hw_sync['regs'])}]")

    # Run Python sim to the sync point.
    cpu = CPU4004Sim(rom)
    cpu.regs = list(POWERUP_REGS)
    cpu.acc = POWERUP_ACC
    cpu.cy = POWERUP_CY
    for _ in range(sync_idx):
        cpu.execute_one()

    # Force-sync the emulator to what LTspice actually shows -- this
    # absorbs any power-up / IO modelling discrepancies. Real
    # divergences appear from this point forward.
    if (cpu.acc != hw_sync['acc'] or cpu.cy != hw_sync['cy']
            or list(cpu.regs) != list(hw_sync['regs'])):
        cpu.acc = hw_sync['acc']
        cpu.cy = hw_sync['cy']
        cpu.regs = list(hw_sync['regs'])

    rows = trace_forward(cpu, instructions, sync_idx, args.max_cycles,
                         resync=args.resync, verbose=not args.quiet)

    # Outputs
    n_div = sum(1 for r in rows if not r['match'])
    print(f"\n# Traced {len(rows)} instructions from sync; "
          f"{n_div} divergence(s).")
    if n_div:
        first = next(r['cycle'] for r in rows if not r['match'])
        print(f"# First divergence at cycle {first} (offset from sync).")

    if not args.no_paper:
        data_dir = PROJECT_ROOT / "paper" / "data"
        csv_path = Path(args.csv) if args.csv else data_dir / f"trace_diff_{program}.csv"
        tex_path = Path(args.tex) if args.tex else data_dir / f"trace_diff_{program}.tex"
        mac_path = Path(args.macros) if args.macros else data_dir / f"trace_diff_{program}_macros.tex"

        write_csv(rows, csv_path)
        write_tex(rows, program, tex_path)
        write_macros(rows, program, mac_path)
        print(f"# Wrote {csv_path}")
        print(f"# Wrote {tex_path}")
        print(f"# Wrote {mac_path}")


if __name__ == "__main__":
    main()
