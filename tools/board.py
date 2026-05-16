#!/usr/bin/env python3
"""Board-level subassembly workflow for the SiC JFET 4004.

This is the tool we use when we want to validate an individual subsystem
(ALU, scratchpad, stack, ...) on its own — either in LTspice with PWL
boundary drivers, or as a physical PCB driven by an AWG / pattern
generator. The same time/voltage data feeds both, so a hardware result
can be diffed directly against the LTspice reference.

Subcommands:

  list                List discoverable subsystems

  extract <subsys>    Read a full-CPU .raw, identify the boundary nets of
                      the subsystem, write one PWL per input net (into
                      boards/<subsys>/inputs/) and one PWL per output net
                      (into boards/<subsys>/expected/).

  testbench <subsys>  Generate a board-level <subsys>_testbench.asc that
                      instantiates the existing subsystem .asc and drives
                      each input boundary net with a V(... PWL filename).

  diff <subsys>       After running the board testbench in LTspice, diff
                      its outputs against expected/ — threshold-crossing
                      comparison, same logic as verify_micro5.

  export-lab <subsys> Re-emit the inputs as AWG-friendly CSV and as a
                      VCD file for digital pattern generators.

Usage examples:

    python tools/board.py list
    python tools/board.py extract alu --raw cpus/4004/4004.raw
    python tools/board.py testbench alu
    python tools/board.py diff alu --raw cpus/4004/boards/alu/alu_testbench.raw
    python tools/board.py export-lab alu

Boundary detection: a "boundary net" of subsystem X is a FLAG name that
appears in X's .asc files AND in at least one other subsystem's .asc.
The user can override which nets are inputs vs outputs via
boards/<subsys>/boundary.yaml; without it we fall back to naming
heuristics (suffix "In" -> input, "Out" -> output, else input).
"""

import argparse
import csv
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_ROOT = PROJECT_ROOT / 'cpus' / '4004'

# Per-CPU subsystem layout. Hand-listed for 4004; mirrors config.py
# COMPONENTS but flattened to just the file paths so we can scan FLAGs.
SUBSYSTEMS_4004 = {
    'alu':                 [CPU_ROOT / 'alu.asc'],
    'instruction_register':[CPU_ROOT / 'instruction_register.asc'],
    'micro_instructions':  [CPU_ROOT / 'micro_instructions.asc'],
    'controls':            [CPU_ROOT / 'controls.asc'],
    'pins':                [CPU_ROOT / 'pins.asc'],
    'program_counter':     [CPU_ROOT / 'program_counter.asc'],
    'step_counter':        [CPU_ROOT / 'step_counter.asc'],
    'scratchpad':          ([CPU_ROOT / 'scratchpad' / f'Bus{i}.asc' for i in (1, 2)] +
                            [CPU_ROOT / 'scratchpad' / 'Controls.asc'] +
                            [CPU_ROOT / 'scratchpad' / f'Pair {i}.asc' for i in range(1, 9)]),
    'stack':               ([CPU_ROOT / 'stack' / 'Bus.asc',
                             CPU_ROOT / 'stack' / 'Controls.asc'] +
                            [CPU_ROOT / 'stack' / f'Level {i}.asc' for i in (1, 2, 3)]),
}

# Power / clock nets are never treated as boundary inputs to drive
# from a PWL — they're shared globally and handled by the testbench rail
# definitions.
GLOBAL_NETS = {'VDD', 'VSS', '0', 'GND', 'CLK', 'ROMCLK', 'RAMCLK'}

THRESH = -2.5  # JFET logic threshold (V); above = high, below = low


# ── FLAG parsing ─────────────────────────────────────────────────────────

FLAG_RE = re.compile(r'^FLAG\s+\S+\s+\S+\s+(\S+)')

def flags_in_file(path: Path) -> set:
    """Return the set of unique FLAG net names in a single .asc file."""
    names = set()
    try:
        text = path.read_text(errors='replace')
    except FileNotFoundError:
        return names
    for line in text.splitlines():
        m = FLAG_RE.match(line.strip())
        if m:
            name = m.group(1)
            # Drop power rails and !-inverted variants of them
            if name in GLOBAL_NETS:
                continue
            if name.startswith('!') and name[1:] in GLOBAL_NETS:
                continue
            names.add(name)
    return names

def flags_in_subsys(subsys: str) -> set:
    """Union of all FLAG net names across the subsystem's .asc files."""
    names = set()
    for p in SUBSYSTEMS_4004[subsys]:
        names |= flags_in_file(p)
    return names


def boundary_nets(subsys: str) -> set:
    """A boundary net of subsys X is one that appears in X AND in at
    least one other subsystem. Internal nets stay inside one subsystem."""
    inside = flags_in_subsys(subsys)
    outside = set()
    for other, files in SUBSYSTEMS_4004.items():
        if other == subsys:
            continue
        for p in files:
            outside |= flags_in_file(p)
    return inside & outside


def classify_direction(net: str, override: dict | None = None) -> str:
    """Return 'input' or 'output' for a boundary net.

    Uses override (from boundary.yaml) when provided, else falls back to
    naming heuristic. Inverted nets like !X follow the same convention
    as X."""
    if override is not None:
        if net in override.get('inputs', []) or net in override.get('outputs', []):
            return 'input' if net in override.get('inputs', []) else 'output'
    base = net[1:] if net.startswith('!') else net
    low = base.lower()
    if low.endswith('out') or low.endswith('_out') or low.endswith('outing'):
        return 'output'
    if low.endswith('in') or low.endswith('_in') or low.endswith('ing'):
        return 'input'
    # default: input (safer — we'll drive it. Outputs accidentally driven
    # are obvious in sim because they fight with the subsystem)
    return 'input'


def load_boundary_override(subsys: str) -> dict | None:
    """Load boards/<subsys>/boundary.yaml if it exists.

    Minimal YAML reader; we only support two top-level keys:
        inputs:  [list of net names]
        outputs: [list of net names]
    Strings can be unquoted. One net per line, indented under either key.
    """
    path = CPU_ROOT / 'boards' / subsys / 'boundary.yaml'
    if not path.exists():
        return None
    out = {'inputs': [], 'outputs': []}
    current = None
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith('#'):
            continue
        if s.endswith(':') and s[:-1] in ('inputs', 'outputs'):
            current = s[:-1]
            continue
        if s.startswith('- ') and current:
            out[current].append(s[2:].strip().strip('"\''))
    return out


# ── .raw reader (lightweight) ────────────────────────────────────────────

class RawReader:
    """Memory-maps the .raw, exposes (times, signal_by_name) lookups."""
    def __init__(self, path: Path):
        self.path = Path(path)
        with open(self.path, 'rb') as f:
            header_raw = f.read(2_000_000)
        text = header_raw.decode('utf-16-le', errors='replace')
        self.n_vars = 0
        self.n_points = 0
        self.var_index = {}  # lowercase v(name) -> index
        for line in text.split('\n'):
            s = line.strip()
            if s.startswith('No. Variables'):
                self.n_vars = int(s.split(':')[1])
            elif s.startswith('No. Points'):
                self.n_points = int(s.split(':')[1])
            elif '\t' in s:
                parts = s.split('\t')
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0])
                        self.var_index[parts[1].strip().lower()] = idx
                    except ValueError:
                        pass
        idx = text.find('Binary:')
        if idx == -1:
            raise RuntimeError(f'Binary: marker not found in {self.path}')
        self.data_start = (idx + len('Binary:') + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4
        fsize = self.path.stat().st_size
        self.n_complete = (fsize - self.data_start) // self.row_size
        self._block = None  # lazy load

    def _ensure_loaded(self):
        if self._block is None:
            self._block = np.fromfile(self.path, dtype=np.uint8,
                                      offset=self.data_start,
                                      count=self.n_complete * self.row_size)
            self._block = self._block.reshape(self.n_complete, self.row_size)

    def times(self) -> np.ndarray:
        self._ensure_loaded()
        return np.abs(self._block[:, :8].copy().view(np.float64).flatten())

    def signal(self, name: str) -> np.ndarray | None:
        key = f'v({name.lower()})'
        if key not in self.var_index:
            return None
        self._ensure_loaded()
        i = self.var_index[key]
        if i == 0:
            return self.times()
        off = 8 + (i - 1) * 4
        return self._block[:, off:off + 4].copy().view(np.float32).flatten()


# ── PWL writing ──────────────────────────────────────────────────────────

def _decimate(times: np.ndarray, voltages: np.ndarray,
              dv_thresh: float = 0.2, dt_max_us: float = 5.0) -> np.ndarray:
    """Return sorted indices into times/voltages to keep.

    Keeps: (a) every logic-threshold crossing plus its two neighbours
    (preserves edge timing), (b) every point whose value differs from
    the last kept point by more than `dv_thresh` volts (preserves slow
    drifts), (c) at least one sample per `dt_max_us` microseconds
    (safety floor), (d) first and last points.
    """
    n = len(times)
    if n == 0:
        return np.array([], dtype=int)
    # Threshold crossings + neighbours
    above = voltages > THRESH
    crossings = np.where(np.diff(above.astype(int)) != 0)[0] + 1
    keep = set()
    for c in crossings:
        keep.add(max(0, c - 1))
        keep.add(c)
        keep.add(min(n - 1, c + 1))
    # Value-change adaptive sampling + time floor
    dt_max = dt_max_us * 1e-6
    last_kept_v = voltages[0]
    last_kept_t = times[0]
    keep.add(0)
    for i in range(1, n):
        if (abs(voltages[i] - last_kept_v) > dv_thresh or
                times[i] - last_kept_t > dt_max):
            keep.add(i)
            last_kept_v = voltages[i]
            last_kept_t = times[i]
    keep.add(n - 1)
    return np.array(sorted(keep), dtype=int)


def write_pwl(path: Path, times: np.ndarray, voltages: np.ndarray) -> int:
    """Write a piecewise-linear file for LTspice. See _decimate for the
    sample-selection policy."""
    path.parent.mkdir(parents=True, exist_ok=True)
    idx = _decimate(times, voltages)
    with open(path, 'w', newline='') as f:
        for i in idx:
            f.write(f'{times[i]*1e6:.6f}u {voltages[i]:.4f}\n')
    return len(idx)


def write_csv(path: Path, times: np.ndarray, voltages: np.ndarray) -> int:
    """Write AWG-friendly CSV (time_s,voltage_v). Same decimation as PWL."""
    idx = _decimate(times, voltages)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time_s', 'voltage_v'])
        for i in idx:
            w.writerow([f'{times[i]:.10e}', f'{voltages[i]:.4f}'])
    return len(idx)


def write_vcd(path: Path, times: np.ndarray, traces: dict, timescale='1us'):
    """Write a single-file VCD covering all named digital traces.

    `traces` is {net_name: voltages_array}; voltages are thresholded at
    THRESH into 0/1. Emits one entry per change. Time is in microseconds
    (timescale `1us`)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Assign short id codes to each net
    ids = {}
    for i, net in enumerate(sorted(traces.keys())):
        ids[net] = chr(33 + i) if i < 94 else f'#{i}'  # printable ASCII
    with open(path, 'w', newline='') as f:
        f.write('$date generated by tools/board.py $end\n')
        f.write('$version deepJFET 4004 board export $end\n')
        f.write(f'$timescale {timescale} $end\n')
        f.write('$scope module subsystem $end\n')
        for net, code in ids.items():
            safe = net.replace('!', 'n_')
            f.write(f'$var wire 1 {code} {safe} $end\n')
            ids[net] = (code, safe)
        f.write('$upscope $end\n$enddefinitions $end\n')
        # Initial dump
        f.write('$dumpvars\n')
        for net, (code, _) in ids.items():
            v = traces[net]
            bit = 1 if v[0] > THRESH else 0
            f.write(f'{bit}{code}\n')
        f.write('$end\n')
        # Find changes across all traces, time-quantised to microseconds
        all_changes = []
        for net, (code, _) in ids.items():
            v = traces[net]
            above = (v > THRESH).astype(int)
            edges = np.where(np.diff(above) != 0)[0] + 1
            for e in edges:
                all_changes.append((times[e], code, above[e]))
        all_changes.sort()
        last_time_us = None
        for t, code, bit in all_changes:
            t_us = int(round(t * 1e6))
            if t_us != last_time_us:
                f.write(f'#{t_us}\n')
                last_time_us = t_us
            f.write(f'{bit}{code}\n')


# ── Sub-commands ─────────────────────────────────────────────────────────

def cmd_list(_args):
    """List subsystems and their auto-detected boundary nets."""
    for name in SUBSYSTEMS_4004:
        boundary = boundary_nets(name)
        override = load_boundary_override(name)
        ins, outs = [], []
        for net in sorted(boundary):
            d = classify_direction(net, override)
            (ins if d == 'input' else outs).append(net)
        print(f'\n{name}: {len(boundary)} boundary nets '
              f'({len(ins)} in, {len(outs)} out)'
              + ('  [boundary.yaml override active]' if override else ''))
        if ins:
            print(f'  inputs:  {", ".join(ins[:12])}'
                  + (' ...' if len(ins) > 12 else ''))
        if outs:
            print(f'  outputs: {", ".join(outs[:12])}'
                  + (' ...' if len(outs) > 12 else ''))


def cmd_extract(args):
    """Extract per-net PWLs from full-CPU .raw for the named subsystem."""
    subsys = args.subsystem
    if subsys not in SUBSYSTEMS_4004:
        print(f'Unknown subsystem: {subsys}. Available: '
              f'{", ".join(SUBSYSTEMS_4004)}')
        sys.exit(1)
    raw_path = Path(args.raw)
    if not raw_path.exists():
        print(f'Raw file not found: {raw_path}')
        sys.exit(1)

    boundary = boundary_nets(subsys)
    override = load_boundary_override(subsys)
    inputs, outputs = [], []
    for net in sorted(boundary):
        (inputs if classify_direction(net, override) == 'input'
                else outputs).append(net)

    print(f'Loading {raw_path} ...')
    raw = RawReader(raw_path)
    times = raw.times()
    print(f'  {raw.n_vars} signals, {raw.n_complete} points, '
          f'{times[-1]*1e3:.2f}ms')
    print(f'Subsystem {subsys}: {len(inputs)} inputs, {len(outputs)} outputs')

    board_dir = CPU_ROOT / 'boards' / subsys
    inputs_dir = board_dir / 'inputs'
    expected_dir = board_dir / 'expected'

    def fetch(net):
        """Look up V(net); fall back to V(X) when net is !X — LTSpice's
        implicit-inverter notation means the base signal is what's .saved.
        Returns (voltages, effective_name_written_to_disk) or (None, None)."""
        v = raw.signal(net)
        if v is not None:
            return v, net
        if net.startswith('!'):
            base = net[1:]
            v = raw.signal(base)
            if v is not None:
                # Write the non-inverted base; the board testbench will
                # reference !net which LTSpice will invert from V(base).
                return v, base
        return None, None

    missing = []
    written = 0
    written_inputs = []
    written_outputs = []
    for net in inputs:
        v, eff = fetch(net)
        if v is None:
            missing.append(net)
            continue
        n = write_pwl(inputs_dir / f'{eff}.pwl', times, v)
        written += 1
        written_inputs.append(eff)
        if args.verbose:
            print(f'  in/{eff}.pwl  ({n} samples)')
    for net in outputs:
        v, eff = fetch(net)
        if v is None:
            missing.append(net)
            continue
        n = write_pwl(expected_dir / f'{eff}.pwl', times, v)
        written += 1
        written_outputs.append(eff)
        if args.verbose:
            print(f'  expected/{eff}.pwl  ({n} samples)')
    # Dedupe: when a boundary set contains both !X and X they both
    # resolve to the same PWL filename. Keep one entry.
    inputs = sorted(set(written_inputs))
    outputs = sorted(set(written_outputs))

    # Write a manifest so testbench/diff know what to expect
    manifest = board_dir / 'manifest.txt'
    with open(manifest, 'w', newline='') as f:
        f.write(f'# Auto-generated by board.py extract\n')
        f.write(f'# Raw source: {raw_path}\n')
        f.write(f'# Duration: {times[-1]:.6e} s\n')
        f.write(f'inputs:\n')
        for n in sorted(inputs):
            if n not in missing:
                f.write(f'  - {n}\n')
        f.write(f'outputs:\n')
        for n in sorted(outputs):
            if n not in missing:
                f.write(f'  - {n}\n')

    print(f'\nWrote {written} PWLs to {board_dir}')
    if missing:
        print(f'WARN: {len(missing)} nets not found in .raw '
              f'(may not have been .saved): {", ".join(missing[:10])}'
              + (' ...' if len(missing) > 10 else ''))


def cmd_testbench(args):
    """Generate <subsys>_testbench.asc that wraps the subsystem .asc and
    drives each input net from boards/<subsys>/inputs/<net>.pwl."""
    subsys = args.subsystem
    if subsys not in SUBSYSTEMS_4004:
        print(f'Unknown subsystem: {subsys}')
        sys.exit(1)
    board_dir = CPU_ROOT / 'boards' / subsys
    if not (board_dir / 'manifest.txt').exists():
        print(f'No manifest found at {board_dir}/manifest.txt. '
              f'Run `board.py extract {subsys}` first.')
        sys.exit(1)

    # Parse manifest
    inputs, outputs, duration = [], [], 100e-6
    section = None
    for line in (board_dir / 'manifest.txt').read_text().splitlines():
        s = line.strip()
        if s.startswith('# Duration:'):
            duration = float(s.split(':')[1].strip().rstrip('s').strip())
        elif s.endswith(':') and s[:-1] in ('inputs', 'outputs'):
            section = s[:-1]
        elif s.startswith('- ') and section:
            (inputs if section == 'inputs' else outputs).append(s[2:].strip())

    # Build the testbench .asc by INLINING the subsystem schematic
    # content, then prepending PWL voltage sources off to the side.
    # This matches build_cpu.py's pattern and keeps the testbench a
    # single self-contained .asc that LTspice opens cleanly.
    tb_path = board_dir / f'{subsys}_testbench.asc'

    # Read & concatenate the subsystem .asc files (skip Version/SHEET
    # headers on all but the first).
    subsys_lines = []
    for i, f in enumerate(SUBSYSTEMS_4004[subsys]):
        with open(f) as fh:
            file_lines = fh.read().splitlines()
        if i == 0:
            # Keep Version + SHEET line from the first file
            subsys_lines.extend(file_lines)
        else:
            # Drop the first two lines (Version 4 + SHEET ...)
            subsys_lines.extend(file_lines[2:])

    # Stitch in PWL voltage sources at a fixed offset so they don't
    # collide with the inlined subsystem geometry. Place them off to
    # the left of x=-8000 (subsystems sit in the 0..+8000 range).
    pwl_lines = []
    base_x = -10000
    for i, net in enumerate(inputs):
        pwl_path = (board_dir / 'inputs' / f'{net}.pwl').as_posix()
        y = -8000 + i * 128
        pwl_lines.append(f'SYMBOL voltage {base_x} {y} R0')
        pwl_lines.append('WINDOW 3 24 96 Left 2')
        pwl_lines.append('WINDOW 123 0 0 Left 0')
        pwl_lines.append('WINDOW 39 0 0 Left 0')
        pwl_lines.append(f'SYMATTR Value PWL file="{pwl_path}"')
        pwl_lines.append(f'SYMATTR InstName Vbnd_{i}')
        # Drive the named net 80 units to the right of the source
        pwl_lines.append(f'FLAG {base_x + 80} {y - 16} {net}')
        # Ground reference
        pwl_lines.append(f'FLAG {base_x + 80} {y + 80} 0')

    # Output observation FLAGs (these should already exist as FLAGs
    # inside the subsystem .asc since they're internal-to-subsystem
    # named nets — we add explicit copies near the PWL strip so they're
    # easy to find in the schematic, and to make sure they're saved).
    obs_y_base = -8000 + len(inputs) * 128 + 200
    for i, net in enumerate(outputs):
        pwl_lines.append(f'FLAG {base_x + 200} {obs_y_base + i * 64} {net}')

    # SPICE directives (.tran, .save, .options) as TEXT entries.
    directive_y = obs_y_base + len(outputs) * 64 + 100
    save_lines = [f'V({n})' for n in inputs + outputs]
    directives = []
    for i in range(0, len(save_lines), 20):
        batch = save_lines[i:i + 20]
        directives.append(
            f'TEXT {base_x} {directive_y + len(directives)*32} '
            f'Left 2 !.save ' + ' '.join(batch))
    directives.append(
        f'TEXT {base_x} {directive_y + len(directives)*32} '
        f'Left 2 !.tran 0 {duration*1e6:.1f}us 1ps 0.01')
    directives.append(
        f'TEXT {base_x} {directive_y + len(directives)*32} '
        f'Left 2 !.options NoOpIter cshunt=50f gminsteps=200 '
        f'itl1=1000 srcsteps=20')

    tb_path.write_text('\n'.join(subsys_lines + pwl_lines + directives) + '\n')
    print(f'Wrote {tb_path}')
    print(f'  {len(inputs)} PWL-driven inputs, {len(outputs)} observed outputs')
    print(f'  .tran covers {duration*1e6:.1f}us')
    print(f'Open in LTspice and run; output .raw lands at '
          f'{tb_path.with_suffix(".raw")}')


def cmd_diff(args):
    """Diff board-testbench outputs vs expected/."""
    subsys = args.subsystem
    raw = Path(args.raw) if args.raw else (CPU_ROOT / 'boards' / subsys /
                                            f'{subsys}_testbench.raw')
    if not raw.exists():
        print(f'No board-testbench raw at {raw}')
        sys.exit(1)
    board_dir = CPU_ROOT / 'boards' / subsys
    expected_dir = board_dir / 'expected'
    if not expected_dir.exists():
        print(f'No expected/ dir at {expected_dir}')
        sys.exit(1)

    print(f'Loading {raw} ...')
    rdr = RawReader(raw)
    board_times = rdr.times()

    pass_count = fail_count = 0
    fails = []
    for pwl in sorted(expected_dir.glob('*.pwl')):
        net = pwl.stem
        actual = rdr.signal(net)
        if actual is None:
            print(f'  MISS: {net} not in board .raw')
            fail_count += 1
            continue
        # Load expected PWL
        exp_times, exp_voltages = [], []
        for line in pwl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            t_str = parts[0].rstrip('u')
            try:
                exp_times.append(float(t_str) * 1e-6)
                exp_voltages.append(float(parts[1]))
            except ValueError:
                continue
        exp_times = np.array(exp_times)
        exp_voltages = np.array(exp_voltages)
        # Interpolate actual onto expected timestamps and threshold-compare
        actual_at_exp = np.interp(exp_times, board_times, actual)
        actual_bits = (actual_at_exp > THRESH).astype(int)
        exp_bits = (exp_voltages > THRESH).astype(int)
        n_diff = int(np.sum(actual_bits != exp_bits))
        n_tot = len(exp_bits)
        if n_diff == 0:
            pass_count += 1
            if args.verbose:
                print(f'  PASS {net}: {n_tot} samples agree')
        else:
            fail_count += 1
            fails.append((net, n_diff, n_tot))
            print(f'  FAIL {net}: {n_diff}/{n_tot} samples disagree '
                  f'({100*n_diff/n_tot:.1f}%)')

    print(f'\n{pass_count} nets match, {fail_count} differ.')
    if fail_count:
        sys.exit(2)


def cmd_export_lab(args):
    """Re-export inputs as CSV + a single VCD covering all digital nets."""
    subsys = args.subsystem
    board_dir = CPU_ROOT / 'boards' / subsys
    inputs_dir = board_dir / 'inputs'
    if not inputs_dir.exists():
        print(f'No inputs/ dir at {inputs_dir}. Run `extract` first.')
        sys.exit(1)
    lab_dir = board_dir / 'lab'
    csv_dir = lab_dir / 'csv'

    # Load each PWL back, write CSV and accumulate VCD traces
    traces = {}
    times_ref = None
    n_csv = 0
    for pwl in sorted(inputs_dir.glob('*.pwl')):
        net = pwl.stem
        ts, vs = [], []
        for line in pwl.read_text().splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            try:
                ts.append(float(parts[0].rstrip('u')) * 1e-6)
                vs.append(float(parts[1]))
            except ValueError:
                continue
        ts = np.array(ts); vs = np.array(vs)
        # Re-densify onto a 100ns grid for VCD (so all nets share timebase)
        if times_ref is None:
            times_ref = np.arange(ts[0], ts[-1], 100e-9)
        vs_dense = np.interp(times_ref, ts, vs)
        traces[net] = vs_dense
        n_csv += write_csv(csv_dir / f'{net}.csv', ts, vs)

    if traces:
        vcd_path = lab_dir / f'{subsys}.vcd'
        write_vcd(vcd_path, times_ref, traces)
        print(f'Wrote {len(traces)} CSV files to {csv_dir}')
        print(f'Wrote combined VCD to {vcd_path}')
    else:
        print('No input PWLs found to export.')


# ── Entry point ──────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = p.add_subparsers(dest='cmd', required=True)

    sp_list = sp.add_parser('list',
        help='List subsystems and their auto-detected boundary nets')
    sp_list.set_defaults(func=cmd_list)

    sp_ex = sp.add_parser('extract',
        help='Write boundary-net PWLs from a full-CPU .raw')
    sp_ex.add_argument('subsystem')
    sp_ex.add_argument('--raw', default=str(CPU_ROOT / '4004.raw'),
        help='Full-CPU .raw to read (default: cpus/4004/4004.raw)')
    sp_ex.add_argument('-v', '--verbose', action='store_true')
    sp_ex.set_defaults(func=cmd_extract)

    sp_tb = sp.add_parser('testbench',
        help='Generate board-level testbench .asc')
    sp_tb.add_argument('subsystem')
    sp_tb.set_defaults(func=cmd_testbench)

    sp_df = sp.add_parser('diff',
        help='Diff board-testbench .raw vs expected/')
    sp_df.add_argument('subsystem')
    sp_df.add_argument('--raw', help='Board testbench .raw '
        '(default: cpus/4004/boards/<subsys>/<subsys>_testbench.raw)')
    sp_df.add_argument('-v', '--verbose', action='store_true')
    sp_df.set_defaults(func=cmd_diff)

    sp_lab = sp.add_parser('export-lab',
        help='Re-emit inputs as CSV + VCD for AWG/pattern-generator use')
    sp_lab.add_argument('subsystem')
    sp_lab.set_defaults(func=cmd_export_lab)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
