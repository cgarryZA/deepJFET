#!/usr/bin/env python3
"""Drive a CLK-frequency sweep over a single 4004 program.

For each requested frequency point this script:
  1. Calls `build_and_run_4004.py <program> --frequency-hz <f>` to
     regenerate the PWLs at the new timebase AND scale every PULSE
     source in the built .asc by the same ratio.
  2. (User opens LTspice and runs the transient — this script does
     NOT auto-run LTspice. Each frequency point is a separate sim,
     which has to be triggered manually.)
  3. After each sim completes (user touches a SIM_COMPLETE sentinel),
     calls `verify_micro5.py` to count ISA-rule pass/fail/CY-artifact.
  4. Aggregates the results into paper/data/frequency_sweep.{csv,tex}
     plus a frequency_sweep_macros.tex file for in-paper inline use.

The frequency at which the first ISA-rule violation appears is the
headline f_max for the design at 27 °C with the supplier model.

Usage:
    # Set up runs (touches PWLs, patches .asc, prints what to do next):
    python tools/frequency_sweep.py setup Fibonacci --frequencies 100e3 200e3 500e3 1e6 2e6 5e6

    # Per frequency, the user then:
    #   * opens cpus/4004/4004.asc in LTspice
    #   * runs the .tran
    #   * copies the .raw into cpus/4004/programs/Fibonacci/runs/<f>/Fibonacci.raw
    #   * touches a SIM_COMPLETE sentinel in that dir
    # then re-runs:
    python tools/frequency_sweep.py collect Fibonacci

    # When done, regen paper data:
    python tools/frequency_sweep.py report Fibonacci

The collect / report phases are gated on the per-frequency SIM_COMPLETE
sentinels so the user can run the LTspice transients at their own pace
across multiple sessions and the partial sweep still produces a partial
paper-data table.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS = PROJECT_ROOT / 'tools'
CPUS = PROJECT_ROOT / 'cpus'
PAPER_DATA = PROJECT_ROOT / 'paper' / 'data'


def freq_label(hz: float) -> str:
    """Stable directory name for a frequency point. 100e3 -> '100kHz'."""
    if hz >= 1e6:
        return f'{hz/1e6:g}MHz'
    if hz >= 1e3:
        return f'{hz/1e3:g}kHz'
    return f'{hz:g}Hz'


def program_runs_dir(cpu: str, program: str) -> Path:
    return CPUS / cpu / 'programs' / program / 'runs'


# ── setup: prepare one frequency point for an LTspice run ───────────────

def cmd_setup(args):
    """Build the schematic + PWLs at the requested frequency and stop —
    the user runs LTspice manually, then triggers collect later."""
    for f_hz in args.frequencies:
        label = freq_label(f_hz)
        run_dir = program_runs_dir(args.cpu, args.program) / label
        run_dir.mkdir(parents=True, exist_ok=True)
        sentinel = run_dir / 'SIM_COMPLETE'
        if sentinel.exists() and not args.force:
            print(f'\n[{label}] already complete (SIM_COMPLETE present); '
                  f'use --force to rebuild')
            continue
        print()
        print('=' * 70)
        print(f'Setting up {args.program} @ {label} ({f_hz:g} Hz)')
        print('=' * 70)
        # Estimate cycles needed — scale by speed so longer programs still
        # cover their natural completion. Default 350 unless overridden.
        cycles = args.cycles
        cmd = [
            sys.executable, str(TOOLS / 'build_and_run_4004.py'),
            args.program,
            '--cpu', args.cpu,
            '--cycles', str(cycles),
            '--frequency-hz', str(f_hz),
            '--no-open',  # don't auto-launch LTspice; the user batches manually
        ]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'  setup failed for {label}; stopping')
            return
        # Record the frequency point so collect knows where to look
        with open(run_dir / 'frequency_hz.txt', 'w') as fh:
            fh.write(f'{f_hz}\n')
        print(f'\n  >>> Now open cpus/{args.cpu}/4004.asc in LTspice and run.')
        print(f'  >>> When .raw is produced, copy:')
        print(f'         cpus/{args.cpu}/4004.raw  ->  {run_dir}/{args.program}.raw')
        print(f'  >>> Then touch the sentinel:')
        print(f'         {run_dir}/SIM_COMPLETE')


# ── collect: run verify_micro5 on every completed frequency point ───────

def cmd_collect(args):
    """For each run_dir with SIM_COMPLETE, call verify_micro5 and store
    results. Skips runs that aren't done yet."""
    base = program_runs_dir(args.cpu, args.program)
    if not base.exists():
        print(f'No runs directory at {base}. Did you `setup` yet?')
        sys.exit(1)
    results = []
    for run_dir in sorted(base.iterdir(), key=lambda p: p.name):
        if not run_dir.is_dir():
            continue
        sentinel = run_dir / 'SIM_COMPLETE'
        if not sentinel.exists():
            print(f'  skip {run_dir.name} (SIM_COMPLETE not present)')
            continue
        raw = run_dir / f'{args.program}.raw'
        if not raw.exists():
            print(f'  skip {run_dir.name} ({args.program}.raw not present)')
            continue
        freq_file = run_dir / 'frequency_hz.txt'
        if freq_file.exists():
            f_hz = float(freq_file.read_text().strip())
        else:
            f_hz = float('nan')
        print(f'  {run_dir.name}: running verify_micro5 on {raw}...')
        cmd = [sys.executable, str(TOOLS / 'verify_micro5.py'),
               str(raw), '--start-time', str(args.start_time)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        # Extract counts from verify_micro5's summary block
        n_pass = n_artifact = n_fail = n_total = 0
        for line in proc.stdout.split('\n'):
            s = line.strip()
            if s.startswith('Pass:'):
                n_pass = int(s.split(':')[1].split()[0])
            elif s.startswith('CY artifact:'):
                n_artifact = int(s.split(':')[1].split()[0])
            elif s.startswith('REAL FAIL:'):
                n_fail = int(s.split(':')[1].split()[0])
            elif s.startswith('Total:'):
                n_total = int(s.split(':')[1].split()[0])
        results.append({
            'label': run_dir.name,
            'frequency_hz': f_hz,
            'instructions_checked': n_total,
            'pass': n_pass,
            'cy_artifact': n_artifact,
            'real_fail': n_fail,
        })
        print(f'    -> {n_pass} pass / {n_artifact} CY-artifact / '
              f'{n_fail} real-fail (of {n_total})')

    if not results:
        print('No completed runs to collect.')
        return

    # Write per-program CSV
    out_csv = base / 'frequency_sweep.csv'
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['label', 'frequency_hz',
            'instructions_checked', 'pass', 'cy_artifact', 'real_fail'])
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f'\nWrote {out_csv}')

    # First-failure summary
    sorted_results = sorted(results, key=lambda r: r['frequency_hz'])
    first_fail = next((r for r in sorted_results if r['real_fail'] > 0), None)
    if first_fail:
        print(f'\nFirst ISA-rule failure at {first_fail["label"]} '
              f'({first_fail["real_fail"]} fail(s) out of '
              f'{first_fail["instructions_checked"]} checked)')
    else:
        passing = [r for r in sorted_results if r['real_fail'] == 0]
        if passing:
            hi = max(passing, key=lambda r: r['frequency_hz'])
            print(f'\nAll completed frequencies pass — '
                  f'highest tested: {hi["label"]} ({hi["frequency_hz"]:g} Hz)')


# ── report: emit paper data tables + macros ─────────────────────────────

def cmd_report(args):
    """Read the per-program CSV and emit paper/data/frequency_sweep.{csv,tex}
    plus a macros file."""
    PAPER_DATA.mkdir(parents=True, exist_ok=True)
    base = program_runs_dir(args.cpu, args.program)
    csv_in = base / 'frequency_sweep.csv'
    if not csv_in.exists():
        print(f'No frequency_sweep.csv at {csv_in}. Run `collect` first.')
        sys.exit(1)

    rows = list(csv.DictReader(open(csv_in)))
    if not rows:
        print('CSV is empty.')
        sys.exit(1)
    rows.sort(key=lambda r: float(r['frequency_hz']))

    # Mirror the CSV into paper/data
    out_csv = PAPER_DATA / f'frequency_sweep_{args.program}.csv'
    with open(out_csv, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Emit a LaTeX table — paper-ready
    out_tex = PAPER_DATA / f'frequency_sweep_{args.program}.tex'
    with open(out_tex, 'w') as fh:
        fh.write('% Auto-generated by tools/frequency_sweep.py — do not edit\n')
        fh.write('\\begin{tabular}{lrrrrr}\n')
        fh.write('\\hline\n')
        fh.write('Frequency & Instr. checked & Pass & CY-artifact & '
                 'Real fail & Outcome \\\\\n')
        fh.write('\\hline\n')
        for r in rows:
            f_hz = float(r['frequency_hz'])
            label = freq_label(f_hz)
            n_tot = int(r['instructions_checked'])
            n_pass = int(r['pass'])
            n_art = int(r['cy_artifact'])
            n_fail = int(r['real_fail'])
            outcome = ('\\checkmark' if n_fail == 0 else f'$\\times$')
            fh.write(f'  {label} & {n_tot} & {n_pass} & {n_art} & '
                     f'{n_fail} & {outcome} \\\\\n')
        fh.write('\\hline\n')
        fh.write('\\end{tabular}\n')

    # Emit macros (paper/data/frequency_sweep_macros.tex)
    # \FreqSweepMaxPassFreq, \FreqSweepFirstFailFreq, etc.
    out_macros = PAPER_DATA / f'frequency_sweep_{args.program}_macros.tex'
    passing = [r for r in rows if int(r['real_fail']) == 0]
    failing = [r for r in rows if int(r['real_fail']) > 0]
    with open(out_macros, 'w') as fh:
        fh.write('% Auto-generated by tools/frequency_sweep.py — do not edit\n')
        if passing:
            f_max = max(passing, key=lambda r: float(r['frequency_hz']))
            f_hz = float(f_max['frequency_hz'])
            fh.write(f'\\renewcommand{{\\FreqSweepMaxPass{args.program}}}{{'
                     f'{freq_label(f_hz)}}}\n')
            fh.write(f'\\renewcommand{{\\FreqSweepMaxPass{args.program}Hz}}{{'
                     f'{f_hz:g}}}\n')
        if failing:
            f_first = min(failing, key=lambda r: float(r['frequency_hz']))
            f_hz = float(f_first['frequency_hz'])
            fh.write(f'\\renewcommand{{\\FreqSweepFirstFail{args.program}}}{{'
                     f'{freq_label(f_hz)}}}\n')
            fh.write(f'\\renewcommand{{\\FreqSweepFirstFail{args.program}Hz}}{{'
                     f'{f_hz:g}}}\n')
        fh.write(f'\\renewcommand{{\\FreqSweepNPoints{args.program}}}{{'
                 f'{len(rows)}}}\n')

    print(f'Wrote {out_csv}')
    print(f'Wrote {out_tex}')
    print(f'Wrote {out_macros}')
    print()
    print('In paper/preamble.tex add provider commands so the document\n'
          'still compiles when these macros are not yet generated:')
    print(f'  \\providecommand{{\\FreqSweepMaxPass{args.program}}}{{TBD}}')
    print(f'  \\providecommand{{\\FreqSweepFirstFail{args.program}}}{{TBD}}')
    print(f'  \\providecommand{{\\FreqSweepNPoints{args.program}}}{{TBD}}')


# ── entry ────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)

    sp_setup = sub.add_parser('setup',
        help='Build PWLs + patch schematic for each frequency point')
    sp_setup.add_argument('program')
    sp_setup.add_argument('--cpu', default='4004')
    sp_setup.add_argument('--cycles', type=int, default=350,
        help='Number of cycles per frequency point (default 350)')
    sp_setup.add_argument('--frequencies', '-f', type=float, nargs='+',
        required=True,
        help='Frequencies in Hz (e.g. --frequencies 100e3 200e3 1e6)')
    sp_setup.add_argument('--force', action='store_true',
        help='Re-setup even if SIM_COMPLETE is present')
    sp_setup.set_defaults(func=cmd_setup)

    sp_coll = sub.add_parser('collect',
        help='Run verify_micro5 on every completed frequency point')
    sp_coll.add_argument('program')
    sp_coll.add_argument('--cpu', default='4004')
    sp_coll.add_argument('--start-time', type=float, default=0.2,
        help='start_time (ms) passed to verify_micro5 — must be small '
             'enough that even fast frequencies have completed startup')
    sp_coll.set_defaults(func=cmd_collect)

    sp_rep = sub.add_parser('report',
        help='Emit paper/data/frequency_sweep_<program>.{csv,tex,_macros.tex}')
    sp_rep.add_argument('program')
    sp_rep.add_argument('--cpu', default='4004')
    sp_rep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
