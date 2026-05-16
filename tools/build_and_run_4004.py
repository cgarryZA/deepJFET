#!/usr/bin/env python3
"""Build the 4004 CPU, append .save directives, and open in LTSpice.

This wraps rom_emulator.py and adds a post-processing step that:
1. Scans 4004.asc for all named FLAG nets (excluding power rails)
2. Appends .save directives so only those named nets are saved in the .raw
3. Optionally appends .ic initial-condition statements

Massively reduces .raw file size and can speed up simulation by reducing
disk I/O. Without .save LTSpice saves voltages for every internal node
(thousands of anonymous P1234 nodes); with it we save only the few hundred
named signals we actually care about.

Usage:
    python tools/build_and_run_4004.py FloatingPoint
    python tools/build_and_run_4004.py FloatingPoint --no-save     # disable
    python tools/build_and_run_4004.py FloatingPoint --no-open     # build only
"""

import argparse
import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LTSPICE = r"C:\Users\z00503ku\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

# Power rails and other nets we don't want to save (no useful info)
EXCLUDE_NETS = {
    'VDD', 'VSS', 'CLK', '0', 'GND', 'ROMCLK', 'RAMCLK',
}

# LTSpice accepts one giant .save directive with all signals on a single line


def find_named_nets(asc_path):
    """Find all named FLAGs in the .asc file. Returns sorted unique list."""
    names = set()
    with open(asc_path) as f:
        for line in f:
            m = re.match(r'FLAG\s+\S+\s+\S+\s+(\S+)', line.strip())
            if m:
                name = m.group(1)
                # Exclude power rails and inverted-rail names like !VDD
                if name in EXCLUDE_NETS:
                    continue
                if name.startswith('!') and name[1:] in EXCLUDE_NETS:
                    continue
                names.add(name)
    return sorted(names)


def append_save_directives(asc_path, named_nets):
    """Append .save TEXT entries to the .asc that save only named nets."""
    with open(asc_path) as f:
        content = f.read()

    # Remove any pre-existing .save directives we added before
    # (so re-running this doesn't accumulate stale entries)
    lines = content.split('\n')
    cleaned = []
    for line in lines:
        if re.match(r'TEXT \S+ \S+ Left 2 !\.save\b', line.strip()):
            continue
        cleaned.append(line)
    content = '\n'.join(cleaned)

    # Filter out !X inverted nets - we have the non-inverted versions anyway
    # and !X net names can break LTSpice's V(!X) parsing
    filtered = [n for n in named_nets if not n.startswith('!')]

    # Batch into multiple .save lines (one giant line can be too long for LTSpice)
    BATCH = 20
    save_text_lines = []
    y_pos = -9400
    for i in range(0, len(filtered), BATCH):
        batch = filtered[i:i + BATCH]
        save_str = '!.save ' + ' '.join(f'V({s})' for s in batch)
        save_text_lines.append(f'TEXT 8552 {y_pos + (i // BATCH) * 32} Left 2 {save_str}')

    # Also patch the .options line to add convergence-helping settings
    # (build_cpu.py resets these to defaults each rebuild)
    content = content.replace(
        'TEXT 8560 -9472 Left 2 !.options NoOpIter',
        'TEXT 8560 -9472 Left 2 !.options NoOpIter cshunt=50f gminsteps=200 itl1=1000 srcsteps=20'
    )

    new_content = content.rstrip() + '\n' + '\n'.join(save_text_lines) + '\n'
    with open(asc_path, 'w') as f:
        f.write(new_content)

    return len(save_text_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('program', help='Program name (e.g. FloatingPoint)')
    parser.add_argument('--cpu', default='4004')
    parser.add_argument('--cycles', type=int, default=500)
    parser.add_argument('--startup-delay', type=float, default=2e-6)
    parser.add_argument('--no-save', action='store_true',
                        help='Disable .save directives (saves all signals - bigger .raw)')
    parser.add_argument('--no-open', action='store_true',
                        help='Build only, do not open LTSpice')
    args = parser.parse_args()

    asc_path = os.path.join(PROJECT_ROOT, 'cpus', args.cpu, f'{args.cpu}.asc')

    # 1. Build the CPU using rom_emulator (handles PWL gen, build_cpu, and .tran patch)
    print('=' * 70)
    print('Step 1: Build CPU and generate PWL')
    print('=' * 70)
    rom_emulator = os.path.join(os.path.dirname(__file__), 'rom_emulator.py')
    result = subprocess.run([
        sys.executable, rom_emulator, args.program,
        '--cpu', args.cpu,
        '--cycles', str(args.cycles),
        '--startup-delay', str(args.startup_delay),
    ], capture_output=True, text=True)
    if result.returncode != 0:
        print('Build failed:')
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    # Show summary line(s)
    for line in result.stdout.split('\n')[-15:]:
        if line.strip():
            print(line)

    if not os.path.isfile(asc_path):
        print(f'ERROR: built file not found at {asc_path}')
        sys.exit(1)

    # 2. Find all named nets in the built .asc
    print()
    print('=' * 70)
    print('Step 2: Scan named nets')
    print('=' * 70)
    named_nets = find_named_nets(asc_path)
    print(f'Found {len(named_nets)} named nets (excluding power rails)')
    # Show a few samples
    print(f'Examples: {", ".join(named_nets[:10])}...')

    # 3. Append .save directives
    if args.no_save:
        print('--no-save flag: skipping .save injection')
    else:
        print()
        print('=' * 70)
        print('Step 3: Append .save directives')
        print('=' * 70)
        n_lines = append_save_directives(asc_path, named_nets)
        print(f'Appended {n_lines} .save TEXT lines covering {len(named_nets)} nets')
        print('This should dramatically reduce .raw file size')

    # 4. Delete stale .raw files so LTSpice doesn't try to use them
    for f in ['4004.raw', '4004.op.raw']:
        p = os.path.join(PROJECT_ROOT, 'cpus', args.cpu, f)
        if os.path.exists(p):
            try:
                os.remove(p)
                print(f'Deleted stale {f}')
            except PermissionError:
                print(f'WARNING: could not delete {f} (file in use?)')

    # 5. Open LTSpice
    if args.no_open:
        print()
        print('--no-open flag: not opening LTSpice')
        print(f'Run: {LTSPICE} {asc_path}')
    else:
        print()
        print('=' * 70)
        print('Step 4: Open LTSpice')
        print('=' * 70)
        # Launch LTSpice directly (no `cmd /c start` shim — that prints
        # "Access is denied." when LTSpice is already running, and even when
        # it succeeds it pops a confusing dialog).
        subprocess.Popen([LTSPICE, asc_path], close_fds=True)
        print(f'Opening {asc_path}')


if __name__ == '__main__':
    main()
