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

# Power rails we don't want to save (no useful info — they're DC).
# CLK/ROMCLK/RAMCLK used to be excluded as "deterministic PULSE sources"
# but for fmax diagnostics we want to see them directly on the waveform
# analyser. Cost is small (3 extra signals out of ~600).
EXCLUDE_NETS = {
    'VDD', 'VSS', '0', 'GND',
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


# ── PULSE-source scaling for the frequency-sweep campaign ───────────────
#
# Every continuous clock in the schematic (CLK, ROMCLK, RAMCLK) is a
# PULSE(...) source on the 10 us baseline. The reset / init one-shots
# (V1, V8, "line 208") are also PULSE sources with shorter periods that
# need to stay aligned to the clock cycle they're keyed off of.
#
# PULSE syntax: PULSE(V1 V2 Td Tr Tf Tpw Tper Ncycles)
#   V1, V2    : low / high values             — DON'T scale (voltages)
#   Td        : initial delay                  — SCALE (phase of clock cycle)
#   Tr, Tf    : rise / fall time               — DON'T scale (JFET slew-limited)
#   Tpw       : pulse width                    — SCALE (duty of clock cycle)
#   Tper      : period                         — SCALE (the clock period itself)
#   Ncycles   : cycle count                    — DON'T scale
#
# We detect "essentially-DC" PULSE sources (Tper > 1 s — the one-shot
# init pulses use Tper=50 s as a sentinel) and only scale Td on those,
# leaving the giant Tper alone.

# Match "PULSE(...)" with the 8 args. SPICE allows commas or whitespace
# between args; we cover whitespace which is what the 4004 schematics use.
PULSE_RE = re.compile(
    r'(SYMATTR\s+Value\s+PULSE\s*\()'        # group 1: prefix
    r'\s*([-\d.eE+]+\w*)'                      # group 2: V1
    r'\s+([-\d.eE+]+\w*)'                      # group 3: V2
    r'\s+([-\d.eE+]+\w*)'                      # group 4: Td
    r'\s+([-\d.eE+]+\w*)'                      # group 5: Tr
    r'\s+([-\d.eE+]+\w*)'                      # group 6: Tf
    r'\s+([-\d.eE+]+\w*)'                      # group 7: Tpw
    r'\s+([-\d.eE+]+\w*)'                      # group 8: Tper
    r'(?:\s+([-\d.eE+]+\w*))?'                 # group 9: Ncycles (optional)
    r'\s*\)'
)

# SI-prefix mapping for SPICE time literals
_SPICE_SUFFIXES = {
    '': 1.0, 's': 1.0,
    'ms': 1e-3, 'us': 1e-6, 'u': 1e-6,
    'ns': 1e-9, 'n': 1e-9,
    'ps': 1e-12, 'p': 1e-12,
    'fs': 1e-15, 'f': 1e-15,
}

def _parse_spice_time(s):
    """Parse a SPICE time literal ("10us", "5e-6", "1n", "12") into seconds."""
    s = s.strip()
    # Strip SPICE suffix
    for suf in sorted(_SPICE_SUFFIXES, key=len, reverse=True):
        if suf and s.lower().endswith(suf):
            try:
                return float(s[:-len(suf)]) * _SPICE_SUFFIXES[suf]
            except ValueError:
                pass
    return float(s)

def _format_spice_time(seconds):
    """Round-trip-safe SPICE time literal. Tries 'us' then 'ns' then 'ps'
    based on magnitude; falls back to bare-seconds or scientific."""
    if seconds == 0:
        return '0'
    abs_s = abs(seconds)
    if abs_s >= 1.0:
        # ≥ 1 second: emit as bare seconds, no SI prefix (avoids "50000ms"
        # for what was originally "50" in the source schematic).
        return f'{seconds:g}'
    if abs_s >= 1e-3:
        return f'{seconds*1e3:g}ms'
    if abs_s >= 1e-6:
        return f'{seconds*1e6:g}us'
    if abs_s >= 1e-9:
        return f'{seconds*1e9:g}ns'
    if abs_s >= 1e-12:
        return f'{seconds*1e12:g}ps'
    return f'{seconds:e}'

def patch_clock_pulses(asc_path, scale):
    """Scale every PULSE(...) source in the built .asc by `scale` (= new
    period / 10us baseline). For "essentially-DC" sources (Tper > 1s, used
    for one-shot init pulses) only Td is scaled, Tper is left alone.

    Returns the count of PULSE sources patched."""
    if scale == 1.0:
        return 0
    with open(asc_path) as f:
        content = f.read()

    n_patched = [0]

    def _patch_one(m):
        prefix = m.group(1)
        v1, v2 = m.group(2), m.group(3)
        td_s = _parse_spice_time(m.group(4))
        tr   = m.group(5)
        tf   = m.group(6)
        tpw_s = _parse_spice_time(m.group(7))
        tper_s = _parse_spice_time(m.group(8))
        ncyc = m.group(9)

        # Three classes of PULSE source need different scaling rules:
        #
        #  (A) "Stay-asserted-forever" one-shots: Tper > 1 s (uses 50-sec
        #      sentinels in controls.asc). Only Td scales (= position
        #      within first clock cycle). Tpw and Tper stay at sentinel.
        #
        #  (B) "Finite one-shot init pulses": Ncycles == 1. Their job is
        #      to assert long enough for an RC-limited internal latch to
        #      settle (Micro1, Reset, etc.). That settling time is
        #      RC-limited (τ ≈ 845 ns), NOT frequency-limited. So we
        #      KEEP Tpw at its absolute value — only Td scales. Tper
        #      is irrelevant for a single-cycle pulse but we keep it
        #      absolute too for consistency.
        #
        #  (C) Continuous clocks: Ncycles == 0 or large. All four time
        #      parameters (Td, Tpw, Tper) scale with the frequency.
        #
        # Empirical confirmation that (B) matters: with V1 (Startup_Pulse,
        # Ncycles=1, Tpw=2us at 100 kHz) scaled to Tpw=1us at 200 kHz,
        # the CPU's Micro1 flip-flop never latched (RC settling needs
        # the original 2us). Keeping it absolute fixes this.
        is_dc_sentinel = tper_s > 1.0
        is_oneshot_init = (ncyc is not None and ncyc.strip() == '1')

        if is_dc_sentinel:
            new_td = td_s * scale
            new_tpw = tpw_s
            new_tper = tper_s
        elif is_oneshot_init:
            # Td scales (phase relative to clocks); Tpw absolute (RC).
            new_td = td_s * scale
            new_tpw = tpw_s
            new_tper = tper_s
        else:
            # Continuous clock — everything scales together.
            new_td = td_s * scale
            new_tpw = tpw_s * scale
            new_tper = tper_s * scale

        n_patched[0] += 1

        rebuilt = (f'{prefix}{v1} {v2} '
                   f'{_format_spice_time(new_td)} {tr} {tf} '
                   f'{_format_spice_time(new_tpw)} '
                   f'{_format_spice_time(new_tper)}')
        if ncyc is not None:
            rebuilt += f' {ncyc}'
        rebuilt += ')'
        return rebuilt

    new_content = PULSE_RE.sub(_patch_one, content)
    with open(asc_path, 'w') as f:
        f.write(new_content)
    return n_patched[0]


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
    parser.add_argument('--clock-period-us', type=float, default=10.0,
                        help='CLK period in microseconds (default 10us = '
                             '100 kHz). Scales every timing constant in '
                             'rom_emulator AND every PULSE source in the '
                             'built .asc by the same ratio.')
    parser.add_argument('--frequency-hz', type=float, default=None,
                        help='CLK frequency in Hz (alternative to '
                             '--clock-period-us). e.g. 1e6 = 1MHz, '
                             '500e3 = 500kHz.')
    args = parser.parse_args()

    # Resolve clock period
    clock_period_us = args.clock_period_us
    if args.frequency_hz is not None:
        clock_period_us = 1e6 / args.frequency_hz
    scale = clock_period_us / 10.0
    if scale != 1.0:
        print(f'>>> Clock period: {clock_period_us:g}us '
              f'(= {1e6/clock_period_us:g} Hz), scale = {scale:g}')

    asc_path = os.path.join(PROJECT_ROOT, 'cpus', args.cpu, f'{args.cpu}.asc')

    # 1. Build the CPU using rom_emulator (handles PWL gen, build_cpu, and .tran patch)
    print('=' * 70)
    print('Step 1: Build CPU and generate PWL')
    print('=' * 70)
    rom_emulator = os.path.join(os.path.dirname(__file__), 'rom_emulator.py')
    rom_emu_cmd = [
        sys.executable, rom_emulator, args.program,
        '--cpu', args.cpu,
        '--cycles', str(args.cycles),
        '--startup-delay', str(args.startup_delay),
        '--clock-period-us', str(clock_period_us),
    ]
    result = subprocess.run(rom_emu_cmd, capture_output=True, text=True)
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

    # 3b. If running off the 100 kHz baseline, scale every PULSE source in
    # the built .asc to match. PWLs are already at the right timebase
    # because rom_emulator was invoked with --clock-period-us above.
    if scale != 1.0:
        print()
        print('=' * 70)
        print(f'Step 3b: Scale PULSE clock sources by {scale:g}')
        print('=' * 70)
        n_pulses = patch_clock_pulses(asc_path, scale)
        print(f'Scaled {n_pulses} PULSE source(s); new CLK period = '
              f'{clock_period_us:g}us ({1e6/clock_period_us:g} Hz)')

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
