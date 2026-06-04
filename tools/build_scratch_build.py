#!/usr/bin/env python3
"""Build the Scratch Build CPU into a single flat .asc.

Differences from tools/build_cpu.py (which targets cpus/4004/):

  - The scratchpad is already pre-assembled into `scratch pad.asc` at
    the top level. We just include it as-is.
  - The stack still comes from a sub-folder (`stack/`) and needs to be
    assembled inline from its 5 component files.
  - Sub-circuits were drawn independently and have overlapping
    coordinate ranges. We compute each one's bounding box and lay them
    out in a vertical column so no two physically overlap.
  - InstNames get a per-source prefix so duplicate `J1`/`R1`/etc. from
    different sub-files don't collide in the merged output.
  - No `.save` SPICE directives are added by default (user wants to
    see every net while debugging the new hardwired ROM/PC/IR
    architecture). Opt back in with --save.

Output: cpus/Scratch Build/CPU.asc

Usage:
    python tools/build_scratch_build.py
    python tools/build_scratch_build.py --save           # include .save
    python tools/build_scratch_build.py --margin 4000    # bigger gaps
"""

import argparse
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CPU_ROOT = PROJECT_ROOT / 'cpus' / 'Scratch Build'


# Top-level sub-circuits. Order roughly matches the data-path flow
# (ROM -> IR -> decode -> ALU -> registers -> back to address).
TOP_LEVEL = [
    ('ALU_',     'alu.asc'),
    ('IR_',      'instruction_register.asc'),
    ('MI_',      'micro_instructions.asc'),
    ('Cont_',    'controls.asc'),
    ('PC_',      'program_counter.asc'),
    ('Counter_', 'step_counter.asc'),
    ('SP_',      'scratch pad.asc'),
]

# Stack sub-folder assembly. Same convention as 4004's stack: Controls
# + Bus + Level 1..3.
STACK_DIR_NAME = 'stack'
STACK_PARTS = [
    'Controls.asc',
    'Bus.asc',
    'Level 1.asc',
    'Level 2.asc',
    'Level 3.asc',
]
STACK_PREFIX = 'Stack_'


# Layout parameters
LTSPICE_GRID = 16   # LTspice's coordinate grid; offsets must be multiples
VERT_MARGIN = 2000  # space between vertically-stacked sub-circuits
LEFT_MARGIN = 0     # leftmost x after translation


# ── Schematic line parsing / translation ────────────────────────────────

INSTNAME_RE = re.compile(r'^(SYMATTR\s+InstName\s+)(.+)$')


def snap(v: int) -> int:
    """Round to LTspice's grid."""
    return int(round(v / LTSPICE_GRID) * LTSPICE_GRID)


def translate_line(line: str, dx: int, dy: int) -> str:
    """Apply (dx, dy) translation to any coordinate in a schematic line.

    LTspice line types that carry coordinates:
      WIRE x1 y1 x2 y2
      SYMBOL name x y rotation
      FLAG x y label
      TEXT x y align size content
      IOPIN x y dir
      LINE Normal x1 y1 x2 y2 [width]
      RECTANGLE Normal x1 y1 x2 y2 [...]
      CIRCLE Normal x1 y1 x2 y2 [...]
    """
    stripped = line.rstrip('\n')
    parts = stripped.split()
    if not parts:
        return line
    cmd = parts[0]
    try:
        if cmd == 'WIRE' and len(parts) == 5:
            x1, y1, x2, y2 = (int(parts[1]) + dx, int(parts[2]) + dy,
                              int(parts[3]) + dx, int(parts[4]) + dy)
            return f'WIRE {x1} {y1} {x2} {y2}\n'
        if cmd == 'SYMBOL' and len(parts) >= 4:
            name = parts[1]
            x = int(parts[2]) + dx
            y = int(parts[3]) + dy
            rest = ' '.join(parts[4:])
            sep = ' ' if rest else ''
            return f'SYMBOL {name} {x} {y}{sep}{rest}\n'
        if cmd == 'FLAG' and len(parts) >= 4:
            x = int(parts[1]) + dx
            y = int(parts[2]) + dy
            rest = ' '.join(parts[3:])
            return f'FLAG {x} {y} {rest}\n'
        if cmd == 'TEXT' and len(parts) >= 3:
            x = int(parts[1]) + dx
            y = int(parts[2]) + dy
            rest = ' '.join(parts[3:])
            return f'TEXT {x} {y} {rest}\n'
        if cmd == 'IOPIN' and len(parts) >= 4:
            x = int(parts[1]) + dx
            y = int(parts[2]) + dy
            rest = ' '.join(parts[3:])
            return f'IOPIN {x} {y} {rest}\n'
        if cmd in ('LINE', 'RECTANGLE', 'CIRCLE') and len(parts) >= 6:
            x1 = int(parts[2]) + dx
            y1 = int(parts[3]) + dy
            x2 = int(parts[4]) + dx
            y2 = int(parts[5]) + dy
            tail = ' '.join(parts[6:])
            sep = ' ' if tail else ''
            return f'{cmd} {parts[1]} {x1} {y1} {x2} {y2}{sep}{tail}\n'
    except ValueError:
        return line
    return line


def content_bbox(lines):
    """Find (xmin, ymin, xmax, ymax) of real schematic content.

    Considers WIRE endpoints, SYMBOL anchors, FLAG positions, and
    LINE/RECTANGLE/CIRCLE corners. Ignores TEXT (often off in the
    margins for comments) and Version/SHEET headers.
    """
    xs, ys = [], []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]
        try:
            if cmd == 'WIRE' and len(parts) >= 5:
                xs += [int(parts[1]), int(parts[3])]
                ys += [int(parts[2]), int(parts[4])]
            elif cmd == 'SYMBOL' and len(parts) >= 4:
                xs.append(int(parts[2])); ys.append(int(parts[3]))
            elif cmd == 'FLAG' and len(parts) >= 4:
                xs.append(int(parts[1])); ys.append(int(parts[2]))
            elif cmd in ('LINE', 'RECTANGLE', 'CIRCLE') and len(parts) >= 6:
                xs += [int(parts[2]), int(parts[4])]
                ys += [int(parts[3]), int(parts[5])]
        except ValueError:
            pass
    if not xs:
        return None
    return min(xs), min(ys), max(xs), max(ys)


def rename_in_lines(lines, prefix):
    """Apply per-(type-letter) renumbering with `prefix` ahead of every
    SYMATTR InstName. Returns (new_lines, n_renamed)."""
    counters = {}
    out = []
    n = 0
    for line in lines:
        m = INSTNAME_RE.match(line.rstrip())
        if m:
            old = m.group(2).strip()
            t_match = re.match(r'^([A-Za-z]+)', old)
            tletter = t_match.group(1) if t_match else 'X'
            counters[tletter] = counters.get(tletter, 0) + 1
            new = f'{prefix}{tletter}{counters[tletter]}'
            out.append(f'{m.group(1)}{new}\n')
            n += 1
        else:
            out.append(line if line.endswith('\n') else line + '\n')
    return out, n


def load_lines(path):
    with open(path) as f:
        return f.readlines()


# ── Sub-circuit composition ─────────────────────────────────────────────

def assemble_stack(stack_dir):
    """Concatenate stack/Controls.asc + Bus.asc + Level 1..3.asc into a
    single block of schematic lines (without Version/SHEET header) with
    a per-part InstName prefix to avoid name collisions WITHIN the
    stack. The outer build then applies STACK_PREFIX on top."""
    out = []
    n_total = 0
    for i, fname in enumerate(STACK_PARTS):
        p = stack_dir / fname
        if not p.exists():
            print(f'  MISSING stack part: {p}')
            continue
        lines = load_lines(p)
        # Per-part prefix within the stack assembly
        part_tag = Path(fname).stem.replace(' ', '')
        renamed, n = rename_in_lines(lines, f'{STACK_PREFIX}{part_tag}_')
        n_total += n
        if i == 0:
            out.extend(renamed)            # keep Version+SHEET on first part
        else:
            out.extend(renamed[2:])         # skip on subsequent parts
    return out, n_total


# ── Main build ──────────────────────────────────────────────────────────

def build(include_save: bool, vert_margin: int):
    out_lines = []
    # Stub Version + SHEET — we'll patch SHEET dimensions after layout
    out_lines.append('Version 4.1\n')
    out_lines.append('SHEET 1 0 0\n')

    y_cursor = 0
    max_width = 0
    stats = []

    # Pre-assemble the stack subassembly into in-memory lines
    stack_lines, stack_n_pre = assemble_stack(CPU_ROOT / STACK_DIR_NAME)

    # Build each top-level + the stack subassembly
    pipeline = list(TOP_LEVEL) + [(STACK_PREFIX, ('__stack__', stack_lines))]

    for prefix, source in pipeline:
        if isinstance(source, str):
            path = CPU_ROOT / source
            if not path.exists():
                print(f'  MISSING: {path}')
                continue
            content = load_lines(path)[2:]   # drop Version+SHEET header
            label = source
        else:
            label, raw_lines = source
            content = raw_lines[2:] if (len(raw_lines) >= 2 and
                raw_lines[0].startswith('Version')) else raw_lines

        bb = content_bbox(content)
        if bb is None:
            print(f'  EMPTY: {label}')
            continue
        xmin, ymin, xmax, ymax = bb

        # Translate so content's top-left lands at (LEFT_MARGIN, y_cursor)
        dx = snap(LEFT_MARGIN - xmin)
        dy = snap(y_cursor - ymin)
        translated = [translate_line(l, dx, dy) for l in content]

        # Rename InstNames so we don't collide across sub-circuits
        # (the stack already has internal per-part prefixes; we strip
        # those and apply STACK_PREFIX once, OR just apply STACK_PREFIX
        # on top of the inner prefix — second option means inner names
        # become Stack_Stack_Controls_J1 etc. Skip outer rename for stack
        # because it's already uniquely prefixed.)
        if label == '__stack__':
            renamed = translated
            n_inst = stack_n_pre
        else:
            renamed, n_inst = rename_in_lines(translated, prefix)

        out_lines.extend(renamed)

        width = xmax - xmin
        height = ymax - ymin
        max_width = max(max_width, width)
        stats.append((label, prefix, n_inst, width, height, y_cursor))
        y_cursor += height + vert_margin

    # Patch SHEET to fit content
    sheet_w = max(60000, max_width + 2000)
    sheet_h = max(60000, y_cursor + 2000)
    out_lines[1] = f'SHEET 1 {sheet_w} {sheet_h}\n'

    # Upgrade .options to the full convergence-helping set we tuned on
    # the 4004. Sub-circuits typically ship with just `NoOpIter` (or
    # nothing at all); add cshunt/gminsteps/itl1/srcsteps so LTspice
    # can actually solve the DC operating point on a 8000+ JFET network.
    new_options = ('!.options NoOpIter cshunt=50f gminsteps=200 '
                   'itl1=1000 srcsteps=20')
    patched = False
    for i, line in enumerate(out_lines):
        if '!.options' in line:
            # Replace whatever options line the sub-circuit had
            out_lines[i] = re.sub(r'!\.options[^"\n]*', new_options, line)
            patched = True
            break
    if not patched:
        # No existing .options — append one as a TEXT directive near origin
        out_lines.append(f'TEXT 0 -1000 Left 2 {new_options}\n')

    # Optional .save injection
    if include_save:
        # Reuse build_and_run_4004's net-finding logic
        import sys
        sys.path.insert(0, str(PROJECT_ROOT / 'tools'))
        from build_and_run_4004 import find_named_nets, append_save_directives
        # Write file first so we can scan it
        out_path = CPU_ROOT / 'CPU.asc'
        out_path.write_text(''.join(out_lines))
        nets = find_named_nets(str(out_path))
        n_save = append_save_directives(str(out_path), nets)
        print(f'\n  Injected {n_save} .save TEXT lines covering {len(nets)} nets')
    else:
        out_path = CPU_ROOT / 'CPU.asc'
        out_path.write_text(''.join(out_lines))

    # Post-build sanity check: detect SYMBOLs sharing identical
    # (type, x, y, rot) tuples — that's exactly what LTspice's "duplicate
    # overlapping components" warning catches. Reports the colliding
    # InstNames so the user can fix the source .asc.
    SYM_RE = re.compile(r'^SYMBOL\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(\S+)')
    INST_RE = re.compile(r'^SYMATTR\s+InstName\s+(\S+)')
    positions = {}   # (type, x, y, rot) -> list of InstNames
    cur_key = None
    out_path = CPU_ROOT / 'CPU.asc'
    out_path.write_text(''.join(out_lines))   # write before reading back
    for line in out_path.read_text().splitlines():
        m = SYM_RE.match(line)
        if m:
            cur_key = (m.group(1), int(m.group(2)), int(m.group(3)),
                       m.group(4))
            continue
        m = INST_RE.match(line)
        if m and cur_key is not None:
            positions.setdefault(cur_key, []).append(m.group(1))
            cur_key = None
    overlaps = {k: v for k, v in positions.items() if len(v) > 1}
    if overlaps:
        print(f'\n*** {len(overlaps)} OVERLAPPING SYMBOL POSITION(S) DETECTED ***')
        for (typ, x, y, rot), names in list(overlaps.items())[:10]:
            print(f'  {typ} @ ({x},{y}) {rot}: {", ".join(names)}')
        if len(overlaps) > 10:
            print(f'  ... and {len(overlaps) - 10} more')
        print('  LTspice will warn about these on file open.')
    else:
        print('\n  No overlapping SYMBOL placements detected.')

    # Report
    print(f'\n{"label":<32} {"prefix":<10} {"InstNames":>10} '
          f'{"width":>8} {"height":>8} {"y_top":>8}')
    print('-' * 84)
    total_inst = 0
    for label, prefix, n, w, h, y in stats:
        print(f'  {label:<30} {prefix:<10} {n:>10} {w:>8} {h:>8} {y:>8}')
        total_inst += n
    print('-' * 84)
    print(f'  TOTAL InstNames: {total_inst}')
    print(f'  Sheet: {sheet_w} × {sheet_h}')
    print(f'  Wrote {out_path}')


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--save', action='store_true',
                   help='Inject .save SPICE directives (omit by default '
                        'so every net is visible for debugging).')
    p.add_argument('--margin', type=int, default=VERT_MARGIN,
                   help=f'Vertical gap between sub-circuits in LTspice units '
                        f'(default {VERT_MARGIN}).')
    args = p.parse_args()
    build(include_save=args.save, vert_margin=args.margin)


if __name__ == '__main__':
    main()
