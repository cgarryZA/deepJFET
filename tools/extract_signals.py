#!/usr/bin/env python3
"""Extract a subset of signals from a large LTSpice .raw file.

Reads the full file (can be 60GB+), extracts only the signals needed
for the reg_viewer, and writes a much smaller .raw file.

Usage:
    python tools/extract_signals.py cpus/4004/4004.raw -o cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os, sys, struct, argparse, re

# Signals we need for the reg_viewer
NEEDED_SIGNALS = (
    # Time is always index 0
    [f"v(acc{i})" for i in range(4)] +
    ["v(cf0)", "v(cf)"] +
    [f"v(ir1_{i})" for i in range(4)] +
    [f"v(ir2_{i})" for i in range(4)] +
    [f"v(pc1{i})" for i in range(4)] +  # try pc1X format
    [f"v(pc2{i})" for i in range(4)] +
    [f"v(pc3{i})" for i in range(4)] +
    [f"v(pc1_{i})" for i in range(4)] +  # try pc1_X format too
    [f"v(pc2_{i})" for i in range(4)] +
    [f"v(pc3_{i})" for i in range(4)] +
    [f"v(bus{i})" for i in range(4)] +
    [f"v(scratch{r}_{b})" for r in range(16) for b in range(4)] +
    ["v(micro1)"] +
    [f"v(index{i})" for i in range(4)] +
    ["v(indexreg_loading)", "v(indexreg_load)"] +
    [f"v(scratch{i:03b})" for i in range(8)] +  # Scratch000-Scratch111
    ["v(cf_out)", "v(cf_inv)"]
)


def main():
    parser = argparse.ArgumentParser(description="Extract signals from large .raw file")
    parser.add_argument("raw_file", help="Input .raw file (can be huge)")
    parser.add_argument("-o", "--output", default=None, help="Output .raw file")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.raw_file)
        args.output = base + "_extracted" + ext

    in_size = os.path.getsize(args.raw_file)
    print(f"Input: {args.raw_file} ({in_size / 1e9:.1f} GB)")

    # ── Parse header ──
    with open(args.raw_file, "rb") as f:
        header_bytes = f.read(4_000_000)  # 4MB should be enough for 27K vars

    header_text = header_bytes.decode("utf-16-le", errors="replace")

    n_vars = 0
    n_points = 0
    var_lines = []  # (index, name, type_str, full_line)

    for line in header_text.split("\n"):
        s = line.strip()
        if s.startswith("No. Variables"):
            n_vars = int(s.split(":")[1].strip())
        elif s.startswith("No. Points"):
            n_points = int(s.split(":")[1].strip())
        elif "\t" in s:
            parts = s.split("\t")
            if len(parts) >= 3:
                try:
                    idx = int(parts[0].strip())
                    name = parts[1].strip()
                    type_str = parts[2].strip() if len(parts) > 2 else "voltage"
                    var_lines.append((idx, name, type_str))
                except ValueError:
                    pass
        elif s == "Binary:":
            break

    binary_marker = header_text.find("Binary:")
    data_start = (binary_marker + len("Binary:") + 1) * 2
    row_size = 8 + (n_vars - 1) * 4

    print(f"  {n_vars} variables, {n_points} points, row_size={row_size}")

    # ── Find which signals to extract ──
    needed_lower = set(s.lower() for s in NEEDED_SIGNALS)
    var_map = {}  # lowercase name -> (orig_idx, name, type_str)
    for idx, name, type_str in var_lines:
        var_map[name.lower()] = (idx, name, type_str)

    # Always include time (index 0)
    extract = [(0, var_lines[0][1], var_lines[0][2])]  # time
    found_names = set()
    for needed in needed_lower:
        if needed in var_map:
            orig_idx, name, type_str = var_map[needed]
            if name.lower() not in found_names:
                extract.append((orig_idx, name, type_str))
                found_names.add(name.lower())

    # Also try to find pc signals by pattern matching if not found
    for name_lower, (orig_idx, name, type_str) in var_map.items():
        if name_lower not in found_names:
            # Match pc1, pc2, pc3 with various naming patterns
            if re.match(r'v\(pc[123]\d?\)', name_lower) or \
               re.match(r'v\(pc[123]_\d\)', name_lower):
                extract.append((orig_idx, name, type_str))
                found_names.add(name_lower)

    # Sort by original index
    extract.sort(key=lambda x: x[0])

    n_extract = len(extract)
    print(f"  Extracting {n_extract} signals (of {n_vars})")
    for orig_idx, name, _ in extract[:10]:
        print(f"    [{orig_idx}] {name}")
    if n_extract > 10:
        print(f"    ... and {n_extract - 10} more")

    # ── Build output header ──
    # Reconstruct header with only extracted variables
    out_header_lines = []
    for line in header_text.split("\n"):
        s = line.strip()
        if s.startswith("No. Variables"):
            out_header_lines.append(f"No. Variables: {n_extract}")
        elif s.startswith("No. Points"):
            out_header_lines.append(f"No. Points: {n_points}")
        elif "\t" in s:
            # Skip variable lines — we'll write our own
            parts = s.split("\t")
            if len(parts) >= 3:
                try:
                    int(parts[0].strip())
                    continue  # skip original var line
                except ValueError:
                    pass
            out_header_lines.append(s)
        elif s == "Variables:":
            out_header_lines.append(s)
            # Write our variable list
            for new_idx, (orig_idx, name, type_str) in enumerate(extract):
                out_header_lines.append(f"\t{new_idx}\t{name}\t{type_str}")
        elif s == "Binary:":
            out_header_lines.append(s)
            break
        else:
            out_header_lines.append(s)

    out_header_text = "\n".join(out_header_lines) + "\n"
    out_header_bytes = out_header_text.encode("utf-16-le")

    out_row_size = 8 + (n_extract - 1) * 4

    # ── Extract data ──
    print(f"\nExtracting {n_points} points...")
    # Build offset list for extraction
    offsets = []
    for orig_idx, name, _ in extract:
        if orig_idx == 0:
            offsets.append(('d', 0))  # time: double at offset 0
        else:
            offsets.append(('f', 8 + (orig_idx - 1) * 4))  # float at offset

    with open(args.raw_file, "rb") as fin, open(args.output, "wb") as fout:
        fout.write(out_header_bytes)

        for pt in range(n_points):
            if pt % 50000 == 0:
                pct = pt / n_points * 100
                print(f"  {pt}/{n_points} ({pct:.0f}%)", end="\r")

            base = data_start + pt * row_size

            # Read time (always first)
            fin.seek(base)
            t_bytes = fin.read(8)
            fout.write(t_bytes)

            # Read each extracted signal
            for i, (orig_idx, name, _) in enumerate(extract):
                if orig_idx == 0:
                    continue  # time already written
                fin.seek(base + 8 + (orig_idx - 1) * 4)
                fout.write(fin.read(4))

    out_size = os.path.getsize(args.output)
    print(f"\nDone: {args.output}")
    print(f"  {in_size / 1e9:.1f} GB -> {out_size / 1e6:.1f} MB "
          f"({n_vars} -> {n_extract} signals, {n_points} points)")


if __name__ == "__main__":
    main()
