#!/usr/bin/env python3
"""Create a smoothed .raw file by averaging values per clock cycle.

For each CLK period (10us), averages all data points within that period
and writes a single point per cycle. Eliminates sampling/transition noise.

Usage:
    python tools/smooth_raw.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import os, sys, struct, argparse
import numpy as np

CLK_PERIOD = 10e-6  # 10us


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--clk-period", type=float, default=CLK_PERIOD)
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.raw_file)
        args.output = base + "_smoothed" + ext

    # ── Parse header ──
    with open(args.raw_file, "rb") as f:
        header_bytes = f.read(2_000_000)

    header_text = header_bytes.decode("utf-16-le", errors="replace")

    n_vars = 0
    n_points = 0
    var_names = []

    for line in header_text.split("\n"):
        s = line.strip()
        if s.startswith("No. Variables"):
            n_vars = int(s.split(":")[1].strip())
        elif s.startswith("No. Points"):
            n_points = int(s.split(":")[1].strip())

    binary_marker = header_text.find("Binary:")
    data_start = (binary_marker + len("Binary:") + 1) * 2
    row_size = 8 + (n_vars - 1) * 4

    print(f"Input: {args.raw_file}")
    print(f"  {n_vars} variables, {n_points} points")
    print(f"  Header ends at byte {data_start}, row size = {row_size}")

    # ── Read all data ──
    print("Reading all data points...")
    times = np.zeros(n_points, dtype=np.float64)
    signals = np.zeros((n_points, n_vars - 1), dtype=np.float32)

    with open(args.raw_file, "rb") as f:
        for i in range(n_points):
            f.seek(data_start + i * row_size)
            times[i] = struct.unpack_from("d", f.read(8), 0)[0]
            row_data = f.read((n_vars - 1) * 4)
            for j in range(n_vars - 1):
                signals[i, j] = struct.unpack_from("f", row_data, j * 4)[0]

    print(f"  Time range: {times[0]*1e6:.1f}us to {times[-1]*1e6:.1f}us")

    # ── Bin into clock cycles and average ──
    t_min = times[0]
    t_max = times[-1]
    clk = args.clk_period
    n_cycles = int((t_max - t_min) / clk) + 1
    print(f"  {n_cycles} clock cycles at {clk*1e6:.1f}us period")

    # Assign each point to a cycle bin
    bin_idx = ((times - t_min) / clk).astype(np.int64)
    bin_idx = np.clip(bin_idx, 0, n_cycles - 1)

    # Compute average per bin
    print("Averaging per clock cycle...")
    avg_times = np.zeros(n_cycles, dtype=np.float64)
    avg_signals = np.zeros((n_cycles, n_vars - 1), dtype=np.float32)
    counts = np.zeros(n_cycles, dtype=np.int64)

    for i in range(n_points):
        b = bin_idx[i]
        avg_times[b] += times[i]
        avg_signals[b] += signals[i]
        counts[b] += 1

    # Avoid divide by zero for empty bins
    mask = counts > 0
    avg_times[mask] /= counts[mask]
    avg_signals[mask] /= counts[mask, np.newaxis]

    # Remove empty bins
    valid = mask
    avg_times = avg_times[valid]
    avg_signals = avg_signals[valid]
    n_out = len(avg_times)

    print(f"  {n_out} output points ({n_points} -> {n_out}, {n_points/n_out:.1f}x reduction)")

    # ── Write output .raw ──
    # Reuse the original header, just update "No. Points"
    print(f"Writing: {args.output}")

    new_header = header_text[:binary_marker + len("Binary:") + 1]
    new_header = new_header.replace(
        f"No. Points:          {n_points}",
        f"No. Points:          {n_out}"
    )
    # Handle different whitespace formats
    import re
    new_header = re.sub(
        r"No\. Points:\s+\d+",
        f"No. Points:          {n_out}",
        new_header
    )

    header_out = new_header.encode("utf-16-le")

    with open(args.output, "wb") as f:
        f.write(header_out)

        for i in range(n_out):
            f.write(struct.pack("d", avg_times[i]))
            for j in range(n_vars - 1):
                f.write(struct.pack("f", float(avg_signals[i, j])))

    out_size = os.path.getsize(args.output)
    in_size = os.path.getsize(args.raw_file)
    print(f"  {in_size/1e6:.1f}MB -> {out_size/1e6:.1f}MB")
    print("Done.")


if __name__ == "__main__":
    main()
