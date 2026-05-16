#!/usr/bin/env python3
"""Read the final register state from the full 60GB .raw file.

Only reads the header and the very last few data points — doesn't load
the whole file.

Usage:
    python tools/read_final_state.py cpus/4004/4004.raw
"""

import os, sys, struct, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_file")
    parser.add_argument("--points-from-end", type=int, default=100,
                        help="How many points from the end to read")
    args = parser.parse_args()

    path = args.raw_file
    file_size = os.path.getsize(path)
    print(f"File: {path} ({file_size / 1e9:.1f} GB)")

    # Parse header
    with open(path, "rb") as f:
        header = f.read(2_000_000)

    text = header.decode("utf-16-le", errors="replace")

    n_vars = 0
    n_points = 0
    var_names = {}

    for line in text.split("\n"):
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
                    var_names[name.lower()] = idx
                except ValueError:
                    pass
        elif s == "Binary:":
            break

    idx = text.find("Binary:")
    data_start = (idx + len("Binary:") + 1) * 2
    row_size = 8 + (n_vars - 1) * 4

    print(f"  {n_vars} variables, {n_points} points")
    print(f"  Row size: {row_size} bytes")
    print(f"  Data start: byte {data_start}")
    expected_size = data_start + n_points * row_size
    print(f"  Expected file size: {expected_size / 1e9:.1f} GB (actual: {file_size / 1e9:.1f} GB)")

    def get_nibble(values, prefix, thresh=-2.5):
        val = 0
        for i in range(4):
            key = f"v({prefix}{i})"
            if key in values and values[key] > thresh:
                val |= (1 << i)
        return val

    def read_point(f, pt_idx):
        f.seek(data_start + pt_idx * row_size)
        row = f.read(row_size)
        if len(row) < row_size:
            return None, None
        t = struct.unpack_from("d", row, 0)[0]
        values = {}
        for name, vidx in var_names.items():
            if vidx == 0:
                values[name] = t
            else:
                values[name] = struct.unpack_from("f", row, 8 + (vidx - 1) * 4)[0]
        return t, values

    # Read last N points
    print(f"\nReading last {args.points_from_end} points...")
    with open(path, "rb") as f:
        for offset in [args.points_from_end, 50, 10, 1]:
            pt = n_points - offset
            if pt < 0:
                continue
            t, vals = read_point(f, pt)
            if t is None:
                print(f"  pt {pt}: FAILED TO READ")
                continue

            acc = get_nibble(vals, "acc")
            ir1 = get_nibble(vals, "ir1_")
            ir2 = get_nibble(vals, "ir2_")
            pc1 = get_nibble(vals, "pc1")
            pc2 = get_nibble(vals, "pc2")
            pc3 = get_nibble(vals, "pc3")
            cy_v = vals.get("v(cf0)", -5)
            cy = 1 if cy_v > -2.5 else 0

            regs = [get_nibble(vals, f"scratch{r}_") for r in range(16)]
            pc = (pc3 << 8) | (pc2 << 4) | pc1
            ir = (ir1 << 4) | ir2

            print(f"\n  pt {pt} (t = {t*1e3:.3f}ms):")
            print(f"    IR = 0x{ir:02X}  PC = 0x{pc:03X}  ACC = 0x{acc:X}  CY = {cy}")
            print(f"    Regs: [{', '.join(f'R{i}=0x{r:X}' for i, r in enumerate(regs))}]")

    # Expected values
    print(f"\n{'='*60}")
    print(f"EXPECTED (from Python sim):")
    print(f"    R12=0x9  R13=0x8  R14=0x6  R15=0x2")
    print(f"    ACC=0x0  IR=0x00 (NOP)  CY=0")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
