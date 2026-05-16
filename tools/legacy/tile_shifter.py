#!/usr/bin/env python3
"""Tile the shift logic from 1-bit to N-bit.

Uses:
    1 bit.asc — base (carry cell + MSB bit cell)
    tile N.asc — template for each additional bit, with placeholders:
        A_QN+1 (shift left input), A_QN-1 (shift right input),
        BusN, TempNIn
    shift_logic.asc — direction control (always appended)

Usage:
    python tools/tile_shifter.py --bits 4 --output shifter_4bit.asc
"""

import argparse
import os
import re


SHIFTER_DIR = os.path.join(os.path.dirname(__file__), "..",
                           "cpus", "tileable", "bits", "ALU", "Shifter")
TILE_OFFSET_X = 912


def read_asc(path):
    with open(path) as f:
        lines = f.readlines()
    return lines[:2], lines[2:]


def offset_lines(lines, dx, dy=0):
    result = []
    for line in lines:
        if line.startswith("WIRE "):
            p = line.split()
            result.append(f"WIRE {int(p[1])+dx} {int(p[2])+dy} "
                          f"{int(p[3])+dx} {int(p[4])+dy}\n")
        elif line.startswith("FLAG "):
            p = line.strip().split(None, 3)
            name = p[3] if len(p) > 3 else ""
            result.append(f"FLAG {int(p[1])+dx} {int(p[2])+dy} {name}\n")
        elif line.startswith("SYMBOL "):
            p = line.split()
            rest = " ".join(p[4:])
            result.append(f"SYMBOL {p[1]} {int(p[2])+dx} {int(p[3])+dy} {rest}\n")
        else:
            result.append(line)
    return result


def rename_flags(lines, renames):
    result = []
    for line in lines:
        if line.startswith("FLAG "):
            p = line.strip().split(None, 3)
            if len(p) >= 4 and p[3] in renames:
                line = f"FLAG {p[1]} {p[2]} {renames[p[3]]}\n"
        result.append(line)
    return result


def renumber_instnames(lines):
    counters = {}
    result = []
    for line in lines:
        if line.startswith("SYMATTR InstName "):
            old = line.split(" ", 2)[2].strip()
            m = re.match(r"([A-Z_!]+?)(\d+)$", old)
            if m:
                pfx = m.group(1)
                counters[pfx] = counters.get(pfx, 0) + 1
                result.append(f"SYMATTR InstName {pfx}{counters[pfx]}\n")
            else:
                result.append(line)
        else:
            result.append(line)
    return result


def tile_shifter(n_bits, output_path):
    print(f"Tiling: {n_bits}-bit shifter")

    base_path = os.path.join(SHIFTER_DIR, "1 bit.asc")
    tile_path = os.path.join(SHIFTER_DIR, "tile N.asc")
    logic_path = os.path.join(SHIFTER_DIR, "shift_logic.asc")

    header, base_body = read_asc(base_path)
    _, tile_body = read_asc(tile_path)
    _, logic_body = read_asc(logic_path)

    msb = n_bits - 1

    # Update the 1-bit base:
    # - Carry cell's Shift_Right A_Q0 (at x=-4240) -> A_Q{MSB}
    # - Bit-0 cell becomes the MSB: Bus0 -> Bus{MSB}, Temp0In -> Temp{MSB}In
    # - Bit-0 cell's Shift_Left CF (at x=-3440) -> A_Q{MSB-1} if n>1
    # - Bit-0 cell's A_Q0 refs stay as A_Q0 (for carry cell connections)
    updated_base = []
    for line in base_body:
        if line.startswith("FLAG "):
            p = line.strip().split(None, 3)
            if len(p) >= 4:
                x = int(p[1])
                name = p[3]

                # Carry cell has two A_Q0 refs:
                # x=-4400 (Shift_Left): carry shift left = receives from LSB (A_Q0)
                # stays A_Q0 — no change
                # x=-4240 (Shift_Right): carry shift right = receives from MSB
                if x == -4240 and name == "A_Q0":
                    line = f"FLAG {p[1]} {p[2]} A_Q{msb}\n"

                # MSB bit cell (x=-3440 and x=-3280):
                # In 1-bit base, both shift inputs are CF (only carry above/below)
                # In multi-bit:
                #   Shift_Left at x=-3440: MSB shift left receives from carry (CF)
                #     -> stays CF, no change
                #   Shift_Right at x=-3280: MSB shift right receives from bit below
                #     -> CF becomes A_Q{MSB-1}
                elif x == -3280 and name == "CF" and n_bits > 1:
                    line = f"FLAG {p[1]} {p[2]} A_Q{msb - 1}\n"

                # Rename Bus0 -> Bus{MSB}
                elif name == "Bus0":
                    line = f"FLAG {p[1]} {p[2]} Bus{msb}\n"

                # Rename Temp0In -> Temp{MSB}In
                elif name == "Temp0In":
                    line = f"FLAG {p[1]} {p[2]} Temp{msb}In\n"

        updated_base.append(line)

    # Tile bits MSB-1 down to 0
    all_tiles = []
    for i in range(1, n_bits):
        bit_idx = msb - i  # MSB-1, MSB-2, ..., 0
        dx = TILE_OFFSET_X * i

        shifted = offset_lines(tile_body, dx)

        # Replace placeholders
        # Shift LEFT: bits move toward MSB, so bit N receives from N-1 (below)
        #   A_QN-1 is the shift LEFT input (bit below feeds in)
        # Shift RIGHT: bits move toward LSB, so bit N receives from N+1 (above)
        #   A_QN+1 is the shift RIGHT input (bit above feeds in)
        if bit_idx == 0:
            shift_left_src = "CF"  # bit 0 shift left receives from carry
        else:
            shift_left_src = f"A_Q{bit_idx - 1}"

        if bit_idx == msb:
            shift_right_src = "CF"  # MSB shift right receives from carry
        else:
            shift_right_src = f"A_Q{bit_idx + 1}"

        renames = {
            "A_QN-1": shift_left_src,   # shift left input = bit below
            "A_QN+1": shift_right_src,  # shift right input = bit above
            "BusN": f"Bus{bit_idx}",
            "TempNIn": f"Temp{bit_idx}In",
        }
        shifted = rename_flags(shifted, renames)
        all_tiles.extend(shifted)

    # Combine
    all_lines = header + updated_base + all_tiles + logic_body
    all_lines = renumber_instnames(all_lines)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(all_lines)

    n_sym = sum(1 for l in all_lines if l.startswith("SYMBOL "))
    print(f"  Output: {output_path} ({n_sym} symbols)")


def main():
    parser = argparse.ArgumentParser(description="Tile shift logic to N bits")
    parser.add_argument("--bits", "-n", type=int, required=True)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()
    tile_shifter(args.bits, args.output)


if __name__ == "__main__":
    main()
