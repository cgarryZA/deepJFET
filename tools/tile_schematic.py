#!/usr/bin/env python3
"""Tile LTSpice schematics from 1-bit to N-bit.

The 1-bit file is the base (becomes MSB, bit N-1).
The 2-bit file is the exact tile piece at the correct coordinates
for the second bit. It always uses _0 for its cell signals and has
a bus output JFET referencing Reg_1.
The 3-bit file is the tile piece at the correct coordinates for the
third bit (used only to determine the per-tile offset).

To generate N bits:
    1. Take 1-bit base, rename _0 -> _{N-1}
    2. Place tile (from 2-bit file) at positions 1..N-1
    3. Each tile K: offset by (K-1)*dx from the 2-bit tile position
    4. Rename _0 -> _{N-1-K} for cell signals
    5. Rename Reg_K (bus JFET) -> Reg_{N-K}
    6. Renumber all InstNames

Usage:
    python tools/tile_schematic.py \\
        --one "register 1 bit.asc" \\
        --two "register 2 bit.asc" \\
        --three "register 3 bit.asc" \\
        --bits 8 --output register_8bit.asc
"""

import argparse
import os
import re


def read_asc(path):
    with open(path) as f:
        lines = f.readlines()
    return lines[:2], lines[2:]


def find_offset(two_path, three_path):
    """Offset = difference in coordinates between tile 2 and tile 3."""
    def first_reg0(path):
        with open(path) as f:
            for line in f:
                if line.startswith("FLAG ") and line.strip().endswith(" Reg_0"):
                    parts = line.split()
                    return int(parts[1]), int(parts[2])
        return 0, 0

    x2, y2 = first_reg0(two_path)
    x3, y3 = first_reg0(three_path)
    dx, dy = x3 - x2, y3 - y2
    print(f"  Tile offset: dx={dx}, dy={dy}")
    return dx, dy


def offset_lines(lines, dx, dy):
    result = []
    for line in lines:
        if line.startswith("WIRE "):
            p = line.split()
            result.append(f"WIRE {int(p[1])+dx} {int(p[2])+dy} {int(p[3])+dx} {int(p[4])+dy}\n")
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


def rename_in_flags(lines, renames):
    """Apply a dict of exact flag name renames: {old_name: new_name}."""
    result = []
    for line in lines:
        if line.startswith("FLAG "):
            parts = line.strip().split(None, 3)
            if len(parts) >= 4:
                name = parts[3]
                if name in renames:
                    name = renames[name]
                line = f"FLAG {parts[1]} {parts[2]} {name}\n"
        result.append(line)
    return result


def apply_prefix(lines, prefix):
    """Rename Reg_ signals with a custom prefix.

    ONLY Reg_* signals are register-specific. Everything else
    (Bus_, CLK, VDD, VSS, etc.) is shared and stays unchanged.

    Reg_0 -> Acc_0, !Reg_0 -> !Acc_0, Reg_Load -> Acc_Load,
    Reg_Out -> Acc_Out, Reg_Loading -> Acc_Loading, etc.
    Reg_Q0 -> Acc_Q0, !Reg_Q0 -> !Acc_Q0.
    InvertReg -> InvertAcc, !InvertReg -> !InvertAcc.
    """
    result = []
    for line in lines:
        if line.startswith("FLAG "):
            parts = line.strip().split(None, 3)
            if len(parts) >= 4:
                name = parts[3]
                new_name = name

                if name.startswith("!Reg_"):
                    new_name = f"!{prefix}_" + name[5:]
                elif name.startswith("Reg_"):
                    new_name = f"{prefix}_" + name[4:]
                elif name == "InvertReg":
                    new_name = f"Invert{prefix}"
                elif name == "!InvertReg":
                    new_name = f"!Invert{prefix}"
                elif name == "Reg0" or re.match(r"Reg\d+$", name):
                    # Stray Reg0, Reg1 etc without underscore
                    new_name = re.sub(r"^Reg", prefix + "_", name)

                if new_name != name:
                    line = f"FLAG {parts[1]} {parts[2]} {new_name}\n"
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
                prefix = m.group(1)
                counters[prefix] = counters.get(prefix, 0) + 1
                result.append(f"SYMATTR InstName {prefix}{counters[prefix]}\n")
            else:
                result.append(line)
        else:
            result.append(line)
    return result


def strip_bus_output_from_tile(lines):
    """Remove bus output JFETs and flags from a tile.

    Bus output lives at y=-1120 to y=-900. The tile has stray bus output
    elements (Bus0In, Reg_N, GND) that must be removed since the base
    handles all bus output.
    """
    result = []
    skip_symbol = False
    for line in lines:
        if line.startswith("SYMBOL njf"):
            p = line.split()
            y = int(p[3])
            if -1250 <= y <= -880:
                skip_symbol = True
                continue
        if skip_symbol:
            if line.startswith("WINDOW ") or line.startswith("SYMATTR "):
                continue
            skip_symbol = False
        if line.startswith("FLAG "):
            p = line.strip().split()
            y = int(p[2])
            name = p[3] if len(p) > 3 else ""
            if -1250 <= y <= -880:
                if re.match(r"Bus\d*In$", name) or name == "Reg_Outing" or name == "0":
                    continue
                if re.match(r"Reg_\d+$", name):
                    continue
        if line.startswith("WIRE "):
            p = line.split()
            y1, y2 = int(p[2]), int(p[4])
            if -1250 <= y1 <= -880 and -1250 <= y2 <= -880:
                continue
        result.append(line)
    return result


def strip_bus_output_from_base(lines):
    """Remove ONLY the per-bit bus output JFETs from the base.

    The bus output section is at x >= -4208, y between -1104 and -912.
    Everything at x < -4208 in that Y range is control logic and must stay.
    """
    BUS_X_MIN = -4220  # bus output starts at x=-4208

    result = []
    skip_symbol = False
    for line in lines:
        if line.startswith("SYMBOL njf"):
            p = line.split()
            x, y = int(p[2]), int(p[3])
            if x >= BUS_X_MIN and y in (-1008, -1104):
                skip_symbol = True
                continue
        if skip_symbol:
            if line.startswith("WINDOW ") or line.startswith("SYMATTR "):
                continue
            skip_symbol = False
        if line.startswith("FLAG "):
            p = line.strip().split()
            x, y = int(p[1]), int(p[2])
            name = p[3] if len(p) > 3 else ""
            if x >= BUS_X_MIN and -1120 <= y <= -900:
                if re.match(r"Bus\d*In$", name) or name == "0" or re.match(r"Reg_\d+$", name):
                    continue
            if x >= BUS_X_MIN and y == -1040 and name == "Reg_Outing":
                continue
        if line.startswith("WIRE "):
            p = line.split()
            x1, y1 = int(p[1]), int(p[2])
            x2, y2 = int(p[3]), int(p[4])
            if x1 >= BUS_X_MIN and x2 >= BUS_X_MIN and -1120 <= y1 <= -930 and -1120 <= y2 <= -930:
                continue
        result.append(line)
    return result


def generate_bus_output(n_bits, reg_prefix="Reg_", base_x=-4208, stride=80):
    """Generate bus output JFETs for N bits at fixed coordinates.

    Args:
        n_bits: number of bits
        reg_prefix: "Reg" for normal registers, "Reg_Q" for invertible
        base_x: X coordinate of MSB bus JFET
        stride: X spacing between bits (default 80)

    From your corrected schematics:
        MSB (bit N-1) at base_x, LSB (bit 0) at base_x + (N-1)*stride
        Reg_Outing flag at base_x, y=-1040 (shared via horizontal wire)
        Per bit: {reg_prefix}X at (x, -944), BusXIn at (x+48, -1104), GND at (x+48, -912)
        Two JFETs per bit at (x, -1008) and (x, -1104)
    """
    lines = []

    # Horizontal wire for Reg_Outing across all pairs
    for i in range(n_bits - 1):
        x = base_x + i * stride
        lines.append(f"WIRE {x + stride} -1040 {x} -1040\n")

    # Reg_Outing flag
    lines.append(f"FLAG {base_x} -1040 Reg_Outing\n")

    # Per-bit elements
    for i in range(n_bits):
        bit_idx = n_bits - 1 - i  # MSB first
        x = base_x + i * stride

        # Vertical wire
        lines.append(f"WIRE {x} -944 {x} -960\n")

        # FLAGs
        lines.append(f"FLAG {x + 48} -912 0\n")
        lines.append(f"FLAG {x + 48} -1104 Bus{bit_idx}In\n")
        lines.append(f"FLAG {x} -944 {reg_prefix}{bit_idx}\n")

        # Bottom JFET
        lines.append(f"SYMBOL njf {x} -1008 R0\n")
        lines.append("WINDOW 0 56 32 Invisible 2\n")
        lines.append("WINDOW 3 56 72 Invisible 2\n")
        lines.append(f"SYMATTR InstName J_BUS{i * 2}\n")
        lines.append("SYMATTR Value DR\n")

        # Top JFET
        lines.append(f"SYMBOL njf {x} -1104 R0\n")
        lines.append("WINDOW 0 56 32 Invisible 2\n")
        lines.append("WINDOW 3 56 72 Invisible 2\n")
        lines.append(f"SYMATTR InstName J_BUS{i * 2 + 1}\n")
        lines.append("SYMATTR Value DR\n")

    return lines


def generate_xor_section(n_bits, xor_template_path):
    """Generate XOR blocks for an invertible register.

    Reads a 3-bit XOR template, extracts the bit-0 block, and tiles it
    N times. Each XOR takes Reg_X/!Reg_X + InvertReg/!InvertReg
    and outputs Reg_QX/!Reg_QX.

    The XOR stride is extracted from the template (bit 0 vs bit 1 positions).
    """
    with open(xor_template_path) as f:
        lines = f.readlines()

    # Find stride from Reg_Q positions
    q_positions = {}
    for line in lines:
        if line.startswith("FLAG "):
            p = line.strip().split()
            name = p[3] if len(p) > 3 else ""
            m = re.match(r"Reg_Q(\d+)$", name)
            if m and "!" not in name:
                q_positions[int(m.group(1))] = int(p[1])

    if 0 in q_positions and 1 in q_positions:
        xor_stride = q_positions[0] - q_positions[1]
    else:
        xor_stride = 816  # fallback

    # Find the X boundary for bit 0 block extraction.
    # Bit 0 leftmost element is at about q_positions[0] - 608 (from Reg_0 flag)
    # Bit 1 rightmost element is at about q_positions[1] (Reg_Q1 flag)
    # Use midpoint between bit 1 rightmost and bit 0 leftmost
    if 0 in q_positions and 1 in q_positions:
        bit0_leftmost = q_positions[0] - 620  # ~-3808 for our template
        bit1_rightmost = q_positions[1]        # ~-4016
        bit0_x_min = (bit0_leftmost + bit1_rightmost) // 2
    else:
        bit0_x_min = -3900

    # Extract bit 0 XOR block
    bit0_lines = []
    skip = False
    for line in lines[2:]:
        if line.startswith("SYMBOL"):
            p = line.split()
            x = int(p[2])
            if x > bit0_x_min:
                bit0_lines.append(line)
                skip = True
            else:
                skip = False
            continue
        if skip and (line.startswith("WINDOW") or line.startswith("SYMATTR")):
            bit0_lines.append(line)
            continue
        skip = False
        if line.startswith("FLAG"):
            p = line.split()
            x = int(p[1])
            if x > bit0_x_min:
                bit0_lines.append(line)
        elif line.startswith("WIRE"):
            p = line.split()
            x1, x2 = int(p[1]), int(p[3])
            if x1 > bit0_x_min or x2 > bit0_x_min:
                bit0_lines.append(line)

    print(f"  XOR template: {len(bit0_lines)} lines, stride={xor_stride}")

    # Tile N copies: MSB (bit N-1) at leftmost, LSB (bit 0) at rightmost.
    #
    # In the 3-bit template:
    #   Bit 2 (MSB) Reg_Q2 at x=-4928
    #   Bit 0 (LSB) Reg_Q0 at x=-3200
    # The MSB XOR should always anchor at the same position as in the
    # 3-bit template. The bit-0 template block has Reg_Q0 at x=-3200.
    # MSB should be at x=-4928, so MSB offset from bit-0 = -4928 - (-3200) = -1728
    # That's -(3-1)*816 = -1632... close but not exact because stride varies.
    #
    # Better: compute where MSB should go based on the template's MSB position.
    # Template MSB (bit 2) is at q_positions[2] if available.
    msb_target_x = q_positions.get(max(q_positions.keys()), -4928)
    bit0_x = q_positions[0]  # where bit 0 output is in template

    all_xor = []
    for i in range(n_bits):
        bit_idx = n_bits - 1 - i  # MSB first (i=0 -> bit N-1)
        # MSB at msb_target_x, each subsequent bit at +xor_stride
        # Offset from bit 0 template position:
        # For MSB (i=0): target_x = msb_target_x, template_x = bit0_x
        #   -> dx = msb_target_x - bit0_x
        # For next (i=1): target_x = msb_target_x + stride
        #   -> dx = msb_target_x + stride - bit0_x
        target_x = msb_target_x + i * xor_stride
        dx = target_x - bit0_x

        shifted = offset_lines(bit0_lines, dx, 0)
        renames = {
            "Reg_0": f"Reg_{bit_idx}", "!Reg_0": f"!Reg_{bit_idx}",
            "Reg_Q0": f"Reg_Q{bit_idx}", "!Reg_Q0": f"!Reg_Q{bit_idx}",
        }
        shifted = rename_in_flags(shifted, renames)
        all_xor.extend(shifted)

    return all_xor


def tile(one_path, two_path, three_path, n_bits, output_path,
         invertible=False, xor_template=None, prefix=None):
    print(f"Tiling: {n_bits}-bit {'(invertible)' if invertible else ''}")

    dx, dy = find_offset(two_path, three_path)

    header, base_body = read_asc(one_path)
    _, tile_body = read_asc(two_path)

    # Strip bus output from both base and tile
    base_body = strip_bus_output_from_base(base_body)
    tile_body = strip_bus_output_from_tile(tile_body)

    print(f"  Base (stripped): {len(base_body)} lines")
    print(f"  Tile (stripped): {len(tile_body)} lines")

    # Rename base: _0 -> _{N-1} (MSB)
    msb = n_bits - 1
    base_renames = {
        "Reg_0": f"Reg_{msb}", "!Reg_0": f"!Reg_{msb}",
        "Bus_0": f"Bus_{msb}", "Bus0In": f"Bus{msb}In",
        "Reg_Q0": f"Reg_Q{msb}", "!Reg_Q0": f"!Reg_Q{msb}",
    }
    base_out = rename_in_flags(base_body, base_renames)

    # Place tiles
    all_tiles = []
    for k in range(1, n_bits):
        bit_idx = n_bits - 1 - k
        shifted = offset_lines(tile_body, dx * (k - 1), dy * (k - 1))
        renames = {
            "Reg_0": f"Reg_{bit_idx}", "!Reg_0": f"!Reg_{bit_idx}",
            "Bus_0": f"Bus_{bit_idx}", "Bus0In": f"Bus{bit_idx}In",
            "Reg_Q0": f"Reg_Q{bit_idx}", "!Reg_Q0": f"!Reg_Q{bit_idx}",
        }
        shifted = rename_in_flags(shifted, renames)
        all_tiles.extend(shifted)

    # Generate bus output — uses Reg_Q for invertible, Reg_ for normal
    # Prefix is applied later by apply_prefix, so use Reg_ here
    if invertible:
        bus_lines = generate_bus_output(n_bits, reg_prefix="Reg_Q")
    else:
        bus_lines = generate_bus_output(n_bits)

    # Generate XOR section and inverter control logic for invertible registers
    xor_lines = []
    inv_logic_lines = []
    if invertible and xor_template:
        xor_lines = generate_xor_section(n_bits, xor_template)

        # Include the register inverter logic (controls InvertReg/!InvertReg)
        inv_logic_path = os.path.join(os.path.dirname(xor_template), "register inverter logic.asc")
        if os.path.isfile(inv_logic_path):
            _, inv_body = read_asc(inv_logic_path)
            inv_logic_lines = list(inv_body)
            print(f"  Inverter logic: {len(inv_logic_lines)} lines")

    # Combine and renumber
    all_lines = header + base_out + all_tiles + bus_lines + xor_lines + inv_logic_lines
    all_lines = renumber_instnames(all_lines)

    # Apply custom prefix: rename Reg_ -> Prefix_, Bus_ -> Prefix_Bus_, etc.
    if prefix:
        all_lines = apply_prefix(all_lines, prefix)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.writelines(all_lines)

    n_sym = sum(1 for l in all_lines if l.startswith("SYMBOL "))
    pfx_str = f" (prefix={prefix}_)" if prefix else ""
    print(f"  Output: {output_path} ({n_sym} symbols{pfx_str})")


def main():
    parser = argparse.ArgumentParser(description="Tile schematic to N bits")
    parser.add_argument("--one", required=True)
    parser.add_argument("--two", required=True)
    parser.add_argument("--three", required=True)
    parser.add_argument("--bits", "-n", type=int, required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--invertible", action="store_true",
                        help="Generate invertible register (adds XOR section, bus uses Reg_QX)")
    parser.add_argument("--xor-template", default=None,
                        help="Path to XOR template .asc (required with --invertible)")
    parser.add_argument("--prefix", default=None,
                        help="Rename Reg_ signals (e.g. --prefix Acc makes Reg_0 -> Acc_0)")
    args = parser.parse_args()
    tile(args.one, args.two, args.three, args.bits, args.output,
         invertible=args.invertible, xor_template=args.xor_template,
         prefix=args.prefix)


if __name__ == "__main__":
    main()
