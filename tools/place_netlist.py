#!/usr/bin/env python3
"""Place a SPICE gate netlist as physical LTSpice .asc schematic.

Reads a .net file (from gen_alu.py) and gate .asc templates (from lib/gates/),
places each gate on a grid, and connects them via LTSpice FLAG labels
(same net name = same wire).

Usage:
    python tools/place_netlist.py \\
        --netlist cpus/tileable/generated/alu_4bit.net \\
        --output cpus/tileable/generated/alu_4bit.asc \\
        --cols 16
"""

import argparse
import os
import re
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from lib import GATES_DIR


# Gate type mapping: netlist name -> .asc filename
GATE_FILES = {
    "INV": "INV.asc",
    "NAND2": "NAND.asc",
    "NOR2": "NOR.asc",
    "NAND3": "3NAND.asc",
    "NOR3": "3NOR.asc",
    "NAND4": "4NAND.asc",
    "NOR4": "4NOR.asc",
}

# Grid spacing
CELL_W = 500   # X spacing between gates
CELL_H = 600   # Y spacing between gates


def parse_netlist(path):
    """Parse a SPICE .net file into gate instances.

    Returns list of dicts: {name, type, inputs, output}
    """
    gates = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("*") or line.startswith("."):
                continue
            # Format: Xname in1 [in2 [in3 [in4]]] out VDD VSS TYPE
            parts = line.split()
            if not parts[0].startswith("X"):
                continue

            name = parts[0][1:]  # strip X prefix
            gate_type = parts[-1]
            # Inputs are between name and output; output is before VDD VSS TYPE
            # VDD and VSS are always the last 3rd and 2nd
            signals = parts[1:-3]  # everything between name and VDD
            output = signals[-1]
            inputs = signals[:-1]

            gates.append({
                "name": name,
                "type": gate_type,
                "inputs": inputs,
                "output": output,
            })
    return gates


def load_gate_template(gate_type):
    """Load a gate .asc template and extract its elements.

    Returns (body_lines, input_flags, output_flag, bbox)
    where input_flags = [(name, x, y), ...] and output_flag = (name, x, y)
    """
    filename = GATE_FILES.get(gate_type)
    if not filename:
        raise ValueError(f"Unknown gate type: {gate_type}")

    path = os.path.join(GATES_DIR, filename)
    with open(path) as f:
        lines = f.readlines()

    body = lines[2:]  # skip header

    # Find input/output FLAG positions
    # Input flags: A, B, C, D
    # Output flag: the gate type name (INV, NAND, NOR, 3NAND, etc.)
    input_flags = []
    output_flag = None
    output_names = {"INV", "NAND", "NOR", "3NAND", "3NOR", "4NAND", "4NOR"}

    for line in body:
        if line.startswith("FLAG "):
            p = line.strip().split()
            x, y = int(p[1]), int(p[2])
            name = p[3] if len(p) > 3 else ""
            if name in ("A", "B", "C", "D"):
                input_flags.append((name, x, y))
            elif name in output_names:
                output_flag = (name, x, y)

    # Sort inputs: A first, then B, C, D
    input_flags.sort(key=lambda f: f[0])

    # Get bounding box
    all_x, all_y = [], []
    for line in body:
        nums = []
        if line.startswith("WIRE "):
            p = line.split()
            all_x.extend([int(p[1]), int(p[3])])
            all_y.extend([int(p[2]), int(p[4])])
        elif line.startswith("FLAG ") or line.startswith("SYMBOL "):
            p = line.split()
            idx = 1 if line.startswith("FLAG") else 2
            all_x.append(int(p[idx]))
            all_y.append(int(p[idx + 1]))

    bbox = (min(all_x), min(all_y), max(all_x), max(all_y))
    return body, input_flags, output_flag, bbox


def offset_body(body_lines, dx, dy):
    """Offset all coordinates in body lines."""
    result = []
    for line in body_lines:
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


def rename_flags(body_lines, renames):
    """Rename FLAG labels."""
    result = []
    for line in body_lines:
        if line.startswith("FLAG "):
            p = line.strip().split(None, 3)
            if len(p) >= 4 and p[3] in renames:
                line = f"FLAG {p[1]} {p[2]} {renames[p[3]]}\n"
        result.append(line)
    return result


def renumber_instnames(lines, prefix_counters):
    """Renumber SYMATTR InstName using shared counters."""
    result = []
    for line in lines:
        if line.startswith("SYMATTR InstName "):
            old = line.split(" ", 2)[2].strip()
            m = re.match(r"([A-Z_!]+?)(\d+)$", old)
            if m:
                pfx = m.group(1)
                prefix_counters[pfx] = prefix_counters.get(pfx, 0) + 1
                result.append(f"SYMATTR InstName {pfx}{prefix_counters[pfx]}\n")
            else:
                result.append(line)
        else:
            result.append(line)
    return result


def place(netlist_path, output_path, cols=16):
    """Place a netlist as physical .asc schematic."""
    gates = parse_netlist(netlist_path)
    print(f"Placing {len(gates)} gates in {cols}-column grid")

    # Cache loaded templates
    templates = {}

    all_body = []
    counters = {}

    for i, gate in enumerate(gates):
        gt = gate["type"]

        # Load template if not cached
        if gt not in templates:
            templates[gt] = load_gate_template(gt)

        body, input_flags, output_flag, bbox = templates[gt]

        # Grid position
        col = i % cols
        row = i // cols
        dx = col * CELL_W
        dy = row * CELL_H

        # Offset the template
        placed = offset_body(list(body), dx, dy)

        # Rename input/output flags to match netlist signals
        renames = {}
        for j, (flag_name, fx, fy) in enumerate(input_flags):
            if j < len(gate["inputs"]):
                renames[flag_name] = gate["inputs"][j]
        if output_flag:
            renames[output_flag[0]] = gate["output"]

        placed = rename_flags(placed, renames)

        # Renumber InstNames
        placed = renumber_instnames(placed, counters)

        all_body.extend(placed)

    # Write output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write("Version 4.1\n")
        f.write("SHEET 1 109444 42780\n")
        f.writelines(all_body)

    n_sym = sum(1 for l in all_body if l.startswith("SYMBOL "))
    n_rows = (len(gates) + cols - 1) // cols
    print(f"  Output: {output_path}")
    print(f"  {n_sym} symbols, {len(gates)} gates, {n_rows} rows x {cols} cols")


def main():
    parser = argparse.ArgumentParser(description="Place gate netlist as .asc schematic")
    parser.add_argument("--netlist", "-i", required=True, help="Input .net file")
    parser.add_argument("--output", "-o", required=True, help="Output .asc file")
    parser.add_argument("--cols", type=int, default=16, help="Grid columns (default: 16)")
    args = parser.parse_args()
    place(args.netlist, args.output, args.cols)


if __name__ == "__main__":
    main()
