#!/usr/bin/env python3
"""Generate an N-bit ALU with Carry Lookahead using NAND/NOR/INV gates.

Produces an LTSpice .asc schematic by placing gate primitives and wiring
them together. Uses your hand-designed SiC JFET gate schematics as the
building blocks.

Architecture:
    - Carry Lookahead Adder (CLA) for fast addition/subtraction
    - XOR via 4 NANDs for sum computation
    - Operation select via mux (NAND/INV based)
    - B input inversion for subtraction (controlled by SUB signal)

Gate basis: NAND2, NOR2, INV (with 3/4-input variants available)

Usage:
    python tools/gen_alu.py --bits 4 --output cpus/tileable/alu_4bit.asc
    python tools/gen_alu.py --bits 8
    python tools/gen_alu.py --bits 16
"""

import argparse
import os
import sys

# ── Gate Templates ──────────────────────────────────────────────────────
# Each gate is defined by its schematic elements, with coordinates
# normalised to origin (0,0). When placing, we offset by (place_x, place_y).
#
# Structure: {
#   "wires": [(x1,y1,x2,y2), ...],
#   "flags": [(x,y,name), ...],       # name is pin label
#   "symbols": [(type,x,y,rot,name,value,windows), ...],
#   "width": int, "height": int,
#   "inputs": {"A": (x,y), "B": (x,y), ...},  # input pin coordinates
#   "output": (x,y),                            # output pin coordinate
#   "vdd": (x,y), "vss": (x,y),                # power pins
# }

# All gates share this base structure (normalised from your schematics):
# Origin at top-left of bounding box, VDD at top, VSS at bottom-ish

def _make_inv():
    """INV gate template, normalised to (0,0) origin."""
    return {
        "type": "INV",
        "wires": [
            (112, 0, 48, 0),       # VDD rail
            (112, 32, 112, 0),     # vertical
            (48, 96, 48, 80),      # internal
            (64, 96, 48, 96),      # horizontal
            (176, 208, 112, 208),  # output
            (112, 288, 96, 288),   # VSS
        ],
        "flags_template": {
            "VSS": (112, 288),
            "A": (0, 160),         # input
            "OUT": (176, 208),     # output
            "GND": (48, 192),      # ground node
            "VDD": (112, 0),       # power
        },
        "symbols": [
            ("res", 32, -16, "R0", "100k"),    # R1 pullup
            ("njf", 64, 32, "R0", "DR"),       # J_F
            ("res", 96, 112, "R0", "1k"),      # R2
            ("res", 96, 192, "R0", "4.8k"),    # R3
            ("njf", 0, 96, "R0", "DR"),        # J_B
        ],
        "width": 176,
        "height": 304,
        "input_pins": ["A"],
        "output_pin": "OUT",
    }

def _make_nand2():
    """NAND2 gate template."""
    return {
        "type": "NAND2",
        "wires": [
            (112, 0, 48, 0),
            (112, 32, 112, 0),
            (48, 96, 48, 80),
            (64, 96, 48, 96),
            (176, 208, 112, 208),
            (112, 288, 96, 288),
        ],
        "flags_template": {
            "VSS": (112, 288),
            "A": (0, 160),
            "B": (0, 256),
            "OUT": (176, 208),
            "GND": (48, 288),
            "VDD": (112, 0),
        },
        "symbols": [
            ("res", 32, -16, "R0", "100k"),
            ("njf", 64, 32, "R0", "DR"),
            ("res", 96, 112, "R0", "1k"),
            ("res", 96, 192, "R0", "4.8k"),
            ("njf", 0, 96, "R0", "DR"),       # J_B (input A)
            ("njf", 0, 192, "R0", "DR"),       # J_A (input B, series)
        ],
        "width": 176,
        "height": 304,
        "input_pins": ["A", "B"],
        "output_pin": "OUT",
    }


# ── Netlist-level ALU generation ────────────────────────────────────────
# Instead of placing physical gate schematics directly, we first generate
# a logic netlist (gate type + connections), then lay it out.

class Net:
    """A named wire in the netlist."""
    _counter = 0

    def __init__(self, name=None):
        if name is None:
            Net._counter += 1
            name = f"n{Net._counter}"
        self.name = name

    def __repr__(self):
        return self.name


class Gate:
    """A logic gate instance."""
    _counter = 0

    def __init__(self, gate_type, inputs, output, name=None):
        Gate._counter += 1
        self.gate_type = gate_type   # "INV", "NAND2", "NOR2", etc.
        self.inputs = inputs          # list of Net
        self.output = output          # Net
        self.name = name or f"G{Gate._counter}"


def make_inv(a, name=None):
    out = Net()
    g = Gate("INV", [a], out, name)
    return out, g

def make_nand2(a, b, name=None):
    out = Net()
    g = Gate("NAND2", [a, b], out, name)
    return out, g

def make_nor2(a, b, name=None):
    out = Net()
    g = Gate("NOR2", [a, b], out, name)
    return out, g

def make_nand3(a, b, c, name=None):
    out = Net()
    g = Gate("NAND3", [a, b, c], out, name)
    return out, g

def make_nand4(a, b, c, d, name=None):
    out = Net()
    g = Gate("NAND4", [a, b, c, d], out, name)
    return out, g

def make_and2(a, b):
    """AND = NAND + INV"""
    nand_out, g1 = make_nand2(a, b)
    out, g2 = make_inv(nand_out)
    return out, [g1, g2]

def make_or2(a, b):
    """OR = NOR + INV"""
    nor_out, g1 = make_nor2(a, b)
    out, g2 = make_inv(nor_out)
    return out, [g1, g2]

def make_xor2(a, b):
    """XOR from 4 NAND gates:
    n1 = NAND(a, b)
    n2 = NAND(a, n1)
    n3 = NAND(b, n1)
    out = NAND(n2, n3)
    """
    n1, g1 = make_nand2(a, b)
    n2, g2 = make_nand2(a, n1)
    n3, g3 = make_nand2(b, n1)
    out, g4 = make_nand2(n2, n3)
    return out, [g1, g2, g3, g4]

def make_mux2(a, b, sel):
    """2:1 MUX: sel=0->a, sel=1->b. Built from NAND/INV.
    sel_bar = INV(sel)
    n1 = NAND(a, sel_bar)
    n2 = NAND(b, sel)
    out = NAND(n1, n2)
    """
    sel_bar, g0 = make_inv(sel)
    n1, g1 = make_nand2(a, sel_bar)
    n2, g2 = make_nand2(b, sel)
    out, g3 = make_nand2(n1, n2)
    return out, [g0, g1, g2, g3]


# ── CLA Adder Generation ───────────────────────────────────────────────

def gen_cla_adder(n_bits, a_nets, b_nets, cin_net):
    """Generate an N-bit Carry Lookahead Adder.

    Args:
        n_bits: bit width
        a_nets: list of Net for A inputs (LSB first)
        b_nets: list of Net for B inputs (LSB first)
        cin_net: carry-in Net

    Returns:
        (sum_nets, cout_net, gates_list)
    """
    gates = []

    # Generate P (propagate) and G (generate) for each bit
    # Pi = Ai XOR Bi (propagate)
    # Gi = Ai AND Bi (generate)
    p_nets = []
    g_nets = []

    for i in range(n_bits):
        pi, xor_gates = make_xor2(a_nets[i], b_nets[i])
        gates.extend(xor_gates)
        p_nets.append(pi)

        gi, and_gates = make_and2(a_nets[i], b_nets[i])
        gates.extend(and_gates)
        g_nets.append(gi)

    # Generate carries using CLA equations
    # C1 = G0 | (P0 & C0)
    # C2 = G1 | (P1 & G0) | (P1 & P0 & C0)
    # etc.
    #
    # For large N, use hierarchical CLA (groups of 4)
    # For now, flat CLA up to 16 bits (works well with NAND/NOR)

    carry_nets = [cin_net]  # C[0] = cin

    if n_bits <= 4:
        # Flat CLA
        carry_nets.extend(_flat_cla(n_bits, p_nets, g_nets, cin_net, gates))
    else:
        # Hierarchical: groups of 4
        group_size = 4
        n_groups = (n_bits + group_size - 1) // group_size
        group_cin = cin_net

        for grp in range(n_groups):
            start = grp * group_size
            end = min(start + group_size, n_bits)
            size = end - start

            grp_p = p_nets[start:end]
            grp_g = g_nets[start:end]

            grp_carries = _flat_cla(size, grp_p, grp_g, group_cin, gates)
            carry_nets.extend(grp_carries)

            # Group carry out = last carry of this group
            group_cin = grp_carries[-1]

    # Generate sums: Si = Pi XOR Ci
    sum_nets = []
    for i in range(n_bits):
        si, xor_gates = make_xor2(p_nets[i], carry_nets[i])
        gates.extend(xor_gates)
        sum_nets.append(si)

    cout_net = carry_nets[n_bits]
    return sum_nets, cout_net, gates


def _flat_cla(n, p, g, cin, gates):
    """Generate flat CLA carry equations for up to n bits.

    Returns list of n carry nets [C1, C2, ..., Cn].
    """
    carries = []

    for i in range(n):
        # Ci+1 = Gi | (Pi & Gi-1) | (Pi & Pi-1 & Gi-2) | ... | (Pi & ... & P0 & Cin)
        # Build this as an OR of AND terms

        terms = []

        # Term 0: Gi (always)
        terms.append(g[i])

        # Term k (for k=1..i+1): product of P[i..i-k+1] AND (G[i-k] or Cin)
        for k in range(1, i + 2):
            # Product of propagates: P[i] & P[i-1] & ... & P[i-k+1]
            props = [p[j] for j in range(i, i - k, -1)]

            # The "seed": either G[i-k] or Cin (if k == i+1)
            if k <= i:
                seed = g[i - k]
            else:
                seed = cin

            # AND all together
            all_inputs = props + [seed]
            term = _make_and_tree(all_inputs, gates)
            terms.append(term)

        # OR all terms
        carry = _make_or_tree(terms, gates)
        carries.append(carry)

    return carries


def _make_and_tree(inputs, gates):
    """Build an AND tree from a list of nets using NAND+INV."""
    if len(inputs) == 1:
        return inputs[0]

    # Use multi-input NANDs where available (up to 4)
    if len(inputs) == 2:
        nand_out, g = make_nand2(inputs[0], inputs[1])
        gates.append(g)
        out, g2 = make_inv(nand_out)
        gates.append(g2)
        return out
    elif len(inputs) == 3:
        nand_out, g = make_nand3(inputs[0], inputs[1], inputs[2])
        gates.append(g)
        out, g2 = make_inv(nand_out)
        gates.append(g2)
        return out
    elif len(inputs) == 4:
        nand_out, g = make_nand4(inputs[0], inputs[1], inputs[2], inputs[3])
        gates.append(g)
        out, g2 = make_inv(nand_out)
        gates.append(g2)
        return out
    else:
        # Split into groups
        mid = len(inputs) // 2
        left = _make_and_tree(inputs[:mid], gates)
        right = _make_and_tree(inputs[mid:], gates)
        nand_out, g = make_nand2(left, right)
        gates.append(g)
        out, g2 = make_inv(nand_out)
        gates.append(g2)
        return out


def _make_or_tree(inputs, gates):
    """Build an OR tree from a list of nets using NOR+INV."""
    if len(inputs) == 1:
        return inputs[0]

    if len(inputs) == 2:
        nor_out, g = make_nor2(inputs[0], inputs[1])
        gates.append(g)
        out, g2 = make_inv(nor_out)
        gates.append(g2)
        return out
    else:
        mid = len(inputs) // 2
        left = _make_or_tree(inputs[:mid], gates)
        right = _make_or_tree(inputs[mid:], gates)
        nor_out, g = make_nor2(left, right)
        gates.append(g)
        out, g2 = make_inv(nor_out)
        gates.append(g2)
        return out


# ── Full ALU Generation ─────────────────────────────────────────────────

def gen_alu(n_bits):
    """Generate a complete N-bit ALU.

    Operations (selected by OP[1:0]):
        00 = ADD (A + B + Cin)
        01 = SUB (A - B = A + ~B + 1)
        10 = AND (A & B)
        11 = OR  (A | B)

    Inputs: A[n-1:0], B[n-1:0], Cin, OP[1:0]
    Outputs: Result[n-1:0], Cout, Zero

    Returns: (input_nets, output_nets, gates)
    """
    Net._counter = 0
    Gate._counter = 0

    gates = []

    # Create input nets
    a_nets = [Net(f"A{i}") for i in range(n_bits)]
    b_nets = [Net(f"B{i}") for i in range(n_bits)]
    cin_net = Net("Cin")
    op0_net = Net("OP0")  # LSB of operation select
    op1_net = Net("OP1")  # MSB of operation select

    # SUB control: for SUB, invert B and set Cin=1
    # SUB = OP0 & ~OP1 (op=01)
    # Actually simpler: OP0 controls B inversion and carry-in for ADD/SUB
    # OP0=0: ADD (B unchanged, Cin from input)
    # OP0=1: SUB (B inverted, Cin forced to 1)

    # Conditionally invert B based on OP0
    b_eff = []
    for i in range(n_bits):
        bi_inv, xor_gates = make_xor2(b_nets[i], op0_net)
        gates.extend(xor_gates)
        b_eff.append(bi_inv)

    # Carry-in for adder: Cin OR OP0 (force carry=1 for subtraction)
    adder_cin, or_gates = make_or2(cin_net, op0_net)
    gates.extend(or_gates)

    # CLA Adder
    sum_nets, cout_net, adder_gates = gen_cla_adder(n_bits, a_nets, b_eff, adder_cin)
    gates.extend(adder_gates)

    # AND operation
    and_nets = []
    for i in range(n_bits):
        ai, and_gates = make_and2(a_nets[i], b_nets[i])
        gates.extend(and_gates)
        and_nets.append(ai)

    # OR operation
    or_nets = []
    for i in range(n_bits):
        oi, or_g = make_or2(a_nets[i], b_nets[i])
        gates.extend(or_g)
        or_nets.append(oi)

    # Result MUX: OP1 selects arithmetic (0) vs logic (1)
    #             OP0 selects within group (ADD/SUB vs AND/OR)
    # For arithmetic: sum_nets (ADD and SUB both use the adder, just different B)
    # For logic: OP0=0 -> AND, OP0=1 -> OR
    result_nets = []
    for i in range(n_bits):
        # Logic mux: OP0=0->AND, OP0=1->OR
        logic_i, mux1_gates = make_mux2(and_nets[i], or_nets[i], op0_net)
        gates.extend(mux1_gates)

        # Final mux: OP1=0->arithmetic (sum), OP1=1->logic
        result_i, mux2_gates = make_mux2(sum_nets[i], logic_i, op1_net)
        gates.extend(mux2_gates)

        result_nets.append(result_i)
        result_i.name = f"Result{i}"

    # Zero flag: NOR tree of all result bits
    zero_net = _make_zero_detect(result_nets, gates)
    zero_net.name = "Zero"

    cout_net.name = "Cout"

    inputs = {
        "A": a_nets,
        "B": b_nets,
        "Cin": cin_net,
        "OP0": op0_net,
        "OP1": op1_net,
    }
    outputs = {
        "Result": result_nets,
        "Cout": cout_net,
        "Zero": zero_net,
    }

    return inputs, outputs, gates


def _make_zero_detect(result_nets, gates):
    """Generate zero flag: 1 when all result bits are 0."""
    # NOR all bits together in a tree
    # First invert each bit, then AND them all
    inv_nets = []
    for r in result_nets:
        ri, g = make_inv(r)
        gates.append(g)
        inv_nets.append(ri)

    return _make_and_tree(inv_nets, gates)


# ── Netlist Output ──────────────────────────────────────────────────────

def print_netlist(inputs, outputs, gates_list):
    """Print the gate-level netlist."""
    print(f"\n=== ALU Netlist ===")
    print(f"Gates: {len(gates_list)}")

    # Count by type
    from collections import Counter
    counts = Counter(g.gate_type for g in gates_list)
    for gt, c in sorted(counts.items()):
        print(f"  {gt}: {c}")

    print(f"\nInputs:")
    for name, nets in inputs.items():
        if isinstance(nets, list):
            print(f"  {name}[{len(nets)-1}:0]: {', '.join(n.name for n in nets)}")
        else:
            print(f"  {name}: {nets.name}")

    print(f"\nOutputs:")
    for name, nets in outputs.items():
        if isinstance(nets, list):
            print(f"  {name}[{len(nets)-1}:0]: {', '.join(n.name for n in nets)}")
        else:
            print(f"  {name}: {nets.name}")

    # Estimate gate delays for critical path (CLA)
    n_bits = len(inputs["A"])
    if n_bits <= 4:
        cla_depth = 3  # AND + OR for carry, then XOR for sum
    else:
        cla_depth = 5  # group CLA + group-level CLA
    mux_depth = 2  # two levels of MUX
    total_depth = 1 + cla_depth + 1 + mux_depth  # XOR(P/G) + CLA + XOR(sum) + MUX
    print(f"\nCritical path: ~{total_depth} gate delays")


# ── LTSpice .asc Output ────────────────────────────────────────────────

def write_spice_netlist(inputs, outputs, gates_list, filepath):
    """Write a SPICE .net netlist (not graphical .asc) for the ALU.

    This can be included in an LTSpice schematic via .include directive
    or used directly.
    """
    with open(filepath, "w") as f:
        f.write(f"* {len(inputs['A'])}-bit ALU with CLA\n")
        f.write(f"* Generated by gen_alu.py\n")
        f.write(f"* Gates: {len(gates_list)}\n\n")

        # Map gate types to subcircuit names
        type_map = {
            "INV": "INV",
            "NAND2": "NAND2",
            "NOR2": "NOR2",
            "NAND3": "NAND3",
            "NOR3": "NOR3",
            "NAND4": "NAND4",
        }

        for g in gates_list:
            subckt = type_map.get(g.gate_type, g.gate_type)
            ins = " ".join(n.name for n in g.inputs)
            out = g.output.name
            f.write(f"X{g.name} {ins} {out} VDD VSS {subckt}\n")

    print(f"Written: {filepath}")


def gen_adder(n_bits, a_prefix="A", b_prefix="B", cin_name="Cin",
              sum_prefix="Sum", cout_name="SUMCF"):
    """Generate a standalone N-bit CLA adder (no ALU mux/logic ops).

    Inputs:  {a_prefix}[n-1:0], {b_prefix}[n-1:0], {cin_name}
    Outputs: {sum_prefix}[n-1:0], {cout_name} (carry out)
    """
    Net._counter = 0
    Gate._counter = 0
    gates = []

    a_nets = [Net(f"{a_prefix}{i}") for i in range(n_bits)]
    b_nets = [Net(f"{b_prefix}{i}") for i in range(n_bits)]
    cin_net = Net(cin_name)

    sum_nets, cout_net, adder_gates = gen_cla_adder(n_bits, a_nets, b_nets, cin_net)
    gates.extend(adder_gates)

    # Name outputs
    for i, s in enumerate(sum_nets):
        s.name = f"{sum_prefix}{i}"
    cout_net.name = cout_name

    inputs = {"A": a_nets, "B": b_nets, "Cin": cin_net}
    outputs = {"Sum": sum_nets, cout_name: cout_net}
    return inputs, outputs, gates


def main():
    parser = argparse.ArgumentParser(description="Generate N-bit CLA ALU or adder")
    parser.add_argument("--bits", "-n", type=int, default=4,
                        help="Bit width (default: 4)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file (.net for SPICE netlist)")
    parser.add_argument("--adder-only", action="store_true",
                        help="Generate just the CLA adder (Sum0..SumN-1, SUMCF)")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    if args.adder_only:
        print(f"Generating {args.bits}-bit CLA adder...")
        inputs, outputs, gates = gen_adder(args.bits)
    else:
        print(f"Generating {args.bits}-bit ALU with CLA...")
        inputs, outputs, gates = gen_alu(args.bits)

    print_netlist(inputs, outputs, gates)

    if args.output:
        write_spice_netlist(inputs, outputs, gates, args.output)


if __name__ == "__main__":
    main()
