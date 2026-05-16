#!/usr/bin/env python3
"""Generate an N-bit incrementer (A + 1) using simplified CLA logic.

Since B=0 and Cin=1:
    P_i = A_i (propagate = input directly)
    G_i = 0   (generate = always zero)
    C_i = A_{i-1} AND A_{i-2} AND ... AND A_0 (running AND chain)
    Sum_0 = NOT(A_0)
    Sum_i = A_i XOR C_i

Only needs INV, NAND2, and the carry chain is just cascaded AND2.

Usage:
    python tools/gen_incrementer.py --bits 4 --output inc_4bit.asc --place
    python tools/gen_incrementer.py --bits 12 --prefix PC --output pc_inc_12bit.asc --place
"""

import argparse
import os
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)


class Net:
    _counter = 0
    def __init__(self, name=None):
        if name is None:
            Net._counter += 1
            name = f"n{Net._counter}"
        self.name = name
    def __repr__(self): return self.name


class Gate:
    _counter = 0
    def __init__(self, gate_type, inputs, output, name=None):
        Gate._counter += 1
        self.gate_type = gate_type
        self.inputs = inputs
        self.output = output
        self.name = name or f"G{Gate._counter}"


def inv(a):
    out = Net(); g = Gate("INV", [a], out); return out, g

def nand2(a, b):
    out = Net(); g = Gate("NAND2", [a, b], out); return out, g

def and2(a, b):
    n, g1 = nand2(a, b); out, g2 = inv(n); return out, [g1, g2]

def xor2(a, b):
    n1, g1 = nand2(a, b)
    n2, g2 = nand2(a, n1)
    n3, g3 = nand2(b, n1)
    out, g4 = nand2(n2, n3)
    return out, [g1, g2, g3, g4]


def gen_incrementer(n_bits, prefix="A", out_prefix="Inc"):
    """Generate an N-bit incrementer.

    Inputs:  {prefix}0 .. {prefix}{N-1}
    Outputs: {out_prefix}0 .. {out_prefix}{N-1}, {out_prefix}CF (carry out)
    """
    Net._counter = 0
    Gate._counter = 0
    gates = []

    a = [Net(f"{prefix}{i}") for i in range(n_bits)]

    sums = []

    # Bit 0: Sum0 = NOT(A0) since A0 XOR 1 = NOT(A0)
    sum0, g = inv(a[0])
    gates.append(g)
    sum0.name = f"{out_prefix}0"
    sums.append(sum0)

    if n_bits == 1:
        # Carry out = A0 (if A0=1, 1+1=10, carry=1)
        cout = a[0]
        cout.name = f"{out_prefix}CF"
        return {"A": a}, {out_prefix: sums, f"{out_prefix}CF": cout}, gates

    # Build carry chain: C1=A0, C2=A0&A1, C3=A0&A1&A2, ...
    carries = [a[0]]  # C1 = A0
    for i in range(1, n_bits - 1):
        c, gs = and2(carries[-1], a[i])
        gates.extend(gs)
        carries.append(c)

    # Bit 1: Sum1 = A1 XOR C1 = A1 XOR A0
    sum1, gs = xor2(a[1], carries[0])
    gates.extend(gs)
    sum1.name = f"{out_prefix}1"
    sums.append(sum1)

    # Bits 2..N-1: Sum_i = A_i XOR C_i
    for i in range(2, n_bits):
        s, gs = xor2(a[i], carries[i - 1])
        gates.extend(gs)
        s.name = f"{out_prefix}{i}"
        sums.append(s)

    # Carry out = A_{N-1} AND C_{N-1} = A_{N-1} AND (A_{N-2} AND ... AND A_0)
    cout, gs = and2(a[n_bits - 1], carries[-1])
    gates.extend(gs)
    cout.name = f"{out_prefix}CF"

    inputs = {"A": a}
    outputs = {out_prefix: sums, f"{out_prefix}CF": cout}
    return inputs, outputs, gates


def write_netlist(inputs, outputs, gates, filepath):
    with open(filepath, "w") as f:
        f.write(f"* {len(inputs['A'])}-bit Incrementer\n")
        f.write(f"* Gates: {len(gates)}\n\n")
        type_map = {"INV": "INV", "NAND2": "NAND2"}
        for g in gates:
            subckt = type_map.get(g.gate_type, g.gate_type)
            ins = " ".join(n.name for n in g.inputs)
            f.write(f"X{g.name} {ins} {g.output.name} VDD VSS {subckt}\n")
    print(f"Written: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate N-bit incrementer")
    parser.add_argument("--bits", "-n", type=int, required=True)
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--prefix", default="A", help="Input signal prefix (default: A)")
    parser.add_argument("--out-prefix", default="Inc", help="Output signal prefix (default: Inc)")
    parser.add_argument("--place", action="store_true")
    parser.add_argument("--cols", type=int, default=12)
    args = parser.parse_args()

    print(f"Generating {args.bits}-bit incrementer...")
    inputs, outputs, gates = gen_incrementer(args.bits, args.prefix, args.out_prefix)

    from collections import Counter
    counts = Counter(g.gate_type for g in gates)
    print(f"  Gates: {len(gates)}")
    for gt, c in sorted(counts.items()):
        print(f"    {gt}: {c}")

    net_path = args.output if args.output.endswith(".net") else args.output.replace(".asc", ".net")
    write_netlist(inputs, outputs, gates, net_path)

    if args.place or args.output.endswith(".asc"):
        from place_netlist import place
        asc_path = args.output if args.output.endswith(".asc") else args.output.replace(".net", ".asc")
        place(net_path, asc_path, args.cols)


if __name__ == "__main__":
    main()
