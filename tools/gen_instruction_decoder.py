#!/usr/bin/env python3
"""Generate an instruction decoder for any instruction width.

Uses a 2-level NAND2 mux structure:
    - Split instruction bits into pairs of 2
    - Each pair produces 4 intermediate signals via 2-to-4 decoders
    - Odd bits use a 1-to-2 decoder (just the bit and its inverse)
    - Each instruction = AND tree of one output per group (NAND2 only)

Reuses existing inverted register outputs (!IR1_2 etc.) instead of
generating redundant inverters. No padding needed for odd bit widths.

Usage:
    python tools/gen_instruction_decoder.py --preset 4004 --output decoder.asc --place
    python tools/gen_instruction_decoder.py --bits 5 --map instructions.json --output decoder.asc --place
"""

import argparse
import json
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

def nor2(a, b):
    out = Net(); g = Gate("NOR2", [a, b], out); return out, g

def or2(a, b):
    n, g1 = nor2(a, b); out, g2 = inv(n); return out, [g1, g2]


def and_tree(inputs, gates):
    if len(inputs) == 1: return inputs[0]
    if len(inputs) == 2:
        out, gs = and2(inputs[0], inputs[1]); gates.extend(gs); return out
    mid = len(inputs) // 2
    left = and_tree(inputs[:mid], gates)
    right = and_tree(inputs[mid:], gates)
    out, gs = and2(left, right); gates.extend(gs); return out


def or_tree(inputs, gates):
    if len(inputs) == 1: return inputs[0]
    if len(inputs) == 2:
        out, gs = or2(inputs[0], inputs[1]); gates.extend(gs); return out
    mid = len(inputs) // 2
    left = or_tree(inputs[:mid], gates)
    right = or_tree(inputs[mid:], gates)
    out, gs = or2(left, right); gates.extend(gs); return out


def decode_2bit(bit_hi, bit_hi_inv, bit_lo, bit_lo_inv, prefix, gates):
    """2-to-4 decoder using provided true and inverted signals. No INV generated."""
    outputs = []
    combos = [(bit_hi_inv, bit_lo_inv), (bit_hi_inv, bit_lo),
              (bit_hi, bit_lo_inv), (bit_hi, bit_lo)]
    for i, (a, b) in enumerate(combos):
        out, gs = and2(a, b); gates.extend(gs)
        out.name = f"{prefix}{i}"
        outputs.append(out)
    return outputs


def decode_1bit(bit, bit_inv, prefix, gates):
    """1-to-2 decoder for an unpaired odd bit. No INV generated."""
    out0 = bit_inv;  out0.name = f"{prefix}0"  # bit=0
    out1 = bit;      out1.name = f"{prefix}1"  # bit=1
    return [out0, out1]


def build_pair_decoders(bits, bits_inv, prefix_names, gates):
    """Build decoders for a list of bits, handling odd counts.

    bits: list of Net (true signals)
    bits_inv: list of Net (inverted signals, from registers)
    prefix_names: list of prefix strings for each pair/single

    Returns list of decoder output lists (each 4 or 2 elements).
    """
    n = len(bits)
    decoders = []
    idx = 0
    p = 0

    while idx < n:
        prefix = prefix_names[p] if p < len(prefix_names) else f"Dec{p}_"
        if idx + 1 < n:
            # Pair decode
            lo = bits[idx]; lo_inv = bits_inv[idx]
            hi = bits[idx + 1]; hi_inv = bits_inv[idx + 1]
            outs = decode_2bit(hi, hi_inv, lo, lo_inv, prefix, gates)
            decoders.append(outs)
            idx += 2
        else:
            # Single bit (odd)
            outs = decode_1bit(bits[idx], bits_inv[idx], prefix, gates)
            decoders.append(outs)
            idx += 1
        p += 1

    return decoders


def generate_decoder(primary_bits, instruction_map, secondary_bits=None,
                     secondary_map=None, group_map=None,
                     primary_signal_names=None, secondary_signal_names=None,
                     primary_inv_names=None, secondary_inv_names=None):
    """Generate instruction decoder.

    If *_inv_names are provided, uses those as inverted inputs directly
    (from register outputs). Otherwise generates INV gates internally.
    """
    Net._counter = 0
    Gate._counter = 0
    gates = []

    # Primary inputs
    if primary_signal_names:
        ir1 = [Net(name) for name in primary_signal_names]
    else:
        ir1 = [Net(f"IR1_{i}") for i in range(primary_bits)]

    # Primary inversions — reuse register outputs or generate
    if primary_inv_names:
        ir1_inv = [Net(name) for name in primary_inv_names]
    else:
        ir1_inv = []
        for i in range(primary_bits):
            inv_out, g = inv(ir1[i]); gates.append(g)
            inv_out.name = f"dec_inv_{ir1[i].name}"
            ir1_inv.append(inv_out)

    # Primary pair decoders
    pri_prefixes = ["Lower", "Upper", "Upper2", "Upper3", "Upper4", "Upper5"]
    pri_decoders = build_pair_decoders(ir1, ir1_inv, pri_prefixes, gates)

    outputs = {}

    # Primary instruction decode: each opcode -> select one from each decoder
    for opcode, name in instruction_map.items():
        selections = []
        idx = 0
        for dec in pri_decoders:
            n_dec = 2 if len(dec) == 2 else 4
            bits_used = 1 if n_dec == 2 else 2
            val = (opcode >> idx) & (n_dec - 1)
            selections.append(dec[val])
            idx += bits_used
        out = and_tree(selections, gates)
        out.name = name
        outputs[name] = out

    # Secondary decode
    if secondary_bits and secondary_map:
        if secondary_signal_names:
            ir2 = [Net(name) for name in secondary_signal_names]
        else:
            ir2 = [Net(f"IR2_{i}") for i in range(secondary_bits)]

        if secondary_inv_names:
            ir2_inv = [Net(name) for name in secondary_inv_names]
        else:
            ir2_inv = []
            for i in range(secondary_bits):
                inv_out, g = inv(ir2[i]); gates.append(g)
                inv_out.name = f"dec_inv_{ir2[i].name}"
                ir2_inv.append(inv_out)

        sec_prefixes = ["2Lower", "2Upper", "2Upper2", "2Upper3"]
        sec_decoders = build_pair_decoders(ir2, ir2_inv, sec_prefixes, gates)

        for (pri_val, sec_val), name in secondary_map.items():
            # Primary match
            pri_sels = []
            idx = 0
            for dec in pri_decoders:
                n_dec = len(dec); bits_used = 1 if n_dec == 2 else 2
                val = (pri_val >> idx) & (n_dec - 1)
                pri_sels.append(dec[val])
                idx += bits_used
            pri_match = and_tree(pri_sels, gates)

            # Secondary match
            sec_sels = []
            idx = 0
            for dec in sec_decoders:
                n_dec = len(dec); bits_used = 1 if n_dec == 2 else 2
                val = (sec_val >> idx) & (n_dec - 1)
                sec_sels.append(dec[val])
                idx += bits_used
            sec_match = and_tree(sec_sels, gates)

            out, gs = and2(pri_match, sec_match); gates.extend(gs)
            out.name = name
            outputs[name] = out

    # Group signals
    if group_map:
        for group_name, inst_list in group_map.items():
            members = [outputs[n] for n in inst_list if n in outputs]
            if members:
                out = or_tree(members, gates)
                out.name = group_name
                outputs[group_name] = out

    inputs_dict = {"IR1": ir1}
    if secondary_bits:
        inputs_dict["IR2"] = ir2
    return inputs_dict, outputs, gates


# ── Presets ──────────────────────────────────────────────────────────────

def preset_4004():
    pri_map = {
        0x0: "NOP", 0x1: "JCN", 0x4: "JUN", 0x5: "JMS",
        0x6: "INC", 0x7: "ISZ", 0x8: "ADD", 0x9: "SUB",
        0xA: "LD", 0xB: "XCH", 0xC: "BBL", 0xD: "LDM",
    }
    sec_map = {}
    io_map = {
        (0xE, 0x0): "WRM", (0xE, 0x1): "WMP", (0xE, 0x2): "WRR",
        (0xE, 0x4): "WR0", (0xE, 0x5): "WR1", (0xE, 0x6): "WR2", (0xE, 0x7): "WR3",
        (0xE, 0x8): "SBM", (0xE, 0x9): "RDM", (0xE, 0xA): "RDR", (0xE, 0xB): "ADM",
        (0xE, 0xC): "RD0", (0xE, 0xD): "RD1", (0xE, 0xE): "RD2", (0xE, 0xF): "RD3",
    }
    acc_map = {
        (0xF, 0x0): "CLB", (0xF, 0x1): "CLC", (0xF, 0x2): "IAC", (0xF, 0x3): "CMC",
        (0xF, 0x4): "CMA", (0xF, 0x5): "RAL", (0xF, 0x6): "RAR", (0xF, 0x7): "TCC",
        (0xF, 0x8): "DAC", (0xF, 0x9): "TCS", (0xF, 0xA): "STC", (0xF, 0xB): "DAA",
        (0xF, 0xC): "KBP", (0xF, 0xD): "DCL",
    }
    sec_map.update(io_map)
    sec_map.update(acc_map)
    grp_map = {
        "WR": ["WR0", "WR1", "WR2", "WR3"],
        "RD": ["RD0", "RD1", "RD2", "RD3"],
        "2Word": ["JCN", "FIM", "JUN", "JMS", "ISZ"],
        "Jump": ["JCN", "JUN", "JMS", "ISZ", "JIN"],
    }
    return 4, pri_map, 4, sec_map, grp_map


def add_fim_src_fin_jin(inputs, outputs, gates):
    ir2_0 = inputs["IR2"][0]
    ir2_0_inv, g = inv(ir2_0); gates.append(g)

    ir1 = inputs["IR1"]
    ir1_inv = []
    for i in range(4):
        for g_existing in gates:
            if hasattr(g_existing, 'output') and g_existing.output.name == f"dec_inv_IR1_{i}":
                ir1_inv.append(g_existing.output); break
        else:
            inv_out, g = inv(ir1[i]); gates.append(g)
            inv_out.name = f"_fim_{i}"; ir1_inv.append(inv_out)

    a, gs = and2(ir1_inv[3], ir1_inv[2]); gates.extend(gs)
    b1, gs = and2(ir1[1], ir1_inv[0]); gates.extend(gs)
    ir1_2, gs = and2(a, b1); gates.extend(gs)
    b2, gs = and2(ir1[1], ir1[0]); gates.extend(gs)
    ir1_3, gs = and2(a, b2); gates.extend(gs)

    fim, gs = and2(ir1_2, ir2_0_inv); gates.extend(gs); fim.name = "FIM"; outputs["FIM"] = fim
    src, gs = and2(ir1_2, ir2_0); gates.extend(gs); src.name = "SRC"; outputs["SRC"] = src
    fin, gs = and2(ir1_3, ir2_0_inv); gates.extend(gs); fin.name = "FIN"; outputs["FIN"] = fin
    jin, gs = and2(ir1_3, ir2_0); gates.extend(gs); jin.name = "JIN"; outputs["JIN"] = jin


# ── Output ───────────────────────────────────────────────────────────────

def write_netlist(inputs, outputs, gates, filepath):
    with open(filepath, "w") as f:
        f.write(f"* Instruction Decoder (NAND2 mux, no padding)\n")
        f.write(f"* Gates: {len(gates)}\n")
        f.write(f"* Outputs: {len(outputs)}\n\n")
        type_map = {"INV": "INV", "NAND2": "NAND2", "NOR2": "NOR2"}
        for g in gates:
            subckt = type_map.get(g.gate_type, g.gate_type)
            ins = " ".join(n.name for n in g.inputs)
            f.write(f"X{g.name} {ins} {g.output.name} VDD VSS {subckt}\n")
    print(f"Written: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Generate instruction decoder")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--preset", choices=["4004"], default=None)
    parser.add_argument("--bits", type=int, default=None)
    parser.add_argument("--map", default=None)
    parser.add_argument("--place", action="store_true")
    parser.add_argument("--cols", type=int, default=14)
    args = parser.parse_args()

    pri_sig_names = None
    sec_sig_names = None
    pri_inv_names = None
    sec_inv_names = None

    if args.preset == "4004":
        pri_bits, pri_map, sec_bits, sec_map, grp_map = preset_4004()
    elif args.bits and args.map:
        with open(args.map) as f:
            data = json.load(f)
        pri_bits = args.bits
        pri_map = {int(k): v for k, v in data.get("primary", {}).items()}
        sec_bits = data.get("secondary_bits")
        sec_map = {(int(k.split(",")[0]), int(k.split(",")[1])): v
                   for k, v in data.get("secondary", {}).items()} if "secondary" in data else None
        grp_map = data.get("groups")
        pri_sig_names = data.get("primary_signal_names")
        sec_sig_names = data.get("secondary_signal_names")
        # Build inverted names from signal names: IR1_2 -> !IR1_2
        if pri_sig_names:
            pri_inv_names = [f"!{n}" for n in pri_sig_names]
        if sec_sig_names:
            sec_inv_names = [f"!{n}" for n in sec_sig_names]
    else:
        print("Specify --preset or --bits + --map")
        return

    print(f"Generating decoder: {pri_bits}-bit primary" +
          (f" + {sec_bits}-bit secondary" if sec_bits else ""))

    inputs, outputs, gates = generate_decoder(
        pri_bits, pri_map, sec_bits, sec_map, grp_map,
        primary_signal_names=pri_sig_names,
        secondary_signal_names=sec_sig_names,
        primary_inv_names=pri_inv_names,
        secondary_inv_names=sec_inv_names)

    if args.preset == "4004":
        add_fim_src_fin_jin(inputs, outputs, gates)
        if "2Word" in outputs and "FIM" in outputs:
            tw = or_tree([outputs["2Word"], outputs["FIM"]], gates)
            tw.name = "2Word"; outputs["2Word"] = tw
        if "Jump" in outputs and "JIN" in outputs:
            jp = or_tree([outputs["Jump"], outputs["JIN"]], gates)
            jp.name = "Jump"; outputs["Jump"] = jp

    from collections import Counter
    counts = Counter(g.gate_type for g in gates)
    print(f"  Gates: {len(gates)}")
    for gt, c in sorted(counts.items()):
        print(f"    {gt}: {c}")
    print(f"  Outputs: {len(outputs)}")

    net_path = args.output if args.output.endswith(".net") else args.output.replace(".asc", ".net")
    write_netlist(inputs, outputs, gates, net_path)

    if args.place or args.output.endswith(".asc"):
        from place_netlist import place
        asc_path = args.output if args.output.endswith(".asc") else args.output.replace(".net", ".asc")
        place(net_path, asc_path, args.cols)


if __name__ == "__main__":
    main()
