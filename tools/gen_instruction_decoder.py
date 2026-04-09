#!/usr/bin/env python3
"""Generate an instruction decoder for any instruction width.

Uses a 2-level NAND2 mux structure:
    - Split instruction bits into pairs of 2
    - Each pair produces 4 intermediate signals via 2-to-4 decoders
    - Each instruction = AND tree of one output per pair (NAND2 only)

For N-bit instructions:
    - N/2 pairs (padded if odd)
    - Each pair: 2 INV + 4 AND2 = 10 gates
    - Each instruction: (N/2 - 1) AND2 stages = 2*(N/2-1) gates
    - Total depth: 2 + log2(N/2) gate delays

Supports:
    - Primary decode (first nibble/byte)
    - Secondary decode (second nibble/byte for sub-instructions)
    - Grouped signals (OR of multiple instructions)
    - Custom instruction maps

Usage:
    # 4004 style (4-bit OPR + 4-bit OPA)
    python tools/gen_instruction_decoder.py --preset 4004 --output decoder.asc --place

    # Custom: 6-bit instructions
    python tools/gen_instruction_decoder.py --bits 6 --map instructions.json --output decoder.asc --place
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
    """AND all inputs using a balanced NAND2 tree."""
    if len(inputs) == 1:
        return inputs[0]
    if len(inputs) == 2:
        out, gs = and2(inputs[0], inputs[1])
        gates.extend(gs)
        return out
    mid = len(inputs) // 2
    left = and_tree(inputs[:mid], gates)
    right = and_tree(inputs[mid:], gates)
    out, gs = and2(left, right)
    gates.extend(gs)
    return out


def or_tree(inputs, gates):
    """OR all inputs using a balanced NOR2 tree."""
    if len(inputs) == 1:
        return inputs[0]
    if len(inputs) == 2:
        out, gs = or2(inputs[0], inputs[1])
        gates.extend(gs)
        return out
    mid = len(inputs) // 2
    left = or_tree(inputs[:mid], gates)
    right = or_tree(inputs[mid:], gates)
    out, gs = or2(left, right)
    gates.extend(gs)
    return out


def decode_2bit(bit_hi, bit_lo, prefix, gates):
    """2-to-4 decoder using NAND2 gates only."""
    hi_inv, g = inv(bit_hi); gates.append(g)
    lo_inv, g = inv(bit_lo); gates.append(g)

    outputs = []
    combos = [(hi_inv, lo_inv), (hi_inv, bit_lo), (bit_hi, lo_inv), (bit_hi, bit_lo)]
    for i, (a, b) in enumerate(combos):
        out, gs = and2(a, b)
        gates.extend(gs)
        out.name = f"{prefix}{i}"
        outputs.append(out)
    return outputs


def decode_nbits(bits, prefix, gates):
    """Decode N bits into 2^N outputs using pairs of 2-bit decoders + AND trees.

    bits: list of Net (LSB first)
    Returns: list of 2^N output nets
    """
    n = len(bits)

    # Pad to even
    if n % 2 == 1:
        # Add a dummy bit tied low (always 0)
        dummy = Net(f"{prefix}_pad")
        # dummy is never driven — treated as always-low
        bits = list(bits) + [dummy]
        n += 1

    # Split into pairs and decode each
    pair_decodes = []
    for p in range(n // 2):
        lo_bit = bits[p * 2]
        hi_bit = bits[p * 2 + 1]
        pair_prefix = f"{prefix}_P{p}_"
        pair_outs = decode_2bit(hi_bit, lo_bit, pair_prefix, gates)
        pair_decodes.append(pair_outs)

    # Each minterm = AND of one output from each pair
    n_outputs = 2 ** len(bits)
    outputs = []
    for val in range(n_outputs):
        # Select one output from each pair
        selections = []
        for p in range(len(pair_decodes)):
            pair_val = (val >> (p * 2)) & 3
            selections.append(pair_decodes[p][pair_val])

        out = and_tree(selections, gates)
        outputs.append(out)

    return outputs


def generate_decoder(primary_bits, instruction_map, secondary_bits=None,
                     secondary_map=None, group_map=None):
    """Generate instruction decoder.

    Args:
        primary_bits: number of bits for primary decode (e.g. 4 for OPR)
        instruction_map: dict {opcode_int: "NAME"} for primary instructions
        secondary_bits: number of bits for secondary decode (e.g. 4 for OPA)
        secondary_map: dict {(primary_val, secondary_val): "NAME"} for sub-instructions
        group_map: dict {"GROUP_NAME": ["INST1", "INST2", ...]} for OR groups
    """
    Net._counter = 0
    Gate._counter = 0
    gates = []

    # Create input nets
    ir1 = [Net(f"IR1_{i}") for i in range(primary_bits)]

    # Inverted copies (useful as outputs)
    ir1_inv = []
    for i in range(primary_bits):
        inv_out, g = inv(ir1[i])
        gates.append(g)
        inv_out.name = f"!IR1_{i}"
        ir1_inv.append(inv_out)

    # Primary decode using pair structure
    n_pairs = (primary_bits + 1) // 2
    pair_decodes = []
    for p in range(n_pairs):
        lo_idx = p * 2
        hi_idx = p * 2 + 1
        lo_bit = ir1[lo_idx]
        hi_bit = ir1[hi_idx] if hi_idx < primary_bits else Net(f"IR1_pad")
        prefix = ["Lower", "Upper", "Upper2", "Upper3"][p] if p < 4 else f"Group{p}"
        pair_outs = decode_2bit(hi_bit, lo_bit, prefix, gates)
        pair_decodes.append(pair_outs)

    outputs = {}

    # Primary instructions
    for opcode, name in instruction_map.items():
        selections = []
        for p in range(len(pair_decodes)):
            pair_val = (opcode >> (p * 2)) & 3
            selections.append(pair_decodes[p][pair_val])
        out = and_tree(selections, gates)
        out.name = name
        outputs[name] = out

    # Secondary decode (if provided)
    if secondary_bits and secondary_map:
        ir2 = [Net(f"IR2_{i}") for i in range(secondary_bits)]

        ir2_inv = []
        for i in range(secondary_bits):
            inv_out, g = inv(ir2[i])
            gates.append(g)
            inv_out.name = f"!IR2_{i}"
            ir2_inv.append(inv_out)

        # Secondary pair decoders
        n_pairs2 = (secondary_bits + 1) // 2
        pair_decodes2 = []
        for p in range(n_pairs2):
            lo_idx = p * 2
            hi_idx = p * 2 + 1
            lo_bit = ir2[lo_idx]
            hi_bit = ir2[hi_idx] if hi_idx < secondary_bits else Net(f"IR2_pad")
            prefix = ["2Lower", "2Upper", "2Upper2", "2Upper3"][p] if p < 4 else f"2Group{p}"
            pair_outs = decode_2bit(hi_bit, lo_bit, prefix, gates)
            pair_decodes2.append(pair_outs)

        # Instructions that need both primary AND secondary
        for (pri_val, sec_val), name in secondary_map.items():
            # Primary match
            pri_sels = []
            for p in range(len(pair_decodes)):
                pair_val = (pri_val >> (p * 2)) & 3
                pri_sels.append(pair_decodes[p][pair_val])
            pri_match = and_tree(pri_sels, gates)

            # Secondary match
            sec_sels = []
            for p in range(len(pair_decodes2)):
                pair_val = (sec_val >> (p * 2)) & 3
                sec_sels.append(pair_decodes2[p][pair_val])
            sec_match = and_tree(sec_sels, gates)

            out, gs = and2(pri_match, sec_match)
            gates.extend(gs)
            out.name = name
            outputs[name] = out

        # Instructions that split on a single secondary bit
        # (like FIM/SRC splitting on IR2 bit 0)
        # These are handled via secondary_map entries

    # Group signals (OR of multiple instructions)
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
    """4004 instruction set: 4-bit OPR + 4-bit OPA."""
    primary_map = {
        0x0: "NOP", 0x1: "JCN", 0x4: "JUN", 0x5: "JMS",
        0x6: "INC", 0x7: "ISZ", 0x8: "ADD", 0x9: "SUB",
        0xA: "LD", 0xB: "XCH", 0xC: "BBL", 0xD: "LDM",
    }

    secondary_map = {
        # FIM/SRC split (IR1=2, IR2 bit 0)
        # FIM = IR1=2 AND IR2 even, SRC = IR1=2 AND IR2 odd
        # Encode as: primary=2, secondary=specific OPA values
        # Actually simpler: just list the full decode
    }

    # For FIM/SRC/FIN/JIN, we need primary AND secondary bit 0
    # Handle these specially after the main decode
    # IO group and ACC group need full secondary decode

    # Build secondary map for IO (IR1=0xE) and ACC (IR1=0xF) groups
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

    # FIM/SRC: IR1=2, even/odd OPA
    fim_src = {}
    for opa in range(0, 16, 2):
        fim_src[(0x2, opa)] = "FIM"
    for opa in range(1, 16, 2):
        fim_src[(0x2, opa)] = "SRC"
    for opa in range(0, 16, 2):
        fim_src[(0x3, opa)] = "FIN"
    for opa in range(1, 16, 2):
        fim_src[(0x3, opa)] = "JIN"

    # Combine all secondary maps — but FIM/SRC/FIN/JIN create duplicates
    # since multiple OPA values map to the same name.
    # Better approach: generate FIM/SRC/FIN/JIN using primary + IR2 bit 0

    # Actually, let's use the simple approach: decode primary, then
    # for IR1=2 and IR1=3, split using IR2 bit 0 separately.
    # This is what the working version did.

    # So: use generate_decoder for primary + IO/ACC secondary,
    # then add FIM/SRC/FIN/JIN manually.

    secondary_map = {}
    secondary_map.update(io_map)
    secondary_map.update(acc_map)

    group_map = {
        "WR": ["WR0", "WR1", "WR2", "WR3"],
        "RD": ["RD0", "RD1", "RD2", "RD3"],
        "2Word": ["JCN", "FIM", "JUN", "JMS", "ISZ"],
        "Jump": ["JCN", "JUN", "JMS", "ISZ", "JIN"],
    }

    return 4, primary_map, 4, secondary_map, group_map


def add_fim_src_fin_jin(inputs, outputs, gates):
    """Add FIM/SRC/FIN/JIN by splitting primary decode on IR2 bit 0."""
    ir2_0 = inputs["IR2"][0]
    ir2_0_inv, g = inv(ir2_0)
    gates.append(g)

    # Need primary decode for IR1=2 and IR1=3
    # These should already exist as intermediate nets from the primary decode
    # But they might not be named. Let's just AND the pair outputs.
    # IR1=2 = 0010: Lower=2(10), Upper=0(00) -> Lower[2] AND Upper[0]
    # We need to find these nets... they exist but aren't named.

    # Simpler: use the primary outputs if they exist, or create them
    # Actually: primary_map doesn't include 2 or 3 (they're split).
    # So we need to create IR1=2 and IR1=3 matches.

    # For now, check if we have named intermediate pair outputs
    # The pair decoders create Lower0-3 and Upper0-3.
    # IR1=2 = Upper0 AND Lower2, IR1=3 = Upper0 AND Lower3

    # These intermediates ARE created but not in outputs dict.
    # We need to either pass them through or recreate.
    # Let's just recreate using IR1 bits directly.

    ir1 = inputs["IR1"]
    # IR1=0010: !IR1_3 AND !IR1_2 AND IR1_1 AND !IR1_0
    # Using pairs: Upper(!IR1_3, !IR1_2)=Upper0 AND Lower(IR1_1, !IR1_0)=Lower2
    # But we don't have direct access. Let's just make AND gates.

    ir1_inv = []
    for i in range(4):
        # Check if we already have inverted versions
        inv_name = f"!IR1_{i}"
        found = None
        for g in gates:
            if g.output.name == inv_name:
                found = g.output
                break
        if found:
            ir1_inv.append(found)
        else:
            inv_out, g = inv(ir1[i])
            gates.append(g)
            inv_out.name = inv_name
            ir1_inv.append(inv_out)

    # IR1=2 (0010)
    ir1_2_a, gs = and2(ir1_inv[3], ir1_inv[2])
    gates.extend(gs)
    ir1_2_b, gs = and2(ir1[1], ir1_inv[0])
    gates.extend(gs)
    ir1_is_2, gs = and2(ir1_2_a, ir1_2_b)
    gates.extend(gs)

    # IR1=3 (0011)
    ir1_3_b, gs = and2(ir1[1], ir1[0])
    gates.extend(gs)
    ir1_is_3, gs = and2(ir1_2_a, ir1_3_b)  # reuse !IR1_3 AND !IR1_2
    gates.extend(gs)

    # FIM = IR1=2 AND !IR2_0
    fim, gs = and2(ir1_is_2, ir2_0_inv)
    gates.extend(gs)
    fim.name = "FIM"
    outputs["FIM"] = fim

    # SRC = IR1=2 AND IR2_0
    src, gs = and2(ir1_is_2, ir2_0)
    gates.extend(gs)
    src.name = "SRC"
    outputs["SRC"] = src

    # FIN = IR1=3 AND !IR2_0
    fin, gs = and2(ir1_is_3, ir2_0_inv)
    gates.extend(gs)
    fin.name = "FIN"
    outputs["FIN"] = fin

    # JIN = IR1=3 AND IR2_0
    jin, gs = and2(ir1_is_3, ir2_0)
    gates.extend(gs)
    jin.name = "JIN"
    outputs["JIN"] = jin


# ── Output ───────────────────────────────────────────────────────────────

def write_netlist(inputs, outputs, gates, filepath):
    with open(filepath, "w") as f:
        f.write(f"* Instruction Decoder (NAND2 mux)\n")
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
    parser.add_argument("--preset", choices=["4004"], default=None,
                        help="Use a preset instruction set")
    parser.add_argument("--bits", type=int, default=None,
                        help="Primary instruction width (if not using preset)")
    parser.add_argument("--map", default=None,
                        help="JSON instruction map file (if not using preset)")
    parser.add_argument("--place", action="store_true")
    parser.add_argument("--cols", type=int, default=14)
    args = parser.parse_args()

    if args.preset == "4004":
        pri_bits, pri_map, sec_bits, sec_map, grp_map = preset_4004()
    elif args.bits and args.map:
        pri_bits = args.bits
        with open(args.map) as f:
            data = json.load(f)
        pri_map = {int(k): v for k, v in data.get("primary", {}).items()}
        sec_bits = data.get("secondary_bits")
        sec_map = {(int(k.split(",")[0]), int(k.split(",")[1])): v
                   for k, v in data.get("secondary", {}).items()} if "secondary" in data else None
        grp_map = data.get("groups")
    else:
        print("Specify --preset or --bits + --map")
        return

    print(f"Generating decoder: {pri_bits}-bit primary" +
          (f" + {sec_bits}-bit secondary" if sec_bits else ""))

    inputs, outputs, gates = generate_decoder(
        pri_bits, pri_map, sec_bits, sec_map, grp_map)

    # Add FIM/SRC/FIN/JIN for 4004 preset
    if args.preset == "4004":
        add_fim_src_fin_jin(inputs, outputs, gates)
        # Update groups that include FIM/JIN
        if "2Word" in outputs and "FIM" in outputs:
            tw = or_tree([outputs["2Word"], outputs["FIM"]], gates)
            tw.name = "2Word"
            outputs["2Word"] = tw
        if "Jump" in outputs and "JIN" in outputs:
            jp = or_tree([outputs["Jump"], outputs["JIN"]], gates)
            jp.name = "Jump"
            outputs["Jump"] = jp

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
