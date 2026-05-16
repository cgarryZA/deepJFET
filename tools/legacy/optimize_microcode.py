#!/usr/bin/env python3
"""Optimize micro-instruction decode logic to minimize JFET count.

Cost model:
- New gate: 5 + N JFETs (1 instruction + 1 micro + 1 level shift + 2 invert + N signal outputs)
- Extra instruction limb on existing gate: +1 JFET
- Extra signal output on existing gate: +1 JFET

Strategy: find instructions that assert identical signal sets at the same
micro-phase and merge them into shared gates.

After hardwiring: IR2->IndexReg, IR3->PC2, IR4->PC3, IR2->PC1
Remove from bus: IR2_Out, IR3_Out, IR4_Out, PC1_Bus_Load, PC2_Bus_Load, PC3_Bus_Load
Replace with direct: PC1_IR_Load, PC2_IR_Load, PC3_IR_Load, IndexReg_Load (all off-bus)
"""

import csv
import os
from collections import defaultdict

# Signals that are now hardwired (not going through bus, not part of micro-instruction ROM)
HARDWIRED_REMOVE = {
    'IR2_Out', 'IR3_Out', 'IR4_Out',
    'PC1_Bus_Load', 'PC2_Bus_Load', 'PC3_Bus_Load',
    'IndexReg_Load',
}

# These become direct-wired control signals (still need micro-instruction gates)
HARDWIRED_ADD = {
    'PC1_IR_Load',   # was IR2->bus->PC1
    'PC2_IR_Load',   # was IR3->bus->PC2
    'PC3_IR_Load',   # was IR4->bus->PC3
    'IndexReg_Load', # was IR2->bus->IndexReg (keep signal name, just hardwired source)
}

# Map old bus signals to new hardwired equivalents
RENAME_MAP = {
    # When IR2_Out + PC1_Bus_Load appeared together -> PC1_IR_Load
    # When IR3_Out + PC2_Bus_Load -> PC2_IR_Load
    # When IR4_Out + PC3_Bus_Load -> PC3_IR_Load
    # When IR2_Out + IndexReg_Load -> IndexReg_Load (already hardwired)
}


def parse_csv(path):
    instructions = []
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row or not row[0].strip():
                continue
            opcode = row[0].strip()
            micros = {}
            for i, cell in enumerate(row[1:], start=2):
                if cell.strip():
                    signals = set(cell.strip().split())
                    micros[i] = signals
            instructions.append((opcode, micros))
    return instructions


def apply_hardwiring(instructions):
    """Replace bus-based IR/PC transfers with hardwired equivalents."""
    updated = []
    for opcode, micros in instructions:
        new_micros = {}
        for micro_n, signals in micros.items():
            new_signals = set()
            for sig in signals:
                if sig in HARDWIRED_REMOVE:
                    # Map to hardwired equivalent
                    if sig == 'PC1_Bus_Load':
                        new_signals.add('PC1_IR_Load')
                    elif sig == 'PC2_Bus_Load':
                        new_signals.add('PC2_IR_Load')
                    elif sig == 'PC3_Bus_Load':
                        new_signals.add('PC3_IR_Load')
                    elif sig == 'IndexReg_Load':
                        new_signals.add('IndexReg_Load')
                    # IR2_Out, IR3_Out, IR4_Out just get removed
                    # (they're now direct wires, no gate needed)
                else:
                    new_signals.add(sig)
            new_micros[micro_n] = new_signals
        updated.append((opcode, new_micros))
    return updated


def find_groups(instructions):
    """Group (instruction, micro) pairs by identical signal sets."""
    # Key: (micro_n, frozenset(signals)) -> [list of opcodes]
    groups = defaultdict(list)
    for opcode, micros in instructions:
        for micro_n, signals in micros.items():
            key = (micro_n, frozenset(signals))
            groups[key].append(opcode)
    return groups


def cost_unoptimized(instructions):
    """Calculate total JFET cost with no sharing."""
    total = 0
    for opcode, micros in instructions:
        for micro_n, signals in micros.items():
            n_signals = len(signals)
            cost = 5 + n_signals  # 1 inst + 1 micro + 1 level + 2 invert + N outputs
            total += cost
    return total


def cost_optimized(groups):
    """Calculate total JFET cost with shared gates for identical signal sets."""
    total = 0
    for (micro_n, signals_frozen), opcodes in groups.items():
        n_signals = len(signals_frozen)
        n_instructions = len(opcodes)
        # First instruction: 5 + N
        # Each additional instruction: +1 (extra limb)
        cost = 5 + n_signals + (n_instructions - 1)
        total += cost
    return total


def further_optimize(instructions):
    """Find signals that can share gates by decomposing into common subsets.

    If instructions A and B at Micro2 need:
      A: {Scratch_Out, Temp_Load, Scratch_Reg_Select}
      B: {Scratch_Out, Temp_Load, Scratch_Reg_Select, EO_Out}

    Then {Scratch_Out, Temp_Load, Scratch_Reg_Select} can be a shared gate for A+B,
    and B gets one extra gate for {EO_Out}.
    """
    # For each micro phase, find common signal subsets
    by_micro = defaultdict(list)  # micro_n -> [(opcode, signals)]
    for opcode, micros in instructions:
        for micro_n, signals in micros.items():
            by_micro[micro_n].append((opcode, signals))

    results = []
    total_cost = 0

    for micro_n in sorted(by_micro.keys()):
        entries = by_micro[micro_n]

        # Strategy: for each signal, find which instructions assert it
        signal_to_insts = defaultdict(set)
        for opcode, signals in entries:
            for sig in signals:
                signal_to_insts[sig].add(opcode)

        # Group signals by their instruction set (signals with identical asserting instructions)
        inst_set_to_signals = defaultdict(set)
        for sig, insts in signal_to_insts.items():
            inst_set_to_signals[frozenset(insts)].add(sig)

        # Each group = one gate
        gates = []
        for inst_set, signal_set in inst_set_to_signals.items():
            n_inst = len(inst_set)
            n_sig = len(signal_set)
            gate_cost = 5 + (n_inst - 1) + n_sig  # base + extra limbs + outputs
            gates.append({
                'micro': micro_n,
                'instructions': sorted(inst_set),
                'signals': sorted(signal_set),
                'cost': gate_cost,
            })
            total_cost += gate_cost

        results.append((micro_n, gates))

    return results, total_cost


def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..',
                            'cpus', 'simplified', 'micro_instructions.csv')

    instructions = parse_csv(csv_path)
    print(f"Loaded {len(instructions)} instruction variants")

    # Apply hardwiring
    instructions = apply_hardwiring(instructions)
    print("Applied hardwiring (IR->PC direct, IR2->IndexReg direct)")

    # Show updated truth table
    print(f"\n{'='*90}")
    print("UPDATED MICRO-INSTRUCTIONS (after hardwiring)")
    print(f"{'='*90}")
    for opcode, micros in instructions:
        print(f"\n  {opcode}:")
        for micro_n in sorted(micros.keys()):
            sigs = sorted(micros[micro_n])
            print(f"    M{micro_n}: {', '.join(sigs)}")

    # Calculate costs
    unopt_cost = cost_unoptimized(instructions)
    groups = find_groups(instructions)
    basic_opt_cost = cost_optimized(groups)

    print(f"\n{'='*90}")
    print("BASIC OPTIMIZATION (merge identical signal sets)")
    print(f"{'='*90}")
    print(f"Unoptimized: {unopt_cost} JFETs ({len([1 for _,m in instructions for _ in m])} gates)")
    print(f"Basic merge: {basic_opt_cost} JFETs ({len(groups)} gates)")

    for (micro_n, signals_frozen), opcodes in sorted(groups.items()):
        if len(opcodes) > 1:
            print(f"  SHARED M{micro_n}: {sorted(signals_frozen)}")
            print(f"         Instructions: {', '.join(opcodes)}")

    # Advanced optimization: group by signal->instruction mapping
    print(f"\n{'='*90}")
    print("ADVANCED OPTIMIZATION (group signals by instruction set)")
    print(f"{'='*90}")

    results, advanced_cost = further_optimize(instructions)

    gate_count = 0
    for micro_n, gates in results:
        print(f"\n--- MICRO {micro_n} ---")
        for gate in sorted(gates, key=lambda g: -len(g['instructions'])):
            gate_count += 1
            insts = ', '.join(gate['instructions'])
            sigs = ', '.join(gate['signals'])
            print(f"  Gate {gate_count:2d} ({gate['cost']:2d} JFETs): "
                  f"[{insts}] -> {{{sigs}}}")

    print(f"\n{'='*90}")
    print(f"COST SUMMARY")
    print(f"{'='*90}")
    print(f"  Unoptimized:    {unopt_cost:4d} JFETs")
    print(f"  Basic merge:    {basic_opt_cost:4d} JFETs")
    print(f"  Advanced:       {advanced_cost:4d} JFETs  ({gate_count} gates)")
    print(f"  Savings:        {unopt_cost - advanced_cost:4d} JFETs ({(unopt_cost-advanced_cost)/unopt_cost*100:.0f}%)")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
