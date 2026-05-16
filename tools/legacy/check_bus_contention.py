#!/usr/bin/env python3
"""Check micro-instructions for bus contention and generate truth table.

Reads the micro_instructions.csv and checks for:
1. Multiple bus drivers in the same micro-phase (contention)
2. Bus driver + bus load that conflict
3. Opportunities for hardwiring

Bus DRIVERS (things that put data ON the bus):
  IR2_Out, IR3_Out, IR4_Out, Scratch_Out, Acc_Out, Temp_Out,
  EO_Out, INC_Out, CF_Out, Bus0In, Bus1In, Bus2In, Bus3In

Bus LOADERS (things that read FROM the bus):
  Acc_Load, Temp_Load, ScratchPad_Load, PC1_Bus_Load, PC2_Bus_Load,
  PC3_Bus_Load

Non-bus signals (directly wired, no contention):
  IndexReg_Load (hardwired from IR2), Scratch_Reg_Select,
  CF_Load, CF_Clear, CF_Inv, CFStage1_Load,
  Shift_Enable, Shift_Direction, Shift_Loading,
  Micro_Reset, PC_INC, PC_Stack_Load, Stack_Load,
  Acc_INV, Temp_INV, Subtract_Set, Subtract_Reset,
  ScratchPadWord1, ScratchPadWord2
"""

import csv
import sys
import os

BUS_DRIVERS = {
    'IR2_Out', 'IR3_Out', 'IR4_Out',
    'Scratch_Out', 'Acc_Out', 'Temp_Out',
    'EO_Out', 'INC_Out', 'CF_Out',
    'Bus0In', 'Bus1In', 'Bus2In', 'Bus3In',
}

BUS_LOADERS = {
    'Acc_Load', 'Temp_Load', 'ScratchPad_Load',
    'PC1_Bus_Load', 'PC2_Bus_Load', 'PC3_Bus_Load',
}

# Proposed hardwired connections (not using bus)
HARDWIRED = {
    'IndexReg_Load',  # hardwired from IR2
}

# BusXIn signals are special - they drive specific bits, not the full bus
# Bus0In drives only bit 0, etc. Multiple BusXIn is fine as long as
# they're different bits. But BusXIn + another full driver = contention.


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


def check_contention(instructions):
    print("=" * 80)
    print("BUS CONTENTION CHECK")
    print("=" * 80)

    issues = []

    for opcode, micros in instructions:
        for micro_n, signals in sorted(micros.items()):
            drivers = signals & BUS_DRIVERS
            loaders = signals & BUS_LOADERS

            # Check for multiple full-bus drivers
            full_drivers = drivers - {'Bus0In', 'Bus1In', 'Bus2In', 'Bus3In'}
            bus_bit_drivers = drivers & {'Bus0In', 'Bus1In', 'Bus2In', 'Bus3In'}

            if len(full_drivers) > 1:
                issue = f"CONTENTION: {opcode} Micro{micro_n}: multiple bus drivers: {full_drivers}"
                issues.append(issue)
                print(f"  !! {issue}")

            if full_drivers and bus_bit_drivers:
                issue = f"CONTENTION: {opcode} Micro{micro_n}: full driver {full_drivers} + bit driver {bus_bit_drivers}"
                issues.append(issue)
                print(f"  !! {issue}")

            # Check for IR_Out driving bus while also loading PC from bus
            # With hardwiring proposal: IR3->PC2, IR4->PC3 directly
            ir_outs = {s for s in signals if s.startswith('IR') and s.endswith('_Out')}
            pc_loads = {s for s in signals if s.startswith('PC') and 'Bus_Load' in s}

            if ir_outs and pc_loads:
                # This is the case where IR drives bus and PC loads from bus
                # With hardwiring, these don't need the bus
                print(f"  NOTE: {opcode} Micro{micro_n}: IR->bus->PC: {ir_outs} + {pc_loads}")
                print(f"         -> Can be hardwired (IR3->PC2, IR4->PC3, IR2->PC1)")

            # Check for Scratch_Out driving bus while also doing other bus operations
            if 'Scratch_Out' in signals:
                other_drivers = full_drivers - {'Scratch_Out'}
                if other_drivers:
                    issue = f"CONTENTION: {opcode} Micro{micro_n}: Scratch_Out + {other_drivers}"
                    issues.append(issue)
                    print(f"  !! {issue}")

    return issues


def check_hardwire_opportunities(instructions):
    print("\n" + "=" * 80)
    print("HARDWIRE ANALYSIS")
    print("=" * 80)

    # Find all (driver, loader) pairs that appear together
    pair_counts = {}
    for opcode, micros in instructions:
        for micro_n, signals in sorted(micros.items()):
            drivers = signals & BUS_DRIVERS
            loaders = signals & BUS_LOADERS
            for d in drivers:
                for l in loaders:
                    key = (d, l)
                    if key not in pair_counts:
                        pair_counts[key] = []
                    pair_counts[key].append(f"{opcode} M{micro_n}")

    print("\nDriver -> Loader pairs (candidates for hardwiring):")
    for (driver, loader), uses in sorted(pair_counts.items(), key=lambda x: -len(x[1])):
        print(f"  {driver:15s} -> {loader:18s}  ({len(uses)} uses): {', '.join(uses[:5])}")

    # Check: does IR2_Out ever drive anything OTHER than IndexReg_Load and Acc_Load?
    print("\n\nIR2_Out usage (what does it load?):")
    for opcode, micros in instructions:
        for micro_n, signals in sorted(micros.items()):
            if 'IR2_Out' in signals:
                loaders = signals & BUS_LOADERS
                others = signals - BUS_DRIVERS - BUS_LOADERS - HARDWIRED - \
                         {'Micro_Reset', 'PC_INC', 'Scratch_Reg_Select',
                          'ScratchPadWord1', 'ScratchPadWord2', 'Temp_INV'}
                print(f"  {opcode:20s} M{micro_n}: loads={loaders}")

    print("\n\nIR3_Out usage:")
    for opcode, micros in instructions:
        for micro_n, signals in sorted(micros.items()):
            if 'IR3_Out' in signals:
                loaders = signals & BUS_LOADERS
                print(f"  {opcode:20s} M{micro_n}: loads={loaders}")

    print("\n\nIR4_Out usage:")
    for opcode, micros in instructions:
        for micro_n, signals in sorted(micros.items()):
            if 'IR4_Out' in signals:
                loaders = signals & BUS_LOADERS
                print(f"  {opcode:20s} M{micro_n}: loads={loaders}")


def print_truth_table(instructions):
    print("\n" + "=" * 80)
    print("MICRO-INSTRUCTION TRUTH TABLE")
    print("=" * 80)

    # Collect all signals
    all_signals = set()
    for _, micros in instructions:
        for _, signals in micros.items():
            all_signals |= signals

    # For each micro phase, show which instructions assert which signals
    for micro_n in range(2, 8):
        signals_this_micro = set()
        for _, micros in instructions:
            if micro_n in micros:
                signals_this_micro |= micros[micro_n]

        if not signals_this_micro:
            continue

        print(f"\n--- MICRO {micro_n} ---")
        # For each signal, list which instructions assert it
        for sig in sorted(signals_this_micro):
            asserting = []
            for opcode, micros in instructions:
                if micro_n in micros and sig in micros[micro_n]:
                    asserting.append(opcode)
            if asserting:
                print(f"  {sig:25s} = {' | '.join(asserting)}")


def main():
    csv_path = os.path.join(os.path.dirname(__file__), '..',
                            'cpus', 'simplified', 'micro_instructions.csv')
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    instructions = parse_csv(csv_path)
    print(f"Loaded {len(instructions)} instruction variants\n")

    issues = check_contention(instructions)
    check_hardwire_opportunities(instructions)
    print_truth_table(instructions)

    print(f"\n{'=' * 80}")
    if issues:
        print(f"FOUND {len(issues)} CONTENTION ISSUES!")
    else:
        print("No bus contention found.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
