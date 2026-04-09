#!/usr/bin/env python3
"""Universal circuit tester for LTSpice schematics.

Takes any .asc file, identifies its input/output signals, generates
PWL stimulus files to exercise all states (or specific test vectors),
builds a test harness, and runs it.

All test artifacts go in a temporary test/ folder that can be cleaned
without touching the original .asc.

Usage:
    # Test an adder — exhaustive (all input combinations)
    python tools/test_circuit.py cpus/tileable/generated/adder_4bit.asc --exhaustive

    # Test specific vectors
    python tools/test_circuit.py some_circuit.asc --vectors "A0=1,A1=0,B0=1,B1=0,Cin=0"

    # Just identify signals (dry run)
    python tools/test_circuit.py some_circuit.asc --dry-run

    # Clean test artifacts
    python tools/test_circuit.py --clean
"""

import argparse
import itertools
import os
import re
import shutil
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

# Logic voltage levels (SiC JFET)
V_HIGH = -0.8
V_LOW = -3.6

# Timing
CLK_PERIOD = 10e-6        # 10us per state
SETTLE_TIME = 5e-6         # 5us settling before measurement
RISE_TIME = 100e-9         # 100ns transition

# LTSpice path
LTSPICE_EXE = r"C:\Users\z00503ku\AppData\Local\Programs\ADI\LTspice\LTspice.exe"


def parse_signals(asc_path):
    """Extract all FLAG signal names from an .asc file.

    Returns dict with:
        inputs: list of signal names that look like inputs
        outputs: list of signal names that look like outputs
        power: list of power/ground signals
        internal: list of internal node signals
    """
    signals = set()
    with open(asc_path) as f:
        for line in f:
            if line.startswith("FLAG "):
                parts = line.strip().split(None, 3)
                if len(parts) >= 4:
                    name = parts[3]
                    signals.add(name)

    # Classify signals
    power = set()
    inputs = set()
    outputs = set()
    internal = set()

    for name in signals:
        if name in ("0", "VDD", "VSS", "VDD2", "VSS2"):
            power.add(name)
        elif name.startswith("!"):
            # Inverted output
            outputs.add(name)
        elif re.match(r"^(A|B|C|D)\d*$", name):
            inputs.add(name)
        elif re.match(r"^(Sum|Result|Cout|SUMCF|Zero|Out|Q|Y)\d*$", name):
            outputs.add(name)
        elif re.match(r"^(Cin|OP\d|sel|clk|CLK|load|enable|InvertReg)\w*$", name, re.IGNORECASE):
            inputs.add(name)
        elif re.match(r"^(Reg_Q|Reg_)\d+$", name):
            outputs.add(name)
        elif re.match(r"^(Bus_|Bus\d+In)\d*$", name):
            # Bus signals could be either
            outputs.add(name)
        elif re.match(r"^n\d+$", name):
            internal.add(name)
        else:
            # Unknown — check if it's likely an input or output
            # If it appears as a FLAG but not driven by any gate, it's an input
            internal.add(name)

    return {
        "inputs": sorted(inputs),
        "outputs": sorted(outputs),
        "power": sorted(power),
        "internal": sorted(internal),
        "all": sorted(signals),
    }


def generate_exhaustive_vectors(input_names):
    """Generate all 2^N input combinations.

    Returns list of dicts: [{name: 0|1, ...}, ...]
    """
    n = len(input_names)
    vectors = []
    for vals in itertools.product([0, 1], repeat=n):
        vec = dict(zip(input_names, vals))
        vectors.append(vec)
    return vectors


def generate_pwl_stimulus(vectors, input_names, output_dir, time_per_vector=None):
    """Generate PWL files for each input signal.

    Each vector holds for CLK_PERIOD, with proper step transitions.
    Returns total simulation time.
    """
    if time_per_vector is None:
        time_per_vector = CLK_PERIOD

    os.makedirs(output_dir, exist_ok=True)

    total_time = len(vectors) * time_per_vector + SETTLE_TIME

    for name in input_names:
        points = []
        t = 0
        prev_v = None

        for vec in vectors:
            v = V_HIGH if vec[name] else V_LOW

            if prev_v is not None and v != prev_v:
                # Step transition
                points.append((t - RISE_TIME, prev_v))
                points.append((t, v))
            elif prev_v is None:
                points.append((t, v))

            prev_v = v
            t += time_per_vector

        # Hold final value
        points.append((total_time, prev_v))

        # Write PWL
        path = os.path.join(output_dir, f"{name}.pwl")
        with open(path, "w") as f:
            last_t = -1
            for pt, pv in points:
                if pt <= last_t:
                    pt = last_t + 10e-9
                f.write(f"{pt:.9e} {pv:.4f}\n")
                last_t = pt

    return total_time


def build_test_harness(asc_path, test_dir, input_names, total_time):
    """Copy the .asc and add voltage sources + .tran directive.

    Returns path to the test .asc file.
    """
    base_name = os.path.splitext(os.path.basename(asc_path))[0]
    test_asc = os.path.join(test_dir, f"{base_name}_test.asc")

    # Copy original
    shutil.copy2(asc_path, test_asc)

    # Append voltage sources for each input and .tran
    # All coordinates on 16-unit grid. Voltage source SYMBOL at (x, y):
    #   Positive pin FLAG at (x, y+16)
    #   Negative pin FLAG at (x, y+96)
    with open(test_asc, "a") as f:
        x_base = 80000  # far right, on grid
        y_base = 0
        spacing = 160   # between voltage sources, on 16-unit grid

        for i, name in enumerate(input_names):
            x = x_base + i * spacing
            y = y_base
            f.write(f"FLAG {x} {y + 16} {name}\n")
            f.write(f"FLAG {x} {y + 96} 0\n")
            f.write(f"SYMBOL voltage {x} {y} R0\n")
            f.write(f"WINDOW 0 56 32 Invisible 2\n")
            f.write(f"WINDOW 3 56 72 Invisible 2\n")
            f.write(f"SYMATTR InstName V_test_{i}\n")
            f.write(f"SYMATTR Value PWL file={name}.pwl\n")

        # VDD source
        vdd_x = x_base
        vdd_y = y_base + 256
        f.write(f"FLAG {vdd_x} {vdd_y + 16} VDD\n")
        f.write(f"FLAG {vdd_x} {vdd_y + 96} 0\n")
        f.write(f"SYMBOL voltage {vdd_x} {vdd_y} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VDD\n")
        f.write(f"SYMATTR Value 24\n")

        # VSS source
        vss_x = x_base + spacing
        vss_y = y_base + 256
        f.write(f"FLAG {vss_x} {vss_y + 16} VSS\n")
        f.write(f"FLAG {vss_x} {vss_y + 96} 0\n")
        f.write(f"SYMBOL voltage {vss_x} {vss_y} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VSS\n")
        f.write(f"SYMATTR Value -20\n")

        # .tran and JFET model
        tran_us = int(total_time * 1e6) + 10
        f.write(f"TEXT {x_base} {vdd_y + 256} Left 2 !.tran {tran_us}u\n")
        f.write(f"TEXT {x_base} {vdd_y + 320} Left 2 !.model DR NJF(Beta=0.135m Betatce=-0.5 "
                f"Vto=-3.45 Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f "
                f"Nr=4 Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
                f"Vk=123.7 M=407m Pb=1 Kf=37860f Af=1 Mfg=Linear_Systems)\n")

    return test_asc


def generate_expected_results(vectors, input_names, output_names, circuit_type="adder"):
    """Compute expected output values for each test vector.

    For adders: Sum = A + B + Cin, SUMCF = carry out.
    Returns list of dicts [{output_name: 0|1, ...}, ...]
    """
    results = []

    if circuit_type == "adder":
        # Figure out bit width from input names
        a_bits = sorted([n for n in input_names if re.match(r"A\d+$", n)])
        b_bits = sorted([n for n in input_names if re.match(r"B\d+$", n)])
        n_bits = len(a_bits)

        for vec in vectors:
            a_val = sum(vec.get(f"A{i}", 0) << i for i in range(n_bits))
            b_val = sum(vec.get(f"B{i}", 0) << i for i in range(n_bits))
            cin = vec.get("Cin", 0)

            total = a_val + b_val + cin
            result = {}
            for i in range(n_bits):
                result[f"Sum{i}"] = (total >> i) & 1
            result["SUMCF"] = (total >> n_bits) & 1
            results.append(result)

    elif circuit_type == "register":
        # For registers, just pass through
        for vec in vectors:
            results.append({})

    else:
        for vec in vectors:
            results.append({})

    return results


def test_circuit(asc_path, vectors=None, exhaustive=False, dry_run=False,
                 circuit_type="auto", custom_inputs=None, custom_outputs=None):
    """Main test flow."""
    print(f"Testing: {asc_path}")

    # Parse signals
    sigs = parse_signals(asc_path)

    if custom_inputs:
        sigs["inputs"] = custom_inputs
    if custom_outputs:
        sigs["outputs"] = custom_outputs

    print(f"  Inputs:  {sigs['inputs']}")
    print(f"  Outputs: {sigs['outputs']}")
    print(f"  Power:   {sigs['power']}")
    print(f"  Internal: {len(sigs['internal'])} nodes")

    if dry_run:
        print("\n  Dry run — no test generated.")
        return

    if not sigs["inputs"]:
        print("  ERROR: No input signals found. Use --inputs to specify them.")
        return

    # Auto-detect circuit type
    if circuit_type == "auto":
        if any("Sum" in o for o in sigs["outputs"]) or "SUMCF" in sigs["outputs"]:
            circuit_type = "adder"
        elif any("Reg_" in o for o in sigs["outputs"]):
            circuit_type = "register"
        else:
            circuit_type = "generic"
    print(f"  Type: {circuit_type}")

    # Generate test vectors
    if exhaustive:
        vectors = generate_exhaustive_vectors(sigs["inputs"])
        print(f"  Exhaustive: {len(vectors)} vectors ({len(sigs['inputs'])} inputs)")
        if len(vectors) > 1024:
            print(f"  WARNING: {len(vectors)} vectors will take a long time!")
    elif vectors is None:
        vectors = generate_exhaustive_vectors(sigs["inputs"])
        if len(vectors) > 256:
            print(f"  Too many vectors ({len(vectors)}), using first 256")
            vectors = vectors[:256]

    # Set up test directory
    base_name = os.path.splitext(os.path.basename(asc_path))[0]
    test_dir = os.path.join(os.path.dirname(asc_path), f"test_{base_name}")
    os.makedirs(test_dir, exist_ok=True)
    print(f"  Test dir: {test_dir}")

    # Generate PWL stimulus
    total_time = generate_pwl_stimulus(vectors, sigs["inputs"], test_dir)
    print(f"  Stimulus: {len(vectors)} vectors, {total_time*1e6:.0f}us total")

    # Generate expected results
    expected = generate_expected_results(vectors, sigs["inputs"], sigs["outputs"], circuit_type)

    # Save expected results for later comparison
    if expected and expected[0]:
        exp_path = os.path.join(test_dir, "expected.txt")
        with open(exp_path, "w") as f:
            f.write(f"# Expected results for {base_name}\n")
            f.write(f"# {len(vectors)} test vectors, {circuit_type}\n")
            f.write(f"# Inputs: {' '.join(sigs['inputs'])}\n")
            f.write(f"# Outputs: {' '.join(sigs['outputs'])}\n\n")

            header = " ".join(sigs["inputs"]) + " | " + " ".join(sorted(expected[0].keys()))
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            for vec, exp in zip(vectors, expected):
                in_str = " ".join(str(vec[n]) for n in sigs["inputs"])
                out_str = " ".join(str(exp[k]) for k in sorted(exp.keys()))
                f.write(f"{in_str} | {out_str}\n")

        print(f"  Expected: {exp_path}")

    # Build test harness
    test_asc = build_test_harness(asc_path, test_dir, sigs["inputs"], total_time)
    print(f"  Test schematic: {test_asc}")

    print(f"\n  Ready! Open {test_asc} in LTSpice and run.")
    print(f"  After simulation, run:")
    print(f"    python tools/test_circuit.py {asc_path} --check {test_dir}")

    return test_dir, vectors, expected


def clean_test_dir(asc_path):
    """Remove test artifacts for a circuit."""
    base_name = os.path.splitext(os.path.basename(asc_path))[0]
    test_dir = os.path.join(os.path.dirname(asc_path), f"test_{base_name}")
    if os.path.isdir(test_dir):
        shutil.rmtree(test_dir)
        print(f"Cleaned: {test_dir}")
    else:
        print(f"Nothing to clean: {test_dir}")


def main():
    parser = argparse.ArgumentParser(description="Test any LTSpice circuit")
    parser.add_argument("asc_file", nargs="?", help="Path to .asc file")
    parser.add_argument("--exhaustive", "-e", action="store_true",
                        help="Test all input combinations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just identify signals, don't generate test")
    parser.add_argument("--type", default="auto",
                        choices=["auto", "adder", "register", "generic"],
                        help="Circuit type (default: auto-detect)")
    parser.add_argument("--inputs", default=None,
                        help="Comma-separated input signal names (override auto-detect)")
    parser.add_argument("--outputs", default=None,
                        help="Comma-separated output signal names (override auto-detect)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove test artifacts")
    parser.add_argument("--check", default=None,
                        help="Check results from test dir against expected")

    args = parser.parse_args()

    if args.clean and args.asc_file:
        clean_test_dir(args.asc_file)
        return

    if not args.asc_file:
        parser.print_help()
        return

    custom_inputs = args.inputs.split(",") if args.inputs else None
    custom_outputs = args.outputs.split(",") if args.outputs else None

    test_circuit(
        args.asc_file,
        exhaustive=args.exhaustive,
        dry_run=args.dry_run,
        circuit_type=args.type,
        custom_inputs=custom_inputs,
        custom_outputs=custom_outputs,
    )


if __name__ == "__main__":
    main()
