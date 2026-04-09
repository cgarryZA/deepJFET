#!/usr/bin/env python3
"""Test an N-bit register schematic.

Tests:
1. Load alternating 10101010 pattern, verify it latches
2. Change inputs to 01010101, verify register HOLDS old value (no clock)
3. Clock with load disabled, verify register HOLDS old value
4. Clock with load enabled, verify new pattern latches
5. Repeat with inverted patterns

Usage:
    python tools/test_register.py cpus/tileable/generated/register_8bit.asc
"""

import os
import re
import shutil
import sys

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

V_HIGH = -0.8
V_LOW = -3.6
RISE = 100e-9


def generate_register_test(asc_path, n_bits=None):
    """Generate test harness for a register."""

    # Detect bit width from signal names
    with open(asc_path) as f:
        content = f.read()

    if n_bits is None:
        bits = set()
        for m in re.finditer(r"FLAG .+ Bus_(\d+)", content):
            bits.add(int(m.group(1)))
        n_bits = max(bits) + 1 if bits else 8

    print(f"Testing {n_bits}-bit register: {asc_path}")

    base_name = os.path.splitext(os.path.basename(asc_path))[0]
    test_dir = os.path.join(os.path.dirname(asc_path), f"test_{base_name}")
    os.makedirs(test_dir, exist_ok=True)

    # Test sequence timing (in us)
    # Each phase is 20us to give plenty of settling time
    P = 20e-6  # phase duration
    CLK_W = 5e-6  # clock pulse width

    # Pattern A = 10101010 (alternating, MSB first)
    pat_a = [(i % 2) for i in range(n_bits)]  # bit0=0, bit1=1, bit2=0...
    # Pattern B = 01010101 (inverted)
    pat_b = [1 - x for x in pat_a]
    # Pattern C = all 1s
    pat_c = [1] * n_bits
    # Pattern D = all 0s
    pat_d = [0] * n_bits

    def v(bit):
        return V_HIGH if bit else V_LOW

    # Build the test sequence:
    # Time    Bus Data    Reg_Load    CLK         Expected Reg    Description
    # 0       pat_a       LOW         LOW         unknown         setup
    # P       pat_a       HIGH        LOW         unknown         enable load
    # 2P      pat_a       HIGH        pulse       pat_a           CLOCK: latch pat_a
    # 3P      pat_b       HIGH        LOW         pat_a           change data, no clock - should HOLD
    # 4P      pat_b       LOW         pulse       pat_a           clock but load disabled - should HOLD
    # 5P      pat_b       HIGH        pulse       pat_b           CLOCK+LOAD: latch pat_b
    # 6P      pat_c       LOW         LOW         pat_b           change data, load off - should HOLD
    # 7P      pat_c       LOW         pulse       pat_b           clock but no load - should HOLD
    # 8P      pat_c       HIGH        pulse       pat_c           CLOCK+LOAD: latch all 1s
    # 9P      pat_d       HIGH        pulse       pat_d           CLOCK+LOAD: latch all 0s
    # 10P     pat_a       HIGH        pulse       pat_a           CLOCK+LOAD: latch pat_a again

    phases = [
        # (time, bus_pattern, reg_load, clk_pulse, expected_after, description)
        (0,    pat_a, 0, 0, None,  "setup: data=pat_a, load=off, no clock"),
        (P,    pat_a, 1, 0, None,  "enable load, no clock yet"),
        (2*P,  pat_a, 1, 1, pat_a, "CLOCK+LOAD: latch pattern A (10101010)"),
        (3*P,  pat_b, 1, 0, pat_a, "change data to B, no clock: should HOLD A"),
        (4*P,  pat_b, 0, 1, pat_a, "clock but load OFF: should HOLD A"),
        (5*P,  pat_b, 1, 1, pat_b, "CLOCK+LOAD: latch pattern B (01010101)"),
        (6*P,  pat_c, 0, 0, pat_b, "change data to all-1s, load off: should HOLD B"),
        (7*P,  pat_c, 0, 1, pat_b, "clock but load OFF: should HOLD B"),
        (8*P,  pat_c, 1, 1, pat_c, "CLOCK+LOAD: latch all 1s"),
        (9*P,  pat_d, 1, 1, pat_d, "CLOCK+LOAD: latch all 0s"),
        (10*P, pat_a, 1, 1, pat_a, "CLOCK+LOAD: latch pattern A again"),
    ]

    total_time = 12 * P

    # Generate Bus PWL files
    for bit in range(n_bits):
        points = []
        prev_v = None
        for t, pattern, _, _, _, _ in phases:
            bv = v(pattern[bit])
            if prev_v is not None and bv != prev_v:
                points.append((t - RISE, prev_v))
            points.append((t, bv))
            prev_v = bv
        points.append((total_time, prev_v))

        path = os.path.join(test_dir, f"Bus_{bit}.pwl")
        with open(path, "w") as f:
            for pt, pv in points:
                f.write(f"{pt:.9e} {pv:.4f}\n")

    # Generate Reg_Load PWL
    points = []
    prev_v = None
    for t, _, load, _, _, _ in phases:
        lv = v(load)
        if prev_v is not None and lv != prev_v:
            points.append((t - RISE, prev_v))
        points.append((t, lv))
        prev_v = lv
    points.append((total_time, prev_v))

    with open(os.path.join(test_dir, "Reg_Load.pwl"), "w") as f:
        for pt, pv in points:
            f.write(f"{pt:.9e} {pv:.4f}\n")

    # Generate CLK PWL (pulses at specific phases)
    clk_points = [(0, V_LOW)]
    for t, _, _, clk_pulse, _, _ in phases:
        if clk_pulse:
            t_rise = t + P * 0.3  # pulse in the middle of the phase
            t_fall = t_rise + CLK_W
            clk_points.append((t_rise - RISE, V_LOW))
            clk_points.append((t_rise, V_HIGH))
            clk_points.append((t_fall, V_HIGH))
            clk_points.append((t_fall + RISE, V_LOW))
    clk_points.append((total_time, V_LOW))

    with open(os.path.join(test_dir, "CLK.pwl"), "w") as f:
        last_t = -1
        for pt, pv in clk_points:
            if pt <= last_t:
                pt = last_t + 10e-9
            f.write(f"{pt:.9e} {pv:.4f}\n")
            last_t = pt

    # Save expected results
    with open(os.path.join(test_dir, "expected.txt"), "w") as f:
        f.write(f"# {n_bits}-bit register test\n")
        f.write(f"# Sample AFTER each phase (at phase_time + 0.8*P)\n\n")
        for t, pattern, load, clk, expected, desc in phases:
            if expected is not None:
                exp_str = "".join(str(b) for b in reversed(expected))  # MSB first
                f.write(f"t={t*1e6:6.0f}us  expect=0b{exp_str}  ({desc})\n")

    # Build test harness
    test_asc = os.path.join(test_dir, f"{base_name}_test.asc")
    shutil.copy2(asc_path, test_asc)

    with open(test_asc, "a") as f:
        x = 80000
        y = 0
        sp = 160

        # Bus data inputs
        for bit in range(n_bits):
            bx = x + bit * sp
            f.write(f"FLAG {bx} {y+16} Bus_{bit}\n")
            f.write(f"FLAG {bx} {y+96} 0\n")
            f.write(f"SYMBOL voltage {bx} {y} R0\n")
            f.write(f"WINDOW 0 56 32 Invisible 2\n")
            f.write(f"WINDOW 3 56 72 Invisible 2\n")
            f.write(f"SYMATTR InstName V_Bus{bit}\n")
            f.write(f"SYMATTR Value PWL file=Bus_{bit}.pwl\n")

        # Reg_Load
        lx = x + n_bits * sp
        f.write(f"FLAG {lx} {y+16} Reg_Load\n")
        f.write(f"FLAG {lx} {y+96} 0\n")
        f.write(f"SYMBOL voltage {lx} {y} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_Load\n")
        f.write(f"SYMATTR Value PWL file=Reg_Load.pwl\n")

        # CLK
        cx = lx + sp
        f.write(f"FLAG {cx} {y+16} CLK\n")
        f.write(f"FLAG {cx} {y+96} 0\n")
        f.write(f"SYMBOL voltage {cx} {y} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_CLK\n")
        f.write(f"SYMATTR Value PWL file=CLK.pwl\n")

        # VDD
        vy = y + 256
        f.write(f"FLAG {x} {vy+16} VDD\n")
        f.write(f"FLAG {x} {vy+96} 0\n")
        f.write(f"SYMBOL voltage {x} {vy} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VDD\n")
        f.write(f"SYMATTR Value 24\n")

        # VSS
        f.write(f"FLAG {x+sp} {vy+16} VSS\n")
        f.write(f"FLAG {x+sp} {vy+96} 0\n")
        f.write(f"SYMBOL voltage {x+sp} {vy} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VSS\n")
        f.write(f"SYMATTR Value -20\n")

        # .tran and model
        tran_us = int(total_time * 1e6) + 10
        f.write(f"TEXT {x} {vy+256} Left 2 !.tran {tran_us}u\n")
        f.write(f"TEXT {x} {vy+320} Left 2 !.model DR NJF(Beta=0.135m Betatce=-0.5 "
                f"Vto=-3.45 Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f "
                f"Nr=4 Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
                f"Vk=123.7 M=407m Pb=1 Kf=37860f Af=1 Mfg=Linear_Systems)\n")

    print(f"  Test dir: {test_dir}")
    print(f"  Phases: {len(phases)} ({total_time*1e6:.0f}us)")
    print(f"  Test schematic: {test_asc}")
    print(f"  Expected: {os.path.join(test_dir, 'expected.txt')}")
    print(f"\n  Open {test_asc} in LTSpice and run.")

    return test_dir, phases, n_bits


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test N-bit register")
    parser.add_argument("asc_file", help="Path to register .asc")
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()

    if args.clean:
        base = os.path.splitext(os.path.basename(args.asc_file))[0]
        test_dir = os.path.join(os.path.dirname(args.asc_file), f"test_{base}")
        if os.path.isdir(test_dir):
            shutil.rmtree(test_dir)
            print(f"Cleaned: {test_dir}")
    else:
        generate_register_test(args.asc_file)
