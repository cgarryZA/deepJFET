#!/usr/bin/env python3
"""Build and test a combined 4-bit ALU (Acc + Temp + CLA Adder)."""

import os
import re
import shutil
import sys

V_HIGH = -0.8
V_LOW = -3.6
RISE = 100e-9
P = 30e-6
CLK_W = 5e-6


def v(bit):
    return V_HIGH if bit else V_LOW


def bits(val, n=4):
    return [(val >> i) & 1 for i in range(n)]


def write_pwl(path, phases_data, total_time):
    """Write a PWL file from (time, value) phase data."""
    pts = []
    prev = None
    for t, bv in phases_data:
        voltage = v(bv)
        if prev is not None and voltage != prev:
            pts.append((t - RISE, prev))
        pts.append((t, voltage))
        prev = voltage
    pts.append((total_time, prev))

    with open(path, "w") as f:
        last_t = -1
        for pt, pv in pts:
            if pt <= last_t:
                pt = last_t + 10e-9
            f.write(f"{pt:.9e} {pv:.4f}\n")
            last_t = pt


def generate_test():
    test_dir = "cpus/tileable/generated/test_alu_combined"
    shutil.rmtree(test_dir, ignore_errors=True)
    os.makedirs(test_dir)

    n_bits = 4

    # Test phases: (bus_val, acc_load, temp_load, clk, inv_temp, carry_in, desc)
    phases = [
        # Load Acc = 5
        (5, 1, 0, 0, 0, 0, "bus=5, acc_load on"),
        (5, 1, 0, 1, 0, 0, "CLK: Acc=5"),
        # Load Temp = 3
        (3, 0, 1, 0, 0, 0, "bus=3, temp_load on"),
        (3, 0, 1, 1, 0, 0, "CLK: Temp=3"),
        # Check: 5+3=8
        (0, 0, 0, 0, 0, 0, "check: 5+3=8, CF=0"),
        # Load Acc=8, Temp=7
        (8, 1, 0, 0, 0, 0, "bus=8"),
        (8, 1, 0, 1, 0, 0, "CLK: Acc=8"),
        (7, 0, 1, 0, 0, 0, "bus=7"),
        (7, 0, 1, 1, 0, 0, "CLK: Temp=7"),
        # Check: 8+7=15
        (0, 0, 0, 0, 0, 0, "check: 8+7=15, CF=0"),
        # Load Acc=15, Temp=1
        (15, 1, 0, 0, 0, 0, "bus=15"),
        (15, 1, 0, 1, 0, 0, "CLK: Acc=15"),
        (1, 0, 1, 0, 0, 0, "bus=1"),
        (1, 0, 1, 1, 0, 0, "CLK: Temp=1"),
        # Check: 15+1=16 -> Sum=0, CF=1
        (0, 0, 0, 0, 0, 0, "check: 15+1=0, CF=1"),
        # SUB: Acc=9, Temp=4
        (9, 1, 0, 0, 0, 0, "bus=9"),
        (9, 1, 0, 1, 0, 0, "CLK: Acc=9"),
        (4, 0, 1, 0, 0, 0, "bus=4"),
        (4, 0, 1, 1, 0, 0, "CLK: Temp=4"),
        # Invert temp + carry=1, clock to latch inversion
        (0, 0, 0, 0, 1, 1, "setup: inv temp, carry=1"),
        (0, 0, 0, 1, 1, 1, "CLK: latch inversion"),
        # Check: 9 + ~4 + 1 = 9+11+1=21 -> Sum=5, CF=1
        (0, 0, 0, 0, 1, 1, "check: 9-4=5, CF=1"),
    ]

    total_time = (len(phases) + 2) * P

    # Generate Bus PWLs
    for bit in range(n_bits):
        data = [(i * P, bits(ph[0])[bit]) for i, ph in enumerate(phases)]
        write_pwl(os.path.join(test_dir, f"Bus_{bit}.pwl"), data, total_time)

    # Control signals
    for sig_idx, name in [(1, "Acc_Load"), (2, "Temp_Load")]:
        data = [(i * P, ph[sig_idx]) for i, ph in enumerate(phases)]
        write_pwl(os.path.join(test_dir, f"{name}.pwl"), data, total_time)

    # Temp_INV
    data = [(i * P, ph[4]) for i, ph in enumerate(phases)]
    write_pwl(os.path.join(test_dir, "Temp_INV.pwl"), data, total_time)

    # CarryInputIn
    data = [(i * P, ph[5]) for i, ph in enumerate(phases)]
    write_pwl(os.path.join(test_dir, "CarryInputIn.pwl"), data, total_time)

    # CLK
    clk_pts = [(0, V_LOW)]
    for i, ph in enumerate(phases):
        if ph[3]:  # clk=1
            t = i * P
            t_rise = t + P * 0.3
            t_fall = t_rise + CLK_W
            clk_pts.append((t_rise - RISE, V_LOW))
            clk_pts.append((t_rise, V_HIGH))
            clk_pts.append((t_fall, V_HIGH))
            clk_pts.append((t_fall + RISE, V_LOW))
    clk_pts.append((total_time, V_LOW))
    with open(os.path.join(test_dir, "CLK.pwl"), "w") as f:
        last_t = -1
        for pt, pv in clk_pts:
            if pt <= last_t:
                pt = last_t + 10e-9
            f.write(f"{pt:.9e} {pv:.4f}\n")
            last_t = pt

    # Expected results
    expected = [
        (4, 8, 0, "5+3=8, CF=0"),
        (9, 15, 0, "8+7=15, CF=0"),
        (14, 0, 1, "15+1=0, CF=1"),
        (22, 5, 1, "9-4=5, CF=1"),
    ]
    with open(os.path.join(test_dir, "expected.txt"), "w") as f:
        for pidx, exp_sum, exp_cf, desc in expected:
            t_check = (pidx + 0.8) * P
            f.write(f"t={t_check*1e6:6.0f}us  Sum={exp_sum:2d}(0b{exp_sum:04b})  CF={exp_cf}  ({desc})\n")

    # Combine Acc + Temp + Adder into single .asc
    test_asc = os.path.join(test_dir, "alu_combined_test.asc")

    with open(test_asc, "w") as f:
        f.write("Version 4.1\n")
        f.write("SHEET 1 109444 42780\n")

        def append_asc(src_path, y_offset=0):
            with open(src_path) as src:
                for line in src.readlines()[2:]:
                    if y_offset == 0:
                        f.write(line)
                    elif line.startswith("WIRE "):
                        p = line.split()
                        f.write(f"WIRE {p[1]} {int(p[2])+y_offset} {p[3]} {int(p[4])+y_offset}\n")
                    elif line.startswith("FLAG "):
                        p = line.strip().split(None, 3)
                        name = p[3] if len(p) > 3 else ""
                        f.write(f"FLAG {p[1]} {int(p[2])+y_offset} {name}\n")
                    elif line.startswith("SYMBOL "):
                        p = line.split()
                        rest = " ".join(p[4:])
                        f.write(f"SYMBOL {p[1]} {p[2]} {int(p[3])+y_offset} {rest}\n")
                    else:
                        f.write(line)

        append_asc("cpus/tileable/generated/alu_acc_4bit.asc", 0)
        append_asc("cpus/tileable/generated/alu_temp_4bit.asc", 2000)
        append_asc("cpus/tileable/generated/alu_adder_4bit.asc", 4000)

        # Test voltage sources
        x = 80000
        sp = 160
        y = 8000
        sources = [
            *[(f"Bus_{i}", f"Bus_{i}.pwl") for i in range(n_bits)],
            ("Acc_Load", "Acc_Load.pwl"),
            ("Temp_Load", "Temp_Load.pwl"),
            ("CLK", "CLK.pwl"),
            ("Temp_INV", "Temp_INV.pwl"),
            ("CarryInputIn", "CarryInputIn.pwl"),
        ]
        for j, (name, pwl) in enumerate(sources):
            sx = x + j * sp
            f.write(f"FLAG {sx} {y+16} {name}\n")
            f.write(f"FLAG {sx} {y+96} 0\n")
            f.write(f"SYMBOL voltage {sx} {y} R0\n")
            f.write(f"WINDOW 0 56 32 Invisible 2\n")
            f.write(f"WINDOW 3 56 72 Invisible 2\n")
            f.write(f"SYMATTR InstName V_t{j}\n")
            f.write(f"SYMATTR Value PWL file={pwl}\n")

        vy = y + 256
        f.write(f"FLAG {x} {vy+16} VDD\n")
        f.write(f"FLAG {x} {vy+96} 0\n")
        f.write(f"SYMBOL voltage {x} {vy} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VDD\n")
        f.write(f"SYMATTR Value 24\n")
        f.write(f"FLAG {x+sp} {vy+16} VSS\n")
        f.write(f"FLAG {x+sp} {vy+96} 0\n")
        f.write(f"SYMBOL voltage {x+sp} {vy} R0\n")
        f.write(f"WINDOW 0 56 32 Invisible 2\n")
        f.write(f"WINDOW 3 56 72 Invisible 2\n")
        f.write(f"SYMATTR InstName V_VSS\n")
        f.write(f"SYMATTR Value -20\n")

        tran_us = int(total_time * 1e6) + 10
        f.write(f"TEXT {x} {vy+256} Left 2 !.tran {tran_us}u\n")
        f.write(f"TEXT {x} {vy+320} Left 2 !.model DR NJF(Beta=0.135m Betatce=-0.5 "
                f"Vto=-3.45 Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f "
                f"Nr=4 Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
                f"Vk=123.7 M=407m Pb=1 Kf=37860f Af=1 Mfg=Linear_Systems)\n")

    # Renumber InstNames
    with open(test_asc) as f:
        lines = f.readlines()

    counters = {}
    new_lines = []
    for line in lines:
        if line.startswith("SYMATTR InstName "):
            old = line.split(" ", 2)[2].strip()
            m = re.match(r"([A-Z_!]+?)(\d+)$", old)
            if m:
                pfx = m.group(1)
                counters[pfx] = counters.get(pfx, 0) + 1
                new_lines.append(f"SYMATTR InstName {pfx}{counters[pfx]}\n")
            else:
                new_lines.append(line)
        else:
            new_lines.append(line)

    with open(test_asc, "w") as f:
        f.writelines(new_lines)

    n_sym = sum(1 for l in new_lines if l.startswith("SYMBOL "))
    print(f"Combined ALU test: {n_sym} symbols")
    print(f"Test dir: {test_dir}")
    print(f"\nExpected results:")
    with open(os.path.join(test_dir, "expected.txt")) as f:
        print(f.read())
    print(f"Open {test_asc} in LTSpice and run.")


if __name__ == "__main__":
    generate_test()
