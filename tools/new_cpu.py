#!/usr/bin/env python3
"""Scaffold a new CPU project under cpus/.

Usage:
    python tools/new_cpu.py <name> [--components comp1,comp2,...]

Examples:
    python tools/new_cpu.py 8008 --components alu,registers,decoder,control,io
    python tools/new_cpu.py 4004   (uses default 4004-style components)

Creates:
    cpus/<name>/
        config.py               — template with JFET model, rails, gate R values
        <component>/__init__.py — stub for each component
"""

import argparse
import os
import sys

DEFAULT_COMPONENTS = [
    "alu",
    "instruction_register",
    "micro_instructions",
    "pins",
    "program_counter",
    "scratchpad",
    "stack",
    "step_counter",
]

CONFIG_TEMPLATE = '''"""{name} CPU design configuration.

Hand-designed gate parameters for the SiC JFET implementation.
Modify R values here to tune individual gate types.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import NChannelJFET, JFETCapacitance

# -- JFET device model (DR NJF from SPICE .model card) --
JFET_MODEL = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)

# -- Junction capacitances --
CAPS = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

# -- Supply rails --
V_POS = 24.0
V_NEG = -20.0

# -- Operating temperature (C) --
TEMP_C = 27.0

# -- Target clock frequency (Hz) --
F_TARGET = 100e3

# -- Hand-designed resistor values per gate type (ohms) --
GATES = {{
    "INV":   {{"r1": 50e3, "r2": 1e3, "r3": 4.5e3}},
    "NAND2": {{"r1": 50e3, "r2": 1e3, "r3": 4.5e3}},
    "NOR2":  {{"r1": 50e3, "r2": 1e3, "r3": 4.5e3}},
}}

# -- Sub-components --
# Each entry: (prefix, asc_filename)
# Prefix makes component names unique when building the combined schematic.
# All .asc files live at the top level of this CPU folder.
COMPONENTS = {components}
'''


def create_cpu(name: str, components: list, base_dir: str = None):
    if base_dir is None:
        base_dir = os.path.join(os.path.dirname(__file__), "..", "cpus")

    cpu_dir = os.path.join(base_dir, name)

    if os.path.exists(cpu_dir):
        print(f"ERROR: {cpu_dir} already exists. Remove it first or choose a different name.")
        sys.exit(1)

    os.makedirs(cpu_dir)
    print(f"Created {cpu_dir}/")

    # Write __init__.py
    with open(os.path.join(cpu_dir, "__init__.py"), "w") as f:
        f.write(f'"""{name} CPU — SiC JFET implementation."""\n')

    # Write config.py — generate COMPONENTS as list of (prefix, asc_file)
    comp_tuples = []
    for comp in components:
        parts = comp.split("_")
        if len(parts) == 1:
            prefix = comp.upper()[:4] + "_"
        else:
            prefix = "".join(p[0].upper() for p in parts) + "_"
        comp_tuples.append(f'    ("{prefix}",{" " * max(1, 10 - len(prefix))}"{comp}.asc"),')

    comp_str = "[\n" + "\n".join(comp_tuples) + "\n]"
    config_content = CONFIG_TEMPLATE.format(name=name, components=comp_str)
    with open(os.path.join(cpu_dir, "config.py"), "w") as f:
        f.write(config_content)
    print(f"  config.py")

    print(f"\nDone! Edit cpus/{name}/config.py to set your gate parameters.")


def main():
    parser = argparse.ArgumentParser(description="Scaffold a new CPU project")
    parser.add_argument("name", help="CPU name (e.g. '8008', 'custom_v1')")
    parser.add_argument(
        "--components", "-c",
        default=None,
        help="Comma-separated component names (default: 4004-style components)"
    )

    args = parser.parse_args()
    components = args.components.split(",") if args.components else DEFAULT_COMPONENTS
    create_cpu(args.name, components)


if __name__ == "__main__":
    main()
