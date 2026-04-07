#!/usr/bin/env python3
"""Open an LTSpice schematic for a CPU component.

Usage:
    python tools/open_ltspice.py <component> [--cpu 4004]
    python tools/open_ltspice.py alu
    python tools/open_ltspice.py program_counter --cpu 4004
    python tools/open_ltspice.py path/to/file.asc

Searches for .asc files in the CPU component folder and opens them in LTSpice.
If no .asc file exists yet, tells you so you can create one.
"""

import argparse
import glob
import os
import subprocess
import sys

LTSPICE_EXE = r"C:\Users\z00503ku\AppData\Local\Programs\ADI\LTspice\LTspice.exe"
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
CPUS_DIR = os.path.join(PROJECT_ROOT, "cpus")


def find_asc_files(component: str, cpu: str = None) -> list:
    """Find .asc files matching a component name.

    Search order:
    1. Exact path (if component is a file path)
    2. cpus/<cpu>/<component>/*.asc
    3. cpus/*/<component>/*.asc (all CPUs)
    4. Anywhere under cpus/ matching *<component>*.asc
    """
    # Direct path
    if os.path.isfile(component):
        return [os.path.abspath(component)]
    if os.path.isfile(component + ".asc"):
        return [os.path.abspath(component + ".asc")]

    results = []

    if cpu:
        # Search specific CPU folder (flat — .asc files at top level)
        pattern = os.path.join(CPUS_DIR, cpu, f"*{component}*.asc")
        results = glob.glob(pattern)
    else:
        # Search all CPU folders
        pattern = os.path.join(CPUS_DIR, "*", f"*{component}*.asc")
        results = glob.glob(pattern)

    # Also check project root and ltspice/output
    if not results:
        for search_dir in [PROJECT_ROOT, os.path.join(PROJECT_ROOT, "ltspice", "output")]:
            pattern = os.path.join(search_dir, f"*{component}*.asc")
            results.extend(glob.glob(pattern))

    return [os.path.abspath(r) for r in results]


def get_component_dir(component: str, cpu: str = None) -> str:
    """Get the directory where a component's .asc file should live."""
    if cpu:
        return os.path.abspath(os.path.join(CPUS_DIR, cpu))
    # Find first CPU folder
    for entry in os.listdir(CPUS_DIR):
        cpu_dir = os.path.join(CPUS_DIR, entry)
        if os.path.isdir(cpu_dir):
            return os.path.abspath(cpu_dir)
    return None


def open_in_ltspice(filepath: str):
    """Open a file in LTSpice."""
    if not os.path.isfile(LTSPICE_EXE):
        print(f"ERROR: LTSpice not found at {LTSPICE_EXE}")
        print("Update LTSPICE_EXE in tools/open_ltspice.py")
        sys.exit(1)

    filepath = os.path.abspath(filepath)
    print(f"Opening in LTSpice: {filepath}")
    subprocess.Popen([LTSPICE_EXE, filepath])


def main():
    parser = argparse.ArgumentParser(description="Open a component schematic in LTSpice")
    parser.add_argument("component", help="Component name (e.g. 'alu', 'program_counter') or path to .asc file")
    parser.add_argument("--cpu", "-c", default=None, help="CPU name (default: search all)")
    parser.add_argument("--list", "-l", action="store_true", help="List all .asc files found, don't open")

    args = parser.parse_args()

    files = find_asc_files(args.component, args.cpu)

    if args.list:
        if files:
            print("Found .asc files:")
            for f in files:
                print(f"  {f}")
        else:
            print("No .asc files found.")
        return

    if not files:
        comp_dir = get_component_dir(args.component, args.cpu)
        print(f"No .asc file found for '{args.component}'.")
        if comp_dir and os.path.isdir(comp_dir):
            expected = os.path.join(comp_dir, f"{args.component}.asc")
            print(f"Expected location: {expected}")
            print(f"Create the schematic there, or use gen_ltspice.py to generate a .net first.")
        else:
            print("Check the component name or specify --cpu.")
        sys.exit(1)

    if len(files) == 1:
        open_in_ltspice(files[0])
    else:
        print("Multiple .asc files found:")
        for i, f in enumerate(files):
            print(f"  [{i}] {f}")
        choice = input("Which one? [0]: ").strip()
        idx = int(choice) if choice else 0
        open_in_ltspice(files[idx])


if __name__ == "__main__":
    main()
