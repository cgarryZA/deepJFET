#!/usr/bin/env python3
"""Build a combined .asc schematic from a CPU project's sub-circuit files.

Reads the project's config.py COMPONENTS list which can contain:
  - Fixed components: ("ALU_", "alu.asc")  — always included as-is
  - Composable components: ("Scratch_", {"folder": "scratchpad", ...})
    Assembled from sub-parts based on a resource profile.

Usage:
    python tools/build_cpu.py 4004                          # full build
    python tools/build_cpu.py 4004 --program rom.bin        # minimal build
    python tools/build_cpu.py 4004 --profile profile.json   # from saved profile
"""

import argparse
import importlib.util
import json
import os
import re
import sys

PROJECT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CPUS_DIR = os.path.join(PROJECT_ROOT, "cpus")

INSTNAME_RE = re.compile(r"^(SYMATTR InstName )(.+)$")
HEADER_LINES = 2


def load_config(cpu_name: str):
    """Load config module from a CPU's config.py."""
    cpu_dir = os.path.join(CPUS_DIR, cpu_name)
    config_path = os.path.join(cpu_dir, "config.py")
    if not os.path.isfile(config_path):
        print(f"ERROR: {config_path} not found.")
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("cpu_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return cpu_dir, mod


def rename_components(lines: list, prefix: str) -> tuple:
    """Rename all SYMATTR InstName entries with prefix + sequential numbering."""
    counters = {}
    rename_map = {}

    for line in lines:
        m = INSTNAME_RE.match(line.rstrip())
        if m:
            old_name = m.group(2)
            if old_name.startswith("J"):
                comp_type = "J"
            elif old_name.startswith("R"):
                comp_type = "R"
            elif old_name.startswith("V"):
                comp_type = "V"
            elif old_name.startswith("C"):
                comp_type = "C"
            elif old_name.startswith("L"):
                comp_type = "L"
            elif old_name.startswith("X"):
                comp_type = "X"
            else:
                comp_type = old_name[0]

            if comp_type not in counters:
                counters[comp_type] = 1
            num = counters[comp_type]
            counters[comp_type] += 1
            rename_map[old_name] = f"{prefix}{comp_type}{num}"

    renamed = []
    for line in lines:
        m = INSTNAME_RE.match(line.rstrip())
        if m:
            renamed.append(f"SYMATTR InstName {rename_map[m.group(2)]}\n")
        else:
            renamed.append(line)

    return renamed, rename_map


def read_asc(filepath: str) -> list:
    """Read an .asc file, strip the 2-line header, return content lines."""
    with open(filepath, "r") as f:
        lines = f.readlines()
    return lines[HEADER_LINES:]


def resolve_composable(spec: dict, cpu_dir: str, profile: dict) -> list:
    """Resolve a composable component into a list of .asc content lines.

    Args:
        spec: the dict from config (folder, common, parts)
        cpu_dir: path to the CPU project folder
        profile: resource profile dict (or None for full build)

    Returns:
        Combined content lines from all included sub-parts.
    """
    folder = os.path.join(cpu_dir, spec["folder"])
    all_lines = []
    included = []

    # Always include common parts
    for common_file in spec.get("common", []):
        path = os.path.join(folder, common_file)
        if os.path.isfile(path):
            all_lines.extend(read_asc(path))
            included.append(common_file)

    # Determine which parts to include
    parts_spec = spec["parts"]
    key = parts_spec["key"]
    files_map = parts_spec["files"]

    if profile and key in profile:
        # Profile tells us exactly which indices are needed
        needed = profile[key]
        if isinstance(needed, (list, set)):
            needed_set = set(needed)
        else:
            # If it's a single int, include all up to that value
            needed_set = set(range(needed))
    else:
        # No profile = full build, include everything
        needed_set = set(files_map.keys())

    for idx in sorted(needed_set):
        if idx in files_map:
            part_file = files_map[idx]
            path = os.path.join(folder, part_file)
            if os.path.isfile(path):
                all_lines.extend(read_asc(path))
                included.append(part_file)
            else:
                print(f"    WARNING: {path} not found, skipping")

    return all_lines, included


def build(cpu_name: str, output_name: str = None,
          program_path: str = None, profile_path: str = None):
    """Build combined .asc for a CPU project."""
    cpu_dir, config = load_config(cpu_name)
    components = config.COMPONENTS

    # Load or generate resource profile
    profile = None
    if profile_path:
        with open(profile_path, "r") as f:
            profile = json.load(f)
        print(f"Loaded profile from {profile_path}\n")
    elif program_path:
        sys.path.insert(0, os.path.join(PROJECT_ROOT, "tools"))
        from analyze_program import analyze, load_binary, load_ihex
        if program_path.endswith(".hex"):
            data = load_ihex(program_path)
        else:
            data = load_binary(program_path)
        result = analyze(data)
        profile = result.to_dict()
        print(f"Analyzed {len(data)} bytes:\n{result.summary()}\n")

    if output_name is None:
        output_name = f"{cpu_name}.asc"
    output_path = os.path.join(cpu_dir, output_name)

    all_names = set()
    total = 0

    with open(output_path, "w") as out:
        out.write("Version 4\n")
        out.write("SHEET 1 109444 42780\n")

    print(f"Building {output_path}...\n")

    for prefix, source in components:
        if isinstance(source, str):
            # Fixed component — single .asc file
            asc_path = os.path.join(cpu_dir, source)
            if not os.path.isfile(asc_path):
                print(f"  SKIP  {source} (not found)")
                continue

            content = read_asc(asc_path)
            renamed, rename_map = rename_components(content, prefix)
            label = source

        elif isinstance(source, dict):
            # Composable component — assembled from sub-parts
            content, included = resolve_composable(source, cpu_dir, profile)

            if not content:
                folder = source["folder"]
                if not included:
                    print(f"  SKIP  {folder}/ (no parts needed or found)")
                continue

            renamed, rename_map = rename_components(content, prefix)
            label = f"{source['folder']}/ ({len(included)} parts: {', '.join(included)})"
        else:
            print(f"  ERROR: unknown source type for {prefix}: {type(source)}")
            continue

        # Check for duplicates
        new_names = set(rename_map.values())
        dupes = new_names & all_names
        if dupes:
            print(f"  WARNING: duplicates after renaming: {dupes}")
        all_names |= new_names
        total += len(rename_map)

        with open(output_path, "a") as out:
            out.writelines(renamed)

        # Stats
        counts = {}
        for new_name in rename_map.values():
            ctype = new_name[len(prefix)]
            counts[ctype] = counts.get(ctype, 0) + 1
        count_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

        print(f"  {prefix:<10s} {label:<55s} {len(rename_map):>5d} ({count_str})")

    print(f"\n  Total: {total} components, {len(all_names)} unique names")
    if total != len(all_names):
        print(f"  WARNING: {total - len(all_names)} duplicates!")
    else:
        print("  No duplicates.")
    print(f"\n  Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Build combined .asc from CPU project")
    parser.add_argument("cpu", help="CPU project name (folder under cpus/)")
    parser.add_argument("--output", "-o", default=None,
                        help="Output filename (default: <cpu>.asc)")
    parser.add_argument("--program", "-p", default=None,
                        help="Program binary to analyze for minimal build")
    parser.add_argument("--profile", default=None,
                        help="JSON resource profile (from analyze_program.py --output)")

    args = parser.parse_args()
    build(args.cpu, args.output, args.program, args.profile)


if __name__ == "__main__":
    main()
