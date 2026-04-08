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


def resolve_composable(spec: dict, cpu_dir: str, prefix: str, profile: dict) -> tuple:
    """Resolve a composable component, renaming each sub-file independently.

    Sub-files often have overlapping component names (e.g. every Pair has J_A1).
    We rename per-file with sub-prefixes to avoid collisions:
        Scratch_Ctrl_J1, Scratch_B1_J1, Scratch_P1_J1, Scratch_P2_J1, ...

    Returns:
        (renamed_lines, rename_map, included_filenames)
        Empty if nothing is needed.
    """
    folder = os.path.join(cpu_dir, spec["folder"])
    key = spec["key"]
    parts_map = spec.get("parts", {})

    # Determine which part indices are needed
    if profile and key in profile:
        needed = profile[key]
        if isinstance(needed, (list, set)):
            needed_set = set(needed)
        else:
            needed_set = set(range(needed))
    else:
        # No profile = full build
        needed_set = set(parts_map.keys())

    if not needed_set:
        return [], {}, []

    all_lines = []
    all_new_names = []  # flat list of all new names (no key collision)
    included = []

    def _include(filename, sub_prefix):
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            content = read_asc(path)
            full_prefix = f"{prefix}{sub_prefix}"
            renamed, rmap = rename_components(content, full_prefix)
            all_lines.extend(renamed)
            all_new_names.extend(rmap.values())
            included.append(filename)
        else:
            print(f"    WARNING: {path} not found, skipping")

    # 1. Common parts
    for i, common_file in enumerate(spec.get("common", [])):
        # Generate short sub-prefix from filename
        stem = os.path.splitext(common_file)[0][:4]
        _include(common_file, f"{stem}_")

    # 2. Group dependencies
    for group_file, group_indices in spec.get("groups", {}).items():
        if needed_set & set(group_indices):
            stem = os.path.splitext(group_file)[0]
            _include(group_file, f"{stem}_")

    # 3. Individual parts
    for idx in sorted(needed_set):
        if idx in parts_map:
            _include(parts_map[idx], f"P{idx}_")

    return all_lines, all_new_names, included


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
            # Composable component — assembled from sub-parts, renamed per-file
            renamed, rename_map, included = resolve_composable(
                source, cpu_dir, prefix, profile)

            if not renamed:
                folder = source["folder"]
                print(f"  SKIP  {folder}/ (no parts needed or found)")
                continue

            label = f"{source['folder']}/ ({len(included)} parts: {', '.join(included)})"
        else:
            print(f"  ERROR: unknown source type for {prefix}: {type(source)}")
            continue

        # Get list of new names (for fixed: from dict values, for composable: already a list)
        if isinstance(source, dict):
            new_name_list = rename_map  # already a list from resolve_composable
        else:
            new_name_list = list(rename_map.values())

        new_names = set(new_name_list)
        dupes = new_names & all_names
        if dupes:
            print(f"  WARNING: duplicates after renaming: {dupes}")
        all_names |= new_names
        total += len(new_name_list)

        with open(output_path, "a") as out:
            out.writelines(renamed)

        # Stats — extract component type (J, R, V, etc.) from renamed values
        counts = {}
        for new_name in new_name_list:
            import re as _re
            m = _re.search(r'([A-Z])\d+$', new_name)
            ctype = m.group(1) if m else '?'
            counts[ctype] = counts.get(ctype, 0) + 1
        count_str = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))

        print(f"  {prefix:<10s} {label:<55s} {len(new_name_list):>5d} ({count_str})")

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
