#!/usr/bin/env python3
"""Build the reproducibility supplement bundle.

Walks an explicit whitelist of source paths under the repo, computes a
SHA-256 for each, writes paper/reproducibility/MANIFEST.csv, and
optionally zips the same file set into
paper/reproducibility/supplement.zip for upload to Zenodo or similar.

The whitelist is deliberately conservative: it includes manuscript
sources, analysis scripts, schematic .asc files, device models, test
programs (.asm + YAML), and the .cir test benches. It explicitly
EXCLUDES large or machine-specific artefacts: LTspice .raw / .log /
.net / Machine.bin / Machine.hex output, __pycache__, the SQLite
optimisation cache, and the entire .git and .claude directories.

This script is safe to run while an LTspice transient is in progress:
no .raw file is ever opened, and the manifest is built from the file
metadata + content of files we explicitly chose to include.

Usage:
    python tools/build_supplement.py                   # MANIFEST + zip
    python tools/build_supplement.py --no-zip          # just MANIFEST
    python tools/build_supplement.py --out path/...    # custom output dir
"""

import argparse
import csv
import hashlib
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper"
REPRO_DIR = PAPER_DIR / "reproducibility"

# Explicit whitelist. Patterns are glob-style relative to REPO_ROOT.
# Order matters for grouping in the MANIFEST; not for correctness.
INCLUDE_GLOBS = [
    # Manuscript sources (shared body + per-variant master files)
    ("manuscript",       "paper/Main.tex"),
    ("manuscript",       "paper/Main_JXCDC.tex"),
    ("manuscript",       "paper/preamble.tex"),
    ("manuscript",       "paper/body.tex"),
    ("manuscript",       "paper/Appendices.tex"),
    ("manuscript",       "paper/AppendixB.tex"),
    ("manuscript",       "paper/AppendixC.tex"),
    ("manuscript",       "paper/bib.bib"),
    ("manuscript",       "paper/bib2.bib"),
    ("manuscript",       "paper/IEEEtran.cls"),
    ("manuscript",       "paper/IEEEtran.bst"),
    ("manuscript",       "paper/supplementary/**/*.tex"),
    ("manuscript",       "paper/BIBLIOGRAPHY_TODO.md"),
    ("manuscript",       "paper/VENUE_TAILORING_TODO.md"),
    ("manuscript",       "paper/LICENSE-CC-BY-4.0"),

    # Auto-generated paper data
    ("paper data",       "paper/data/*.csv"),
    ("paper data",       "paper/data/*.tex"),

    # Figures
    ("figures",          "paper/Images/*.pdf"),
    ("figures",          "paper/Images/*.png"),
    ("figures",          "paper/Images/*.jpg"),
    ("figures",          "paper/Appendices/*.pdf"),

    # Build + reproducibility tooling
    ("build scripts",    "paper/build.py"),
    ("build scripts",    "paper/regen_data.py"),
    ("build scripts",    "paper/reproducibility/README.md"),
    ("build scripts",    "paper/.zenodo.json"),
    ("build scripts",    "paper/CITATION.cff"),

    # Analysis scripts
    ("tools",            "tools/*.py"),

    # CPU schematics and config (4004 only; other CPUs out of scope)
    ("CPU schematics",   "cpus/4004/*.asc"),
    ("CPU schematics",   "cpus/4004/config.py"),
    ("CPU schematics",   "cpus/4004/__init__.py"),
    ("CPU schematics",   "cpus/4004/scratchpad/*.asc"),
    ("CPU schematics",   "cpus/4004/stack/*.asc"),
    ("CPU schematics",   "cpus/4004/traces_instructions.txt"),

    # Test programs (asm + expected state + README); NO Machine.bin/.hex/.raw/PWLs
    ("test programs",    "cpus/4004/programs/Assembler.py"),
    ("test programs",    "cpus/4004/programs/FloatingPoint/FloatingPoint.asm"),
    ("test programs",    "cpus/4004/programs/FloatingPoint/README.md"),
    ("test programs",    "cpus/4004/programs/BitwiseAND/BitwiseAND.asm"),
    ("test programs",    "cpus/4004/programs/BitwiseAND/expected_state.yaml"),
    ("test programs",    "cpus/4004/programs/BitwiseAND/README.md"),
    ("test programs",    "cpus/4004/programs/Load5/Load5.asm"),

    # Primitive gate library
    ("gate library",     "lib/gates/*.asc"),
    ("gate library",     "lib/components/**/*.asc"),

    # Device model + gate solver + simulator engine
    ("device model",     "model/*.py"),
    ("device model",     "blocks/*.py"),
    ("device model",     "simulator/*.py"),
    ("device model",     "transient/*.py"),

    # LTspice corner-sweep test benches
    ("corner sweeps",    "analysis/sweeps/*.cir"),
    ("corner sweeps",    "analysis/sweeps/README.md"),
    ("corner sweeps",    "analysis/sweeps/*.log"),  # included only if user has run them

    # Top-level repo docs
    ("repo docs",        "CLAUDE.md"),
    ("repo docs",        "README.md"),
    ("repo docs",        "STATUS.md"),
    ("repo docs",        "LICENSE"),
    ("repo docs",        "LICENSE.md"),
    ("repo docs",        "LICENSE.txt"),
]


# Map output file -> generator (for the MANIFEST "generator" column).
# Anything not in this map is reported as "static".
GENERATORS = {
    "paper/data/transistor_count.csv":         "tools/count_transistors.py",
    "paper/data/transistor_count.tex":         "tools/count_transistors.py",
    "paper/data/transistor_count_macros.tex":  "tools/count_transistors.py",
    "paper/data/trace_diff_FloatingPoint.csv": "tools/trace_synced.py",
    "paper/data/trace_diff_FloatingPoint.tex": "tools/trace_synced.py",
    "paper/data/trace_diff_FloatingPoint_macros.tex": "tools/trace_synced.py",
    "paper/data/trace_diff_BitwiseAND.csv":    "tools/trace_synced.py",
    "paper/data/trace_diff_BitwiseAND.tex":    "tools/trace_synced.py",
    "paper/data/trace_diff_BitwiseAND_macros.tex": "tools/trace_synced.py",
    "paper/data/freq_sweep.csv":               "tools/analyze_ltspice_log.py",
    "paper/data/freq_sweep.tex":               "tools/analyze_ltspice_log.py",
    "paper/data/freq_sweep_macros.tex":        "tools/analyze_ltspice_log.py",
    "paper/data/sensitivity.csv":              "tools/analyze_ltspice_log.py",
    "paper/data/sensitivity.tex":              "tools/analyze_ltspice_log.py",
    "paper/data/sensitivity_macros.tex":       "tools/analyze_ltspice_log.py",
    "analysis/sweeps/freq_testbench.cir":      "tools/gen_freq_testbench.py",
    "analysis/sweeps/sensitivity_r1.cir":      "tools/gen_sensitivity_testbench.py",
    "analysis/sweeps/sensitivity_r2.cir":      "tools/gen_sensitivity_testbench.py",
    "analysis/sweeps/sensitivity_r3.cir":      "tools/gen_sensitivity_testbench.py",
    "analysis/sweeps/sensitivity_vpos.cir":    "tools/gen_sensitivity_testbench.py",
    "analysis/sweeps/sensitivity_vneg.cir":    "tools/gen_sensitivity_testbench.py",
    "cpus/4004/programs/BitwiseAND/expected_state.yaml":
                                                "tools/derive_expected_state.py",
}


def collect_files() -> list:
    """Resolve the whitelist to a sorted list of (group, repo_relative_path)."""
    seen = set()
    out = []
    for group, pattern in INCLUDE_GLOBS:
        # Path.glob doesn't support "**/" unless we use rglob; handle both.
        if "**" in pattern:
            base, _, sub = pattern.partition("**/")
            base_path = REPO_ROOT / base
            if not base_path.exists():
                continue
            for p in base_path.rglob(sub):
                if p.is_file():
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        out.append((group, rel))
        else:
            # Treat as a normal glob relative to REPO_ROOT.
            for p in REPO_ROOT.glob(pattern):
                if p.is_file():
                    rel = p.relative_to(REPO_ROOT).as_posix()
                    if rel not in seen:
                        seen.add(rel)
                        out.append((group, rel))
    # Stable order: by group then path.
    out.sort(key=lambda gr: (gr[0], gr[1]))
    return out


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=None,
                   help="output directory (default: paper/reproducibility/)")
    p.add_argument("--no-zip", action="store_true",
                   help="write MANIFEST.csv but skip the supplement.zip bundle")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else REPRO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "MANIFEST.csv"
    zip_path = out_dir / "supplement.zip"

    files = collect_files()
    if not files:
        sys.exit("ERROR: no files matched the whitelist; check INCLUDE_GLOBS")

    rows = []
    total_bytes = 0
    for group, rel in files:
        abs_path = REPO_ROOT / rel
        size = abs_path.stat().st_size
        digest = sha256_of(abs_path)
        gen = GENERATORS.get(rel, "static")
        rows.append({
            "group":     group,
            "path":      rel,
            "bytes":     size,
            "sha256":    digest,
            "generator": gen,
        })
        total_bytes += size

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["group", "path", "bytes", "sha256", "generator"])
        w.writeheader()
        w.writerows(rows)

    if not args.quiet:
        print(f"# Manifest: {len(rows)} files, {total_bytes/1e6:.2f} MB total")
        print(f"# Wrote    {manifest_path}")

    if not args.no_zip:
        # Exclude supplement.zip itself from the zip if it already exists
        # (don't pack the previous run's bundle inside this run's bundle).
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            # MANIFEST first.
            z.write(manifest_path, arcname="MANIFEST.csv")
            for row in rows:
                rel = row["path"]
                if rel.startswith("paper/reproducibility/supplement.zip"):
                    continue
                z.write(REPO_ROOT / rel, arcname=rel)

        zip_size = zip_path.stat().st_size
        if not args.quiet:
            print(f"# Bundle:   {zip_path} ({zip_size/1e6:.2f} MB)")
            ratio = zip_size / total_bytes if total_bytes else 0
            print(f"#          compression ratio {ratio:.2f}")


if __name__ == "__main__":
    main()
