#!/usr/bin/env python3
"""Build the deepJFET manuscript variants with up-to-date data.

Single entry point for rebuilding the paper. By default this:

  1. Regenerates everything under paper/data/ (transistor count today;
     timing/sensitivity/trace-diff in later weeks) by invoking
     paper/regen_data.py.
  2. Runs pdflatex -> biber -> pdflatex -> pdflatex on Main.tex
     (and/or Main_JXCDC.tex, depending on --variant).

Two variants share `preamble.tex` and `body.tex`; the \\ifjxcdc
boolean in each variant's master file selects between full content
and compressed JXCDC content. Edit body.tex once; both variants pick
up the change on the next build.

Output layout:
  paper/Main.pdf            (full ~14 page journal version)
  paper/jxcdc/Main_JXCDC.pdf (compressed 4-8 page JXCDC variant)

The macros emitted under paper/data/ are \\input{} by both variants,
so prose numbers (\\TotalJFETCount etc.) and the transistor-count
table all refresh automatically -- you do not have to touch any .tex
file when the CPU changes.

Usage:
    python paper/build.py                       # both variants
    python paper/build.py --variant master      # full version only
    python paper/build.py --variant jxcdc       # JXCDC version only
    python paper/build.py --skip-data           # pdflatex only, reuse paper/data/
    python paper/build.py --clean               # wipe aux files first
    python paper/build.py --quiet               # less stdout from pdflatex
    python paper/build.py --open                # open the PDF(s) after build
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parent
REPO_ROOT = PAPER_DIR.parent

AUX_SUFFIXES = (
    ".aux", ".bbl", ".bcf", ".blg", ".log", ".out",
    ".run.xml", ".toc", ".lof", ".lot", ".synctex.gz",
)

# Variant table.
#   tex_basename : root of the .tex file (no extension)
#   out_subdir   : where pdflatex writes its outputs (relative to PAPER_DIR);
#                  '' means in PAPER_DIR itself
VARIANTS = {
    "master": {
        "tex_basename": "Main",
        "out_subdir":   "",
        "label":        "master (full ~14pp journal version)",
    },
    "jxcdc": {
        "tex_basename": "Main_JXCDC",
        "out_subdir":   "jxcdc",
        "label":        "JXCDC (compressed 4-8pp variant)",
    },
}


def run(label, cmd, *, cwd, quiet):
    if not quiet:
        print(f"\n=== {label} ===")
        print("$ " + " ".join(str(c) for c in cmd))
    stdout = subprocess.DEVNULL if quiet else None
    result = subprocess.run(cmd, cwd=cwd, stdout=stdout)
    if result.returncode != 0:
        print(f"!! {label} failed (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def clean_aux():
    removed = 0
    search_dirs = [PAPER_DIR] + [
        PAPER_DIR / v["out_subdir"] for v in VARIANTS.values() if v["out_subdir"]
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for suffix in AUX_SUFFIXES:
            for path in d.glob(f"*{suffix}"):
                path.unlink()
                removed += 1
    print(f"Cleaned {removed} aux file(s).")


def open_pdf(path: Path):
    if sys.platform.startswith("win"):
        os.startfile(path)  # noqa: SIM115
    elif sys.platform == "darwin":
        subprocess.run(["open", path])
    else:
        subprocess.run(["xdg-open", path])


def check_tool(name):
    if shutil.which(name) is None:
        sys.exit(f"ERROR: '{name}' not found on PATH. "
                 f"Install TeX Live or MiKTeX and rerun.")


def build_variant(name: str, *, quiet: bool, open_after: bool) -> Path:
    """Run pdflatex -> biber -> pdflatex -> pdflatex for one variant.

    Returns the resulting PDF path on success, exits non-zero on failure.
    """
    v = VARIANTS[name]
    tex_base = v["tex_basename"]
    tex_file = f"{tex_base}.tex"
    out_subdir = v["out_subdir"]
    out_dir = PAPER_DIR / out_subdir if out_subdir else PAPER_DIR

    out_dir.mkdir(parents=True, exist_ok=True)

    # -output-directory tells pdflatex to write .aux .log .pdf into out_dir
    # while still reading source files relative to PAPER_DIR (which is cwd).
    out_flag = [f"-output-directory={out_subdir}"] if out_subdir else []

    pdflatex_cmd = [
        "pdflatex", "-interaction=nonstopmode", "-halt-on-error",
        *out_flag, tex_file,
    ]
    # biber wants the .bcf next to its working dir. With -output-directory,
    # the .bcf lands in out_dir, so we point biber at it explicitly.
    biber_target = str(out_dir / tex_base) if out_subdir else tex_base
    biber_cmd = ["biber", biber_target]

    steps = [
        (f"[{name}] pdflatex pass 1 (resolve labels)", pdflatex_cmd),
        (f"[{name}] biber (bibliography)",             biber_cmd),
        (f"[{name}] pdflatex pass 2 (insert refs)",    pdflatex_cmd),
        (f"[{name}] pdflatex pass 3 (finalise)",       pdflatex_cmd),
    ]

    for label, cmd in steps:
        if not run(label, cmd, cwd=PAPER_DIR, quiet=quiet):
            print(f"\nBuild aborted at: {label}", file=sys.stderr)
            tail_log(out_dir / f"{tex_base}.log")
            sys.exit(1)

    pdf_path = out_dir / f"{tex_base}.pdf"
    if not pdf_path.is_file():
        sys.exit(f"!! Build finished but {pdf_path.name} not produced")
    size_mb = pdf_path.stat().st_size / 1_048_576
    print(f"OK [{name}] -- {pdf_path.relative_to(REPO_ROOT)} ({size_mb:.2f} MB)")

    report_warnings(out_dir / f"{tex_base}.log", variant_name=name)

    if open_after:
        open_pdf(pdf_path)

    return pdf_path


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variant", choices=list(VARIANTS) + ["all"], default="all",
                   help="which variant to build (default: all)")
    p.add_argument("--skip-data", action="store_true",
                   help="don't regenerate paper/data/, just compile LaTeX")
    p.add_argument("--clean", action="store_true",
                   help="delete aux files (.aux .bbl .log ...) before building")
    p.add_argument("--quiet", action="store_true",
                   help="suppress pdflatex/biber stdout")
    p.add_argument("--open", action="store_true",
                   help="open the resulting PDF(s) when the build succeeds")
    args = p.parse_args()

    check_tool("pdflatex")
    check_tool("biber")

    if args.clean:
        clean_aux()

    # Step 1: regenerate data artefacts the paper depends on.
    if not args.skip_data:
        regen_script = PAPER_DIR / "regen_data.py"
        ok = run("Regenerating paper/data/",
                 [sys.executable, str(regen_script)],
                 cwd=REPO_ROOT, quiet=args.quiet)
        if not ok:
            sys.exit("Data regeneration failed; not running pdflatex.")

    # Step 2: pdflatex pipeline, per variant.
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    for name in variants:
        print(f"\n=== Building {name}: {VARIANTS[name]['label']} ===")
        build_variant(name, quiet=args.quiet, open_after=args.open)


def tail_log(log_path: Path):
    if not log_path.is_file():
        return
    print(f"\n--- last 30 lines of {log_path.name} ---", file=sys.stderr)
    text = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for line in text[-30:]:
        print(line, file=sys.stderr)


# Filter out warnings that are pre-existing and noise (caption-package
# class-unknown, gensymb \perthousand, float-too-large, font shape size
# substitutions). Everything else is genuinely worth surfacing.
NOISY = (
    "obsolete",
    "epstopdf",
    "Unknown document class",
    "Not defining \\perthousand",
    "Not defining \\micro",
    "Float too large",
    "`h' float specifier",
    "Size substitutions",
    "Font shape",
)


def report_warnings(log_path: Path, *, variant_name: str = ""):
    if not log_path.is_file():
        return
    interesting = []
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if any(k in line for k in ("Warning", "undefined", "multiply defined", "??")):
            if not any(n in line for n in NOISY):
                interesting.append(line.strip())
    if interesting:
        tag = f" [{variant_name}]" if variant_name else ""
        print(f"\nLaTeX flagged the following{tag} (review):")
        for line in interesting[:20]:
            print(f"  {line}")
        if len(interesting) > 20:
            print(f"  ... and {len(interesting) - 20} more in {log_path.name}")


if __name__ == "__main__":
    main()
