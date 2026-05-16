# deepJFET — a SiC JFET implementation of the Intel 4004

This repository contains a complete gate-level implementation of the
Intel 4004 microprocessor built from discrete silicon-carbide JFET logic,
simulated in LTspice. The aim is a working CPU on a wide-bandgap
technology that tolerates the temperatures (200–300 °C) where
conventional silicon CMOS fails.

The repository is the artefact backing an in-progress manuscript; see
[`paper/reproducibility/README.md`](paper/reproducibility/README.md) for
the canonical entry point if you are trying to reproduce the figures.

## What's in here

```
cpus/4004/         Working SiC JFET 4004 (LTspice .asc, JFET config,
                   composable scratchpad + stack)
cpus/4004/programs/   FloatingPoint, AdderTest, BitwiseAND, Load5
cpus/simplified/   Microcode-optimised variant (experimental)
cpus/tileable/     RV32I work in progress for a follow-up paper

blocks/            Parameterised Python building blocks (DFF, register,
                   mux, decoder) that compose into gate-level netlists
model/             SiC JFET physics + RC timing model
simulator/         Event-driven gate-level Python simulator
transient/         ODE-based transient solver
optimization/      Isolated noise-margin / R-value sensitivity studies

tools/             Currently-used CLI tools (build, verify, assemble,
                   reproducibility regen). 18 scripts.
tools/legacy/      Archived debugging scripts from the FloatingPoint
                   bring-up. Not part of the reproducibility flow.

paper/             Manuscript (LaTeX), figures, regenerable data tables,
                   reproducibility plan, supplement bundler.
analysis/          Corner-sweep testbenches and result tables.
ltspice/templates/ LTspice symbol/component templates for generators.
lib/gates/         Primitive .asc gate symbols (INV, NAND, NOR, NAND3,
                   NAND4, NOR3) used by hand-drawn schematics.
```

## Quickstart

```
# Assemble a 4004 program, build the flat .asc, open in LTspice
python tools/asm_ide.py                       # GUI editor + build button
# or, headless:
python tools/build_and_run_4004.py AdderTest --cycles 500
```

The IDE writes `cpus/4004/programs/<name>/<name>.asm`, generates the
ROM PWLs, rebuilds `cpus/4004/4004.asc` with the correct `.tran`
duration, and launches LTspice on the result.

After simulation completes, verify instruction-by-instruction:

```
python tools/verify_micro5.py cpus/4004/4004.raw --start-time 2
```

## Reproducing the paper

See [`paper/reproducibility/README.md`](paper/reproducibility/README.md)
for the one-command rebuild, the script-by-script table of headline
results, the supplement-zip workflow, and the Zenodo upload procedure.

## Licence

Code and netlists under MIT (see [`LICENSE`](LICENSE)).
Manuscript and figures under CC-BY-4.0 (see
[`paper/LICENSE-CC-BY-4.0`](paper/LICENSE-CC-BY-4.0)).
SiC JFET model parameters in `cpus/4004/config.py` originate with the
project supervisor; redistribution beyond academic reproduction of this
work requires their permission. See
[`paper/reproducibility/README.md`](paper/reproducibility/README.md#sic-jfet-model-parameters-provenance-and-redistribution).
