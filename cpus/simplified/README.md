# Simplified 4004 (microcode-optimised variant)

This directory holds an experimental variant of `cpus/4004/` with a
**reduced micro-instruction set**. The schematics that differ from the
baseline are:

- `alu.asc` — combinational simplifications
- `micro_instructions.asc` — fewer micro-cycle states
- (PWLs are regenerated; ignore those)

Everything else (scratchpad, stack, controls, PC, step counter, IR, pins)
is unchanged from `cpus/4004/`.

## Status

The simplified microcode passed unit-level checks but **has not been
revalidated against the full FloatingPoint and AdderTest programs**
since the baseline `cpus/4004/` was bug-fixed in May 2026. Treat this
directory as an exploratory branch, not a second working CPU.

## Why both exist

The two trees coexist for an intended before/after JFET-count comparison
in the paper (see `paper/data/transistor_count.csv`). Once the simplified
variant has been revalidated end-to-end on the same programs as the
baseline, the comparison table in the manuscript can quote both numbers.

## If you are reading as a reviewer

The paper's headline numbers come from `cpus/4004/` (the baseline). The
simplified variant is, today, a supplementary curiosity, not a load-bearing
result.
