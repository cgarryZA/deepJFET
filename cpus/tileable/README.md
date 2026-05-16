# Tileable CPU (work in progress)

This directory contains an in-progress RV32I implementation built from
parameterised "tileable" blocks (bit-slices, generated registers,
generated ALU). **It is not part of the published paper.**

It is preserved here because several pieces are working and verified
(register tiling, 4-bit ALU, instruction decoder all pass their bespoke
tests in `cpus/tileable/generated/`) and form the seed of a planned
follow-up paper on a JFET-native RISC ISA.

If you are reading this directory as a reviewer of the SiC JFET 4004
paper: skip it. The paper is built entirely from `cpus/4004/`.

If you want to resume the work:

- `cpus/tileable/generated/` holds the working `.asc` outputs of the
  generators in `tools/legacy/gen_*.py` and `tools/legacy/tile_*.py`.
- `cpus/tileable/config.py` is the JFET model + supply rails — identical
  to `cpus/4004/config.py` so cross-comparison is direct.
- The RV32I instruction decoder (`generated/rv32i_decoder.asc`) and
  instruction register (`generated/rv32i_instruction_register.asc`) are
  the most-recently-modified files; everything else has been stable for
  weeks.
