# Legacy / archival scripts

Exploratory and one-off debug scripts kept for archival reference. **None
of these are part of the paper-reproducibility flow** — see the table in
`paper/reproducibility/README.md` for the currently-used tool list.

They are retained because they document the debugging path that led to
several schematic-level bug fixes (the SUMCF false-carry, the 5SPMux1 typo,
the CMC redesign, the RAL/RAR rotate-IN of CY) and may be useful for
revisiting individual signals in a captured `.raw` file.

| Script | What it did | Why it's legacy |
|---|---|---|
| `check_bus_contention.py` | Looked for simultaneous high-impedance drivers on shared 4004 busses during FloatingPoint debugging | Was a one-shot diagnostic |
| `check_final_state.py` | Read the last sample of `4004.raw` to verify FP's R12:R13 product | Superseded by `verify_micro5.py` |
| `check_index_decode.py` | Sanity-checked R-index decoding on XCH/LD | Bug found, no longer needed |
| `read_final_state.py` | Random-access reader for the 60 GB pre-`.save` raws | Obsoleted by `.save` injection in `build_and_run_4004.py` |
| `scan_ir_transitions.py` | Found IR boundaries via raw transitions before Micro5 sampling | Superseded by `verify_micro5.py` |
| `smooth_raw.py` | Resampled noisy traces to a uniform grid for plotting | Used for one figure; ad-hoc |
| `trace_from_sync.py`, `trace_divergence.py`, `trace_synced.py.bak` | Earlier iterations of `tools/trace_synced.py` | Superseded |
| `verify_hw_ops.py` | First-pass instruction verifier sampling at IR edges | Superseded by `verify_micro5.py` |
| `analyze_program.py` | Static analysis of a binary to pick composable variants | Workflow simplified; rarely needed |
| `extract_signals.py` | Bulk reader of named nets from raw | Inline in `verify_micro5.py` now |
| `gen_alu.py`, `gen_incrementer.py`, `gen_instruction_decoder.py` | Standalone subsystem generators that predated the full CPU | Subsystems are now hand-drawn `.asc` |
| `optimize_final.py`, `optimize_microcode.py` | Microcode optimisation experiments | Belongs in `optimization/`, kept here for git provenance |
| `reg_viewer.py` | Tk GUI to inspect scratchpad state over time | Replaced by `asm_ide.py` for editor work |
| `test_alu_combined.py`, `test_circuit.py`, `test_register.py` | Bespoke ALU/register test harnesses | Per-subsystem testing now goes through the board-level workflow |
| `tile_schematic.py`, `tile_shifter.py` | Schematic-tiling experiments toward the RV32I work in `cpus/tileable/` | Tileable CPU is paused |
