# Board-level subassembly workflow

This directory holds the artefacts for validating individual 4004
subsystems on their own — either as standalone LTspice simulations
with PWL-driven boundary inputs, or as physical PCBs driven by an
AWG / pattern generator.

## Why this exists

A full SiC JFET 4004 has 10 259 JFETs and resistors. If you only have
enough devices to populate one subassembly at a time, you can still
validate it end-to-end against the rest of the (simulated) CPU by:

1. Running the full CPU sim once to capture a known-good `.raw`
2. Extracting the boundary nets of the target subsystem at that
   `.raw`'s sample timestamps
3. Driving those nets onto the subsystem-only sim or the physical
   board, and diffing the resulting outputs against the same `.raw`'s
   captured outputs

The same time/voltage data drives both flows — what passes in sim
should pass on the bench.

## Subsystems

The tool auto-discovers boundary nets for each of the 4004 subsystems
in `cpus/4004/config.py`:

| Subsystem | Boundary nets (auto) |
|---|---|
| `alu` | 39 (35 in, 4 out) |
| `instruction_register` | 62 (59 in, 3 out) |
| `micro_instructions` | 102 (89 in, 13 out) |
| `controls` | 19 |
| `pins` | 28 (26 in, 2 out) |
| `program_counter` | 40 (37 in, 3 out) |
| `step_counter` | 15 |
| `scratchpad` | 24 (23 in, 1 out) |
| `stack` | 29 |

A boundary net is one that appears in this subsystem AND in at least
one other subsystem. Internal nets (only referenced inside the
subsystem) stay internal.

Direction is auto-classified by name suffix (`*In` -> input, `*Out` ->
output). Override per subsystem with a `boundary.yaml`:

```yaml
# cpus/4004/boards/alu/boundary.yaml
inputs:
  - net_a
  - net_b
outputs:
  - net_c
```

## End-to-end flow, illustrated with the ALU

```
# 1. Run the full CPU sim and capture cpus/4004/4004.raw
python tools/build_and_run_4004.py FloatingPoint --cycles 500
# (then run it in LTspice and let it finish)

# 2. Extract ALU boundary PWLs at every saved timestamp
python tools/board.py extract alu --raw cpus/4004/4004.raw

# 3. Generate the board-level testbench .asc that inlines alu.asc
#    and prepends PWL voltage sources for every input
python tools/board.py testbench alu

# 4. Open cpus/4004/boards/alu/alu_testbench.asc in LTspice and run
#    -> writes alu_testbench.raw

# 5. Diff the testbench outputs vs the expected captured from step 1
python tools/board.py diff alu
```

A pass on step 5 means: this ALU schematic, given the exact boundary
inputs the full CPU produced in step 1, produces the same outputs the
full CPU did. That's stronger than `verify_micro5` (which only checks
instruction-level effects, not the per-cycle wiring) and is what you'd
want to compare a physical board against.

## Lab signal export

For driving the physical board with an AWG or digital pattern
generator, re-emit the same boundary inputs as CSV + VCD:

```
python tools/board.py export-lab alu
```

Writes:

```
boards/alu/lab/csv/<net>.csv        # one CSV per input, time_s + voltage_v
boards/alu/lab/alu.vcd              # single VCD covering all digital inputs
```

CSV is the universal AWG file format (Keysight, Tektronix, Rigol,
Siglent all import it). VCD is the universal digital-pattern format
(PulseView, sigrok, DSLogic, Saleae, Digilent WaveForms all import it).

## What's in each subsystem's board directory

```
cpus/4004/boards/alu/
├── boundary.yaml            (optional, user-curated direction overrides)
├── manifest.txt             (auto, lists which nets got captured)
├── inputs/<net>.pwl         (one per driven boundary net)
├── expected/<net>.pwl       (one per observed boundary net, the reference
│                             output that diff compares against)
├── alu_testbench.asc        (auto-generated, inlines alu.asc + PWL sources)
├── alu_testbench.raw        (after you run it in LTspice — gitignored)
└── lab/
    ├── csv/<net>.csv        (AWG-ready)
    └── alu.vcd              (pattern-generator-ready)
```

The `.pwl` and `.csv` files compress to one sample per logic-threshold
crossing plus one per 200 mV change plus one per 5 µs floor — typically
~2000 samples per net for a 10 ms capture.

## What this doesn't do

- Symbol generation for hierarchical schematic includes. The testbench
  inlines the subsystem .asc verbatim rather than referencing it as a
  sub-circuit symbol. That's simpler and matches `build_cpu.py`'s
  pattern for the full CPU.
- Power/clock-rail PWL extraction. Rails (`VDD`, `VSS`, `CLK`,
  `ROMCLK`, `RAMCLK`) are global; the testbench leaves their existing
  pulse sources from the subsystem .asc in place.
- Automatic boundary-direction detection from `.asc` symbol pin types.
  The current implementation relies on naming convention (`*In` /
  `*Out`) plus optional `boundary.yaml` overrides. Adding LTspice
  symbol introspection would let it auto-classify more reliably.
