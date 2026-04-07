# Session Status — 2026-04-07

## What was done this session

### 1. Major cleanup
- Moved all optimisation/analysis code into isolated `optimization/` folder
- Removed `static/` directory entirely (contents moved to optimization/)
- Stripped `simulator/__init__.py` of optimizer imports
- Cleaned all `__pycache__` and `plots/` directories
- Saved pre-cleanup state on `future-work` branch

### 2. Building block library (`blocks/`)
- `dff.py` — D flip-flop (master-slave, 10 gates: 8 NAND2 + 2 INV)
- `register.py` — N-bit register (tiles N DFFs, shared clock)
- `mux.py` — 2:1 multiplexer, N-bit bus width
- `decoder.py` — N-to-2^N line decoder (NAND2 + INV trees)
- `registry.py` — `list_blocks()`, `get_block(name, **params)`
- All verified: 4-bit register flattens to 40 gates correctly

### 3. CPU project structure (`cpus/4004/`)
- Flat layout: all `.asc` sub-circuit files at top level of CPU folder
- `config.py` with JFET model, supply rails, gate R values, and COMPONENTS list
- Copied all 9 sub-circuit `.asc` files from the original JFET project

### 4. Tools
- **`tools/new_cpu.py`** — scaffold a new CPU project with auto-generated config
- **`tools/open_ltspice.py`** — find and open `.asc` files in LTSpice
- **`tools/gen_ltspice.py`** — export blocks/modules to SPICE netlists + PWL stimulus
- **`tools/build_cpu.py`** — combine sub-circuit `.asc` files into single CPU schematic, with component renaming (prefixes like ALU_J1, IR_R1 etc.) to avoid duplicates
- **`tools/isa_4004.py`** — full 4004 instruction set decoder/disassembler
- **`tools/analyze_program.py`** — analyzes a 4004 binary to determine which hardware resources are actually used (registers, stack depth, RAM, ROM, ALU ops)

### 5. Composable sub-component system
- Config supports two types of COMPONENTS entry:
  - **Fixed**: `("ALU_", "alu.asc")` — single file, always included
  - **Composable**: `("Scratch_", {"folder": "scratchpad", "parts": {...}})` — assembled from sub-parts based on program analysis
- `build_cpu.py` accepts `--program rom.bin` to auto-select minimal variants
- Scratchpad and stack are configured as composable in `config.py`

### 6. Claude Code integration
- `.claude/commands/open.md` — `/open alu` slash command
- `CLAUDE.md` — project instructions for Claude Code

## What still needs to be done

### Immediate next steps
1. **Provide sub-component .asc files** — the composable system is built but needs the actual LTSpice schematics:
   - `cpus/4004/scratchpad/pair_0.asc` through `pair_7.asc` (one per register pair)
   - `cpus/4004/scratchpad/common.asc` (shared bus wiring, address decode)
   - `cpus/4004/stack/level_0.asc` through `level_2.asc`
   - `cpus/4004/stack/common.asc` (shared stack pointer logic)

2. **Test with a real program** — write a simple 4004 program, analyze it, build a minimal schematic, and verify it runs in LTSpice

3. **PWL stimulus for testing** — use `gen_ltspice.py` to drive specific nets with waveforms instead of needing a full ROM

### Future
- More building blocks (full adder, counter, shift register)
- Program counter composable variants (by address space size)
- GUI wrapper for the build/analyze workflow
- Integration between the Python block library and LTSpice .asc generation
