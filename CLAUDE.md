# deepJFET — SiC JFET CPU Design Framework

## LTSpice Integration
- LTSpice is at: `C:\Users\z00503ku\AppData\Local\Programs\ADI\LTspice\LTspice.exe`
- When the user says "open the ALU" or "edit the program counter" etc., run `python tools/open_ltspice.py <component>` to find and open the .asc schematic in LTSpice.
- Use `/open <component>` as a shortcut.
- .asc files live inside `cpus/<cpu>/<component>/<component>.asc`

## Project Structure
- `model/` — JFET physics, DC gate solvers (do not modify without good reason)
- `simulator/` — gate-level netlist, modules, event-driven sim engine
- `transient/` — ODE transient solver
- `optimization/` — isolated analysis/optimizer code (not used in normal workflow)
- `blocks/` — parameterized building blocks (dff, register, mux, decoder)
- `cpus/` — CPU designs (currently: 4004)
- `tools/` — standalone scripts: new_cpu.py, gen_ltspice.py, open_ltspice.py
- `ltspice/` — templates and generated output

## Gate Parameters
- Hand-designed R1/R2/R3 values per gate type are in each CPU's `config.py`
- No automated optimization in the normal workflow — edit config.py directly

## Workflow
1. Define components using blocks from `blocks/`
2. Export to LTSpice with `tools/gen_ltspice.py` (drives any nets with PWL)
3. Open in LTSpice with `tools/open_ltspice.py` or `/open`
4. New CPU projects: `python tools/new_cpu.py <name> --components a,b,c`
