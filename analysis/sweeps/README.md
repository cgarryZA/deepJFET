# LTspice corner sweeps

Two paper-facing analyses live here:

1. **Frequency sweep** (`freq_testbench.cir`) — what's the highest clock
   frequency at which a representative SiC JFET RTL gate chain still
   produces clean logic levels? Used in §V (Results) to bound the
   simulated CPU's f_max.

2. **Sensitivity sweep** (`sensitivity_{r1,r2,r3,vpos,vneg}.cir`) — how
   do the noise margins V_OH − V_th and V_th − V_OL respond to ±20 %
   variation of each hand-tuned resistor and ±2 V on each supply rail?
   Used in §VI (Limitations) to bound the design's tolerance budget.

Neither sweep touches the CPU schematic in `cpus/4004/4004.asc`. They
operate on a small standalone test article (a 4-inverter chain for the
frequency sweep, a single 2-JFET inverter for the sensitivity sweep),
so you can run them in parallel with anything else without risk to your
in-flight CPU simulations.

## End-to-end workflow

```bash
# 1. (Re)generate the test-bench .cir files from cpus/4004/config.py.
#    Re-run after editing config.py so the sweeps track the live design.
python tools/gen_freq_testbench.py
python tools/gen_sensitivity_testbench.py

# 2. Open each .cir in LTspice and run the simulation.
#    GUI:    File / Open / <file>.cir / hit Run
#    Batch:  LTspice.exe -b analysis/sweeps/freq_testbench.cir
#            LTspice.exe -b analysis/sweeps/sensitivity_r1.cir   (etc.)
#    LTspice writes a .log next to each .cir.

# 3. Parse the .log files into paper/data/*.{csv,tex,macros}.
python tools/analyze_ltspice_log.py --sweep freq
python tools/analyze_ltspice_log.py --sweep sensitivity

# 4. Rebuild the paper. The new tables and prose macros are picked up
#    automatically via \input{} statements in Main.tex.
python paper/build.py
```

Steps 1 and 3 are also wired into `paper/regen_data.py`, so once the
.log files exist the wrapper `python paper/build.py` does steps 3 and 4
in one shot.

## What's inside each .cir

### `freq_testbench.cir`
- 4-inverter chain (`X1..X4`) of the 2-JFET RTL inverter subcircuit.
  Default depth=4 matches the 4-bit ripple-carry adder critical path in
  the ALU; pass `--chain N` to `gen_freq_testbench.py` to override.
- Clock input is a `PULSE(v_low, v_high, ...)` parameterised by `{freq}`.
- `.step dec param freq 10k 2Meg 10` — log decade sweep.
- Per step, `.meas tran` extracts V_OH, V_OL, swing, t_pd (rise/fall).

### `sensitivity_r1.cir` (similar for r2, r3, vpos, vneg)
- Single inverter `X1` driven by a slow 10 kHz square wave (so the
  output settles fully at every step).
- `.step param r1 list 40k 45k 50k 55k 60k` (R values ±20 % from
  nominal). The other resistors and the supplies stay at config.py
  values.
- Per step, the same V_OH / V_OL / t_pd `.meas` directives.

## Parameter source of truth

Everything in these test benches — the JFET .model card, supply
voltages, hand-tuned resistor values — is emitted from
`cpus/4004/config.py` at generation time. If you change a resistor in
the CPU config, re-run the two `gen_*` scripts and the sweeps update.

## What lands in the paper

| File | Where it shows up |
|---|---|
| `paper/data/freq_sweep.csv`  | reproducibility supplement |
| `paper/data/freq_sweep.tex`  | Results §V (in-paper table) |
| `paper/data/freq_sweep_macros.tex` | `\FmaxKHz`, `\NominalSwingV` used inline in §V prose |
| `paper/data/sensitivity.csv` | reproducibility supplement |
| `paper/data/sensitivity.tex` | Limitations §VI (in-paper table) |
| `paper/data/sensitivity_macros.tex` | `\WorstNoiseMargin`, `\LogicThresholdV` inline |

## Troubleshooting the parser

`tools/analyze_ltspice_log.py` accepts two LTspice .log formats —
inline (`name: ... = value`) and summary-table (`Measurement: name`
followed by indented rows). If your LTspice version emits something
neither pattern catches, run the analyzer with `--log path/to/file.log`
on a single file and inspect the printed step / measurement counts; if
they're empty, send me a sample of the .log and I'll patch the regex.
