# Extension B — Temperature-scaled CPU re-simulation with a public SiC JFET model

**STATUS: BLOCKED pending Neudeck email response.** The original
brief assumed Neudeck/Spry/Chen 2016 published a drop-in NJF SPICE
card; on actually pulling the paper (saved at
`paper/precedent/neudeck2016_4HSiC_JFET.pdf`) it turns out they
publish an *NMOS LEVEL 1* model, not NJF, with 9 discrete cards
(3 temperatures × 3 wafer positions) rather than continuous
temperature scaling. They explicitly state in §III that they switched
to NMOS because the SPICE NJF model crashes above ~300 °C.

Two consequences:

1. **The model card is not drop-in.** Different device type, different
   parameter names (VTO/KP/GAMMA/LAMBDA/CJ/PB/PHI/RD/RS instead of
   Beta/Vto/Lambda/Is/N/Isr/Nr), no `Vtotc`/`Betatce`/`Xti`/`Eg`
   continuous temperature coefficients.
2. **The threshold voltage differs by ~5.4 V.** Our supplier card has
   `Vto = -3.45 V`; Neudeck's nominal at 25 °C is `VTO = -8.85 V`.
   Our hand-tuned R1/R2/R3 produce logic levels around the supplier's
   threshold. With Neudeck's threshold, the gates sit in cutoff and
   the CPU does not function without an R-ladder redesign — i.e.
   a different CPU.

## Working title (if unblocked)

*"Temperature-scaled program-level validation of a SiC JFET 4-bit
microprocessor from 27 °C to 500 °C"*.

## Original one-line pitch (preserved for reference)

Replace the proprietary supplier SiC JFET parameter card with the
publicly-released Neudeck/Spry/Chen 2016 extreme-temperature 4H-SiC
JFET SPICE model, re-run the unchanged 4004-compatible CPU schematic
through the existing `verify_micro5.py` pipeline at three documented
temperature points, and report ISA-rule pass/fail counts per
temperature. Two wins in one paper: removes the proprietary-model
caveat from the master paper *and* turns the high-temperature
motivation into a demonstrated result rather than extrapolation.

## Current options (post-finding)

**Option B.D — Email Neudeck for NJF-form parameters.** Concrete ask:
NJF-form fits for the same 4H-SiC JFETs in the 2016 paper, with
standard SPICE `Vtotc`/`Betatce`/`Xti`/`Eg` temperature coefficients,
even as preliminary or hand-extracted fits. Contact:
`neudeck@nasa.gov`. Cost: a week of waiting, zero work. **Recommended
first action.** If positive response, the brief below applies as
originally written.

**Option B.C — Supplier model + built-in NJF temperature scaling at
27 / 300 / 500 °C.** Cheapest fallback. Risks: doesn't address either
the proprietary-model caveat or the extrapolation caveat (still using
the supplier's published `Vtotc`), and may not run at all at 500 °C
(SPICE NJF >300 °C crash that Neudeck documents). Six LTspice
transients, ~70 h wall clock. Recommend deferring unless B.D fails.

**Option B.B — Cross-validate primitives against Neudeck measured
data.** Compare our LTspice NAND/NOR I-V/switching against the
measured curves in Neudeck 2015 (wafer DC variations,
`paper/precedent/neudeck2015_wafer_characterization.pdf`). Adds a
paragraph to §V of the current paper. Not a standalone paper.

**Option B.A — Re-tune R1/R2/R3 for Neudeck's threshold and
re-validate.** Effectively a different CPU. Multi-week schematic +
sim effort. Not recommended.

## Suggested email to Neudeck

```
Subject: NJF-form 4H-SiC JFET SPICE parameters for academic temperature-sweep work

Dr Neudeck,

I'm an MEng student at Durham University (supervisor: Alton Horsfall,
ORCID 0000-0003-2772-2006) working on a transistor-level SiC JFET
implementation of an Intel-4004-compatible 4-bit microprocessor,
captured in LTspice. The current manuscript validates the design at
27 °C using a proprietary SiC JFET parameter card; for a follow-on
paper we would like to swap to a publicly-citable model and run a
temperature sweep through verify_micro5.py at 27 / 300 / 500 °C.

Your 2016 SPICE-modeling paper publishes NMOS LEVEL 1 parameters
because the SPICE NJF model is unstable above ~300 °C. Our CPU is
designed around the NJF formulation (it relies on Vtotc/Betatce/Xti/Eg
temperature scaling and on the buried-gate junction-diode currents
that NMOS LEVEL 1 doesn't characterise), and the threshold voltage
difference rules out a straight drop-in.

Would you be willing to share NJF-form fits for the 4H-SiC JFETs in
your 2016 paper, even as preliminary or hand-extracted parameter
sets — specifically Beta, Vto, Lambda, Is, N, Isr, Nr, Alpha, Vk,
Rd, Rs plus Vtotc / Betatce / Xti / Eg, with junction capacitances
Cgs0 / Cgd0? Anything we use will be cited to your published work
and attributed to your group.

If the NJF formulation genuinely doesn't work at the temperatures we
need, that's also useful to know — it would help us bound the claim
in the limitations section honestly.

Thank you for considering it. Happy to share more about our setup if
useful.

— Christian Garry
  ORCID 0009-0001-3931-9401
  Durham University Department of Engineering
```

Cost of sending: zero. Worst case: a definitive "no, NJF doesn't work
above 300 °C even with our fits, that's why we publish NMOS" — which
is itself useful information for the §VI Limitations text. Best case:
the original brief plan works.

## What's already on disk

- `paper/precedent/neudeck2016_4HSiC_JFET.pdf` — the actual paper.
- `paper/precedent/neudeck2015_wafer_characterization.pdf` — wafer
  DC parameter variations, useful for cross-validating any model card
  against measured envelopes regardless of which path B takes.
- `cpus/4004/config.py` — dual-model selector (env var
  `JFET_MODEL=supplier` / `neudeck`), Neudeck slot still has TBD
  placeholder values awaiting the email response.
- `tools/build_cpu.py` — `--temperature <C>` flag wired in (works
  regardless of which model is selected).

## Live alternative if B stays blocked

The LTspice corner-sweep characterisation work — already-generated
`analysis/sweeps/*.cir` test benches plus `tools/analyze_ltspice_log.py`
— produces a real publishable noise-margin / fanout / f_max
characterisation result that doesn't depend on a public device model.
Tracked in `paper/VENUE_TAILORING_TODO.md` as Option F.

## Why this is the cheapest viable extension

Almost every piece is already in place:

- The CPU schematics in `cpus/4004/*.asc` are unchanged --- no
  redrawing.
- Test programs (`FloatingPoint`, `BitwiseAND`) are already assembled
  and PWLs already generated.
- `verify_micro5.py` already does the ISA-rule verification correctly
  (454 / 0 confirmed for FloatingPoint at 27 °C).
- `paper/build.py` already regenerates Table III from the
  per-program macros files.
- The `\providecommand{TBD}` defaults in `paper/preamble.tex` mean a
  partially-populated Table III still compiles.

The only real work is the LTspice run time (each transient takes the
same wall time at 500 °C as at 27 °C, so the cost is roughly *N temperatures
× M programs* of your current FP sim cost).

## Concrete steps

### 1. Acquire the Neudeck 2016 SPICE model card

**Reference**: P. Neudeck, D. Spry, L. Chen, *"First-Order SPICE Modeling
of Extreme-Temperature 4H-SiC JFET Integrated Circuits"*, 2016.
Semantic Scholar ID `81183fea1000d3f2ddee9c14fd3402ceaa70c5b5`.

Check:
- Does the paper publish the full NJF `.model` line in text or as
  supplementary material? (Most NASA technical reports of this kind do.)
- If yes, transcribe the parameters into a new file under
  `cpus/4004/models/neudeck2016_4HSiC_JFET.lib`.
- If the parameters are given only as plots/tables, contact Neudeck's
  group at NASA Glenn for the SPICE deck (they have historically been
  responsive to academic requests).

Cross-check against:
- Neudeck/Chen/Spry 2015 *"Electrical Characterization of a 4H-SiC JFET
  Wafer"* (`7886430a6858d2cfac6d834867bfb90d3720f646`) for measured
  parameter variation envelopes.
- Neudeck 2008 EDL *"Stable Electrical Operation of 6H-SiC JFETs and ICs
  for Thousands of Hours at 500 °C"* (`5fc2d95f2c37a1fdd2f735ea856e6d4a14625d08`)
  for the longest-running operating point you'll claim.

### 2. Wire the model swap into the build

Add a new entry to `cpus/4004/config.py`:

```python
# Public extreme-temperature parameterisation from Neudeck et al. 2016.
# Used as the canonical model for the temperature-scaled validation
# campaign (Extension B). Replaces the proprietary supplier model
# previously named here.
JFET_MODEL_NEUDECK_2016 = NChannelJFET(
    beta=...,  vto=..., lmbda=..., is_=...,
    # values from Neudeck/Spry/Chen 2016 Table II / supplementary deck
)

# The default `JFET_MODEL` symbol used by build_cpu.py / count_transistors.py
# is whichever is selected for the current campaign.
JFET_MODEL = JFET_MODEL_NEUDECK_2016
```

Keep the previous (supplier) model card alongside it as
`JFET_MODEL_SUPPLIER_<rev>` so the original transistor-count and
verify_micro5 numbers can be regenerated on demand for the current
paper.

### 3. Add temperature as a sweep axis

Two options.

**Option B.1 — One CPU schematic, model parameters re-evaluated at each T.**
The `NChannelJFET.at_temp(T)` method already exists in
`model/jfet.py` (see `optimization/README.md`). Generate the flat CPU
schematic once with the model card; LTspice's NJF model handles
temperature via the standard `.temp` directive or per-instance
`temp=...` override. Cleanest workflow.

**Option B.2 — Pre-evaluate the device model in Python and emit per-T schematics.**
Use `JFET_MODEL_NEUDECK_2016.at_temp(T)` to bake temperature-scaled
parameters in at netlist-build time. Useful only if LTspice's built-in
temperature scaling diverges from the published Neudeck temperature
trend.

Recommend B.1 first. Add a `--temperature` flag to `tools/build_cpu.py`
that emits the schematic with a `.temp <T>` directive at the top.

### 4. Run the campaign

Three temperature points cover the headline claim:

| Temperature | Why | Expected LTspice time |
|---|---|---|
| 27 °C | Baseline, matches current paper | already done for FP |
| 300 °C | Mid-range, covers terrestrial extreme-environment apps (boreholes, near-engine sensing) | same as FP at 27 °C |
| 500 °C | Validated upper limit of Neudeck 2008 EDL paper (thousands of hours stable) | same as FP at 27 °C |

Two programs (`FloatingPoint`, `BitwiseAND`) × three temperatures =
six transient simulations. Probably 4-6 hours per sim on the current
machine → roughly one wall-clock day to run all six.

Mark each with `SIM_COMPLETE` and re-run `python paper/build.py`. The
new tool `tools/temperature_sweep.py` (see step 5) collates the
results.

### 5. New tool: `tools/temperature_sweep.py`

Thin wrapper that:

1. Walks `cpus/4004/programs/<program>/runs/<T>/<program>.raw` for
   each program × temperature combination.
2. Calls `verify_micro5.py` on each raw and collects
   (program, temperature, instructions_checked, ISA_failures,
    CY_artefacts) tuples.
3. Emits:
   - `paper/data/temperature_sweep.csv` --- one row per (program, T).
   - `paper/data/temperature_sweep.tex` --- summary table.
   - `paper/data/temperature_sweep_macros.tex` --- macros like
     `\TraceFailures<Program>At<T>` for inline prose use.

Wire it into `paper/regen_data.py` gated on the per-temperature
`SIM_COMPLETE` sentinels.

### 6. Manuscript changes

New `Main_TemperatureB.tex` variant *or* extend the master:

Either:

**6a) Extend the master manuscript**. Add §V.D *"Temperature-scaled
validation"* showing:
- Table: programs × temperatures × ISA-rule pass/fail counts.
- Figure: NAND/NOR primitive switching time vs temperature
  (LTspice-extracted from the new model at the three temperatures,
  cross-referenced to Neudeck 2008 measured data where available).
- One paragraph noting that the proprietary-supplier-model
  reproducibility caveat from the original §VI Limitations is now
  removed.

**6b) Spin as a short standalone**. Same content, framed as a
follow-on letter to *IEEE EDL* or a short *Microelectronics Journal*
note. Cite the master paper for the CPU itself.

I'd suggest **6a** first --- it's the cleaner research story
("here's the CPU, here's the temperature characterisation") and
strengthens the existing paper rather than fragmenting it. 6b is the
better fit only if a short-paper venue lines up.

## Decision points / risks

- **Model parameter availability**. If Neudeck 2016 only publishes
  parameter *trends* rather than a full NJF card, you'll need to
  contact NASA Glenn or transcribe from plots. This is the only
  potential blocker; everything else is mechanical.

- **Quantitative model drift at 27 °C**. The Neudeck card will not
  produce exactly the same I-V curves as the proprietary supplier
  card. Expect the room-temperature \TotalJFETCount{}-transistor CPU
  to produce slightly different bus voltages, possibly different
  noise margins. The verify_micro5 ISA-rule check is robust to
  voltage shifts (it only thresholds at -2.5 V), so a 0 / 0
  result at 27 °C with the new model would actually be the
  cleanest possible "model swap is valid" headline.

- **High-T device behaviour invalidating gate operation**. At 500 °C
  the JFET threshold voltage shifts, channel mobility drops, and
  leakage currents grow. The hand-tuned R1/R2/R3 values were chosen
  for room temperature. The most interesting *negative* result here
  would be: "at 500 °C the existing CPU still passes for AND but
  fails for FP because of accumulated leakage in long arithmetic
  loops" --- that's a publishable finding either way.

- **Run time blow-up**. If LTspice's iteration tolerance has to be
  tightened at 500 °C to stop the solver from diverging, sim time
  could grow significantly. Budget for 2× headroom on the FP run.

## Expected timeline

| Step | Effort |
|---|---|
| Locate / transcribe Neudeck 2016 model card | 1 evening |
| Wire model swap into `config.py` + `build_cpu.py` `--temperature` flag | 1 evening |
| Sanity-check at 27 °C: FP and AND should still produce 0 ISA failures | 1 LTspice run cycle |
| Run FP and AND at 300 °C and 500 °C | 1-2 LTspice run cycles (overnight) |
| Write `tools/temperature_sweep.py` + wire into `regen_data.py` | 1 evening |
| Add §V.D + table + figure to master | 2 evenings |
| Polish, rebuild, send to Alton | 1 evening |

Realistic: **2-3 weeks of evenings**, mostly waiting on LTspice. The
human time is roughly equal to one week of focused effort.

## What this lands you

- Master paper goes from "room-temperature simulation only" to
  "validated 27 °C / 300 °C / 500 °C with a publicly-cited device model."
- §VI Limitations drops one of its three named limitations.
- The CPU design is then fully reproducible from public sources, which
  unlocks the strongest possible reproducibility statement.
- A standalone short paper option (6b) is available if a follow-on
  venue is preferred over extending the master.

## Pointers in this repo

- Current paper that this extends: `paper/Main.tex` (master) /
  `paper/Main_JXCDC.tex` (compressed).
- Build pipeline: `paper/build.py` --variant master|jxcdc|all.
- ISA-rule verifier: `tools/verify_micro5.py` --program <name>.
- CPU configuration: `cpus/4004/config.py`.
- Device model code that already supports `at_temp(T)`:
  `model/jfet.py`.
- Test programs (no changes): `cpus/4004/programs/FloatingPoint/`,
  `cpus/4004/programs/BitwiseAND/`.

## Cross-references

- Neudeck 2016 (the model): SS ID
  `81183fea1000d3f2ddee9c14fd3402ceaa70c5b5`.
- Neudeck/Chen/Spry 2015 (wafer DC variations): SS ID
  `7886430a6858d2cfac6d834867bfb90d3720f646`.
- Neudeck/Spry/Chen 2008 (500 °C operation, IEEE EDL, 127 citations):
  SS ID `5fc2d95f2c37a1fdd2f735ea856e6d4a14625d08`.
- Hunter/Kremic/Neudeck 2021 (Venus surface electronics roadmap, motivation):
  OpenAlex W3153203516.
