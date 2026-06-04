# Venue tailoring + flagged extensions

## Next-paper options (flagged, not yet started)

### Option A (parked) — RTL vs CJFET 4004 comparison

Re-implement the existing 4004-compatible CPU using complementary
SiC JFET primitives, based on the SPICE models published by Maeda,
Kaneko, Tanaka et al. (2025, *APL Electronic Devices*, doi
10.1063/5.0254971). Direct architecture-scale RTL-vs-CJFET comparison.

**Status**: parked. The current CPU is hand-drawn at the schematic
level (not script-generated), so re-implementing every primitive +
every subsystem in CJFET is multi-week schematic-redraw work. Worth
revisiting once a code-generated schematic flow exists, or once
someone in the group has the time for the redraw. The opportunity
is real (Maeda 2025's models are now public and nobody has built a
CJFET CPU yet) but the activation energy is high.

### Option B (blocked) — Temperature-scaled CPU re-simulation

See `paper/EXTENSION_B_TEMPERATURE.md`. **Currently blocked**: on
inspection, the Neudeck/Spry/Chen 2016 paper publishes an *NMOS
LEVEL 1* model rather than an NJF card (because the SPICE NJF model
crashes above ~300 °C), with a threshold voltage 5.4 V different
from our supplier card. Direct model swap is not viable without
redesigning R1/R2/R3 — i.e. a different CPU. **Path forward**:
email Neudeck (`neudeck@nasa.gov`) for NJF-form parameters with
standard temperature coefficients. If unsuccessful, Extension B
likely gets replaced or downgraded.

### Option F (live alternative) — LTspice corner-sweep characterisation

The `analysis/sweeps/*.cir` test benches and `tools/analyze_ltspice_log.py`
are already in place but the LTspice runs haven't been done.
Running them produces a real noise-margin / fanout / f_max corner
sweep characterisation of the SiC JFET RTL family — nothing
comparable exists in the literature, and unlike Option B this
doesn't depend on a public device model. Could be folded into the
current master paper as a new §V.D table, or spun as a short
standalone characterisation paper for HiTEC/IMAPS. If Extension B
stays blocked, this is the natural substitute.

---

## Venue tailoring — work deferred until venue choice is finalised

Once the arXiv-master draft is settled and a venue is picked, the items
below need attention. They are intentionally **not** in `Main.tex` yet,
because the trade-offs depend on the target venue's page budget and
expected audience.

## Figure budget — main paper vs supplementary

The current draft carries every CPU subsystem as a full transistor-level
schematic in the main body. These are dense and hard to read at
column scale; reviewers will read them as "proof of work" rather than
explanatory figures. For any of {JXCDC 4–8\,pp, conference 4–8\,pp,
HiTEC short manuscript}, the cleanest cut is:

**Keep in main paper:**
- `Inverter.pdf` — the NOT/NAND/NOR logic primitive (Fig.~\ref{fig:Inverter}).
- `MultiGate.pdf` — the condensed Boolean gate technique (Fig.~\ref{fig:Multigate}).
- `4004Architecture.pdf` — the top-level 4004 block diagram (Fig.~\ref{fig:4004Architecture}).
- `DFlipFlop.pdf` — the storage primitive (Fig.~\ref{DFlopLogi}).
- One results-flow / validation-flow diagram (does not exist yet — would
  need to be drawn; suggestion: emulator $\leftrightarrow$ LTspice diff
  block diagram from §III).
- Two or three measured-vs-simulated waveform figures (NAND/NOR at
  100\,kHz and at the highest tested frequency).

**Move to supplementary PDF:**
- `ALUAnnotated.pdf`, `InstructionRegisterAnnotated.pdf`,
  `Step Counter.pdf`, `ProgramCounterAnnotated.pdf`,
  `StackAnnotated.pdf`, `ScratchPadAnnotated.pdf`, `PinsAnnotated.pdf`,
  `ROM.pdf` — full transistor-level subsystem schematics.
- The two intermediate FP-mult register-trace plots (`Mult.pdf`,
  `Mult2.pdf`) — will be replaced in the main paper by the register-level
  diff *table* once `trace_synced.py` runs on the FP `.raw`.
- The current-draw plots (`NAND2kI.pdf`, `NAND100kI.pdf`) — keep one
  inline, push the other to supplementary.

For each subsystem in §II.B, replace its full-schematic figure
reference with a one-sentence prose pointer: e.g. ``the full ALU
schematic is given in Supplementary Figure~S\#''. The architectural
discussion in §II.B does not depend on the reader seeing the full
transistor netlist.

## Length

Current draft: 14 pages. Sample targets:

| Target               | Page budget | Approach                                         |
| -------------------- | ----------- | ------------------------------------------------ |
| JXCDC                | 4--8        | aggressive figure cut + §II.B compression        |
| Microelectronics J.  | 10--14      | figure cut only; keep architectural prose        |
| IEEE Access          | $\leq$20    | as-is, possibly extend supplement                |
| HiTEC abstract       | 250\,words  | abstract only; figures in conference slides      |
| arXiv master         | unrestricted| as-is                                            |

For the JXCDC cut specifically: §II.B currently runs ~1 page per
subsystem; compress to one page total with a subsystem-summary table
(name, function, JFET count from \texttt{tab:transistor-count}, key
design quirk) and prose only on the two or three most non-obvious
subsystems (ALU two-stage carry; instruction-decode AND tree; tri-state
bus workaround). The rest move to supplementary.

## Claim audit

Search-and-destroy targets when targeting a published venue:

- Any remaining use of "first", "demonstrated", "successfully" without
  qualification.
- Any remaining unqualified "radiation-hardened" / "rad-hard" claim
  about *this work* (motivation context is fine; the index terms
  already say "Radiation-Environment Electronics").
- Any remaining reliance on the analytical/back-of-envelope
  frequency-limit argument once the LTspice corner sweep .log
  populates `paper/data/freq_sweep.tex` --- update prose to cite the
  measured-from-LTspice $f_{\max}$ then.

## Reproducibility statement

The §V results section should end with a one-line reproducibility
statement once `paper/build.py` is publicly runnable:

> The complete reproducibility package, including the LTspice
> schematics, Python emulator, assembler, and analysis scripts that
> produce every table and figure in this paper, is available at
> [DOI / GitHub URL once minted]. Running `python paper/build.py`
> regenerates the manuscript end-to-end.
