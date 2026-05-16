# Venue tailoring — work deferred until venue choice is finalised

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
