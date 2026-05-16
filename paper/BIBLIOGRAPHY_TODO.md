# Bibliography references to track down

Citations the manuscript would benefit from but I haven't added as
bibtex stubs because I can't verify the exact authors, year, or DOI
without literature access. Each row below is something the
**prior-art table** in §I, or one of the §IV / §V / §VI paragraphs,
would cite if we added it.

When you find the actual reference, paste a verified bibtex entry into
`bib.bib` and the corresponding `\cite{}` into `Main.tex`. The TODO
items are roughly ordered by how much the missing citation hurts the
paper's defensibility.

## High priority — would strengthen the prior-art table directly

- [ ] **KTH SiC integrated logic (Lanni, Zetterling et al., ~2013-2015).**
  4H-SiC bipolar / lateral-JFET integrated logic at 500 °C+. The most
  obvious peer to Neudeck's NASA work. Either *Lanni 2013* (IEEE TED
  on 4H-SiC bipolar integrated logic) or *Lanni/Zetterling 2015* (SiC
  integrated NOR/oscillator paper).

- [ ] **Fraunhofer IISB SiC integrated logic / digital control.**
  Multiple groups have published SiC integrated digital control IC
  results in the 2015-2022 window; worth a single citation for
  industrial / non-NASA SiC IC work.

- [ ] **Honeywell HX5000 or BAE RAD750 silicon-on-insulator (SOI)
  rad-hard processor.** A representative cite for SOI radiation
  hardness as the historical industry baseline that SiC has to beat.

- [ ] **Diamond logic (Nebel / AIST or comparable).** A representative
  cite if we want a non-SiC wide-bandgap row in the prior-art table.
  Mostly research demos, no commercial digital ICs.

## Lower priority — nice to have

- [ ] **GaAs rad-hard digital (Triquint / Vitesse historical).**
  Mentioned as motivation in §II's RTL-vs-CJFET paragraph; not strictly
  needed if Zolper~\cite{ZOLPER19982153} stays as the umbrella
  citation.

- [ ] **NASA Glenn Neudeck/Spry 2009 follow-on to the 2018 1000 °C
  paper.** Would strengthen the temperature-extended evidence chain
  but Neudeck2018 alone is probably enough.

- [ ] **A modern primitive-gate measurement paper that documents
  similar baseline-shift behaviour in SiC JFETs.** If one exists, it
  would let the Limitations §VI paragraph on the 1.5-2 V offset cite
  a prior observation rather than presenting it as novel-and-unsolved.

## Already cited — for cross-reference

The current `bib.bib` / `bib2.bib` covers:

- SiC JFET / IC: `ZOLPER19982153`, `RAYNAUD19971504`, `Neudeck2000`,
  `Neudeck2018`, `9785630`, `Habib2013`.
- Radiation hardness: `RadHard`, `RADHARD2`, `Nava2003645`,
  `Iwamoto2013`, `LEE2003489`.
- Bandgap / materials: `SiBandGap`, `SiCBandGap`, `DiamondHard`,
  `SIHard`.
- 4004 reference: `Intel1973MCS4`, `Intel4004Specs`, `Archi`, `Emu`,
  `ReadingSilicon`.

22 entries total. Adding the 3-4 high-priority items above is enough
to make the prior-art table fully populated and the §VI claims well
sourced.
