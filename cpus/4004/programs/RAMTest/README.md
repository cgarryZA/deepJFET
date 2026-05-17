# RAMTest — first round-trip exercise of the off-CPU RAM/IO bus

## What it does

Three small tests that round-trip data through the **emulated 4002 RAM**
(the new `tools/ram4002.py` state model driven by `tools/rom_emulator.py`'s
extended PWL generator):

| # | Instruction sequence | What it tests | Expected end-state |
|--:|---|---|---|
| 1 | `SRC; LDM; WRM; CLB; RDM; XCH R2` | Main-memory write→read round trip on character `[0,0,0,5]` | R2 = 9 |
| 2 | `SRC; LDM; WRM; LDM 5; CLC; ADM; XCH R3; TCC; XCH R4` | `ADM` reading from char `[0,0,0,7]` and adding to ACC | R3 = 9, R4 = 0 (CY) |
| 3 | `SRC; LDM 11; WR2; CLB; RD2; XCH R5` | Status-character `MS2` write→read round trip | R5 = B |

Each test does its own SRC so the latched (chip, register, character)
address changes between tests — which exercises the SRC bus-drive
microcode as well as the RAM bus-data reception microcode.

## Why this matters

Before this program, every demo (FloatingPoint, AdderTest, BitwiseAND,
Fibonacci) stayed entirely inside the CPU's 16 internal index registers.
None of the **RAM/IO opcode group** (WRM, RDM, WR0..3, RD0..3, ADM, SBM,
WMP, WRR, RDR, SRC, DCL) had ever been exercised on the LTspice
schematic.

This is the smallest program that lets us validate the off-CPU bus
protocol *without* needing to build a 4002 RAM chip in JFETs. The
emulator (in `tools/ram4002.py`) plays the role of "everything the CPU
talks to over D0–D3 that isn't ROM" — it watches what the CPU drives
during `SRC` and `WRM` cycles, updates its in-memory model, and drives
the right value back onto D0In..D3In during `RDM`/`ADM`/`RD0..3`/`SBM`/
`RDR` cycles at the X2/X3 micro-phase.

## Expected final state

After the terminating `NOP`:

```
R0  = 0     (last char-index loaded, test 3)
R1  = 0
R2  = 9     ← Test 1: WRM/RDM round trip
R3  = 9     ← Test 2: ADM (5 + 4 + 0)
R4  = 0     ← Test 2: CY captured by TCC after ADM (5+4 didn't overflow)
R5  = B     ← Test 3: WR2/RD2 round trip on status char 2
R6..R15 = F  (DFF power-up state, never written)
ACC = F     (the value that was in R5 before XCH R5 — DFF power-up)
CY  = 0
```

Virtual RAM state at end:

```
main[bank=0, chip=0, reg=0, char=5] = 9
main[bank=0, chip=0, reg=0, char=7] = 4
status[bank=0, chip=0, reg=0, status=2] = B (= 11)
```

See `expected_state.yaml` for the machine-readable snapshot.

## What this verifies vs what it doesn't

**Verifies** (assuming the LTspice run matches):

- SRC's X2/X3 bus-drive timing
- The CPU's RAM-data latch in cycle X2/X3 of WRM
- The CPU's data-from-bus latch on RDM, RD0..3, ADM
- ALU integration with bus-supplied operand (ADM)
- The status-character vs main-character addressing distinction (which
  the CPU decodes from OPA, not from SRC)

**Doesn't verify** (deferred to later programs):

- `SBM` (would mirror ADM with complement)
- `RDR`, `WRR` (ROM I/O ports — needs preload of `rom_port_in` and
  a board-level check of `rom_port_out`)
- `WMP` (RAM I/O port — same pattern as WRR)
- `DCL` bank switching (a multi-bank test would write to two different
  banks and read them back)
- `FIN` indirect ROM fetch
- Cross-page SRC address with multiple chips

## Execution stats

- 35 instructions, 36 ROM bytes
- ~280 cycles to terminating NOP (per Python emulator)
- `.tran` covers ~330 cycles to be safe
- Expected wall-clock sim time: ~1.5–2 hours

## Workflow

```
# Build, regenerate PWLs (now with RAM data injected at the right times),
# and open LTspice
python tools/build_and_run_4004.py RAMTest --cycles 330

# After sim finishes, verify per-instruction
python tools/verify_micro5.py cpus/4004/4004.raw --start-time 0.2

# Then read out the final scratchpad and confirm R2..R5 match the table
```

If `verify_micro5` reports all-pass and the final R2..R5 match, the
CPU's RAM-bus microcode is working end-to-end and we know the 4002 chip
(if built in JFETs later) can be slotted in without re-validating the
CPU side.

## Implementation notes

The PWL generator drives RAM-read data at micro-phase 7 (= Micro7→Micro8
boundary, minus the `ROMCLK_LEAD` propagation delay). This mirrors the
existing ROM-fetch timing (phases 5 and 6 for OPR/OPA respectively) but
shifted one CLK later, matching the X2/X3 timing slot where the real
4002 would drive the bus.

If a verify_micro5 failure points specifically at RAM instructions, the
likely culprit is the `RAM_READ_PHASE` constant in `tools/rom_emulator.py`
needing adjustment by ±1 CLK.
