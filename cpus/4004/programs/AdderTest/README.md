# AdderTest — 4-bit Adder Verification

## What it does

Runs **8 specific ADD operations** chosen to exercise every carry-out path
of the CLA adder. Each test stores its result and carry-out in a pair of
scratchpad registers so the entire SUMCF logic can be verified by reading
the final register state.

## The bug this catches

Originally found at t=22.684ms in FloatingPoint: `ADD R4` with `ACC=0, R4=8, CY=0`
produced ACC=8 (correct) but **CY=1** (wrong — 0+8 doesn't overflow 4 bits).

Likely cause: a path in SUMCF where T3 can reach the output without being
AND'd with anything on the A side. The user's written SUMCF logic was:

```
SUMCF = CFin·P1·P2·P3 + G0·P1·P2·P3 + G1·P2·P3 + G2·P3 + G3
```

The correct CLA carry-out is:
```
C4 = G3 + P3·G2 + P3·P2·G1 + P3·P2·P1·G0 + P3·P2·P1·P0·Cin
```

**Two bugs:** (1) something is making SUMCF=1 for 0+8+0 (T3-only path?), and
(2) Term 3 is missing P0, so Cin can incorrectly propagate when P0=0.

## Tests

Each test: clears, sets CY, loads operand into R14 (or R15 for test 8),
loads accumulator, runs `ADD`, saves ACC to even-indexed register, captures
CY via `TCC` and saves to next register.

| #  | Operation     | ACC reg | CY reg | Expected ACC | Expected CY | Notes                          |
|----|---------------|---------|--------|--------------|-------------|--------------------------------|
| 1  | 0 + 8 + 0     | R0      | R1     | 8            | 0           | **The original bug case**      |
| 2  | 0 + 0 + 1     | R2      | R3     | 1            | 0           | Cin alone (tests missing P0)   |
| 3  | F + 0 + 0     | R4      | R5     | F            | 0           | Max A, no overflow             |
| 4  | F + 1 + 0     | R6      | R7     | 0            | 1           | Overflow from G                |
| 5  | F + 0 + 1     | R8      | R9     | 0            | 1           | Overflow from Cin              |
| 6  | 8 + 8 + 0     | R10     | R11    | 0            | 1           | Overflow from G3 (MSB)         |
| 7  | F + F + 1     | R12     | R13    | F            | 1           | Full carry chain               |
| 8  | E + 1 + 1     | R14     | R15    | 0            | 1           | Cin overflow from low bit      |

## Expected final register state

```
R0  = 8    R1  = 0    (Test 1: 0+8+0)
R2  = 1    R3  = 0    (Test 2: 0+0+1)
R4  = F    R5  = 0    (Test 3: F+0+0)
R6  = 0    R7  = 1    (Test 4: F+1+0)
R8  = 0    R9  = 1    (Test 5: F+0+1)
R10 = 0    R11 = 1    (Test 6: 8+8+0)
R12 = F    R13 = 1    (Test 7: F+F+1)
R14 = 0    R15 = 1    (Test 8: E+1+1)
```

## Running

```
python tools/build_and_run_4004.py AdderTest
```

Then open `cpus/4004/4004.asc` in LTSpice and run. Final state should match
the table above. Any deviation pinpoints which carry path is broken:

- **R1=1** instead of 0 → SUMCF generating false carry on 0+8 (the original bug)
- **R3=1** instead of 0 → also same false-carry path
- **R5=1** instead of 0 → SUMCF generating false carry from G2 or G3
- **R7=0** instead of 1 → carry chain broken (G3 not reaching SUMCF)
- **R9=0** instead of 1 → Cin propagation broken (likely missing P0 issue)
- **R11=0** instead of 1 → G3 path broken
- **R13=0** instead of 1 → full chain broken
- **R15=0** instead of 1 → low-bit Cin propagation broken (missing P0)

## Notes

- The program is short (~80 instructions) so simulation should complete in
  about 1/6th the time of FloatingPoint (~15 minutes vs ~90 minutes).
- All tests are deliberately ordered to load specific bit patterns into both
  ACC and operand to exercise different combinations of G/P bits.
