# Fibonacci — first 8 Fibonacci numbers in scratchpad

## What it does
Computes F(0) through F(7) of the Fibonacci sequence and stores each
in its own scratchpad register so the entire sequence is visible in
the final CPU state.

```
F(0) = 0    F(1) = 1    F(2) = 1    F(3) = 2
F(4) = 3    F(5) = 5    F(6) = 8    F(7) = 13 (0xD)
```

F(7) = 13 is the largest Fibonacci number that fits in a 4-bit nibble.
The next, F(8) = 21, would overflow, so this program stops one step
short of multi-precision arithmetic and keeps the algorithm to a clean
single-nibble loop.

## Algorithm

Straight-line "compute the next Fibonacci, store it, advance":

```asm
START   LDM 0     ; seed R0 = F(0) = 0
        XCH 0
        LDM 1     ; seed R1 = F(1) = 1
        XCH 1
        JMS Compute
        NOP

Compute CLC       ; F(2) = F(0) + F(1)
        LD 0
        ADD 1
        XCH 2
        CLC       ; F(3) = F(1) + F(2)
        LD 1
        ADD 2
        XCH 3
        ... (four more (CLC, LD, ADD, XCH) groups for F(4)..F(7))
        BBL 0
```

Six ADD instructions, each summing the previous two Fibonacci numbers
into the next register. Six CLC instructions reset CY before each ADD
so the addition is unsigned (no incoming carry).

## What this exercises (vs the existing programs)

| Behaviour | FloatingPoint | AdderTest | BitwiseAND | **Fibonacci** |
|---|:--:|:--:|:--:|:--:|
| Multiplication / partial products | ✓ | | | |
| Adder edge cases (carry chain) | | ✓ | | |
| Rotate + AND via shift-and-add | | | ✓ | |
| `JMS` + `BBL` (stack push/pop) | ✓ | | ✓ | ✓ |
| `ISZ` (looping) | ✓ | | ✓ | |
| **Long chain of ADDs across 8 different registers** | | | | ✓ |
| **CLC + ADD + XCH sequence × 6** | partial | | | ✓ |
| **Whole-sequence visible in final state** | | | | ✓ |

The straight-line code is intentional — it makes the per-instruction
correctness directly readable from the final scratchpad and avoids the
sim cost of an ISZ loop (which is already covered by the other two
programs).

## Expected final state

After the terminating `NOP`:

```
R0  = 0     ← F(0)
R1  = 1     ← F(1)
R2  = 1     ← F(2)
R3  = 2     ← F(3)
R4  = 3     ← F(4)
R5  = 5     ← F(5)
R6  = 8     ← F(6)
R7  = D     ← F(7) = 13
R8..R15 = F (0xF, DFF power-up state, never written by this program)
ACC = 0     (loaded by BBL 0)
CY  = 0     (last ADD: 5 + 8 = 13, no overflow)
```

See `expected_state.yaml` for the machine-readable snapshot.

## Execution stats

- 34 bytes of ROM, 26 instructions
- Python emulator: 28 dynamic instruction cycles to terminating NOP
- Total clock cycles to halt: ~294
  (16 JUN START + 34 setup + 18 JMS + 6 × 35 compute + 8 BBL + 8 NOP)
- LTspice simulation time: ~3.5 ms wall-clock (.tran covers 300 cycles)
- Expected wall-clock sim time: order of one hour

## Workflow

```
# Open in the IDE, edit & build in one click
python tools/asm_ide.py cpus/4004/programs/Fibonacci/Fibonacci.asm

# Or headless
python tools/build_and_run_4004.py Fibonacci --cycles 350

# After LTspice run completes, verify
python tools/verify_micro5.py cpus/4004/4004.raw --start-time 0.2
```

Then check the final scratchpad against the table above — every
register from R0 through R7 should match, R8..R15 should still read
0xF, ACC = 0, CY = 0.
