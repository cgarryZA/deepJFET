# BitwiseAND — Bitwise AND via Shift-and-Add (MCS-4 Manual Routine)

## What it does
Computes the bitwise AND of two 4-bit operands using the routine
published in the Intel MCS-4 Assembly Language Programming Manual,
without an AND opcode (the 4004 has none).

The trick is that **AND of two single bits equals the carry out of
adding them with no incoming carry**. The routine repeatedly shifts
both operands left, adds the high bits, and the carry register is
the AND of the bits that just shifted out. The accumulator collects
those carries as the result, then shifts back into R0 at the end.

## Input operands (loaded at START)
```
R0 = 11   (0b1011)
R1 = 6    (0b0110)
Expected AND = 11 & 6 = 1011 & 0110 = 0010 = 2
```

## Register setup at START (address 0x002)
```
START LDM 11   ; ACC = 11
      XCH 0    ; R0 = 11
      LDM 6    ; ACC = 6
      XCH 1    ; R1 = 6
      JMS AND  ; call subroutine
```

## Algorithm
1. `JUN START` at address 0x000 jumps to the data setup.
2. START loads R0 = 11, R1 = 6, calls `JMS AND`.
3. AND subroutine `FIM 1 0 11` sets R2 = 0 (working bit), R3 = 11
   (loop counter: counts up; loop ends when R3 wraps to 0).
4. Loop body:
   - Shift R0 left through carry; the bit that fell off is now in CY.
   - Increment R3; if R3 overflowed to 0, return.
   - Rotate CY into the MSB of R2 (via `RAR; XCH 2`).
   - Shift R1 left through carry; that bit goes back into R1's LSB.
   - Rotate the new R1-MSB into CY (via `RAR`), then `ADD 2`.
   - The carry out of that ADD is the AND of the two MSBs.
   - The accumulator collects this carry into its own MSB on the
     next loop iteration's `XCH 0; RAL`.
5. After 5 iterations the counter overflows; `BBL 0` returns.
6. `NOP` at the call site marks the end of execution.

## Expected final register state (from Python simulation)
See `expected_state.yaml` for the full machine-readable snapshot.

```
R0  = 2     ← AND RESULT (1011 & 0110 = 0010)
R1  = 0     (operand shifted out)
R2  = 8     (working-bit register, last value held)
R3  = 0     (loop counter overflowed)
R4..R15 = 15 (0xF, never touched -- power-up state retained)
ACC = 0     (cleared by BBL 0)
CY  = 0
```

The "untouched" 0xF in R4..R15 is the power-up state of the D
flip-flops in the LTspice CPU; the routine never writes to those
registers, so they keep their initial high value on a real SiC
processor as well.

## Execution stats (Python reference emulator)
- 80 instruction cycles from power-up to terminating NOP
- ~0.8 ms simulation time at 100 kHz clock
- LTspice transient wall time: order of magnitude shorter than the
  FloatingPoint run (which takes ~90 min for 451 cycles), because
  the cycle count is ~5.5x smaller and the program touches fewer
  registers.

## Workflow to validate against LTspice

```
# 1. Assembler.py converts BitwiseAND.asm into Machine.bin and the
#    1-bit-per-file PWL stimulus under PROGRAM/.
cd cpus/4004/programs/BitwiseAND
python ../Assembler.py BitwiseAND.asm    # (script reads Assembled.asm
                                          #  by convention -- copy first)

# 2. Build the flat CPU schematic with this program loaded.
cd ../../..
python tools/build_cpu.py 4004 --program BitwiseAND

# 3. Open cpus/4004/4004.asc in LTspice and run the transient.
#    Save the .raw next to this README as BitwiseAND.raw.

# 4. Run the register-level diff against the Python emulator.
python tools/trace_synced.py --program BitwiseAND

# 5. Rebuild the paper to pick up the new trace_diff table.
python paper/build.py
```

Step 1 is the only manual file-juggling -- once `Assembler.py` accepts a
program path on the command line (or once we replace it with a thin
wrapper around `rom_emulator.assemble`), the workflow becomes a single
`python paper/build.py` invocation.
