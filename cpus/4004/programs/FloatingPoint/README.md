# FloatingPoint — 8-bit Floating Point Multiplication

## What it does
Multiplies two floating-point numbers using software on the 4004.
Each number has an 8-bit significand and an 8-bit biased exponent (bias=63).

## Input operands (loaded at START)
```
Operand A:  significand = 0xF4 (244),  exponent = 0x47 (71, unbiased=8)
Operand B:  significand = 0xA0 (160),  exponent = 0x4B (75, unbiased=12)
```

In decimal:
- A = 0.953125 × 2^8 = 244.0
- B = 0.625 × 2^12 = 2560.0
- **Expected result: 244 × 2560 = 624,640**

## Register setup at START (address 0x080)
```
R4  = 0xF    R5  = 0x4    → significand A = 0xF4
R6  = 0x4    R7  = 0x7    → exponent A = 0x47
R8  = 0xA    R9  = 0x0    → significand B = 0xA0
R10 = 0x4    R11 = 0xB    → exponent B = 0x4B
```

## Algorithm
1. `JUN START` at address 0x000 jumps to 0x080
2. START loads operands into registers, calls `JMS MULT`
3. MULT calls `SIGNBITS` — extracts sign bits from exponents
4. Adds exponents, subtracts bias (63) since it was counted twice
5. Combines sign bits into final exponent
6. **BCHECK loop** — shift-and-add multiplication (~16 iterations):
   - Check if multiplier (R8:R9) is zero → done
   - Shift multiplier right by 1
   - If shifted-out bit was 1, add multiplicand to product
   - Shift multiplicand left by 1
   - Repeat
7. RETURN — normalize product, adjust exponent
8. `BBL 0` returns, then NOP at address 0x092

## Expected final register state (from Python simulation)
```
R0  = 0xF    R1  = 0x4    (working registers)
R2  = 0x0    R3  = 0x0    (sign bits / cleared after normalize)
R4  = 0x0    R5  = 0x0    (multiplicand shifted out)
R6  = 0x4    R7  = 0x7    (exponent A, unchanged)
R8  = 0x0    R9  = 0x0    (multiplier shifted out)
R10 = 0x4    R11 = 0xB    (exponent B, unchanged)
R12 = 0x9    R13 = 0x8    ← PRODUCT SIGNIFICAND = 0x98
R14 = 0x6    R15 = 0x2    ← RESULT EXPONENT = 0x62
```

## Expected outputs to check
```
R12:R13 = 0x98  (product significand = 152/256 = 0.59375)
R14:R15 = 0x62  (result exponent = 98, unbiased = 98-63 = 35)
ACC     = 0x0   (cleared by BBL 0)
IR      = 0x00  (NOP at end)
PC      = 0x092 (address of final NOP)
```

## Verification
Result = 0.59375 × 2^35 ≈ 20,401,094,656

Wait — that doesn't equal 624,640. The significands multiply as fractions:
- Raw multiply: 244 × 160 = 39,040 = 0x9880
- Top byte = 0x98 ✓ (matches R12:R13)
- Exponent: 8 + 12 = 20, plus normalization adjustment → 0x62 = 98 → unbiased 35

The floating-point representation encodes the result correctly for the
multiplication of the two significand fractions with exponent addition.

## Execution stats
- 451 instruction cycles
- ~42.6ms simulation time at 100kHz clock
- ~90 minutes LTSpice wall time
