; SmokeTest — Scratch Build CPU minimal verification
;
; Does R0 = 5, R1 = 3, then ACC = R0 + R1 = 8.
; If all 16 IR_In lines and the ALU are wired correctly, final state
; should be: R0 = 5, R1 = 3, ACC = 8, CY = 0.
;
; Each instruction is 2 bytes (16 bits) under the new ISA. Total ROM
; = 14 bytes = 7 instructions. The 7th NOP gives the CPU one cycle
; to settle before trailing zero-fills (which all decode as NOP too).
LDM 5
XCH 0
LDM 3
XCH 1
LD  0
ADD 1
NOP
