# Simplified 4004 — Optimized Micro-Instruction Gate Table

Total: **53 gates, 413 JFETs**

Cost per gate: 5 + (n_instructions - 1) + n_signals

Hardwired (no bus, no gate needed): IR2->IndexReg, IR3->PC2, IR4->PC3, IR2->PC1

## MICRO 2 (25 gates, 192 JFETs)

| Gate | Instructions | Signals | JFETs |
|------|-------------|---------|-------|
| G1 | ADD, INC, ISZ_F, ISZ_T, LD, SUB, XCH (7) | IndexReg_Load, Scratch_Out, Scratch_Reg_Select | 14 |
| G2 | FIM, FIN, JIN (3) | IndexReg_Load | 8 |
| G3 | FIN, JIN (2) | Scratch_Out | 7 |
| G4 | ADD, DAA, DAC, IAC, INC, ISZ_F, ISZ_T, RAL, RAR, SUB, XCH (11) | Temp_Load | 16 |
| G5 | BBL, CLB, LD, LDM, TCC, TCS_CF, TCS_!CF (7) | Acc_Load | 12 |
| G6 | FIN, JCN_jump, JIN, JUN (4) | PC2_IR_Load | 9 |
| G7 | DAC, IAC, STC, TCS_!CF (4) | Bus0In | 9 |
| G8 | FIM, FIN, JIN (3) | ScratchPadWord1 | 8 |
| G9 | RAL, RAR (2) | CFStage1_Load, Shift_Enable | 8 |
| G10 | JCN_jump, JUN (2) | PC3_IR_Load | 7 |
| G11 | CLB, CLC (2) | CF_Clear | 7 |
| G12 | TCS_CF, TCS_!CF (2) | Bus3In | 7 |
| G13 | DAA, TCS_CF (2) | Bus1In | 7 |
| G14 | CMA (1) | Acc_INV | 6 |
| G15 | CMC (1) | CF_Inv | 6 |
| G16 | DAA (1) | Bus2In | 6 |
| G17 | DAC (1) | Subtract_Set | 6 |
| G18 | FIM (1) | ScratchPad_Load | 6 |
| G19 | JMS (1) | Shift_Loading | 6 |
| G20 | JUN (1) | PC1_IR_Load | 6 |
| G21 | NOP (1) | Micro_Reset, PC_INC | 7 |
| G22 | RAL (1) | Shift_Direction | 6 |
| G23 | STC (1) | CF_Load | 6 |
| G24 | SUB (1) | Temp_INV | 6 |
| G25 | TCC (1) | CF_Out | 6 |

## MICRO 3 (15 gates, 121 JFETs)

| Gate | Instructions | Signals | JFETs |
|------|-------------|---------|-------|
| G26 | CLB, CLC, CMA, CMC, JCN_!jump, LD, LDM, STC, TCS_CF, TCS_!CF (10) | Micro_Reset, PC_INC | 16 |
| G27 | JCN_jump, JUN (2) | Micro_Reset | 7 |
| G28 | ADD, DAA, DAC, IAC, SUB (5) | Acc_Load, EO_Out | 11 |
| G29 | RAL, RAR (2) | Acc_Load, CF_Load, Temp_Out | 9 |
| G30 | INC, ISZ_F, ISZ_T (3) | INC_Out, ScratchPad_Load, Scratch_Reg_Select | 10 |
| G31 | FIM, XCH (2) | ScratchPad_Load | 7 |
| G32 | XCH (1) | Scratch_Reg_Select | 6 |
| G33 | FIN, ISZ_F, ISZ_T, JIN (4) | PC3_IR_Load | 9 |
| G34 | FIM, FIN, JIN (3) | ScratchPadWord2 | 8 |
| G35 | FIN, JIN (2) | Scratch_Out | 7 |
| G36 | ISZ_F, ISZ_T (2) | PC2_IR_Load | 7 |
| G37 | BBL (1) | PC_Stack_Load | 6 |
| G38 | JMS (1) | Stack_Load | 6 |
| G39 | TCC (1) | CF_Clear | 6 |
| G40 | XCH (1) | Acc_Out | 6 |

## MICRO 4 (9 gates, 73 JFETs)

| Gate | Instructions | Signals | JFETs |
|------|-------------|---------|-------|
| G41 | ADD, BBL, DAA, DAC, FIM, IAC, INC, ISZ_T, RAL, RAR, SUB, TCC (12) | Micro_Reset, PC_INC | 18 |
| G42 | ISZ_F, JIN (2) | Micro_Reset | 7 |
| G43 | ADD, SUB (2) | CF_Load | 7 |
| G44 | BBL (1) | Shift_Direction, Shift_Loading | 7 |
| G45 | DAC (1) | Subtract_Reset | 6 |
| G46 | FIN (1) | ScratchPadWord1, ScratchPad_Load | 7 |
| G47 | JMS (1) | PC1_IR_Load, PC2_IR_Load, PC3_IR_Load | 8 |
| G48 | SUB (1) | Temp_INV | 6 |
| G49 | XCH (1) | Acc_Load, Temp_Out | 7 |

## MICRO 5 (3 gates, 20 JFETs)

| Gate | Instructions | Signals | JFETs |
|------|-------------|---------|-------|
| G50 | JMS, XCH (2) | Micro_Reset | 7 |
| G51 | FIN (1) | ScratchPadWord2, ScratchPad_Load | 7 |
| G52 | XCH (1) | PC_INC | 6 |

## MICRO 6 (1 gate, 7 JFETs)

| Gate | Instructions | Signals | JFETs |
|------|-------------|---------|-------|
| G53 | FIN (1) | Micro_Reset, PC_INC | 7 |

## SUMMARY

| Section | Gates | JFETs |
|---------|-------|-------|
| Micro 2 | 25 | 192 |
| Micro 3 | 15 | 120 |
| Micro 4 | 9 | 73 |
| Micro 5 | 3 | 20 |
| Micro 6 | 1 | 7 |
| **TOTAL** | **53** | **413** |

Verified by script: 53 gates, 413 JFETs (M2:192, M3:121, M4:73, M5:20, M6:7)
