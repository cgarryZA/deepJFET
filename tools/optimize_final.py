#!/usr/bin/env python3
"""Final optimized micro-instruction gate table with all merges applied."""

gates = []
gate_num = 0

def g(micro, insts, sigs, note=""):
    global gate_num
    gate_num += 1
    n_i = len(insts)
    n_s = len(sigs)
    cost = 5 + (n_i - 1) + n_s
    kind = "shared" if n_i > 1 else "solo"
    gates.append((gate_num, micro, insts, sigs, cost, kind, note))

# MICRO 2

# Triple merge: IndexReg_Load + Scratch_Out + Scratch_Reg_Select
g(2, ["ADD","INC","ISZ_F","ISZ_T","LD","SUB","XCH"],
     ["IndexReg_Load","Scratch_Out","Scratch_Reg_Select"], "triple merge")
g(2, ["FIM","FIN","JIN"], ["IndexReg_Load"], "remainder of triple")
g(2, ["FIN","JIN"], ["Scratch_Out"], "remainder of triple")

g(2, ["ADD","DAA","DAC","IAC","INC","ISZ_F","ISZ_T","RAL","RAR","SUB","XCH"],
     ["Temp_Load"])
g(2, ["BBL","CLB","LD","LDM","TCC","TCS_CF","TCS_!CF"],
     ["Acc_Load"])
g(2, ["FIN","JCN_jump","JIN","JUN"], ["PC2_IR_Load"])
g(2, ["DAC","IAC","STC","TCS_!CF"], ["Bus0In"])
g(2, ["FIM","FIN","JIN"], ["ScratchPadWord1"])

# Identical merge: CFStage1_Load + Shift_Enable
g(2, ["RAL","RAR"], ["CFStage1_Load","Shift_Enable"], "identical merge")

g(2, ["JCN_jump","JUN"], ["PC3_IR_Load"])
g(2, ["CLB","CLC"], ["CF_Clear"])
g(2, ["TCS_CF","TCS_!CF"], ["Bus3In"])
g(2, ["DAA","TCS_CF"], ["Bus1In"])

# Solo M2
g(2, ["CMA"], ["Acc_INV"])
g(2, ["CMC"], ["CF_Inv"])
g(2, ["DAA"], ["Bus2In"])
g(2, ["DAC"], ["Subtract_Set"])
g(2, ["FIM"], ["ScratchPad_Load"])
g(2, ["JMS"], ["Shift_Loading"])
g(2, ["JUN"], ["PC1_IR_Load"])
g(2, ["NOP"], ["Micro_Reset","PC_INC"])
g(2, ["RAL"], ["Shift_Direction"])
g(2, ["STC"], ["CF_Load"])
g(2, ["SUB"], ["Temp_INV"])
g(2, ["TCC"], ["CF_Out"])

# MICRO 3

# Micro_Reset + PC_INC merge
g(3, ["CLB","CLC","CMA","CMC","JCN_!jump","LD","LDM","STC","TCS_CF","TCS_!CF"],
     ["Micro_Reset","PC_INC"], "merged")
g(3, ["JCN_jump","JUN"], ["Micro_Reset"], "remainder")

# Acc_Load + EO_Out merge
g(3, ["ADD","DAA","DAC","IAC","SUB"], ["Acc_Load","EO_Out"], "merged")
g(3, ["RAL","RAR"], ["Acc_Load"], "remainder")

# CF_Load + Temp_Out identical merge
g(3, ["RAL","RAR"], ["CF_Load","Temp_Out"], "identical merge")

# ScratchPad_Load + Scratch_Reg_Select + INC_Out triple
g(3, ["INC","ISZ_F","ISZ_T"], ["INC_Out","ScratchPad_Load","Scratch_Reg_Select"], "triple merge")
g(3, ["FIM","XCH"], ["ScratchPad_Load"], "remainder")
g(3, ["XCH"], ["Scratch_Reg_Select"], "remainder")

g(3, ["FIN","ISZ_F","ISZ_T","JIN"], ["PC3_IR_Load"])
g(3, ["FIM","FIN","JIN"], ["ScratchPadWord2"])
g(3, ["FIN","JIN"], ["Scratch_Out"])
g(3, ["ISZ_F","ISZ_T"], ["PC2_IR_Load"])

# Solo M3
g(3, ["BBL"], ["PC_Stack_Load"])
g(3, ["JMS"], ["Stack_Load"])
g(3, ["TCC"], ["CF_Clear"])
g(3, ["XCH"], ["Acc_Out"])

# MICRO 4

# Micro_Reset + PC_INC merge
g(4, ["ADD","BBL","DAA","DAC","FIM","IAC","INC","ISZ_T","RAL","RAR","SUB","TCC"],
     ["Micro_Reset","PC_INC"], "merged")
g(4, ["ISZ_F","JIN"], ["Micro_Reset"], "remainder")

g(4, ["ADD","SUB"], ["CF_Load"])

# Solo M4
g(4, ["BBL"], ["Shift_Direction","Shift_Loading"])
g(4, ["DAC"], ["Subtract_Reset"])
g(4, ["FIN"], ["ScratchPadWord1","ScratchPad_Load"])
g(4, ["JMS"], ["PC1_IR_Load","PC2_IR_Load","PC3_IR_Load"])
g(4, ["SUB"], ["Temp_INV"])
g(4, ["XCH"], ["Acc_Load","Temp_Out"])

# MICRO 5
g(5, ["JMS","XCH"], ["Micro_Reset"])
g(5, ["FIN"], ["ScratchPadWord2","ScratchPad_Load"])
g(5, ["XCH"], ["PC_INC"])

# MICRO 6
g(6, ["FIN"], ["Micro_Reset","PC_INC"])


# PRINT
current_micro = None
total = 0
shared_c = 0
shared_j = 0
solo_c = 0
solo_j = 0

for num, micro, insts, sigs, cost, kind, note in gates:
    if micro != current_micro:
        if current_micro is not None:
            print()
        current_micro = micro
        print(f"MICRO {micro}:")
        print(f"| Gate | Instructions | Signals | JFETs | Note |")
        print(f"|------|-------------|---------|-------|------|")

    n_i = len(insts)
    n_s = len(sigs)
    check = 5 + (n_i - 1) + n_s
    flag = " *** MATH ERROR ***" if check != cost else ""

    print(f"| G{num:<2d} | {', '.join(insts)} ({n_i}) | {', '.join(sigs)} | {check} | {note} |")
    total += check
    if kind == "shared":
        shared_c += 1
        shared_j += check
    else:
        solo_c += 1
        solo_j += check

print()
print("SUMMARY:")
print(f"  Shared: {shared_c} gates, {shared_j} JFETs")
print(f"  Solo:   {solo_c} gates, {solo_j} JFETs")
print(f"  TOTAL:  {shared_c + solo_c} gates, {total} JFETs")
