"""4004 CPU design configuration.

Hand-designed gate parameters for the SiC JFET implementation of the
Intel 4004 architecture. Modify R values here to tune individual gate
types without affecting the rest of the design.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import NChannelJFET, JFETCapacitance

# -- JFET device model (DR NJF from SPICE .model card) --
JFET_MODEL = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)

# -- Junction capacitances --
CAPS = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

# -- Supply rails --
V_POS = 24.0
V_NEG = -20.0

# -- Operating temperature (C) --
TEMP_C = 27.0

# -- Target clock frequency (Hz) --
F_TARGET = 100e3

# -- Hand-designed resistor values per gate type --
# Each gate type gets its own R1, R2, R3 (ohms).
# These are the values you've validated in LTSpice.
GATES = {
    "INV":   {"r1": 50e3, "r2": 1e3, "r3": 4.5e3},
    "NAND2": {"r1": 50e3, "r2": 1e3, "r3": 4.5e3},
    "NOR2":  {"r1": 50e3, "r2": 1e3, "r3": 4.5e3},
}

# -- Sub-components --
# Two kinds of entry:
#
# 1. Fixed component — a single .asc file, always included.
#      ("ALU_", "alu.asc")
#
# 2. Composable component — a subfolder of parts assembled at build time.
#      ("Scratch_", {"folder": "scratchpad", "parts": {...}})
#
#    The build tool reads the subfolder, finds which parts are needed from
#    the resource profile, and concatenates only those + any "common" parts.
#
COMPONENTS = [
    ("ALU_",     "alu.asc"),
    ("IR_",      "instruction_register.asc"),
    ("MI_",      "micro_instructions.asc"),
    ("Cont_",    "controls.asc"),
    ("Pin_",     "pins.asc"),
    ("PC_",      "program_counter.asc"),
    ("Counter_", "step_counter.asc"),

    # -- Composable: Scratchpad --
    # The subfolder scratchpad/ contains one .asc per register pair plus
    # a common.asc for shared bus wiring, decoders, etc.
    # The analyzer determines which register pairs the program uses;
    # only those pair .asc files are included in the build.
    ("Scratch_", {
        "folder": "scratchpad",
        # "common" parts are always included (bus wiring, address decode, etc.)
        "common": [
            # "common.asc",  # uncomment when you create it
        ],
        # "parts" maps a profile field to individual .asc files.
        # Key = field name from ResourceProfile (set of ints)
        # Value = dict mapping each int to its .asc filename
        "parts": {
            "key": "register_pairs_used",
            "files": {
                0: "pair_0.asc",   # R0, R1
                1: "pair_1.asc",   # R2, R3
                2: "pair_2.asc",   # R4, R5
                3: "pair_3.asc",   # R6, R7
                4: "pair_4.asc",   # R8, R9
                5: "pair_5.asc",   # R10, R11
                6: "pair_6.asc",   # R12, R13
                7: "pair_7.asc",   # R14, R15
            },
        },
    }),

    # -- Composable: Stack --
    ("Stack_", {
        "folder": "stack",
        "common": [
            # "common.asc",
        ],
        "parts": {
            # Stack levels: program uses JMS/BBL, analyzer tracks max depth.
            # key is a set-like field; we synthesise it from stack_depth_needed.
            "key": "stack_levels_needed",   # [0], [0,1], or [0,1,2]
            "files": {
                0: "level_0.asc",
                1: "level_1.asc",
                2: "level_2.asc",
            },
        },
    }),
]
