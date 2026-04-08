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
    # subfolder scratchpad/ contains:
    #   Controls.asc       — address decode, always needed if any pair is used
    #   Bus1.asc           — data bus for pairs 1-4 (R0-R7)
    #   Bus2.asc           — data bus for pairs 5-8 (R8-R15)
    #   Pair 1.asc .. Pair 8.asc — one per register pair
    #
    # If no registers are used, the whole scratchpad is omitted.
    # If any are used, Controls is always included, plus the relevant
    # bus(es) and pair(s).
    ("Scratch_", {
        "folder": "scratchpad",
        "key": "register_pairs_used",  # set of pair indices 0-7 from profile

        # Always included when ANY part is needed
        "common": ["Controls.asc"],

        # Group dependencies: if any part in the group is needed, include the dep
        "groups": {
            "Bus1.asc": [0, 1, 2, 3],   # pairs 1-4 (indices 0-3)
            "Bus2.asc": [4, 5, 6, 7],   # pairs 5-8 (indices 4-7)
        },

        # Individual parts, keyed by pair index
        "parts": {
            0: "Pair 1.asc",   # R0, R1
            1: "Pair 2.asc",   # R2, R3
            2: "Pair 3.asc",   # R4, R5
            3: "Pair 4.asc",   # R6, R7
            4: "Pair 5.asc",   # R8, R9
            5: "Pair 6.asc",   # R10, R11
            6: "Pair 7.asc",   # R12, R13
            7: "Pair 8.asc",   # R14, R15
        },
    }),

    # -- Composable: Stack --
    # Not yet split into sub-parts. Using full stack.asc for now.
    ("Stack_", "stack.asc"),
]
