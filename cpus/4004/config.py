"""4004 CPU design configuration.

Hand-designed gate parameters for the SiC JFET implementation of the
Intel 4004 architecture. Modify R values here to tune individual gate
types without affecting the rest of the design.

Two JFET parameterisations are maintained side-by-side:

  JFET_MODEL_SUPPLIER_v1     — proprietary supplier parameter card.
                                Validated baseline at 27 °C; used for
                                all results in the current manuscript.
  JFET_MODEL_NEUDECK_2016    — public NASA Glenn parameter card from
                                Neudeck, Spry, Chen 2016. Used for the
                                temperature-sweep extension and for any
                                fully-reproducible-from-public-sources
                                manuscript variant.

The active model (the bound name `JFET_MODEL`) is selected by the
environment variable `JFET_MODEL`:

    JFET_MODEL=supplier  python tools/build_cpu.py 4004   # default
    JFET_MODEL=neudeck   python tools/build_cpu.py 4004

This lets the existing supplier-model results in the current paper be
reproduced from this repo at any time without committing a model
swap, while the new temperature-sweep campaign can run with a
publicly-citable card by setting the env var.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import NChannelJFET, JFETCapacitance

# ── Supplier-provided proprietary parameterisation (current paper) ──────
#
# The DR-NJF card from the project's supplier SPICE deck. Validated
# baseline; not redistributable beyond academic reproduction of this
# work (see paper/reproducibility/README.md for the provenance
# statement).
JFET_MODEL_SUPPLIER_v1 = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)

# ── Public NASA Glenn parameterisation (Neudeck/Spry/Chen 2016) ─────────
#
# *** PLACEHOLDER — pending paper acquisition ***
#
# Source: P. Neudeck, D. Spry, L. Chen, "First-Order SPICE Modeling of
# Extreme-Temperature 4H-SiC JFET Integrated Circuits", NASA Glenn
# Research Center, ~2016. Semantic Scholar paper ID:
#   81183fea1000d3f2ddee9c14fd3402ceaa70c5b5
#
# Values below are TBD; the supplier card is duplicated here as a
# stand-in so the module still imports cleanly. Replace with the
# real Neudeck parameters once the model card has been extracted
# from the paper's text / supplementary deck.
JFET_MODEL_NEUDECK_2016 = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,   # TBD: from Neudeck 2016
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,   # TBD
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,   # TBD
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,   # TBD
)
# The temperature range the Neudeck card has been validated against
# (from the originating paper). Used by tools/temperature_sweep.py to
# refuse runs outside the device's qualified range. Update once the
# paper is acquired.
JFET_MODEL_NEUDECK_2016_VALIDATED_RANGE_C = (None, None)  # TBD: (Tmin, Tmax)

# ── Active selection ────────────────────────────────────────────────────
_MODEL_NAME = os.environ.get("JFET_MODEL", "supplier").lower()
if _MODEL_NAME in ("neudeck", "neudeck2016", "neudeck_2016", "nasa"):
    JFET_MODEL = JFET_MODEL_NEUDECK_2016
    JFET_MODEL_PROVENANCE = "Neudeck/Spry/Chen 2016 (NASA Glenn, public)"
elif _MODEL_NAME in ("supplier", "default", "v1", ""):
    JFET_MODEL = JFET_MODEL_SUPPLIER_v1
    JFET_MODEL_PROVENANCE = "supplier-provided proprietary v1"
else:
    raise ValueError(
        f"Unknown JFET_MODEL env value {_MODEL_NAME!r}; "
        f"expected 'supplier' or 'neudeck'"
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
    # subfolder stack/ contains:
    #   Controls.asc  — always needed
    #   Bus.asc       — always needed
    #   Level 1.asc   — always needed (program counter return)
    #   Level 2.asc   — needed if JMS is used (1st subroutine call)
    #   Level 3.asc   — needed if nested JMS before BBL (2nd nesting level)
    #
    # stack_depth_needed from the analyzer:
    #   0 = no subroutines  -> still need Level 1 (PC lives here)
    #   1 = one JMS/BBL     -> need Level 2
    #   2 = nested JMS      -> need Level 2 + Level 3
    ("Stack_", {
        "folder": "stack",
        "key": "stack_depth_needed",

        # Always included
        "common": ["Controls.asc", "Bus.asc", "Level 1.asc"],

        # No groups needed — levels are independent
        "groups": {},

        # Level 2 needed at depth >= 1, Level 3 at depth >= 2
        "parts": {
            1: "Level 2.asc",
            2: "Level 3.asc",
        },
    }),
]
