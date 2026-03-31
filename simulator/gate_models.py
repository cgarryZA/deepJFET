"""Gate type utilities for the simulator.

Gate solvers and GateType live in model/gate.py. This module adds
truth table logic used by the simulator's precompute and optimizer.
"""

import itertools

from model import GateType, gate_input_count


def truth_table(gate_type: GateType) -> list:
    """Return the truth table for a gate type.

    Returns list of (input_tuple_of_bools, output_is_high).
    """
    n_in = gate_input_count(gate_type)
    table = []

    for combo in itertools.product([False, True], repeat=n_in):
        if gate_type == GateType.INV:
            out_high = not combo[0]
        elif gate_type in (GateType.NAND2, GateType.NAND3, GateType.NAND4):
            out_high = not all(combo)
        elif gate_type in (GateType.NOR2, GateType.NOR3, GateType.NOR4):
            out_high = not any(combo)
        else:
            raise ValueError(f"No truth table for {gate_type}")
        table.append((combo, out_high))

    return table
