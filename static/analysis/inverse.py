"""Inverse solver — find resistor values for desired output logic levels."""

import numpy as np
from scipy.optimize import minimize

from model import NChannelJFET, solve_gate


def find_resistors(
    v_in_low: float,
    v_in_high: float,
    v_out_target_when_low: float,
    v_out_target_when_high: float,
    v_pos: float,
    v_neg: float,
    jfet: NChannelJFET,
    temp_c: float = 27.0,
    r_bounds: tuple = (100.0, 200_000.0),
    r0: tuple = (50_000.0, 1_000.0, 4_500.0),
) -> dict:
    """Find R1, R2, R3 that produce the desired output logic levels.

    Returns dict with optimized R1/R2/R3, actual output levels, and error.
    """
    def cost(x):
        r1, r2, r3 = x
        try:
            lo = solve_gate(v_in_low, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
            hi = solve_gate(v_in_high, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
        except Exception:
            return 1e6
        return ((lo["v_out"] - v_out_target_when_low) ** 2 +
                (hi["v_out"] - v_out_target_when_high) ** 2)

    result = minimize(cost, r0, method="L-BFGS-B",
                      bounds=[r_bounds, r_bounds, r_bounds],
                      options={"ftol": 1e-12, "maxiter": 500})

    r1, r2, r3 = result.x
    lo = solve_gate(v_in_low, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
    hi = solve_gate(v_in_high, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)

    return {
        "r1": r1, "r2": r2, "r3": r3,
        "v_out_at_low_input": lo["v_out"],
        "v_out_at_high_input": hi["v_out"],
        "error_low_mV": (lo["v_out"] - v_out_target_when_low) * 1e3,
        "error_high_mV": (hi["v_out"] - v_out_target_when_high) * 1e3,
        "converged": result.success,
    }
