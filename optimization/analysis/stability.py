"""Contraction mapping stability analysis for JFET logic gates.

Tests whether a gate's logic levels are stable fixed points: if you chain
gates infinitely, do the voltage levels converge or diverge?

This is useful for investigating gate topologies during redesign — checking
whether a proposed R1/R2/R3 + rail combination produces non-degenerate,
self-consistent logic levels.

TODO: Extend to handle mixed gate chains (e.g. INV -> NAND2 -> INV)
TODO: Add numerical derivative of double-inversion transfer function
      to quantify contraction rate (how fast perturbations shrink)
TODO: Add voltage sweep to map the full f(f(v)) transfer curve and
      visualize the fixed points and their basins of attraction
TODO: Add temperature sweep to check stability across operating range
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from model import NChannelJFET, solve_gate


def find_fixed_points_inv(v_pos, v_neg, r1, r2, r3, jfet, temp_c=27.0,
                          n_iters=30):
    """Find stable HIGH and LOW fixed points for an inverter by iterating
    double-inversion.

    Starts from two initial voltages (0V and -5V) and iterates:
      v -> INV -> v1 -> INV -> v2 (repeat)

    After n_iters, checks if the fixed points are attractive (perturbations
    shrink on each pass).

    Returns (v_high, v_low, stable).
    """
    def double_inv(v):
        res1 = solve_gate(v, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
        res2 = solve_gate(res1["v_out"], v_pos, v_neg, r1, r2, r3,
                          jfet, jfet, temp_c)
        return res2["v_out"], res1["v_out"]

    # Find HIGH fixed point (start high)
    v = 0.0
    for _ in range(n_iters):
        v, _ = double_inv(v)
    v_high = v

    # Find LOW fixed point (start low)
    v = -5.0
    for _ in range(n_iters):
        v, _ = double_inv(v)
    v_low = v

    # Stability: perturb and check contraction
    eps = 0.05
    v_h_perturbed, _ = double_inv(v_high + eps)
    v_l_perturbed, _ = double_inv(v_low - eps)
    stable_h = abs(v_h_perturbed - v_high) < eps
    stable_l = abs(v_l_perturbed - v_low) < eps

    return v_high, v_low, (stable_h and stable_l)


def find_fixed_points_any(gate_type, v_pos, v_neg, r1, r2, r3, jfet,
                          temp_c=27.0, n_iters=30):
    """Find stable fixed points for any gate type.

    Requires simulator.gate_models to be importable.
    """
    from model import gate_input_count, solve_any_gate

    n_in = gate_input_count(gate_type)

    def double_inv(v):
        res1 = solve_any_gate(gate_type, [v] * n_in,
                              v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
        res2 = solve_any_gate(gate_type, [res1["v_out"]] * n_in,
                              v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
        return res2["v_out"], res1["v_out"]

    v = 0.0
    for _ in range(n_iters):
        v, _ = double_inv(v)
    v_high = v

    v = -5.0
    for _ in range(n_iters):
        v, _ = double_inv(v)
    v_low = v

    eps = 0.05
    v_h_perturbed, _ = double_inv(v_high + eps)
    v_l_perturbed, _ = double_inv(v_low - eps)
    stable_h = abs(v_h_perturbed - v_high) < eps
    stable_l = abs(v_l_perturbed - v_low) < eps

    return v_high, v_low, (stable_h and stable_l)
