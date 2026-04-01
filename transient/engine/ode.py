"""ODE system for N-input JFET gate transient simulation.

Supports arbitrary pull-down network topologies (INV, NOR-N, NAND-N, and
any series/parallel combination).

State variables: [v_a, v_b, v_out, mid_0, mid_1, ...]
  v_a   — Node A (J1 drains / R1 / J2 gate)
  v_b   — Node B (J2 source / top of R2)
  v_out — Output node (between R2 and R3)
  mid_k — internal midpoint voltages from series connections in the topology

The ODE is C * dv/dt = KCL residual at each node.
"""

from model import jfet_ids, jfet_gate_current
from model.network import (
    Leaf, Series, Parallel, PulldownNetwork,
    count_midpoints, network_current_with_gate, network_current,
)


def _network_current_ode(net, v_top, v_bot, v_inputs, j1, vt, midpoints, residuals):
    """Compute J1 network current + igd for the ODE.

    Same interface as network_current_with_gate but adapted for scalar ODE use.
    """
    return network_current_with_gate(
        net, v_top, v_bot, v_inputs, j1,
        jfet_ids, jfet_gate_current, vt,
        midpoints, residuals,
    )


def gate_ode(t, state, circuit):
    """RHS of the transient ODE for any gate topology.

    Args:
        t: Current time (s).
        state: node voltage vector.
        circuit: Circuit dataclass with network topology.

    Returns:
        derivative vector (same length as state).
    """
    c = circuit
    vt = c.vt

    v_a = state[0]
    v_b = state[1]
    v_out = state[2]

    # Get input voltages at time t
    v_ins_list = [fn(t) for fn in c.v_in_funcs]

    # Build input dict from names
    names = c.input_names
    v_ins_dict = {names[i]: v_ins_list[i] for i in range(len(names))}

    # Midpoint voltages from state
    n_mids = c.n_midpoints
    v_mids = list(state[3:3 + n_mids])

    # --- J1 network current ---
    midpoints_copy = list(v_mids)
    mid_residuals = []
    i_j1, igd_j1 = _network_current_ode(
        c.network, v_a, 0.0, v_ins_dict, c.j1, vt,
        midpoints_copy, mid_residuals,
    )

    # --- J2 (same for all topologies) ---
    vgs2 = v_a - v_b
    vds2 = c.v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=c.j2)
    vgs2_int = v_a - (v_b + i_j2 * c.j2.rs)
    vgd2_int = v_a - (c.v_pos - i_j2 * c.j2.rd)
    igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, c.j2, vt)

    # --- KCL at each node ---
    i_kcl_a = (c.v_pos - v_a) / c.r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
    i_kcl_b = i_j2 + igs_j2 - (v_b - v_out) / c.r2
    i_kcl_out = (v_b - v_out) / c.r2 - (v_out - c.v_neg) / c.r3

    dv_a = i_kcl_a / c.c_a
    dv_b = i_kcl_b / c.c_b
    dv_out = i_kcl_out / c.c_out

    derivs = [dv_a, dv_b, dv_out]

    # Midpoint derivatives from KCL residuals
    c_mid = c.caps.cgd0 + c.caps.cgs0  # capacitance at each midpoint
    for res in mid_residuals:
        derivs.append(res / c_mid)

    return derivs
