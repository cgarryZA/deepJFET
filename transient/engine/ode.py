"""ODE system for the 2-JFET gate transient simulation.

State variables: [v_a, v_b, v_out]
  v_a   — Node A (J1 drain / R1 / J2 gate)
  v_b   — Node B (J2 source / top of R2)
  v_out — Output node (between R2 and R3, drives load capacitance)

The ODE is derived from KCL at each node with capacitive currents:
  C * dv/dt = sum of resistive and device currents (the static KCL residual)

At DC steady state, dv/dt = 0 and we recover the static solution.
"""

import numpy as np
from model import jfet_ids, jfet_gate_current, thermal_voltage


def gate_ode(t, state, circuit):
    """Right-hand side of the transient ODE system.

    Args:
        t: Current time (s).
        state: [v_a, v_b, v_out] node voltages.
        circuit: Circuit dataclass with all parameters.

    Returns:
        [dv_a/dt, dv_b/dt, dv_out/dt]
    """
    v_a, v_b, v_out = state
    c = circuit
    v_in = c.v_in_func(t)
    vt = c.vt

    # -- J1: gate=v_in, source=GND(0), drain=v_a --
    i_j1 = jfet_ids(vgs=v_in, vds=v_a, j=c.j1)

    # J1 gate junction currents (internal voltages)
    vgs1_int = v_in - (i_j1 * c.j1.rs)
    vgd1_int = v_in - (v_a - i_j1 * c.j1.rd)
    igs_j1, igd_j1 = jfet_gate_current(vgs1_int, vgd1_int, c.j1, vt)

    # -- J2: gate=v_a, source=v_b, drain=V_POS --
    vgs2 = v_a - v_b
    vds2 = c.v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=c.j2)

    # J2 gate junction currents (internal voltages)
    vgs2_int = v_a - (v_b + i_j2 * c.j2.rs)
    vgd2_int = v_a - (c.v_pos - i_j2 * c.j2.rd)
    igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, c.j2, vt)

    # -- KCL residuals (= C * dv/dt) --

    # Node A: current from R1 + Igd_J1 - I_J1_channel - J2 gate current
    i_kcl_a = (c.v_pos - v_a) / c.r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)

    # Node B: J2 channel + Igs_J2 - current into R2
    i_kcl_b = i_j2 + igs_j2 - (v_b - v_out) / c.r2

    # V_OUT: current from R2 - current through R3 (no gate load current in ODE;
    # load is modeled as C_out capacitance)
    i_kcl_out = (v_b - v_out) / c.r2 - (v_out - c.v_neg) / c.r3

    # dv/dt = I_kcl / C
    dv_a = i_kcl_a / c.c_a
    dv_b = i_kcl_b / c.c_b
    dv_out = i_kcl_out / c.c_out

    return [dv_a, dv_b, dv_out]
