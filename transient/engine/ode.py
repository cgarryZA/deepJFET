"""ODE system for N-input JFET gate transient simulation.

Supports INV (1 input), NOR-N (parallel J1s), and NAND-N (series J1s).

INV / NOR-N state variables: [v_a, v_b, v_out]
  v_a   — Node A (J1 drains / R1 / J2 gate)
  v_b   — Node B (J2 source / top of R2)
  v_out — Output node (between R2 and R3)

NAND-N state variables: [v_a, v_b, v_out, v_mid_0, ..., v_mid_{N-2}]
  v_mid_k — midpoint between J1[k] and J1[k+1] in series chain

The ODE is C * dv/dt = KCL residual at each node.
At DC steady state, dv/dt = 0 and we recover the static solution.
"""

from model import jfet_ids, jfet_gate_current


def gate_ode(t, state, circuit):
    """RHS of the transient ODE for any gate type.

    Args:
        t: Current time (s).
        state: node voltage vector (length depends on gate type).
        circuit: Circuit dataclass.

    Returns:
        derivative vector (same length as state).
    """
    c = circuit
    vt = c.vt
    gate_type = c.gate_type  # "INV", "NOR", "NAND"
    n_inputs = c.n_inputs

    v_a = state[0]
    v_b = state[1]
    v_out = state[2]

    # Get input voltages at time t
    v_ins = [fn(t) for fn in c.v_in_funcs]

    if gate_type == "NAND" and n_inputs > 1:
        # --- NAND: series J1 chain with midpoint nodes ---
        v_mids = list(state[3:])  # N-1 midpoints
        nodes = [v_a] + v_mids + [0.0]  # [v_a, mid_0, ..., mid_{N-2}, GND]

        # Compute chain currents
        i_j1 = []
        for k in range(n_inputs):
            vgs_k = v_ins[k] - nodes[k + 1]
            vds_k = nodes[k] - nodes[k + 1]
            i_k = jfet_ids(vgs=vgs_k, vds=vds_k, j=c.j1)
            i_j1.append(i_k)

        # Gate current of topmost J1 (for KCL at node A)
        i_top = i_j1[0]
        vgs_int = v_ins[0] - (nodes[1] + i_top * c.j1.rs)
        vgd_int = v_ins[0] - (v_a - i_top * c.j1.rd)
        _, igd_j1_top = jfet_gate_current(vgs_int, vgd_int, c.j1, vt)

        i_j1_into_a = i_top
        igd_j1_into_a = igd_j1_top

    else:
        # --- INV / NOR: parallel J1s, all drain to node A, source to GND ---
        i_j1_into_a = 0.0
        igd_j1_into_a = 0.0
        for v_in_k in v_ins:
            i_k = jfet_ids(vgs=v_in_k, vds=v_a, j=c.j1)
            i_j1_into_a += i_k
            vgs_int = v_in_k - (i_k * c.j1.rs)
            vgd_int = v_in_k - (v_a - i_k * c.j1.rd)
            _, igd_k = jfet_gate_current(vgs_int, vgd_int, c.j1, vt)
            igd_j1_into_a += igd_k

    # --- J2: gate=v_a, source=v_b, drain=V_POS (same for all gate types) ---
    vgs2 = v_a - v_b
    vds2 = c.v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=c.j2)
    vgs2_int = v_a - (v_b + i_j2 * c.j2.rs)
    vgd2_int = v_a - (c.v_pos - i_j2 * c.j2.rd)
    igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, c.j2, vt)

    # --- KCL at each node ---

    # Node A
    i_kcl_a = (c.v_pos - v_a) / c.r1 + igd_j1_into_a - i_j1_into_a - (igs_j2 + igd_j2)
    dv_a = i_kcl_a / c.c_a

    # Node B
    i_kcl_b = i_j2 + igs_j2 - (v_b - v_out) / c.r2
    dv_b = i_kcl_b / c.c_b

    # Output node
    i_kcl_out = (v_b - v_out) / c.r2 - (v_out - c.v_neg) / c.r3
    dv_out = i_kcl_out / c.c_out

    derivs = [dv_a, dv_b, dv_out]

    # NAND midpoint nodes
    if gate_type == "NAND" and n_inputs > 1:
        for k in range(n_inputs - 1):
            # KCL at midpoint k: current from J1[k] above minus J1[k+1] below
            i_kcl_mid = i_j1[k] - i_j1[k + 1]
            # Capacitance at midpoint: Cgd of J1[k] + Cgs of J1[k+1]
            c_mid = c.caps.cgd0 + c.caps.cgs0
            dv_mid = i_kcl_mid / c_mid
            derivs.append(dv_mid)

    return derivs
