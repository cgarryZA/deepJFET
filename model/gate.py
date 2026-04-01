"""Forward solvers for 2-JFET logic gates (INV, NOR, NAND)."""

from enum import Enum
import numpy as np
from scipy.optimize import fsolve

from .jfet import (
    NChannelJFET, jfet_ids, jfet_gate_current, thermal_voltage, region_name,
)


# ---------------------------------------------------------------------------
# Gate type definitions
# ---------------------------------------------------------------------------

class GateType(Enum):
    INV = "INV"
    NAND2 = "NAND2"
    NAND3 = "NAND3"
    NAND4 = "NAND4"
    NOR2 = "NOR2"
    NOR3 = "NOR3"
    NOR4 = "NOR4"
    CUSTOM = "CUSTOM"


def gate_input_count(gt: GateType) -> int:
    """Return the number of inputs for a gate type."""
    return {
        GateType.INV: 1, GateType.NAND2: 2, GateType.NAND3: 3, GateType.NAND4: 4,
        GateType.NOR2: 2, GateType.NOR3: 3, GateType.NOR4: 4,
    }.get(gt, None)


# ---------------------------------------------------------------------------
# Inverter solver (single J1)
# ---------------------------------------------------------------------------

def solve_gate(
    v_in: float,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Solve all node voltages for the 2-JFET inverter gate.

    Circuit:
        +V_POS -- R1 -- Node_A -- J1(drain)   J1(source) -- GND   J1(gate) -- V_IN
        +V_POS -- J2(drain)   J2(source) -- Node_B -- R2 -- V_OUT -- R3 -- V_NEG
        J2(gate) -- Node_A

    Returns dict with all node voltages, branch currents, and region info.
    """
    r_load = r2 + r3
    vt = thermal_voltage(temp_c + 273.15)

    def equations(x):
        v_a, v_b = x

        i_j1 = jfet_ids(vgs=v_in, vds=v_a, j=j1)
        vgs1_int = v_in - (i_j1 * j1.rs)
        vgd1_int = v_in - (v_a - i_j1 * j1.rd)
        igs_j1, igd_j1 = jfet_gate_current(vgs1_int, vgd1_int, j1, vt)

        vgs2_ext = v_a - v_b
        vds2_ext = v_pos - v_b
        i_j2 = jfet_ids(vgs=vgs2_ext, vds=vds2_ext, j=j2)
        vgs2_int = v_a - (v_b + i_j2 * j2.rs)
        vgd2_int = v_a - (v_pos - i_j2 * j2.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, j2, vt)

        eq_a = (v_pos - v_a) / r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
        eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load
        return [eq_a, eq_b]

    best = None
    for guess in [(5.0, 3.0), (v_pos / 2.0, 0.0), (1.0, -1.0)]:
        sol, info, ier, msg = fsolve(equations, guess, full_output=True, maxfev=100)
        if ier == 1:
            best = sol
            break
    if best is None:
        best = sol
    v_a, v_b = best

    i_j1 = jfet_ids(vgs=v_in, vds=v_a, j=j1)
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
    igs_j2, igd_j2 = jfet_gate_current(vgs2, v_a - v_pos, j2, vt)
    i_r1 = (v_pos - v_a) / r1
    i_load = (v_b - v_neg) / r_load
    v_out = v_neg + i_load * r3

    return {
        "v_in": v_in, "v_a": v_a, "v_b": v_b, "v_out": v_out,
        "j1_gate": v_in, "j1_source": 0.0, "j1_drain": v_a,
        "j2_gate": v_a, "j2_source": v_b, "j2_drain": v_pos,
        "i_r1_mA": i_r1 * 1e3, "i_j1_mA": i_j1 * 1e3,
        "i_j2_mA": i_j2 * 1e3, "i_load_mA": i_load * 1e3,
        "i_gate_j2_mA": (igs_j2 + igd_j2) * 1e3,
        "j1_region": region_name(v_in, v_a, j1),
        "j2_region": region_name(vgs2, vds2, j2),
    }


# ---------------------------------------------------------------------------
# NOR solver (parallel J1s)
# ---------------------------------------------------------------------------

def solve_nor(
    v_ins: list,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Solve NOR gate: parallel J1s sharing Node A.

    Each J1 has gate=v_ins[k], source=GND, drain=v_a.
    KCL at Node A sums all J1 drain currents.
    """
    r_load = r2 + r3
    vt = thermal_voltage(temp_c + 273.15)

    def equations(x):
        v_a, v_b = x

        i_j1_total = 0.0
        igd_j1_total = 0.0
        for v_in_k in v_ins:
            i_k = jfet_ids(vgs=v_in_k, vds=v_a, j=j1)
            i_j1_total += i_k
            vgs_int = v_in_k - (i_k * j1.rs)
            vgd_int = v_in_k - (v_a - i_k * j1.rd)
            _, igd_k = jfet_gate_current(vgs_int, vgd_int, j1, vt)
            igd_j1_total += igd_k

        vgs2 = v_a - v_b
        vds2 = v_pos - v_b
        i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
        vgs2_int = v_a - (v_b + i_j2 * j2.rs)
        vgd2_int = v_a - (v_pos - i_j2 * j2.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, j2, vt)

        eq_a = (v_pos - v_a) / r1 + igd_j1_total - i_j1_total - (igs_j2 + igd_j2)
        eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load
        return [eq_a, eq_b]

    best = None
    for guess in [(5.0, 3.0), (v_pos / 2.0, 0.0), (1.0, -1.0)]:
        sol, info, ier, msg = fsolve(equations, guess, full_output=True, maxfev=100)
        if ier == 1:
            best = sol
            break
    if best is None:
        best = sol
    v_a, v_b = best

    i_j1_total = sum(jfet_ids(vgs=v_k, vds=v_a, j=j1) for v_k in v_ins)
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
    i_r1 = (v_pos - v_a) / r1
    i_load = (v_b - v_neg) / r_load
    v_out = v_neg + i_load * r3

    return {
        "v_ins": list(v_ins), "v_a": v_a, "v_b": v_b, "v_out": v_out,
        "i_r1_mA": i_r1 * 1e3, "i_j1_total_mA": i_j1_total * 1e3,
        "i_j2_mA": i_j2 * 1e3, "i_load_mA": i_load * 1e3,
        "j2_region": region_name(vgs2, vds2, j2),
    }


# ---------------------------------------------------------------------------
# NAND solver (series J1s)
# ---------------------------------------------------------------------------

def solve_nand(
    v_ins: list,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Solve NAND gate: series J1s between Node A and GND.

    J1 chain: Node A -- J1[0] -- v_mid[0] -- J1[1] -- ... -- J1[N-1] -- GND
    J1[k]: gate=v_ins[k], drain=node_above[k], source=node_below[k]
    """
    n = len(v_ins)
    r_load = r2 + r3
    vt = thermal_voltage(temp_c + 273.15)
    n_mids = n - 1

    def equations(x):
        v_a = x[0]
        v_b = x[1]
        v_mids = list(x[2:2 + n_mids])

        nodes = [v_a] + v_mids + [0.0]

        i_j1 = []
        igd_j1_top = 0.0
        for k in range(n):
            vgs_k = v_ins[k] - nodes[k + 1]
            vds_k = nodes[k] - nodes[k + 1]
            i_k = jfet_ids(vgs=vgs_k, vds=vds_k, j=j1)
            i_j1.append(i_k)

            if k == 0:
                vgs_int = v_ins[k] - (nodes[k + 1] + i_k * j1.rs)
                vgd_int = v_ins[k] - (nodes[k] - i_k * j1.rd)
                _, igd_k = jfet_gate_current(vgs_int, vgd_int, j1, vt)
                igd_j1_top = igd_k

        vgs2 = v_a - v_b
        vds2 = v_pos - v_b
        i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
        vgs2_int = v_a - (v_b + i_j2 * j2.rs)
        vgd2_int = v_a - (v_pos - i_j2 * j2.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, j2, vt)

        eq_a = (v_pos - v_a) / r1 + igd_j1_top - i_j1[0] - (igs_j2 + igd_j2)
        eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load

        eq_mids = []
        for k in range(n_mids):
            eq_mids.append(i_j1[k] - i_j1[k + 1])

        return [eq_a, eq_b] + eq_mids

    best = None
    for v_a_g, v_b_g in [(5.0, 3.0), (v_pos / 2.0, 0.0), (1.0, -1.0), (10.0, 5.0)]:
        mids_g = [v_a_g * (n_mids - k) / n for k in range(n_mids)]
        guess = [v_a_g, v_b_g] + mids_g
        sol, info, ier, msg = fsolve(equations, guess, full_output=True, maxfev=200)
        if ier == 1:
            best = sol
            break
    if best is None:
        best = sol

    v_a = best[0]
    v_b = best[1]
    v_mids = list(best[2:2 + n_mids])
    nodes = [v_a] + v_mids + [0.0]

    i_j1_chain = jfet_ids(vgs=v_ins[0] - nodes[1], vds=nodes[0] - nodes[1], j=j1)
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
    i_r1 = (v_pos - v_a) / r1
    i_load = (v_b - v_neg) / r_load
    v_out = v_neg + i_load * r3

    return {
        "v_ins": list(v_ins), "v_a": v_a, "v_b": v_b, "v_out": v_out,
        "v_mids": v_mids,
        "i_r1_mA": i_r1 * 1e3, "i_j1_mA": i_j1_chain * 1e3,
        "i_j2_mA": i_j2 * 1e3, "i_load_mA": i_load * 1e3,
        "j2_region": region_name(vgs2, vds2, j2),
    }


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def solve_any_gate(
    gate_type: GateType,
    v_ins: list,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Dispatch to the appropriate solver based on gate type."""
    if gate_type == GateType.INV:
        if len(v_ins) != 1:
            raise ValueError(f"INV expects 1 input, got {len(v_ins)}")
        return solve_gate(v_ins[0], v_pos, v_neg, r1, r2, r3, j1, j2, temp_c)

    if gate_type in (GateType.NOR2, GateType.NOR3, GateType.NOR4):
        return solve_nor(v_ins, v_pos, v_neg, r1, r2, r3, j1, j2, temp_c)

    if gate_type in (GateType.NAND2, GateType.NAND3, GateType.NAND4):
        return solve_nand(v_ins, v_pos, v_neg, r1, r2, r3, j1, j2, temp_c)

    if gate_type == GateType.CUSTOM:
        raise ValueError("Use solve_network() for CUSTOM gate types")

    raise ValueError(f"Unsupported gate type: {gate_type}")


# ---------------------------------------------------------------------------
# Generic network solver (arbitrary series/parallel topology)
# ---------------------------------------------------------------------------

def solve_network(
    network,
    v_ins_dict: dict,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Solve a gate with arbitrary pull-down network topology.

    Args:
        network: PulldownNetwork (Leaf, Series, or Parallel)
        v_ins_dict: dict of input_name -> voltage
        v_pos, v_neg: supply rails
        r1, r2, r3: resistors
        j1, j2: JFET models (all J1s identical)
        temp_c: temperature

    Returns dict with v_a, v_b, v_out, v_mids, currents, etc.
    """
    from .network import (
        count_midpoints, network_current_with_gate,
        network_current,
    )

    r_load = r2 + r3
    vt = thermal_voltage(temp_c + 273.15)
    n_mids = count_midpoints(network)
    n_vars = 2 + n_mids  # v_a, v_b, midpoints

    def equations(x):
        v_a = x[0]
        v_b = x[1]
        mids_list = list(x[2:])

        # J1 network current + gate-drain current of topmost JFET
        residuals = []
        mids_copy = list(mids_list)
        i_j1, igd_j1 = network_current_with_gate(
            network, v_a, 0.0, v_ins_dict, j1,
            jfet_ids, jfet_gate_current, vt,
            mids_copy, residuals,
        )

        # J2 (load transistor) — same for all topologies
        vgs2 = v_a - v_b
        vds2 = v_pos - v_b
        i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
        vgs2_int = v_a - (v_b + i_j2 * j2.rs)
        vgd2_int = v_a - (v_pos - i_j2 * j2.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, j2, vt)

        # KCL at Node A
        eq_a = (v_pos - v_a) / r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
        # KCL at Node B
        eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load

        return [eq_a, eq_b] + residuals

    # Try multiple initial guesses
    best = None
    for v_a_g, v_b_g in [(5.0, 3.0), (v_pos / 2.0, 0.0), (1.0, -1.0), (10.0, 5.0)]:
        mids_g = [v_a_g * (n_mids - k) / max(n_mids, 1)
                  for k in range(n_mids)]
        guess = [v_a_g, v_b_g] + mids_g
        sol, info, ier, msg = fsolve(equations, guess, full_output=True,
                                      maxfev=200)
        if ier == 1:
            best = sol
            break
    if best is None:
        best = sol

    v_a = best[0]
    v_b = best[1]
    v_mids = list(best[2:2 + n_mids])

    # Extract results
    mids_copy = list(v_mids)
    residuals = []
    i_j1 = network_current(network, v_a, 0.0, v_ins_dict, j1,
                           jfet_ids, mids_copy, residuals)

    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=j2)
    i_r1 = (v_pos - v_a) / r1
    i_load = (v_b - v_neg) / r_load
    v_out = v_neg + i_load * r3

    return {
        "v_ins": v_ins_dict, "v_a": v_a, "v_b": v_b, "v_out": v_out,
        "v_mids": v_mids,
        "i_r1_mA": i_r1 * 1e3, "i_j1_mA": i_j1 * 1e3,
        "i_j2_mA": i_j2 * 1e3, "i_load_mA": i_load * 1e3,
        "j2_region": region_name(vgs2, vds2, j2),
    }


# ---------------------------------------------------------------------------
# Sweep utility
# ---------------------------------------------------------------------------

def sweep(
    v_in_range: np.ndarray,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    temp_c: float = 27.0,
) -> dict:
    """Sweep V_IN over a range and collect all results."""
    keys = [
        "v_in", "v_a", "v_b", "v_out",
        "j1_gate", "j1_source", "j1_drain",
        "j2_gate", "j2_source", "j2_drain",
    ]
    results = {k: [] for k in keys}
    for v_in in v_in_range:
        r = solve_gate(v_in, v_pos, v_neg, r1, r2, r3, j1, j2, temp_c)
        for k in keys:
            results[k].append(r[k])
    return {k: np.array(v) for k, v in results.items()}
