"""Fan-out and cascading analysis for the 2-JFET gate."""

import numpy as np
from scipy.optimize import fsolve

from model import (
    NChannelJFET, jfet_ids, jfet_gate_current, thermal_voltage, region_name,
    solve_gate,
)


def solve_gate_with_fanout(
    v_in: float,
    v_pos: float, v_neg: float,
    r1: float, r2: float, r3: float,
    j1: NChannelJFET, j2: NChannelJFET,
    n_load: int = 0,
    load_jfet: NChannelJFET = None,
    temp_c: float = 27.0,
) -> dict:
    """Solve the driving gate with N load gates connected to V_OUT."""
    vt = thermal_voltage(temp_c + 273.15)
    if load_jfet is None:
        load_jfet = j1

    if n_load == 0:
        return solve_gate(v_in, v_pos, v_neg, r1, r2, r3, j1, j2, temp_c)

    def equations(x):
        v_a, v_b, v_out = x[0], x[1], x[2]
        v_a_loads = x[3:3 + n_load]

        i_j1 = jfet_ids(vgs=v_in, vds=v_a, j=j1)
        vgs1_int = v_in - (i_j1 * j1.rs)
        vgd1_int = v_in - (v_a - i_j1 * j1.rd)
        igs_j1, igd_j1 = jfet_gate_current(vgs1_int, vgd1_int, j1, vt)

        i_j2 = jfet_ids(vgs=v_a - v_b, vds=v_pos - v_b, j=j2)
        vgs2_int = v_a - (v_b + i_j2 * j2.rs)
        vgd2_int = v_a - (v_pos - i_j2 * j2.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, j2, vt)

        eq_a = (v_pos - v_a) / r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
        eq_b = i_j2 + igs_j2 - (v_b - v_out) / r2

        total_ig_load = 0.0
        load_eqs = []
        for k in range(n_load):
            v_a_lk = v_a_loads[k]
            i_j1_lk = jfet_ids(vgs=v_out, vds=v_a_lk, j=load_jfet)
            vgs_lk = v_out - (i_j1_lk * load_jfet.rs)
            vgd_lk = v_out - (v_a_lk - i_j1_lk * load_jfet.rd)
            igs_lk, igd_lk = jfet_gate_current(vgs_lk, vgd_lk, load_jfet, vt)
            total_ig_load += (igs_lk + igd_lk)
            load_eqs.append((v_pos - v_a_lk) / r1 - i_j1_lk + igd_lk)

        eq_vout = (v_b - v_out) / r2 - (v_out - v_neg) / r3 - total_ig_load
        return [eq_a, eq_b, eq_vout] + load_eqs

    x0 = [5.0, 3.0, -1.0] + [5.0] * n_load
    best = None
    for scale in [1.0, 0.5, 2.0]:
        guess = [5.0 * scale, 3.0 * scale, -1.0] + [5.0 * scale] * n_load
        sol, info, ier, msg = fsolve(equations, guess, full_output=True)
        if ier == 1:
            best = sol
            break
    if best is None:
        best = sol
    v_a, v_b, v_out = best[0], best[1], best[2]

    i_j1 = jfet_ids(vgs=v_in, vds=v_a, j=j1)
    i_j2 = jfet_ids(vgs=v_a - v_b, vds=v_pos - v_b, j=j2)
    i_r2 = (v_b - v_out) / r2
    i_r3 = (v_out - v_neg) / r3

    return {
        "v_in": v_in, "v_a": v_a, "v_b": v_b, "v_out": v_out,
        "i_r1_mA": (v_pos - v_a) / r1 * 1e3, "i_j1_mA": i_j1 * 1e3,
        "i_j2_mA": i_j2 * 1e3, "i_r2_mA": i_r2 * 1e3, "i_r3_mA": i_r3 * 1e3,
        "i_gate_load_mA": (i_r2 - i_r3) * 1e3,
        "j1_region": region_name(v_in, v_a, j1),
        "j2_region": region_name(v_a - v_b, v_pos - v_b, j2),
        "n_load": n_load,
    }


def cascade_test(v_in, v_pos, v_neg, r1, r2, r3, jfet, temp_c=27.0):
    """Two gates in series: double inversion should give identity."""
    res1 = solve_gate_with_fanout(v_in, v_pos, v_neg, r1, r2, r3, jfet, jfet,
                                   n_load=1, load_jfet=jfet, temp_c=temp_c)
    res2 = solve_gate_with_fanout(res1["v_out"], v_pos, v_neg, r1, r2, r3,
                                   jfet, jfet, n_load=0, temp_c=temp_c)
    return {"v_in": v_in, "gate1_v_out": res1["v_out"],
            "gate2_v_out": res2["v_out"], "gate1": res1, "gate2": res2}


def fanout_sweep(v_pos, v_neg, r1, r2, r3, jfet, v_in_low, v_in_high,
                 max_fanout=20, temp_c=27.0):
    """Sweep fan-out from 0 to max_fanout and measure degradation."""
    results = {"n": [], "v_out_high": [], "v_out_low": [], "swing": [],
               "i_gate_load_high_mA": [], "i_gate_load_low_mA": []}
    for n in range(0, max_fanout + 1):
        try:
            lo = solve_gate_with_fanout(v_in_low, v_pos, v_neg, r1, r2, r3,
                                        jfet, jfet, n, jfet, temp_c)
            hi = solve_gate_with_fanout(v_in_high, v_pos, v_neg, r1, r2, r3,
                                        jfet, jfet, n, jfet, temp_c)
            results["n"].append(n)
            results["v_out_high"].append(lo["v_out"])
            results["v_out_low"].append(hi["v_out"])
            results["swing"].append(lo["v_out"] - hi["v_out"])
            results["i_gate_load_high_mA"].append(lo["i_gate_load_mA"])
            results["i_gate_load_low_mA"].append(hi["i_gate_load_mA"])
        except Exception:
            break
    return {k: np.array(v) for k, v in results.items()}
