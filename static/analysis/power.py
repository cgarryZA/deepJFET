"""Power optimizer — minimum-power circuit parameters for a target frequency."""

import numpy as np
from scipy.optimize import differential_evolution
from dataclasses import dataclass

from model import (
    NChannelJFET, jfet_ids, thermal_voltage, solve_gate,
    estimate_prop_delay, max_r_out_for_freq,
)


@dataclass
class PowerResult:
    freq_hz: float
    v_pos: float
    v_neg: float
    r1: float
    r2: float
    r3: float
    v_out_high: float
    v_out_low: float
    swing: float
    noise_margin: float
    power_mW: float
    prop_delay_ns: float
    max_freq_hz: float
    converged: bool



def _total_power(v_pos, v_neg, r1, r2, r3, jfet, v_in_low, v_in_high, temp_c):
    """Average static power across both logic states (watts)."""
    def state_power(res):
        i_r1 = res["i_r1_mA"] * 1e-3
        i_j2 = res["i_j2_mA"] * 1e-3
        i_load = res["i_load_mA"] * 1e-3
        return v_pos * (i_r1 + i_j2) + (-v_neg) * i_load

    lo = solve_gate(v_in_low, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
    hi = solve_gate(v_in_high, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
    return (state_power(lo) + state_power(hi)) / 2.0


def optimize_for_frequency(f_target, jfet_nom, temp_c=27.0,
                            min_noise_margin=0.3,
                            cgd=16.9e-12, cgs=16.9e-12, n_fanout=4):
    """Find minimum-power V_POS, V_NEG, R1, R2, R3 for a target frequency."""
    jfet = jfet_nom.at_temp(temp_c)
    max_delay = 1.0 / (2.0 * f_target)
    r_out_max = max_r_out_for_freq(f_target, cgd, cgs, n_fanout)
    r23_max = min(2.0 * r_out_max, 100_000)

    def objective(x):
        v_pos, v_neg, r1, r2, r3 = x
        try:
            if (r2 * r3) / (r2 + r3) > r_out_max:
                return 1e6

            v_in_lo = jfet.vto * 1.2
            res_lo = solve_gate(v_in_lo, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
            res_hi = solve_gate(0.0, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
            v_h, v_l = res_lo["v_out"], res_hi["v_out"]

            res_lo2 = solve_gate(v_l, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
            res_hi2 = solve_gate(v_h, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)

            swing = v_h - v_l
            if swing < 2 * min_noise_margin:
                return 1e6

            delay = estimate_prop_delay(r1, r2, r3, cgd, cgs, n_fanout)
            if delay > max_delay:
                return 1e6

            cascade_err = (v_h - res_lo2["v_out"])**2 + (v_l - res_hi2["v_out"])**2
            power = _total_power(v_pos, v_neg, r1, r2, r3, jfet,
                                 v_in_lo, 0.0, temp_c)
            return power + 100.0 * cascade_err
        except Exception:
            return 1e6

    bounds = [(3, 30), (-25, -1), (1000, 500000), (100, r23_max), (100, r23_max)]
    result = differential_evolution(objective, bounds, seed=42, maxiter=200,
                                    tol=1e-8, popsize=20)
    v_pos, v_neg, r1, r2, r3 = result.x

    v_in_lo = jfet.vto * 1.2
    res_lo = solve_gate(v_in_lo, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
    res_hi = solve_gate(0.0, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
    v_h, v_l = res_lo["v_out"], res_hi["v_out"]
    delay = estimate_prop_delay(r1, r2, r3, res_lo["v_a"], res_hi["v_a"],
                                cgd, cgs, n_fanout)
    power = _total_power(v_pos, v_neg, r1, r2, r3, jfet, v_in_lo, 0.0, temp_c)

    return PowerResult(
        freq_hz=f_target, v_pos=v_pos, v_neg=v_neg, r1=r1, r2=r2, r3=r3,
        v_out_high=v_h, v_out_low=v_l, swing=v_h - v_l,
        noise_margin=(v_h - v_l) / 2.0, power_mW=power * 1e3,
        prop_delay_ns=delay * 1e9,
        max_freq_hz=1.0 / (2.0 * delay) if delay > 0 else float("inf"),
        converged=result.success,
    )


def sweep_frequencies(freqs, jfet_nom, temp_c=27.0, **kwargs):
    """Run the optimizer at each target frequency."""
    results = []
    for i, f in enumerate(freqs):
        print(f"\r  [{i+1}/{len(freqs)}] f={f/1e3:.0f} kHz ...", end="", flush=True)
        r = optimize_for_frequency(f, jfet_nom, temp_c, **kwargs)
        results.append(r)
        print(f" -> {r.power_mW:.3f} mW, R1={r.r1/1e3:.1f}k, "
              f"R2={r.r2/1e3:.1f}k, R3={r.r3/1e3:.1f}k, swing={r.swing:.2f}V")
    return results
