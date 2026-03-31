"""Design space explorer — maps achievable output levels over (R2, R3)."""

import numpy as np
from model import NChannelJFET, solve_gate


def sweep_design_space(r1, r2_range, r3_range, v_in_low, v_in_high,
                        v_pos, v_neg, jfet, temp_c=27.0):
    """Sweep R2/R3 grid, solving at both input states."""
    nr2, nr3 = len(r2_range), len(r3_range)
    v_out_high = np.full((nr2, nr3), np.nan)
    v_out_low = np.full((nr2, nr3), np.nan)
    total = nr2 * nr3

    for i, r2 in enumerate(r2_range):
        for j, r3 in enumerate(r3_range):
            try:
                lo = solve_gate(v_in_low, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
                hi = solve_gate(v_in_high, v_pos, v_neg, r1, r2, r3, jfet, jfet, temp_c)
                v_out_high[i, j] = lo["v_out"]
                v_out_low[i, j] = hi["v_out"]
            except Exception:
                pass
        print(f"\r  R1={r1/1e3:.0f}k: {(i+1)*nr3}/{total} ({100*(i+1)/nr2:.0f}%)",
              end="", flush=True)
    print()

    return {"r2": r2_range, "r3": r3_range, "r1": r1,
            "v_out_high": v_out_high, "v_out_low": v_out_low,
            "swing": v_out_high - v_out_low}


def fit_heuristic(data, v_neg=-20.0):
    """Back-calculate Node B to check if a simple divider model works."""
    R2g, R3g = np.meshgrid(data["r2"], data["r3"], indexing="ij")
    ratio = R3g / (R2g + R3g)
    v_b_high = v_neg + (data["v_out_high"] - v_neg) / ratio
    v_b_low = v_neg + (data["v_out_low"] - v_neg) / ratio

    result = {
        "v_b_high_mean": np.nanmean(v_b_high),
        "v_b_high_std": np.nanstd(v_b_high),
        "v_b_low_mean": np.nanmean(v_b_low),
        "v_b_low_std": np.nanstd(v_b_low),
    }
    print(f"\n  Heuristic (R1={data['r1']/1e3:.0f}k): "
          f"Node B HIGH={result['v_b_high_mean']:.2f}+/-{result['v_b_high_std']:.2f}V, "
          f"LOW={result['v_b_low_mean']:.2f}+/-{result['v_b_low_std']:.2f}V")
    return result
