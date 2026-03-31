"""Plotting for transient simulation results."""

import os
import numpy as np
import matplotlib.pyplot as plt

PLOT_DIR = "plots"


def _save(fig, filename):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = f"{PLOT_DIR}/{filename}"
    fig.savefig(path, dpi=150)
    print(f"  Saved {path}")
    plt.close(fig)


def plot_transient(result: dict, title: str = "Transient Simulation"):
    """Plot all node voltages vs time."""
    t_us = result["t"] * 1e6

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Input and output
    axes[0].plot(t_us, result["v_in"], "r-", lw=2, label="V_IN")
    axes[0].plot(t_us, result["v_out"], "b-", lw=2, label="V_OUT")
    axes[0].set(ylabel="Voltage (V)", title=title)
    axes[0].grid(True, alpha=0.3); axes[0].legend()

    # Node A
    axes[1].plot(t_us, result["v_a"], "r-", lw=2, label="Node A (J1 drain)")
    axes[1].set(ylabel="Voltage (V)")
    axes[1].grid(True, alpha=0.3); axes[1].legend()

    # Node B
    axes[2].plot(t_us, result["v_b"], "g-", lw=2, label="Node B (J2 source)")
    axes[2].set(xlabel="Time (us)", ylabel="Voltage (V)")
    axes[2].grid(True, alpha=0.3); axes[2].legend()

    fig.tight_layout()
    _save(fig, "transient_nodes.png")


def plot_transient_comparison(result: dict, spice_data: dict = None,
                               title: str = "Transient: Solver vs LTSpice"):
    """Plot V_IN and V_OUT overlaid with LTSpice reference."""
    t_us = result["t"] * 1e6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t_us, result["v_in"], "r-", lw=2, label="V_IN (transient)")
    ax.plot(t_us, result["v_out"], "b-", lw=2, label="V_OUT (transient)")

    if spice_data is not None:
        t_sp = spice_data["time"] * 1e6
        ax.plot(t_sp, spice_data["v_in"], "r--", lw=1, alpha=0.5, label="V_IN (LTSpice)")
        ax.plot(t_sp, spice_data["v_out"], "b--", lw=1, alpha=0.5, label="V_OUT (LTSpice)")

    ax.set(xlabel="Time (us)", ylabel="Voltage (V)", title=title)
    ax.grid(True, alpha=0.3); ax.legend()
    fig.tight_layout()
    _save(fig, "transient_vs_spice.png")


def measure_timing(result: dict, v_threshold: float = None) -> dict:
    """Measure rise time, fall time, and propagation delay from transient result.

    Uses the midpoint of V_OUT swing as threshold if not specified.
    """
    v_out = result["v_out"]
    t = result["t"]
    v_in = result["v_in"]

    v_high = np.max(v_out)
    v_low = np.min(v_out)
    if v_threshold is None:
        v_threshold = (v_high + v_low) / 2.0

    v_10 = v_low + 0.1 * (v_high - v_low)
    v_90 = v_low + 0.9 * (v_high - v_low)

    # Find first falling edge on V_OUT (output goes low when input goes high)
    # Look for V_OUT crossing threshold downward
    crossings_down = []
    crossings_up = []
    for i in range(1, len(v_out)):
        if v_out[i-1] > v_threshold and v_out[i] <= v_threshold:
            # Interpolate
            frac = (v_threshold - v_out[i]) / (v_out[i-1] - v_out[i])
            t_cross = t[i] - frac * (t[i] - t[i-1])
            crossings_down.append(t_cross)
        elif v_out[i-1] < v_threshold and v_out[i] >= v_threshold:
            frac = (v_threshold - v_out[i-1]) / (v_out[i] - v_out[i-1])
            t_cross = t[i-1] + frac * (t[i] - t[i-1])
            crossings_up.append(t_cross)

    # Find input crossings for prop delay
    v_in_high = np.max(v_in)
    v_in_low = np.min(v_in)
    v_in_thresh = (v_in_high + v_in_low) / 2.0
    in_crossings_up = []
    in_crossings_down = []
    for i in range(1, len(v_in)):
        if v_in[i-1] < v_in_thresh and v_in[i] >= v_in_thresh:
            frac = (v_in_thresh - v_in[i-1]) / (v_in[i] - v_in[i-1])
            in_crossings_up.append(t[i-1] + frac * (t[i] - t[i-1]))
        elif v_in[i-1] > v_in_thresh and v_in[i] <= v_in_thresh:
            frac = (v_in_thresh - v_in[i]) / (v_in[i-1] - v_in[i])
            in_crossings_down.append(t[i] - frac * (t[i] - t[i-1]))

    # Propagation delay: input rising -> output falling (inverting gate)
    tpd_hl = None
    if in_crossings_up and crossings_down:
        for t_in in in_crossings_up:
            for t_out in crossings_down:
                if t_out > t_in:
                    tpd_hl = t_out - t_in
                    break
            if tpd_hl is not None:
                break

    tpd_lh = None
    if in_crossings_down and crossings_up:
        for t_in in in_crossings_down:
            for t_out in crossings_up:
                if t_out > t_in:
                    tpd_lh = t_out - t_in
                    break
            if tpd_lh is not None:
                break

    return {
        "v_high": v_high, "v_low": v_low, "swing": v_high - v_low,
        "tpd_hl_ns": tpd_hl * 1e9 if tpd_hl else None,
        "tpd_lh_ns": tpd_lh * 1e9 if tpd_lh else None,
        "tpd_avg_ns": ((tpd_hl + tpd_lh) / 2 * 1e9
                       if tpd_hl and tpd_lh else None),
    }
