"""All plotting functions for the static solver suite."""

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


# ── Gate node plots ──────────────────────────────────────────────────────

def plot_gate_nodes(data: dict, spice_data: dict = None):
    """J1 nodes, J2 nodes, V_OUT, and combined overview."""
    v_in = data["v_in"]

    for name, keys, title in [
        ("j1_nodes", [("j1_gate", "J1 Gate (V_IN)"), ("j1_source", "J1 Source (GND)"),
                       ("j1_drain", "J1 Drain (Node A)")], "J1 Node Voltages vs V_IN"),
        ("j2_nodes", [("j2_gate", "J2 Gate (Node A)"), ("j2_source", "J2 Source (Node B)"),
                       ("j2_drain", "J2 Drain (+V_POS)")], "J2 Node Voltages vs V_IN"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for k, label in keys:
            ax.plot(v_in, data[k], lw=2, label=label)
        ax.set(xlabel="V_IN (V)", ylabel="Voltage (V)", title=title)
        ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
        _save(fig, f"{name}.png")

    # V_OUT
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v_in, data["v_out"], "b-", lw=2, label="V_OUT (solver)")
    if spice_data is not None:
        ax.plot(spice_data["v_in"], spice_data["v_out"], "r--", lw=1.5,
                label="V_OUT (LTSpice)", alpha=0.8)
    ax.set(xlabel="V_IN (V)", ylabel="V_OUT (V)", title="Output Node Voltage vs V_IN")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    _save(fig, "v_out.png")

    # Combined
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v_in, data["j1_drain"], "--", lw=1.5, label="Node A (J1 drain / J2 gate)")
    ax.plot(v_in, data["j2_source"], "--", lw=1.5, label="Node B (J2 source)")
    ax.plot(v_in, data["v_out"], "b-", lw=2, label="V_OUT")
    if spice_data is not None:
        ax.plot(spice_data["v_in"], spice_data["v_out"], "r--", lw=1.5,
                label="V_OUT (LTSpice)", alpha=0.8)
    ax.set(xlabel="V_IN (V)", ylabel="Voltage (V)", title="Full Gate Transfer Characteristic")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    _save(fig, "transfer_overview.png")


# Alias for backward compat
plot_transfer = plot_gate_nodes


# ── Time-domain ──────────────────────────────────────────────────────────

def plot_time_domain(t, v_in_t, v_out_t, spice_data=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    t_us = t * 1e6
    ax.plot(t_us, v_in_t, "r-", lw=2, label="V_IN (solver)")
    ax.plot(t_us, v_out_t, "g-", lw=2, label="V_OUT (solver)")
    if spice_data is not None:
        ax.plot(spice_data["time"] * 1e6, spice_data["v_in"], "r--", lw=1, alpha=0.6,
                label="V_IN (LTSpice)")
        ax.plot(spice_data["time"] * 1e6, spice_data["v_out"], "g--", lw=1, alpha=0.6,
                label="V_OUT (LTSpice)")
    ax.set(xlabel="Time (us)", ylabel="Voltage (V)",
           title="JFET Gate -- Time Domain (Solver vs LTSpice)")
    ax.grid(True, alpha=0.3); ax.legend(); fig.tight_layout()
    _save(fig, "time_domain.png")


# ── Temperature ──────────────────────────────────────────────────────────

def plot_temperature_sweep(temp_data):
    """temp_data: list of (temp_c, v_out_low, v_out_high, swing) tuples."""
    tc = [d[0] for d in temp_data]
    v_lo = [d[1] for d in temp_data]
    v_hi = [d[2] for d in temp_data]
    swings = [d[3] for d in temp_data]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(tc, v_lo, "ro-", lw=2, label="V_OUT (input LOW / J1 off)")
    ax1.plot(tc, v_hi, "bo-", lw=2, label="V_OUT (input HIGH / J1 on)")
    ax1.fill_between(tc, v_hi, v_lo, alpha=0.15, color="green")
    ax1.set(ylabel="V_OUT (V)", title="Logic Gate Output vs Temperature")
    ax1.grid(True, alpha=0.3); ax1.legend()
    ax2.plot(tc, swings, "go-", lw=2)
    ax2.set(xlabel="Temperature (C)", ylabel="Output Swing (V)",
            title="Logic Swing vs Temperature")
    ax2.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "temperature_sweep.png")


# ── Fan-out ──────────────────────────────────────────────────────────────

def plot_fanout(data: dict, r1, r2, r3):
    """Plot fan-out degradation for a single R2/R3 config."""
    n = data["n"]
    label = f"R1={r1/1e3:.0f}k, R2={r2/1e3:.1f}k, R3={r3/1e3:.1f}k"
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(n, data["v_out_high"], "ro-", ms=5, label="V_OUT HIGH")
    axes[0, 0].plot(n, data["v_out_low"], "bo-", ms=5, label="V_OUT LOW")
    axes[0, 0].set(xlabel="Fan-out", ylabel="V_OUT (V)", title=f"Output Levels\n{label}")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].legend()

    axes[0, 1].plot(n, data["swing"], "go-", ms=5)
    axes[0, 1].axhline(y=0.6, color="r", ls="--", alpha=0.5, label="Min margin")
    axes[0, 1].set(xlabel="Fan-out", ylabel="Swing (V)", title="Output Swing")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(n, data["i_gate_load_high_mA"], "ro-", ms=5, label="Output HIGH")
    axes[1, 0].plot(n, data["i_gate_load_low_mA"], "bo-", ms=5, label="Output LOW")
    axes[1, 0].set(xlabel="Fan-out", ylabel="Load Current (mA)", title="Gate Load Current")
    axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    if len(n) > 1 and data["swing"][0] != 0:
        pct = 100.0 * (1.0 - data["swing"] / data["swing"][0])
        axes[1, 1].plot(n, pct, "mo-", ms=5)
    axes[1, 1].set(xlabel="Fan-out", ylabel="Swing Loss (%)", title="Degradation")
    axes[1, 1].grid(True, alpha=0.3)

    fig.tight_layout()
    _save(fig, f"fanout_r2_{r2/1e3:.1f}k_r3_{r3/1e3:.1f}k.png")


def plot_fanout_comparison(configs_data: list):
    """Plot fan-out comparison across R2/R3 configs.
    configs_data: list of (r2, r3, sweep_data) tuples."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    for r2, r3, data in configs_data:
        label = f"R2={r2/1e3:.1f}k, R3={r3/1e3:.1f}k"
        ax1.plot(data["n"], data["swing"], "o-", ms=4, label=label)
        ax2.plot(data["n"], data["v_out_high"], "o-", ms=3, label=f"HIGH {label}")
        ax2.plot(data["n"], data["v_out_low"], "s--", ms=3, alpha=0.6)
    ax1.axhline(y=0.6, color="r", ls="--", alpha=0.5, label="Min margin")
    ax1.set(xlabel="Fan-out", ylabel="Swing (V)", title="Swing vs Fan-out")
    ax1.grid(True, alpha=0.3); ax1.legend(fontsize=8)
    ax2.set(xlabel="Fan-out", ylabel="V_OUT (V)", title="Levels vs Fan-out")
    ax2.grid(True, alpha=0.3); ax2.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, "fanout_comparison.png")


# ── Design space ─────────────────────────────────────────────────────────

def plot_design_space(data: dict):
    """Contour plots from design space sweep."""
    r2 = data["r2"] / 1e3
    r3 = data["r3"] / 1e3
    R2g, R3g = np.meshgrid(r2, r3, indexing="ij")
    r1_k = data["r1"] / 1e3

    for field, title_part, cmap, fname in [
        ("v_out_high", "Output HIGH Level (input LOW)", "RdYlBu_r", "design_vout_high"),
        ("v_out_low", "Output LOW Level (input HIGH)", "RdYlBu_r", "design_vout_low"),
        ("swing", "Output Voltage Swing", "viridis", "design_swing"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 8))
        cs = ax.contourf(R2g, R3g, data[field], levels=30, cmap=cmap)
        ax.contour(R2g, R3g, data[field], levels=15, colors="k", linewidths=0.5, alpha=0.4)
        plt.colorbar(cs, ax=ax, label="V" if "swing" not in field else "Swing (V)")
        ax.set(xlabel="R2 (kohm)", ylabel="R3 (kohm)",
               title=f"{title_part} | R1={r1_k:.0f}k")
        fig.tight_layout()
        _save(fig, f"{fname}_r1_{r1_k:.0f}k.png")


# ── Power sweep ──────────────────────────────────────────────────────────

def plot_power_sweep(results: list):
    """Plot power-frequency Pareto curve and parameter trends."""
    f = [r.freq_hz / 1e3 for r in results]
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    axes[0, 0].plot(f, [r.power_mW for r in results], "ro-", lw=2, ms=6)
    axes[0, 0].set(xlabel="Freq (kHz)", ylabel="Power (mW)", title="Min Power vs Frequency")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(f, [r.v_pos for r in results], "r^-", lw=2, ms=6, label="V_POS")
    axes[0, 1].plot(f, [r.v_neg for r in results], "bv-", lw=2, ms=6, label="V_NEG")
    axes[0, 1].plot(f, [r.v_pos - r.v_neg for r in results], "g--", lw=1.5, label="Span")
    axes[0, 1].set(xlabel="Freq (kHz)", ylabel="Voltage (V)", title="Optimal Rails")
    axes[0, 1].grid(True, alpha=0.3); axes[0, 1].legend()

    axes[1, 0].plot(f, [r.r1/1e3 for r in results], "s-", lw=2, ms=5, label="R1")
    axes[1, 0].plot(f, [r.r2/1e3 for r in results], "o-", lw=2, ms=5, label="R2")
    axes[1, 0].plot(f, [r.r3/1e3 for r in results], "^-", lw=2, ms=5, label="R3")
    axes[1, 0].set(xlabel="Freq (kHz)", ylabel="R (kohm)", title="Optimal Resistors")
    axes[1, 0].set_yscale("log"); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    axes[1, 1].plot(f, [r.swing for r in results], "go-", lw=2, ms=6)
    axes[1, 1].set(xlabel="Freq (kHz)", ylabel="Swing (V)", title="Output Swing")
    axes[1, 1].grid(True, alpha=0.3)

    axes[2, 0].plot(f, [r.prop_delay_ns for r in results], "mo-", lw=2, ms=6)
    axes[2, 0].set(xlabel="Freq (kHz)", ylabel="Delay (ns)", title="Propagation Delay")
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(f, [r.power_mW for r in results], "ro-", lw=2, ms=6, label="Total")
    axes[2, 1].set(xlabel="Freq (kHz)", ylabel="Power (mW)", title="Power per Gate")
    axes[2, 1].grid(True, alpha=0.3); axes[2, 1].legend()

    fig.tight_layout()
    _save(fig, "power_frequency_sweep.png")
