"""Transient simulation of the inverter gate with a pulse input.

Matches the LTSpice simulation: PULSE(-4, 0, 10us, 0.1us, 0.1us, 5us, 10us)
Compares output waveform, rise/fall times, and propagation delay to LTSpice.
"""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

import numpy as np
from model import NChannelJFET, JFETCapacitance
from transient.engine import simulate, Circuit
from transient.util import plot_transient, plot_transient_comparison, measure_timing

# -- JFET model (DR NJF from SPICE card) --
JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)

# -- Capacitances from SPICE model card --
caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12, pb=1.0, m=0.407, fc=0.5)

# -- Pulse input (matching LTSpice) --
# PULSE(-4 0 10u 0.1u 0.1u 5u 10u 2)
def v_in_pulse(t):
    v1, v2 = -4.0, 0.0
    td, tr, tf, pw, per = 10e-6, 0.1e-6, 0.1e-6, 5e-6, 10e-6
    if t < td:
        return v1
    tc = (t - td) % per
    if tc < tr:
        return v1 + (v2 - v1) * tc / tr
    elif tc < tr + pw:
        return v2
    elif tc < tr + pw + tf:
        return v2 + (v1 - v2) * (tc - tr - pw) / tf
    else:
        return v1

# -- Circuit --
circuit = Circuit(
    v_pos=24.0, v_neg=-20.0,
    r1=50_000, r2=1_000, r3=4_500,
    j1=jfet, j2=jfet,
    caps=caps,
    v_in_func=v_in_pulse,
    n_fanout=1,
    temp_c=27.0,
)

print(f"Circuit: V+={circuit.v_pos}, V-={circuit.v_neg}")
print(f"R1={circuit.r1/1e3:.0f}k, R2={circuit.r2/1e3:.1f}k, R3={circuit.r3/1e3:.1f}k")
print(f"C_A={circuit.c_a*1e12:.1f}pF, C_B={circuit.c_b*1e12:.1f}pF, "
      f"C_out={circuit.c_out*1e12:.1f}pF")
print(f"Fan-out: {circuit.n_fanout}")

# -- Run simulation --
print("\nRunning transient simulation (0 to 40us)...")
t_eval = np.linspace(0, 40e-6, 4000)
result = simulate(circuit, t_span=(0, 40e-6), t_eval=t_eval, max_step=0.1e-6)

print(f"  Solver: {result['message']}")
print(f"  Function evaluations: {result['n_eval']}")

# -- Timing measurements --
timing = measure_timing(result)
print(f"\nTiming:")
print(f"  V_OUT high: {timing['v_high']:.4f} V")
print(f"  V_OUT low:  {timing['v_low']:.4f} V")
print(f"  Swing:      {timing['swing']:.4f} V")
if timing["tpd_hl_ns"]:
    print(f"  t_pd (H->L): {timing['tpd_hl_ns']:.1f} ns")
if timing["tpd_lh_ns"]:
    print(f"  t_pd (L->H): {timing['tpd_lh_ns']:.1f} ns")
if timing["tpd_avg_ns"]:
    print(f"  t_pd (avg):   {timing['tpd_avg_ns']:.1f} ns")

# -- Plot --
print("\nGenerating plots:")
plot_transient(result, title="Inverter Transient — PULSE(-4, 0, 10us)")

# Load SPICE reference if available
spice_file = os.path.join(_root, "Multi-Or.txt")
spice = None
if os.path.exists(spice_file):
    raw = np.loadtxt(spice_file, skiprows=1)
    spice = {"time": raw[:, 0], "v_out": raw[:, 1], "v_in": raw[:, 2]}
    print(f"  Loaded LTSpice reference from {spice_file}")

plot_transient_comparison(result, spice_data=spice)

# -- Compare DC levels to static solver --
from model import solve_gate
r_dc_lo = solve_gate(-4.0, 24, -20, 50e3, 1e3, 4.5e3, jfet, jfet)
r_dc_hi = solve_gate(0.0, 24, -20, 50e3, 1e3, 4.5e3, jfet, jfet)
print(f"\nDC comparison:")
print(f"  Static V_OUT (in=-4V): {r_dc_lo['v_out']:.4f} V")
print(f"  Static V_OUT (in=0V):  {r_dc_hi['v_out']:.4f} V")
print(f"  Transient V_OUT high:  {timing['v_high']:.4f} V")
print(f"  Transient V_OUT low:   {timing['v_low']:.4f} V")
