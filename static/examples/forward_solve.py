"""Basic forward solver: compute output for given inputs and plot transfer curve."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))
import numpy as np

from model import NChannelJFET, solve_gate, sweep
from util import plot_gate_nodes, plot_time_domain, plot_temperature_sweep
from util import pulse_waveform, load_spice_data

# -- JFET (DR NJF from SPICE .model card) --
JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)

# -- Circuit --
V_POS, V_NEG = 24.0, -20.0
R1, R2, R3 = 50_000, 1_000, 4_500
TEMP_C = 27.0

jfet = JFET_NOM.at_temp(TEMP_C)

# -- Single-point solves --
print(f"T={TEMP_C}C  V+={V_POS}  V-={V_NEG}  R1={R1/1e3:.0f}k  R2={R2/1e3:.1f}k  R3={R3/1e3:.1f}k")
for vin in [-4.0, -3.45, -2.0, -1.0, 0.0]:
    r = solve_gate(vin, V_POS, V_NEG, R1, R2, R3, jfet, jfet, TEMP_C)
    print(f"  V_IN={vin:+.2f}  V_OUT={r['v_out']:+.4f}  "
          f"J1:{r['j1_region']}  J2:{r['j2_region']}")

# -- Transfer curve --
data = sweep(np.linspace(-5, 1, 500), V_POS, V_NEG, R1, R2, R3, jfet, jfet, TEMP_C)

spice = None
spice_file = os.path.join("..", "..", "Multi-Or.txt")
if os.path.exists(spice_file):
    spice = load_spice_data(spice_file)

plot_gate_nodes(data, spice_data=spice)

# -- Time domain --
t = np.linspace(0, 40e-6, 2000)
v_in_t = pulse_waveform(t, -4.0, 0.0, 10e-6, 0.1e-6, 0.1e-6, 5e-6, 10e-6)
v_out_t = np.array([solve_gate(vi, V_POS, V_NEG, R1, R2, R3, jfet, jfet, TEMP_C)["v_out"]
                     for vi in v_in_t])
plot_time_domain(t, v_in_t, v_out_t, spice_data=spice)

# -- Temperature sweep --
temps = [27, 100, 200, 300, 400, 500, 600]
temp_data = []
for tc in temps:
    jt = JFET_NOM.at_temp(tc)
    lo = solve_gate(-4.0, V_POS, V_NEG, R1, R2, R3, jt, jt, tc)
    hi = solve_gate(0.0, V_POS, V_NEG, R1, R2, R3, jt, jt, tc)
    temp_data.append((tc, lo["v_out"], hi["v_out"], lo["v_out"] - hi["v_out"]))
plot_temperature_sweep(temp_data)
