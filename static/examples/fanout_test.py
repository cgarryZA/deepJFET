"""Fan-out and cascade analysis."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))
from model import NChannelJFET
from analysis import cascade_test, fanout_sweep
from util import plot_fanout, plot_fanout_comparison

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)
V_POS, V_NEG, R1 = 24.0, -20.0, 50_000

# Cascade test
print("Cascade test (double inversion):")
for vin in [-4.0, 0.0]:
    c = cascade_test(vin, V_POS, V_NEG, R1, 1000, 4500, jfet)
    print(f"  V_IN={vin:+.1f} -> G1={c['gate1_v_out']:+.3f} -> G2={c['gate2_v_out']:+.3f}")

# Fan-out with default R2/R3
print("\nFan-out sweep (R2=1k, R3=4.5k):")
data = fanout_sweep(V_POS, V_NEG, R1, 1000, 4500, jfet, -4.0, 0.0, max_fanout=15)
for i in range(len(data["n"])):
    print(f"  N={data['n'][i]:>2.0f}  swing={data['swing'][i]:.3f}V")
plot_fanout(data, R1, 1000, 4500)

# Comparison across R2/R3 configs
configs = [(1000, 4500), (2000, 9000), (5000, 22000)]
all_data = []
for r2, r3 in configs:
    d = fanout_sweep(V_POS, V_NEG, R1, r2, r3, jfet, -4.0, 0.0, max_fanout=15)
    all_data.append((r2, r3, d))
plot_fanout_comparison(all_data)
