"""Find resistor values for target logic levels."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))
from model import NChannelJFET
from analysis import find_resistors

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)

targets = [
    ("Paper levels", -4.0, 0.0, -0.8, -3.6),
    ("5V swing",     -4.0, 0.0,  0.0, -5.0),
    ("Tight swing",  -4.0, 0.0, -1.0, -3.0),
]

for name, v_lo, v_hi, target_h, target_l in targets:
    r = find_resistors(v_lo, v_hi, target_h, target_l,
                       v_pos=24.0, v_neg=-20.0, jfet=jfet)
    print(f"{name}: R1={r['r1']/1e3:.2f}k  R2={r['r2']/1e3:.2f}k  R3={r['r3']/1e3:.2f}k  "
          f"OUT=({r['v_out_at_low_input']:+.3f}, {r['v_out_at_high_input']:+.3f})  "
          f"err=({r['error_low_mV']:+.1f}, {r['error_high_mV']:+.1f}) mV")
