"""Design space exploration — R2/R3 contour maps."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))
import numpy as np
from model import NChannelJFET
from analysis import sweep_design_space, fit_heuristic
from util import plot_design_space

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)

r2_range = np.linspace(200, 20_000, 40)
r3_range = np.linspace(200, 20_000, 40)

for r1 in [20_000, 50_000]:
    print(f"\nR1={r1/1e3:.0f}k:")
    data = sweep_design_space(r1, r2_range, r3_range, -4.0, 0.0,
                               24.0, -20.0, jfet)
    plot_design_space(data)
    fit_heuristic(data)
