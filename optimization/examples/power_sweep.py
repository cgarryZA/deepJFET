"""Power optimizer frequency sweep with output RC constraint."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))
import numpy as np
from model import NChannelJFET
from analysis import sweep_frequencies
from model import max_r_out_for_freq
from util import plot_power_sweep

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)

CGD, CGS = 16.9e-12, 16.9e-12
N_FANOUT = 4
freqs = np.array([10e3, 25e3, 50e3, 100e3, 200e3, 400e3, 600e3, 800e3, 1e6])

print(f"Fan-out: {N_FANOUT}, Load cap: {N_FANOUT*(CGD+CGS)*1e12:.1f}pF")
for f in freqs:
    ro = max_r_out_for_freq(f, CGD, CGS, N_FANOUT)
    print(f"  {f/1e3:>6.0f} kHz: max R_out = {ro/1e3:.1f}k")

results = sweep_frequencies(freqs, JFET_NOM, cgd=CGD, cgs=CGS, n_fanout=N_FANOUT)

print(f"\n{'Freq':>8} {'Power':>8} {'V+':>6} {'V-':>6} "
      f"{'R1':>7} {'R2':>7} {'R3':>7} {'Swing':>6}")
for r in results:
    print(f"{r.freq_hz/1e3:>7.0f}k {r.power_mW:>7.1f}mW {r.v_pos:>6.1f} {r.v_neg:>6.1f} "
          f"{r.r1/1e3:>6.1f}k {r.r2/1e3:>6.1f}k {r.r3/1e3:>6.1f}k {r.swing:>6.2f}")

plot_power_sweep(results)
