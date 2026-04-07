"""Compare power draw for three 100kHz-target circuit configurations."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)
sys.path.insert(0, os.path.join(_root, "static"))

from model import NChannelJFET, solve_gate
from analysis.power import _total_power, optimize_for_frequency
from model import max_r_out_for_freq

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)

V_IN_LO = jfet.vto * 1.2  # ~-4.14V
V_IN_HI = 0.0

CGD, CGS = 16.9e-12, 16.9e-12
N_FANOUT = 4
F_TARGET = 100e3

configs = {
    "Rounded (100kHz target)":  dict(v_pos=10.0,  v_neg=-10.0, r1=20_000, r2=2_000, r3=2_800),
    "Original (paper circuit)": dict(v_pos=24.0,  v_neg=-20.0, r1=50_000, r2=1_000, r3=4_500),
}

print(f"JFET: Vto={jfet.vto:.3f}V, Beta={jfet.beta*1e3:.4f}mA/V²")
print(f"V_IN: lo={V_IN_LO:.3f}V, hi={V_IN_HI:.1f}V")
print(f"Fan-out: {N_FANOUT}, Target: {F_TARGET/1e3:.0f} kHz")
r_out_max = max_r_out_for_freq(F_TARGET, CGD, CGS, N_FANOUT)
print(f"Max R_out (R2||R3): {r_out_max/1e3:.2f} kohm\n")

# Compute for static configs
rows = []
for name, cfg in configs.items():
    v_pos, v_neg, r1, r2, r3 = cfg["v_pos"], cfg["v_neg"], cfg["r1"], cfg["r2"], cfg["r3"]
    lo = solve_gate(V_IN_LO, v_pos, v_neg, r1, r2, r3, jfet, jfet)
    hi = solve_gate(V_IN_HI, v_pos, v_neg, r1, r2, r3, jfet, jfet)
    v_out_hi = lo["v_out"]
    v_out_lo = hi["v_out"]
    swing = v_out_hi - v_out_lo
    noise_margin = swing / 2.0

    # Cascade check: use output as input
    casc_lo = solve_gate(v_out_lo, v_pos, v_neg, r1, r2, r3, jfet, jfet)
    casc_hi = solve_gate(v_out_hi, v_pos, v_neg, r1, r2, r3, jfet, jfet)
    casc_ok = (casc_lo["v_out"] > v_out_hi - 0.2 and casc_hi["v_out"] < v_out_lo + 0.2)

    power_W = _total_power(v_pos, v_neg, r1, r2, r3, jfet, V_IN_LO, V_IN_HI, 27.0)
    r_out = (r2 * r3) / (r2 + r3)
    speed_ok = r_out <= r_out_max

    rows.append({
        "name": name,
        "v_pos": v_pos, "v_neg": v_neg, "r1": r1, "r2": r2, "r3": r3,
        "v_out_hi": v_out_hi, "v_out_lo": v_out_lo,
        "swing": swing, "noise_margin": noise_margin,
        "power_mW": power_W * 1e3, "r_out": r_out,
        "speed_ok": speed_ok, "casc_ok": casc_ok,
    })

# Run optimizer for exact 100kHz values
print("Running 100kHz optimizer for exact values (this may take ~30s)...")
opt = optimize_for_frequency(F_TARGET, JFET_NOM, temp_c=27.0,
                              min_noise_margin=0.3, cgd=CGD, cgs=CGS, n_fanout=N_FANOUT)
print(f"  Done. V+={opt.v_pos:.1f}V, V-={opt.v_neg:.1f}V, "
      f"R1={opt.r1/1e3:.1f}k, R2={opt.r2/1e3:.1f}k, R3={opt.r3/1e3:.1f}k\n")

r2o, r3o = opt.r2, opt.r3
r_out_o = (r2o * r3o) / (r2o + r3o)
power_o = _total_power(opt.v_pos, opt.v_neg, opt.r1, r2o, r3o, jfet, V_IN_LO, V_IN_HI, 27.0)
casc_lo_o = solve_gate(opt.v_out_low,  opt.v_pos, opt.v_neg, opt.r1, r2o, r3o, jfet, jfet)
casc_hi_o = solve_gate(opt.v_out_high, opt.v_pos, opt.v_neg, opt.r1, r2o, r3o, jfet, jfet)
casc_ok_o = (casc_lo_o["v_out"] > opt.v_out_high - 0.2 and
             casc_hi_o["v_out"] < opt.v_out_low  + 0.2)
rows.append({
    "name": f"Exact optimiser (100kHz)",
    "v_pos": opt.v_pos, "v_neg": opt.v_neg, "r1": opt.r1, "r2": r2o, "r3": r3o,
    "v_out_hi": opt.v_out_high, "v_out_lo": opt.v_out_low,
    "swing": opt.swing, "noise_margin": opt.noise_margin,
    "power_mW": power_o * 1e3, "r_out": r_out_o,
    "speed_ok": r_out_o <= r_out_max, "casc_ok": casc_ok_o,
})

# Print table
W = 28
print(f"{'Config':<{W}} {'V+':>5} {'V-':>6} {'R1':>7} {'R2':>6} {'R3':>6} "
      f"{'Vhi':>6} {'Vlo':>6} {'Swing':>6} {'NM':>5} {'Power':>8} {'Speed':>5} {'Casc':>5}")
print("-" * (W + 70))
for r in rows:
    print(f"{r['name']:<{W}} "
          f"{r['v_pos']:>5.1f} {r['v_neg']:>6.1f} "
          f"{r['r1']/1e3:>6.1f}k {r['r2']/1e3:>5.1f}k {r['r3']/1e3:>5.1f}k "
          f"{r['v_out_hi']:>6.3f} {r['v_out_lo']:>6.3f} {r['swing']:>6.3f} "
          f"{r['noise_margin']:>5.3f} "
          f"{r['power_mW']:>7.3f}mW "
          f"{'OK' if r['speed_ok'] else 'FAIL':>5} "
          f"{'OK' if r['casc_ok'] else 'FAIL':>5}")

print()
rounded = rows[0]
exact   = rows[2]
pct = (rounded["power_mW"] - exact["power_mW"]) / exact["power_mW"] * 100
print(f"Rounded vs exact optimiser: {pct:+.1f}% power difference")
orig = rows[1]
pct2 = (rounded["power_mW"] - orig["power_mW"]) / orig["power_mW"] * 100
print(f"Rounded vs original:        {pct2:+.1f}% power difference")
