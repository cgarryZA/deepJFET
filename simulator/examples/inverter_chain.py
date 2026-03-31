"""Inverter chain: basic end-to-end simulator test."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

from model import NChannelJFET, JFETCapacitance, solve_gate
from simulator import (
    GateType, Gate, Netlist, CircuitParams,
    precompute_uniform, SimulationEngine, Stimulus,
    timing_report, waveform_table,
)

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)
caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

params = CircuitParams(v_pos=24.0, v_neg=-20.0, r1=50e3, r2=1e3, r3=4.5e3,
                       jfet=jfet, caps=caps)

# Determine logic levels from INV
res_lo = solve_gate(jfet.vto * 1.2, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
res_hi = solve_gate(0.0, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
v_high, v_low = res_lo["v_out"], res_hi["v_out"]
print(f"Logic levels: HIGH={v_high:.4f}V, LOW={v_low:.4f}V")

# Precompute
print("Precomputing gate profiles...")
profiles = precompute_uniform(params, v_high, v_low, [GateType.INV])

# Build 4-inverter chain
gates = [
    Gate("inv1", GateType.INV, ["clk"], "n1"),
    Gate("inv2", GateType.INV, ["n1"], "n2"),
    Gate("inv3", GateType.INV, ["n2"], "n3"),
    Gate("inv4", GateType.INV, ["n3"], "out"),
]
nl = Netlist.from_gates(gates, primary_outputs={"out"})
ordered, feedback = nl.topological_sort()
print(f"Topological order: {ordered}")
print(f"Fan-out: {nl.fan_out_map()}")

# Simulate
print("\nRunning simulation...")
eng = SimulationEngine(nl, profiles, v_high=v_high, v_low=v_low)
eng.add_stimulus(Stimulus("clk",
    times=[0.0, 5e-6, 15e-6, 25e-6],
    values=[True, False, True, False]))
result = eng.run(end_time=35e-6)

print(f"Events processed: {result.events_processed}")
print(f"\n{timing_report(result, nl)}")
print(f"\n{waveform_table(result, ['clk','n1','n2','n3','out'], time_step=0.5e-6, end_time=35e-6)}")
