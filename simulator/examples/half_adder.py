"""Half-adder: tests hierarchical module flattening + simulation."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

from model import NChannelJFET, JFETCapacitance, solve_gate
from simulator import (
    GateType, Gate, Netlist, CircuitParams,
    Module, Port, PortDir, flatten_top,
    precompute_uniform, SimulationEngine, Stimulus,
    timing_report, waveform_table, dump_vcd,
)

IN, OUT = PortDir.IN, PortDir.OUT

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

# Logic levels
res_lo = solve_gate(jfet.vto * 1.2, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
res_hi = solve_gate(0.0, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
v_high, v_low = res_lo["v_out"], res_hi["v_out"]

print("Precomputing gate profiles...")
profiles = precompute_uniform(params, v_high, v_low, [GateType.INV, GateType.NAND2])

# Half-adder module
ha = Module("half_adder",
    ports=[Port("a", IN), Port("b", IN), Port("sum", OUT), Port("carry", OUT)],
    gates=[
        Gate("n1", GateType.NAND2, ["a", "b"], "nab"),
        Gate("n2", GateType.NAND2, ["a", "nab"], "p"),
        Gate("n3", GateType.NAND2, ["b", "nab"], "q"),
        Gate("n4", GateType.NAND2, ["p", "q"], "sum"),
        Gate("i1", GateType.INV,   ["nab"], "carry"),
    ])

flat_gates = flatten_top(ha, {"a": "a", "b": "b", "sum": "sum", "carry": "carry"})
nl = Netlist.from_gates(flat_gates, primary_outputs={"sum", "carry"})
print(f"Flattened: {len(flat_gates)} gates, inputs: {nl.primary_inputs}")

# Truth table verification
print("\n--- Truth table ---")
for a_val, b_val in [(False, False), (False, True), (True, False), (True, True)]:
    eng = SimulationEngine(nl, profiles, v_high=v_high, v_low=v_low)
    eng.add_stimulus(Stimulus("a", times=[0.0], values=[a_val]))
    eng.add_stimulus(Stimulus("b", times=[0.0], values=[b_val]))
    result = eng.run(end_time=20e-6)

    s = result.net_states["sum"].value
    c = result.net_states["carry"].value
    expected_s = int(a_val) ^ int(b_val)
    expected_c = int(a_val) & int(b_val)
    ok = (int(s) == expected_s and int(c) == expected_c)
    print(f"  A={int(a_val)} B={int(b_val)} -> SUM={int(s)} CARRY={int(c)}  "
          f"(expected {expected_s},{expected_c}) {'OK' if ok else 'FAIL'}")

# Waveform
print("\n--- Waveform ---")
eng = SimulationEngine(nl, profiles, v_high=v_high, v_low=v_low)
eng.add_stimulus(Stimulus("a", [0.0, 10e-6, 20e-6, 30e-6], [False, True, False, True]))
eng.add_stimulus(Stimulus("b", [0.0, 10e-6, 15e-6, 25e-6], [False, False, True, True]))
result = eng.run(end_time=40e-6)
print(f"Events: {result.events_processed}")
print(f"\n{waveform_table(result, ['a','b','sum','carry'], time_step=0.5e-6, end_time=40e-6)}")

os.makedirs("plots", exist_ok=True)
dump_vcd(result, "plots/half_adder.vcd", ["a", "b", "sum", "carry"])
