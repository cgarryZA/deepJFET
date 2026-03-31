"""SR latch: tests feedback loop handling in the event-driven engine."""

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

# Logic levels from INV
res_lo = solve_gate(jfet.vto * 1.2, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
res_hi = solve_gate(0.0, 24.0, -20.0, 50e3, 1e3, 4.5e3, jfet, jfet)
v_high, v_low = res_lo["v_out"], res_hi["v_out"]

print("Precomputing gate profiles...")
profiles = precompute_uniform(params, v_high, v_low, [GateType.NOR2])

# SR NOR latch: Q = NOR(R, Q_bar), Q_bar = NOR(S, Q)
gates = [
    Gate("nor_q",    GateType.NOR2, ["r", "q_bar"], "q"),
    Gate("nor_qbar", GateType.NOR2, ["s", "q"],     "q_bar"),
]
nl = Netlist.from_gates(gates, primary_outputs={"q", "q_bar"})
ordered, feedback = nl.topological_sort()
print(f"Feedback gates (expected): {feedback}")

eng = SimulationEngine(nl, profiles, v_high=v_high, v_low=v_low)
eng.set_initial_state({"q": False, "q_bar": True})

# Set at 5us, release at 8us. Reset at 15us, release at 18us.
eng.add_stimulus(Stimulus("s", [0.0, 5e-6, 8e-6], [False, True, False]))
eng.add_stimulus(Stimulus("r", [0.0, 15e-6, 18e-6], [False, True, False]))

print("\nRunning SR latch simulation...")
result = eng.run(end_time=25e-6)
print(f"Events processed: {result.events_processed}")
print(f"\n{timing_report(result, nl)}")
print(f"\n{waveform_table(result, ['s','r','q','q_bar'], time_step=0.5e-6, end_time=25e-6)}")

q_final = result.net_states["q"].value
qbar_final = result.net_states["q_bar"].value
print(f"\nFinal: Q={'H' if q_final else 'L'}, Q_bar={'H' if qbar_final else 'L'}")
print(f"Expected after R=1: Q=L, Q_bar=H")
