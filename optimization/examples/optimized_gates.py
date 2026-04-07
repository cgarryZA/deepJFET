"""Board-level optimization demo using E-series grid search."""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

from model import NChannelJFET, JFETCapacitance, GateType
from simulator import (
    Gate, Netlist, BoardConfig, optimize_board,
    precompute_from_designs,
    SimulationEngine, Stimulus, waveform_table, db_summary,
)

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)
caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

board = BoardConfig(
    v_high=-0.8, v_low=-4.0,
    v_pos=24.0, v_neg=-20.0,
    jfet=jfet, caps=caps, temp_c=27.0,
    f_target=100e3, n_fanout=4,
)

print("=" * 70)
print("BOARD-LEVEL GATE OPTIMIZATION (E-series grid search)")
print("=" * 70)

gate_types = [GateType.INV, GateType.NAND2, GateType.NOR2]
designs = optimize_board(board, gate_types)

# Precompute profiles and simulate
print("\nPrecomputing profiles...")
profiles = precompute_from_designs(designs, board)

print("\n" + "=" * 70)
print("SIMULATION: NAND2 -> INV -> NOR2")
print("=" * 70)

gates = [
    Gate("nand1", GateType.NAND2, ["a", "b"], "n1"),
    Gate("inv1",  GateType.INV,   ["n1"],     "n2"),
    Gate("nor1",  GateType.NOR2,  ["n2", "c"], "out"),
]
nl = Netlist.from_gates(gates, primary_outputs={"out"})

eng = SimulationEngine(nl, profiles, v_high=board.v_high, v_low=board.v_low)
eng.add_stimulus(Stimulus("a", [0.0, 5e-6, 15e-6], [True, False, True]))
eng.add_stimulus(Stimulus("b", [0.0, 5e-6, 10e-6], [True, True, False]))
eng.add_stimulus(Stimulus("c", [0.0], [False]))
result = eng.run(end_time=25e-6)

print(f"Events: {result.events_processed}")
print(f"\n{waveform_table(result, ['a','b','c','n1','n2','out'], time_step=0.5e-6, end_time=25e-6)}")

print("\n" + "=" * 70)
print("DATABASE CONTENTS")
print("=" * 70)
db_summary()
