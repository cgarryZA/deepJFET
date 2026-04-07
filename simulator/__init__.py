"""Gate-level netlist simulator for SiC JFET logic."""

from model import (
    GateType, gate_input_count,
    solve_gate, solve_nor, solve_nand, solve_any_gate,
    estimate_prop_delay, max_r_out_for_freq,
)
from .gate_models import truth_table
from .netlist import Gate, Net, Netlist
from .module import Module, ModuleInstance, Port, PortDir, flatten, flatten_top
from .precompute import CircuitParams, GateProfile, precompute_gate, precompute_uniform, precompute_from_designs, profile_custom_gate
from .engine import SimulationEngine, Stimulus, SimResult, Event, NetState
from .report import timing_report, critical_path, waveform_table, dump_vcd
