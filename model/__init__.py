"""JFET device model, gate solvers, and timing utilities."""

from .jfet import (
    NChannelJFET,
    jfet_ids,
    jfet_gate_current,
    thermal_voltage,
    region_name,
    K_BOLTZ, Q_ELEC, TNOM_K,
)
from .gate import (
    GateType, gate_input_count,
    solve_gate, solve_nor, solve_nand, solve_any_gate,
    sweep,
)
from .capacitance import JFETCapacitance
from .timing import max_r_out_for_freq, estimate_prop_delay
from .resistors import e_series_values, nearest_e_series, e_series_neighbourhood
