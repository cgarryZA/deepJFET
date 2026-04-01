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
    solve_gate, solve_nor, solve_nand, solve_any_gate, solve_network,
    sweep,
)
from .network import (
    Leaf, Series, Parallel, PulldownNetwork,
    count_midpoints, count_jfets, input_names, n_solver_vars,
    canonical_str, to_dict, from_dict,
    network_truth_table, gate_type_to_network,
    network_current, network_current_with_gate,
)
from .capacitance import JFETCapacitance
from .timing import max_r_out_for_freq, estimate_prop_delay
from .resistors import e_series_values, nearest_e_series, e_series_neighbourhood
