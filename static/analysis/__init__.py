"""Analysis tools: inverse solver, design space, fan-out, power optimization."""

from .inverse import find_resistors
from .fanout import solve_gate_with_fanout, fanout_sweep, cascade_test
from .power import optimize_for_frequency, sweep_frequencies
from .design_space import sweep_design_space, fit_heuristic
from .stability import find_fixed_points_inv, find_fixed_points_any
