"""Transient simulator for any JFET gate topology."""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, List

from model import (
    NChannelJFET, JFETCapacitance, GateType,
    solve_any_gate, solve_network, thermal_voltage,
)
from model.network import (
    PulldownNetwork, input_names as net_input_names,
    count_midpoints, gate_type_to_network,
)
from .ode import gate_ode


@dataclass
class Circuit:
    """Circuit description for transient simulation.

    Accepts either a GateType enum (backward compat) or a PulldownNetwork.

    For INV: v_in_funcs=[func_a]
    For NOR2: v_in_funcs=[func_a, func_b]
    For A NOR (B NAND C): v_in_funcs=[func_a, func_b, func_c]
    """
    v_pos: float
    v_neg: float
    r1: float
    r2: float
    r3: float
    j1: NChannelJFET
    j2: NChannelJFET
    caps: JFETCapacitance
    v_in_func: Callable[[float], float] = None     # legacy single input
    v_in_funcs: List[Callable[[float], float]] = None
    gate_type_enum: GateType = None
    network: PulldownNetwork = None
    n_fanout: int = 1
    temp_c: float = 27.0

    # Derived fields
    input_names: list = field(init=False)
    n_inputs: int = field(init=False)
    n_midpoints: int = field(init=False)
    c_a: float = field(init=False)
    c_b: float = field(init=False)
    c_out: float = field(init=False)
    vt: float = field(init=False)

    def __post_init__(self):
        # Handle input functions
        if self.v_in_funcs is None:
            if self.v_in_func is not None:
                self.v_in_funcs = [self.v_in_func]
            else:
                raise ValueError("Must provide v_in_func or v_in_funcs")

        # Determine topology
        if self.network is None:
            if self.gate_type_enum is not None:
                self.network = gate_type_to_network(self.gate_type_enum)
            else:
                # Default: single input = INV
                from model.network import Leaf
                n = len(self.v_in_funcs)
                if n == 1:
                    self.network = Leaf("A")
                else:
                    from model.network import Parallel
                    names = [chr(ord('A') + i) for i in range(n)]
                    self.network = Parallel(tuple(Leaf(nm) for nm in names))

        self.input_names = net_input_names(self.network)
        self.n_inputs = len(self.input_names)
        self.n_midpoints = count_midpoints(self.network)

        # Capacitances
        self.c_a = self.caps.cgd0 + self.caps.cgs0
        self.c_b = self.caps.cgd0
        self.c_out = self.n_fanout * self.caps.c_per_input
        self.vt = thermal_voltage(self.temp_c + 273.15)


def compute_initial_conditions(circuit: Circuit) -> np.ndarray:
    """DC initial conditions via static solver.

    Returns [v_a, v_b, v_out, mid_0, ..., mid_{N-1}].
    """
    c = circuit
    t0_inputs = [fn(0.0) for fn in c.v_in_funcs]
    v_ins_dict = {c.input_names[i]: t0_inputs[i] for i in range(c.n_inputs)}

    res = solve_network(c.network, v_ins_dict, c.v_pos, c.v_neg,
                        c.r1, c.r2, c.r3, c.j1, c.j2, c.temp_c)

    y0 = [res["v_a"], res["v_b"], res["v_out"]]
    mids = res.get("v_mids", [])
    # Clamp midpoint values to physical range (between 0 and v_a)
    v_a = res["v_a"]
    for m in mids:
        y0.append(np.clip(float(m), 0.0, max(v_a, 0.0)))
    return np.array(y0)


def simulate(
    circuit: Circuit,
    t_span: tuple,
    t_eval: np.ndarray = None,
    method: str = "Radau",
    max_step: float = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict:
    """Run transient simulation for any gate topology."""
    y0 = compute_initial_conditions(circuit)

    def rhs(t, y):
        return gate_ode(t, y, circuit)

    kwargs = dict(method=method, t_eval=t_eval, rtol=rtol, atol=atol,
                  dense_output=True)
    if max_step is not None:
        kwargs["max_step"] = max_step

    sol = solve_ivp(rhs, t_span, y0, **kwargs)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    n_ins = circuit.n_inputs
    v_ins_arr = np.zeros((n_ins, len(sol.t)))
    for k in range(n_ins):
        v_ins_arr[k] = [circuit.v_in_funcs[k](ti) for ti in sol.t]

    result = {
        "t": sol.t,
        "v_a": sol.y[0],
        "v_b": sol.y[1],
        "v_out": sol.y[2],
        "v_in": v_ins_arr[0],
        "v_ins": v_ins_arr,
        "success": sol.success,
        "message": sol.message,
        "n_eval": sol.nfev,
    }

    if circuit.n_midpoints > 0:
        result["v_mids"] = sol.y[3:3 + circuit.n_midpoints]

    return result
