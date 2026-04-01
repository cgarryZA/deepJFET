"""Transient simulator for any N-input JFET gate (INV, NOR-N, NAND-N)."""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, List

from model import (
    NChannelJFET, JFETCapacitance, GateType,
    solve_any_gate, thermal_voltage,
)
from .ode import gate_ode


@dataclass
class Circuit:
    """Complete circuit description for transient simulation.

    Supports INV, NOR-N (parallel J1s), and NAND-N (series J1s).

    For INV: provide v_in_funcs=[func] (single input)
    For NOR2: provide v_in_funcs=[func_a, func_b]
    For NAND3: provide v_in_funcs=[func_a, func_b, func_c]

    Legacy: v_in_func (single callable) still works for INV.
    """
    v_pos: float
    v_neg: float
    r1: float
    r2: float
    r3: float
    j1: NChannelJFET
    j2: NChannelJFET
    caps: JFETCapacitance
    v_in_func: Callable[[float], float] = None
    v_in_funcs: List[Callable[[float], float]] = None
    gate_type_enum: GateType = None
    n_fanout: int = 1
    temp_c: float = 27.0

    # Derived fields
    gate_type: str = field(init=False)
    n_inputs: int = field(init=False)
    c_a: float = field(init=False)
    c_b: float = field(init=False)
    c_out: float = field(init=False)
    vt: float = field(init=False)

    def __post_init__(self):
        # Handle legacy single v_in_func
        if self.v_in_funcs is None:
            if self.v_in_func is not None:
                self.v_in_funcs = [self.v_in_func]
            else:
                raise ValueError("Must provide v_in_func or v_in_funcs")

        self.n_inputs = len(self.v_in_funcs)

        # Auto-detect gate type from enum or n_inputs
        if self.gate_type_enum is not None:
            gt = self.gate_type_enum.value
            if gt == "INV":
                self.gate_type = "INV"
            elif gt.startswith("NOR"):
                self.gate_type = "NOR"
            elif gt.startswith("NAND"):
                self.gate_type = "NAND"
            else:
                self.gate_type = "INV"
        elif self.n_inputs == 1:
            self.gate_type = "INV"
        else:
            # Default to NOR for multiple inputs (user should set gate_type_enum)
            self.gate_type = "NOR"

        # Capacitances
        self.c_a = self.caps.cgd0 + self.caps.cgs0   # Cgd_J1 + Cgs_J2
        self.c_b = self.caps.cgd0                     # Cgd_J2
        self.c_out = self.n_fanout * self.caps.c_per_input
        self.vt = thermal_voltage(self.temp_c + 273.15)


def compute_initial_conditions(circuit: Circuit) -> np.ndarray:
    """Get DC initial conditions using the static solver.

    Returns state vector at t=0:
      INV/NOR: [v_a, v_b, v_out]
      NAND-N:  [v_a, v_b, v_out, v_mid_0, ..., v_mid_{N-2}]
    """
    c = circuit
    t0_inputs = [fn(0.0) for fn in c.v_in_funcs]

    if c.gate_type == "INV":
        gt = GateType.INV
    elif c.gate_type == "NOR":
        gt = GateType(f"NOR{c.n_inputs}")
    elif c.gate_type == "NAND":
        gt = GateType(f"NAND{c.n_inputs}")
    else:
        gt = GateType.INV

    res = solve_any_gate(gt, t0_inputs, c.v_pos, c.v_neg,
                         c.r1, c.r2, c.r3, c.j1, c.j2, c.temp_c)

    y0 = [res["v_a"], res["v_b"], res["v_out"]]

    # NAND midpoints
    if c.gate_type == "NAND" and c.n_inputs > 1:
        v_mids = res.get("v_mids", [])
        y0.extend(v_mids)

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
    """Run a transient simulation for any gate type.

    Returns dict with 't', 'v_a', 'v_b', 'v_out', 'v_ins' arrays,
    plus 'v_mids' for NAND gates.
    """
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

    # Build input voltage arrays
    n_ins = circuit.n_inputs
    v_ins_arr = np.zeros((n_ins, len(sol.t)))
    for k in range(n_ins):
        v_ins_arr[k] = [circuit.v_in_funcs[k](ti) for ti in sol.t]

    result = {
        "t": sol.t,
        "v_a": sol.y[0],
        "v_b": sol.y[1],
        "v_out": sol.y[2],
        "v_in": v_ins_arr[0],  # legacy: first input
        "v_ins": v_ins_arr,     # all inputs: (n_inputs, n_time)
        "success": sol.success,
        "message": sol.message,
        "n_eval": sol.nfev,
    }

    # NAND midpoints
    if circuit.gate_type == "NAND" and circuit.n_inputs > 1:
        n_mids = circuit.n_inputs - 1
        result["v_mids"] = sol.y[3:3 + n_mids]

    return result
