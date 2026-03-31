"""Transient simulator — wraps the ODE system with initial conditions and solve_ivp."""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable

from model import NChannelJFET, JFETCapacitance, solve_gate, thermal_voltage
from .ode import gate_ode


@dataclass
class Circuit:
    """Complete circuit description for transient simulation."""
    v_pos: float
    v_neg: float
    r1: float
    r2: float
    r3: float
    j1: NChannelJFET
    j2: NChannelJFET
    caps: JFETCapacitance
    v_in_func: Callable[[float], float] = None  # v_in(t) -> voltage
    n_fanout: int = 1
    temp_c: float = 27.0

    # Computed capacitances (set in __post_init__)
    c_a: float = field(init=False)   # Node A: Cgd_J1 + Cgs_J2
    c_b: float = field(init=False)   # Node B: Cgd_J2
    c_out: float = field(init=False) # Output: N_fanout * (Cgs + Cgd)
    vt: float = field(init=False)    # Thermal voltage

    def __post_init__(self):
        self.c_a = self.caps.cgd0 + self.caps.cgs0   # Cgd_J1 + Cgs_J2
        self.c_b = self.caps.cgd0                     # Cgd_J2
        self.c_out = self.n_fanout * self.caps.c_per_input  # load
        self.vt = thermal_voltage(self.temp_c + 273.15)


def compute_initial_conditions(circuit: Circuit, v_in_0: float) -> np.ndarray:
    """Get DC initial conditions using the static solver.

    Returns [v_a, v_b, v_out] at the initial input voltage.
    """
    r = solve_gate(
        v_in_0, circuit.v_pos, circuit.v_neg,
        circuit.r1, circuit.r2, circuit.r3,
        circuit.j1, circuit.j2, circuit.temp_c,
    )
    return np.array([r["v_a"], r["v_b"], r["v_out"]])


def simulate(
    circuit: Circuit,
    t_span: tuple,
    t_eval: np.ndarray = None,
    method: str = "Radau",
    max_step: float = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> dict:
    """Run a transient simulation.

    Args:
        circuit: Circuit with v_in_func set to a time-varying input.
        t_span: (t_start, t_end) in seconds.
        t_eval: Time points to evaluate at. If None, solver picks its own.
        method: ODE solver method. 'Radau' (implicit) recommended for stiff circuits.
        max_step: Maximum step size (s). Set to ~1/10 of shortest time constant.
        rtol, atol: Relative and absolute tolerances.

    Returns:
        Dict with 't', 'v_a', 'v_b', 'v_out', 'v_in' arrays.
    """
    # Initial conditions from DC at t=0
    v_in_0 = circuit.v_in_func(t_span[0])
    y0 = compute_initial_conditions(circuit, v_in_0)

    # Wrap the ODE to pass circuit as parameter
    def rhs(t, y):
        return gate_ode(t, y, circuit)

    # Solve
    kwargs = dict(method=method, t_eval=t_eval, rtol=rtol, atol=atol, dense_output=True)
    if max_step is not None:
        kwargs["max_step"] = max_step

    sol = solve_ivp(rhs, t_span, y0, **kwargs)

    if not sol.success:
        print(f"Warning: solver did not converge: {sol.message}")

    # Build input voltage array at evaluation times
    v_in_arr = np.array([circuit.v_in_func(ti) for ti in sol.t])

    return {
        "t": sol.t,
        "v_a": sol.y[0],
        "v_b": sol.y[1],
        "v_out": sol.y[2],
        "v_in": v_in_arr,
        "success": sol.success,
        "message": sol.message,
        "n_eval": sol.nfev,
    }
