"""Multi-gate coupled transient simulator.

Builds a single ODE system for all gates in a netlist. Gate outputs directly
drive gate inputs as continuous voltage signals — no logic-level abstraction.

State vector: [gate0_va, gate0_vb, gate0_vout, (gate0_mids...),
               gate1_va, gate1_vb, gate1_vout, (gate1_mids...), ...]

Coupling: gate_i.v_out IS gate_j.v_in — same state variable in the ODE.
"""

import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass, field
from typing import Callable, Dict, List

from model import (
    NChannelJFET, JFETCapacitance, jfet_ids, jfet_gate_current,
    thermal_voltage, solve_network, solve_any_gate,
)
from model.network import (
    PulldownNetwork, Leaf, Parallel, gate_type_to_network,
    count_midpoints, input_names as net_input_names,
    network_current_with_gate, network_current,
)
from model import GateType
from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams


@dataclass
class MultiGateCircuit:
    """Multi-gate circuit for coupled transient simulation.

    Each gate has its own R1/R2/R3 (from optimization) and a network topology.
    Gates are connected via a Netlist. Primary inputs have stimulus functions.
    """
    netlist: Netlist
    gate_params: Dict[str, CircuitParams]  # gate_name -> CircuitParams
    gate_networks: Dict[str, PulldownNetwork]  # gate_name -> topology
    stimuli: Dict[str, Callable]  # primary_input_net -> v(t) function
    jfet: NChannelJFET
    caps: JFETCapacitance
    n_default_fanout: int = 4
    temp_c: float = 27.0

    # Computed in __post_init__
    gate_order: list = field(init=False)
    state_map: dict = field(init=False)  # gate_name -> (start_idx, n_states)
    net_to_state: dict = field(init=False)  # net_name -> state index of v_out
    total_states: int = field(init=False)
    vt: float = field(init=False)

    def __post_init__(self):
        self.vt = thermal_voltage(self.temp_c + 273.15)

        # Topological order (feedback gates at end)
        ordered, feedback = self.netlist.topological_sort()
        self.gate_order = ordered + feedback

        # Build state map: each gate gets [v_a, v_b, v_out, mids...]
        self.state_map = {}
        self.net_to_state = {}
        idx = 0
        for gname in self.gate_order:
            net = self.gate_networks[gname]
            n_mids = count_midpoints(net)
            n_states = 3 + n_mids  # v_a, v_b, v_out, midpoints
            self.state_map[gname] = (idx, n_states)
            # Map output net to v_out state index
            gate = self.netlist.gates[gname]
            self.net_to_state[gate.output] = idx + 2  # v_out is at offset 2
            idx += n_states

        self.total_states = idx

    def get_input_voltage(self, net_name: str, state: np.ndarray, t: float) -> float:
        """Get voltage on a net — either from a driving gate's v_out or a stimulus."""
        if net_name in self.net_to_state:
            return state[self.net_to_state[net_name]]
        if net_name in self.stimuli:
            return self.stimuli[net_name](t)
        return 0.0  # undriven net defaults to 0


def compute_multi_ic(circuit: MultiGateCircuit) -> np.ndarray:
    """Compute DC initial conditions for all gates in topological order.

    Each gate is solved using the settled output of its driving gates.
    """
    y0 = np.zeros(circuit.total_states)
    # Track settled DC voltages per net
    net_voltages = {}

    # Initialize primary inputs at t=0
    for net_name, stim_fn in circuit.stimuli.items():
        net_voltages[net_name] = stim_fn(0.0)

    for gname in circuit.gate_order:
        gate = circuit.netlist.gates[gname]
        params = circuit.gate_params[gname]
        network = circuit.gate_networks[gname]
        names = net_input_names(network)
        start, n_states = circuit.state_map[gname]

        # Build input voltage dict from settled net voltages
        v_ins_dict = {}
        for i, inp_net in enumerate(gate.inputs):
            v = net_voltages.get(inp_net, 0.0)
            v_ins_dict[names[i]] = v

        # Solve DC
        try:
            res = solve_network(network, v_ins_dict,
                                params.v_pos, params.v_neg,
                                params.r1, params.r2, params.r3,
                                circuit.jfet, circuit.jfet, circuit.temp_c)
            y0[start] = res["v_a"]
            y0[start + 1] = res["v_b"]
            y0[start + 2] = res["v_out"]
            mids = res.get("v_mids", [])
            for k, m in enumerate(mids):
                y0[start + 3 + k] = np.clip(float(m), 0.0, max(res["v_a"], 0.0))

            net_voltages[gate.output] = res["v_out"]
        except Exception:
            # Fallback: rough guess
            y0[start] = params.v_pos * 0.5
            y0[start + 1] = params.v_pos * 0.3
            y0[start + 2] = 0.0
            net_voltages[gate.output] = 0.0

    return y0


def multi_gate_ode(t: float, state: np.ndarray, circuit: MultiGateCircuit):
    """ODE RHS for coupled multi-gate system."""
    derivs = np.zeros_like(state)
    vt = circuit.vt
    fan_out_map = circuit.netlist.fan_out_map()
    c_per_input = circuit.caps.c_per_input

    for gname in circuit.gate_order:
        gate = circuit.netlist.gates[gname]
        params = circuit.gate_params[gname]
        network = circuit.gate_networks[gname]
        names = net_input_names(network)
        start, n_states = circuit.state_map[gname]

        v_a = state[start]
        v_b = state[start + 1]
        v_out = state[start + 2]
        n_mids = n_states - 3
        v_mids = list(state[start + 3: start + 3 + n_mids])

        # Build input voltages from connected nets
        v_ins_dict = {}
        for i, inp_net in enumerate(gate.inputs):
            v_ins_dict[names[i]] = circuit.get_input_voltage(inp_net, state, t)

        # J1 network current
        midpoints_copy = list(v_mids)
        mid_residuals = []
        i_j1, igd_j1 = network_current_with_gate(
            network, v_a, 0.0, v_ins_dict, circuit.jfet,
            jfet_ids, jfet_gate_current, vt,
            midpoints_copy, mid_residuals,
        )

        # J2
        vgs2 = v_a - v_b
        vds2 = params.v_pos - v_b
        i_j2 = jfet_ids(vgs=vgs2, vds=vds2, j=circuit.jfet)
        vgs2_int = v_a - (v_b + i_j2 * circuit.jfet.rs)
        vgd2_int = v_a - (params.v_pos - i_j2 * circuit.jfet.rd)
        igs_j2, igd_j2 = jfet_gate_current(vgs2_int, vgd2_int, circuit.jfet, vt)

        # Capacitances
        c_a = circuit.caps.cgd0 + circuit.caps.cgs0
        c_b = circuit.caps.cgd0
        fo = max(fan_out_map.get(gname, 1), 1)
        c_out = fo * c_per_input
        c_mid = circuit.caps.cgd0 + circuit.caps.cgs0

        # KCL -> dv/dt
        i_kcl_a = (params.v_pos - v_a) / params.r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
        i_kcl_b = i_j2 + igs_j2 - (v_b - v_out) / params.r2
        i_kcl_out = (v_b - v_out) / params.r2 - (v_out - params.v_neg) / params.r3

        derivs[start] = i_kcl_a / c_a
        derivs[start + 1] = i_kcl_b / c_b
        derivs[start + 2] = i_kcl_out / c_out

        for k, res in enumerate(mid_residuals):
            derivs[start + 3 + k] = res / c_mid

    return derivs


def simulate_multi(
    circuit: MultiGateCircuit,
    t_span: tuple,
    t_eval: np.ndarray = None,
    method: str = "Radau",
    max_step: float = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> dict:
    """Run coupled multi-gate transient simulation.

    Returns dict with 't', per-gate state traces, and per-net voltage traces.
    """
    y0 = compute_multi_ic(circuit)

    def rhs(t, y):
        return multi_gate_ode(t, y, circuit)

    kwargs = dict(method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    if max_step is not None:
        kwargs["max_step"] = max_step

    print(f"  Multi-gate transient: {len(circuit.gate_order)} gates, "
          f"{circuit.total_states} state vars...")
    sol = solve_ivp(rhs, t_span, y0, **kwargs)

    if not sol.success:
        print(f"  Warning: {sol.message}")
    print(f"  Done: {sol.nfev} evaluations, {len(sol.t)} time points")

    # Build per-net voltage traces
    net_traces = {}
    for net_name in circuit.netlist.nets:
        if net_name in circuit.net_to_state:
            idx = circuit.net_to_state[net_name]
            net_traces[net_name] = sol.y[idx]
        elif net_name in circuit.stimuli:
            net_traces[net_name] = np.array([circuit.stimuli[net_name](ti)
                                              for ti in sol.t])

    # Per-gate traces
    gate_traces = {}
    for gname in circuit.gate_order:
        start, n_states = circuit.state_map[gname]
        gate_traces[gname] = {
            "v_a": sol.y[start],
            "v_b": sol.y[start + 1],
            "v_out": sol.y[start + 2],
        }

    return {
        "t": sol.t,
        "net_traces": net_traces,
        "gate_traces": gate_traces,
        "success": sol.success,
        "message": sol.message,
        "n_eval": sol.nfev,
    }
