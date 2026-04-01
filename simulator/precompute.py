"""Precompute DC output levels and propagation delays for each gate type.

All gates share the same board-level logic levels (v_high, v_low).
Each gate type has its own R1/R2/R3 (from the optimizer) but produces
the same output levels.
"""

import itertools
from dataclasses import dataclass, field

from model import (
    NChannelJFET, JFETCapacitance, GateType, gate_input_count,
    solve_gate, solve_any_gate, estimate_prop_delay,
)


@dataclass
class CircuitParams:
    """Circuit parameters for a specific gate type on a board."""
    v_pos: float
    v_neg: float
    r1: float
    r2: float
    r3: float
    jfet: NChannelJFET
    caps: JFETCapacitance
    temp_c: float = 27.0


@dataclass
class GateProfile:
    """Precomputed characteristics for one gate type."""
    gate_type: GateType
    n_inputs: int
    v_high: float                         # board-level logic HIGH
    v_low: float                          # board-level logic LOW
    params: CircuitParams = None          # this gate type's circuit params
    dc_table: dict = field(default_factory=dict)    # (bool,...) -> v_out
    delay_table: dict = field(default_factory=dict) # fan_out -> delay_s


def _delay_for_params(params, n_fanout, delay_model=None):
    """Delay estimate. Uses delay NN if available, else RC estimate."""
    if delay_model is not None:
        import numpy as np
        from nn.delay_model import predict_delay
        X = np.array([[params.r1, params.r2, params.r3,
                        params.v_pos, params.v_neg, params.temp_c]], dtype=np.float32)
        delays = predict_delay(delay_model, X)  # (1, 2) in ns
        # Scale by fanout (delay NN trained at fanout=4, scale linearly)
        base_fanout = 4
        scale = max(n_fanout, 1) / base_fanout
        return float(delays[0].max()) * scale * 1e-9  # convert ns to seconds
    return estimate_prop_delay(params.r1, params.r2, params.r3,
                               params.caps.cgd0, params.caps.cgs0, n_fanout)


def precompute_gate(
    gate_type: GateType,
    params: CircuitParams,
    v_high: float,
    v_low: float,
    max_fanout: int = 10,
    delay_model=None,
) -> GateProfile:
    """Build DC table and delay table for a gate type.

    v_high/v_low are the board-level logic levels (same for all gates).
    params contains this gate type's specific R1/R2/R3/rails.
    """
    n_in = gate_input_count(gate_type)
    profile = GateProfile(
        gate_type=gate_type, n_inputs=n_in,
        v_high=v_high, v_low=v_low, params=params,
    )

    v_map = {False: v_low, True: v_high}
    v_a_high = None
    v_a_low = None

    for combo in itertools.product([False, True], repeat=n_in):
        v_ins = [v_map[b] for b in combo]
        res = solve_any_gate(gate_type, v_ins,
                             params.v_pos, params.v_neg,
                             params.r1, params.r2, params.r3,
                             params.jfet, params.jfet, params.temp_c)
        profile.dc_table[combo] = res["v_out"]

        if v_a_high is None or res["v_a"] > v_a_high:
            v_a_high = res["v_a"]
        if v_a_low is None or res["v_a"] < v_a_low:
            v_a_low = res["v_a"]

    for fo in range(max_fanout + 1):
        delay = _delay_for_params(params, max(fo, 1), delay_model=delay_model)
        profile.delay_table[fo] = delay

    return profile


def save_profiles(profiles: dict, path: str):
    """Save precomputed gate profiles to disk."""
    import json
    from pathlib import Path
    data = {}
    for gt, prof in profiles.items():
        key = gt.value if hasattr(gt, 'value') else str(gt)
        data[key] = {
            "n_inputs": prof.n_inputs,
            "v_high": prof.v_high,
            "v_low": prof.v_low,
            "dc_table": {str(k): v for k, v in prof.dc_table.items()},
            "delay_table": {str(k): v for k, v in prof.delay_table.items()},
            "params": {
                "v_pos": prof.params.v_pos, "v_neg": prof.params.v_neg,
                "r1": prof.params.r1, "r2": prof.params.r2, "r3": prof.params.r3,
                "temp_c": prof.params.temp_c,
            } if prof.params else None,
        }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {len(profiles)} profiles to {path}")


def load_profiles(path: str, jfet=None, caps=None) -> dict:
    """Load precomputed gate profiles from disk."""
    import json
    import ast
    with open(path) as f:
        data = json.load(f)

    profiles = {}
    for key, d in data.items():
        gt = GateType(key)
        params = None
        if d["params"] and jfet and caps:
            params = CircuitParams(
                v_pos=d["params"]["v_pos"], v_neg=d["params"]["v_neg"],
                r1=d["params"]["r1"], r2=d["params"]["r2"], r3=d["params"]["r3"],
                jfet=jfet, caps=caps, temp_c=d["params"]["temp_c"],
            )
        prof = GateProfile(
            gate_type=gt, n_inputs=d["n_inputs"],
            v_high=d["v_high"], v_low=d["v_low"], params=params,
        )
        prof.dc_table = {ast.literal_eval(k): v for k, v in d["dc_table"].items()}
        prof.delay_table = {int(k): v for k, v in d["delay_table"].items()}
        profiles[gt] = prof
    print(f"  Loaded {len(profiles)} profiles from {path}")
    return profiles


def precompute_from_designs(
    designs: dict,
    board_config,
    v_high: float = None,
    v_low: float = None,
    max_fanout: int = 10,
) -> dict:
    """Precompute profiles from optimizer results.

    Args:
        designs: dict[GateType, GateDesign] from optimize_board()
        board_config: BoardConfig with jfet, caps, temp_c
        v_high/v_low: board-wide logic levels for DC table input mapping.
            If not provided, uses the INV design's converged levels
            (or the first design if no INV).

    Returns dict[GateType, GateProfile].
    """
    if v_high is None or v_low is None:
        # Use INV levels as baseline, or first available
        ref = designs.get(GateType.INV, next(iter(designs.values())))
        v_high = ref.v_high
        v_low = ref.v_low
        print(f"  Using reference levels: HIGH={v_high:.4f}V, LOW={v_low:.4f}V")

    profiles = {}
    for gt, design in designs.items():
        params = CircuitParams(
            v_pos=design.v_pos, v_neg=design.v_neg,
            r1=design.r1, r2=design.r2, r3=design.r3,
            jfet=board_config.jfet, caps=board_config.caps,
            temp_c=board_config.temp_c,
        )
        profile = precompute_gate(gt, params, v_high, v_low, max_fanout)
        profiles[gt] = profile

        n = profile.n_inputs
        all_lo = tuple([False] * n)
        all_hi = tuple([True] * n)
        print(f"  {gt.value}: all-LOW->{profile.dc_table[all_lo]:.4f}V, "
              f"all-HIGH->{profile.dc_table[all_hi]:.4f}V, "
              f"delay(fo=1)={profile.delay_table[1]*1e9:.1f}ns")

    return profiles


def precompute_uniform(
    params: CircuitParams,
    v_high: float,
    v_low: float,
    gate_types: list = None,
    max_fanout: int = 10,
) -> dict:
    """Precompute all gate types with the same CircuitParams and explicit levels.

    Use for quick testing or when all gates share the same circuit.
    """
    if gate_types is None:
        gate_types = [GateType.INV, GateType.NAND2, GateType.NOR2]

    profiles = {}
    for gt in gate_types:
        profile = precompute_gate(gt, params, v_high, v_low, max_fanout)
        profiles[gt] = profile
        n = profile.n_inputs
        all_lo = tuple([False] * n)
        all_hi = tuple([True] * n)
        print(f"  {gt.value}: all-LOW->{profile.dc_table[all_lo]:.4f}V, "
              f"all-HIGH->{profile.dc_table[all_hi]:.4f}V, "
              f"delay(fo=1)={profile.delay_table[1]*1e9:.1f}ns")
    return profiles


def profile_custom_gate(
    name: str,
    solver_func,
    n_inputs: int,
    params: CircuitParams,
    v_high: float,
    v_low: float,
    max_fanout: int = 10,
    delay_model=None,
) -> GateProfile:
    """Profile a custom (Tier 2) gate using an arbitrary solver function."""
    profile = GateProfile(
        gate_type=GateType.CUSTOM, n_inputs=n_inputs,
        v_high=v_high, v_low=v_low, params=params,
    )

    v_map = {False: v_low, True: v_high}
    v_a_high = None
    v_a_low = None

    for combo in itertools.product([False, True], repeat=n_inputs):
        v_ins = [v_map[b] for b in combo]
        res = solver_func(v_ins, params.v_pos, params.v_neg,
                          params.r1, params.r2, params.r3,
                          params.jfet, params.jfet, params.temp_c)
        profile.dc_table[combo] = res["v_out"]
        if v_a_high is None or res["v_a"] > v_a_high:
            v_a_high = res["v_a"]
        if v_a_low is None or res["v_a"] < v_a_low:
            v_a_low = res["v_a"]

    for fo in range(max_fanout + 1):
        delay = _delay_for_params(params, max(fo, 1), delay_model=delay_model)
        profile.delay_table[fo] = delay

    return profile
