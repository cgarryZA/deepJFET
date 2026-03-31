"""Board-level gate optimizer using coarse-to-fine grid search.

Given board-level parameters (V+, V-, V_HIGH, V_LOW, f_target), finds
R1/R2/R3 from standard E-series resistor values for each gate type.

Search strategy:
  1. E12 full grid — rank by accuracy, take top N
  2. E24 around top N — rank by accuracy, take top N
  3. E96 around top N — among results within tolerance, pick lowest power

Every evaluation result is stored in the SQLite database.
"""

from dataclasses import dataclass

from model import (
    NChannelJFET, JFETCapacitance, GateType, gate_input_count, solve_any_gate,
    estimate_prop_delay, max_r_out_for_freq,
    e_series_values, e_series_neighbourhood,
)
from .gate_models import truth_table


@dataclass
class BoardConfig:
    """Board-level configuration shared by all gates."""
    v_high: float
    v_low: float
    v_pos: float
    v_neg: float
    jfet: NChannelJFET
    caps: JFETCapacitance
    temp_c: float = 27.0
    f_target: float = 100e3
    n_fanout: int = 4

    @property
    def v_threshold(self):
        return (self.v_high + self.v_low) / 2.0


@dataclass
class GateDesign:
    """Optimized circuit parameters for one gate type."""
    gate_type: GateType
    r1: float
    r2: float
    r3: float
    v_pos: float
    v_neg: float
    v_high: float
    v_low: float
    swing: float
    power_mW: float
    delay_ns: float
    max_error_mV: float
    converged: bool


def _evaluate_combo(gate_type, r1, r2, r3, board, use_db=True):
    """Evaluate one R1/R2/R3 combo. Checks DB cache first.

    Returns (power_W, max_error_V, v_out_high, v_out_low, delay_s) or None.
    """
    # Check cache
    if use_db:
        from .results_db import lookup_evaluation, cache_evaluation
        cached = lookup_evaluation(gate_type, r1, r2, r3, board)
        if cached is not None:
            return cached

    table = truth_table(gate_type)
    v_map = {False: board.v_low, True: board.v_high}

    max_err = 0.0
    total_power = 0.0
    n_states = 0
    v_out_high = None
    v_out_low = None

    try:
        for combo, out_high in table:
            v_ins = [v_map[b] for b in combo]
            target = board.v_high if out_high else board.v_low
            res = solve_any_gate(gate_type, v_ins,
                                 board.v_pos, board.v_neg, r1, r2, r3,
                                 board.jfet, board.jfet, board.temp_c)

            err = abs(res["v_out"] - target)
            max_err = max(max_err, err)

            i_r1 = res["i_r1_mA"] * 1e-3
            i_j2 = res["i_j2_mA"] * 1e-3
            i_load = res["i_load_mA"] * 1e-3
            total_power += board.v_pos * (i_r1 + i_j2) + (-board.v_neg) * i_load
            n_states += 1

            if all(not b for b in combo):
                v_out_high = res["v_out"]
            if all(b for b in combo):
                v_out_low = res["v_out"]
    except Exception:
        return None

    avg_power = total_power / n_states if n_states > 0 else 0
    delay = estimate_prop_delay(r1, r2, r3, board.caps.cgd0, board.caps.cgs0,
                                board.n_fanout)

    result = (avg_power, max_err, v_out_high, v_out_low, delay)

    # Save to cache
    if use_db:
        cache_evaluation(gate_type, r1, r2, r3, board,
                         avg_power, max_err, v_out_high, v_out_low, delay)

    return result


def _grid_search(gate_type, r1_values, r2_values, r3_values, board,
                 label="", use_db=True):
    """Evaluate a grid of R1/R2/R3 combos with speed pre-filter.

    Returns ALL valid results sorted by max_error (most accurate first).
    Result tuples: (max_err, power, r1, r2, r3, v_high, v_low, delay)
    """
    cgd, cgs = board.caps.cgd0, board.caps.cgs0
    r_out_max = max_r_out_for_freq(board.f_target, cgd, cgs, board.n_fanout)
    max_half_period = 1.0 / (2.0 * board.f_target)

    results = []
    n_total = len(r1_values) * len(r2_values) * len(r3_values)
    n_filtered = 0
    n_evaluated = 0
    n_cached = 0

    for r1 in r1_values:
        if 0.7 * r1 * (cgd + cgs) > max_half_period:
            continue
        for r2 in r2_values:
            for r3 in r3_values:
                r_out = (r2 * r3) / (r2 + r3)
                if r_out > r_out_max:
                    continue
                delay = estimate_prop_delay(r1, r2, r3, cgd, cgs, board.n_fanout)
                if delay > max_half_period:
                    continue

                n_filtered += 1
                result = _evaluate_combo(gate_type, r1, r2, r3, board, use_db)
                if result is None:
                    continue

                n_evaluated += 1
                power, max_err, v_high, v_low, delay = result
                results.append((max_err, power, r1, r2, r3, v_high, v_low, delay))

    # Flush any pending DB writes
    if use_db:
        from .results_db import flush_eval_cache
        flush_eval_cache()

    results.sort(key=lambda x: x[0])

    if label:
        best_err = results[0][0] * 1e3 if results else float('inf')
        print(f"    {label}: {n_total} grid, {n_filtered} speed-ok, "
              f"{n_evaluated} solved, best err={best_err:.1f}mV")

    return results


def optimize_gate_resistors(
    gate_type: GateType,
    board: BoardConfig,
    max_error_tol: float = 0.2,
    top_n: int = 10,
    use_db: bool = True,
) -> GateDesign:
    """Find optimal R1/R2/R3 using coarse-to-fine E-series grid search.

    Stages 1-2 rank by accuracy to find the neighbourhood.
    Stage 3 applies the tolerance filter and picks lowest power.
    If nothing meets tolerance, returns the most accurate result found.
    """
    if use_db:
        from .results_db import find_design
        cached = find_design(gate_type, board.v_pos, board.v_neg, board)
        if cached is not None:
            return cached

    cgd, cgs = board.caps.cgd0, board.caps.cgs0
    r_out_max = max_r_out_for_freq(board.f_target, cgd, cgs, board.n_fanout)
    max_half_period = 1.0 / (2.0 * board.f_target)
    r1_max = max_half_period / (10.0 * (cgd + cgs))

    r1_range = (100, min(r1_max, 500_000))
    r23_range = (100, min(2 * r_out_max, 500_000))

    print(f"  {gate_type.value}: R1 up to {r1_range[1]/1e3:.1f}k, "
          f"R23 up to {r23_range[1]/1e3:.1f}k")

    # Stage 1: E12 full grid — find the neighbourhood
    r1_e12 = e_series_values("E12", r1_range[0], r1_range[1])
    r2_e12 = e_series_values("E12", r23_range[0], r23_range[1])
    r3_e12 = e_series_values("E12", r23_range[0], r23_range[1])

    results_e12 = _grid_search(gate_type, r1_e12, r2_e12, r3_e12, board, "E12")

    if not results_e12:
        print(f"    WARNING: no results at all for {gate_type.value}")
        return GateDesign(gate_type=gate_type, r1=1e3, r2=1e3, r3=1e3,
                          v_pos=board.v_pos, v_neg=board.v_neg,
                          v_high=0, v_low=0, swing=0, power_mW=0,
                          delay_ns=0, max_error_mV=9999, converged=False)

    # Stage 2: E24 around top N most accurate E12 results
    r1_e24, r2_e24, r3_e24 = set(), set(), set()
    for _, _, r1, r2, r3, _, _, _ in results_e12[:top_n]:
        for v in e_series_neighbourhood(r1, "E24", n_steps=2):
            if r1_range[0] <= v <= r1_range[1]:
                r1_e24.add(v)
        for v in e_series_neighbourhood(r2, "E24", n_steps=2):
            if r23_range[0] <= v <= r23_range[1]:
                r2_e24.add(v)
        for v in e_series_neighbourhood(r3, "E24", n_steps=2):
            if r23_range[0] <= v <= r23_range[1]:
                r3_e24.add(v)

    results_e24 = _grid_search(gate_type, sorted(r1_e24), sorted(r2_e24),
                               sorted(r3_e24), board, "E24")

    best_so_far = results_e24 if results_e24 else results_e12

    # Stage 3: E96 around top N most accurate E24 results
    r1_e96, r2_e96, r3_e96 = set(), set(), set()
    for _, _, r1, r2, r3, _, _, _ in best_so_far[:top_n]:
        for v in e_series_neighbourhood(r1, "E96", n_steps=2):
            if r1_range[0] <= v <= r1_range[1]:
                r1_e96.add(v)
        for v in e_series_neighbourhood(r2, "E96", n_steps=2):
            if r23_range[0] <= v <= r23_range[1]:
                r2_e96.add(v)
        for v in e_series_neighbourhood(r3, "E96", n_steps=2):
            if r23_range[0] <= v <= r23_range[1]:
                r3_e96.add(v)

    results_e96 = _grid_search(gate_type, sorted(r1_e96), sorted(r2_e96),
                               sorted(r3_e96), board, "E96")

    final = results_e96 if results_e96 else best_so_far

    # Among results within tolerance, pick lowest power.
    # If none meet tolerance, just take the most accurate.
    within_tol = [r for r in final if r[0] <= max_error_tol]
    if within_tol:
        # Sort by power (index 1) among those within tolerance
        within_tol.sort(key=lambda x: x[1])
        pick = within_tol[0]
        print(f"    Best (min power within {max_error_tol*1e3:.0f}mV tol): "
              f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k R3={pick[4]/1e3:.2f}k "
              f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW")
    else:
        pick = final[0]  # most accurate
        print(f"    Best (closest, outside {max_error_tol*1e3:.0f}mV tol): "
              f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k R3={pick[4]/1e3:.2f}k "
              f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW")

    max_err, power, r1, r2, r3, v_high, v_low, delay = pick

    design = GateDesign(
        gate_type=gate_type,
        r1=r1, r2=r2, r3=r3,
        v_pos=board.v_pos, v_neg=board.v_neg,
        v_high=float(v_high), v_low=float(v_low),
        swing=float(v_high - v_low),
        power_mW=float(power * 1e3),
        delay_ns=float(delay * 1e9),
        max_error_mV=float(max_err * 1e3),
        converged=bool(max_err <= max_error_tol),
    )

    if use_db:
        from .results_db import save_design
        save_design(design, board)

    return design


def optimize_board(
    board: BoardConfig,
    gate_types: list,
    max_error_tol: float = 0.2,
    use_db: bool = True,
) -> dict:
    """Optimize R1/R2/R3 for all gate types on a board."""
    designs = {}
    print(f"Board: V_HIGH={board.v_high:.2f}V  V_LOW={board.v_low:.2f}V  "
          f"+{board.v_pos:.0f}/{board.v_neg:.0f}V  "
          f"f={board.f_target/1e3:.0f}kHz  fanout={board.n_fanout}")
    print()

    for gt in gate_types:
        if use_db:
            from .results_db import find_design
            cached = find_design(gt, board.v_pos, board.v_neg, board)
            if cached is not None:
                print(f"  {gt.value} [FROM DB] "
                      f"R1={cached.r1/1e3:.2f}k R2={cached.r2/1e3:.2f}k "
                      f"R3={cached.r3/1e3:.2f}k "
                      f"err={cached.max_error_mV:.1f}mV "
                      f"P={cached.power_mW:.2f}mW")
                designs[gt] = cached
                continue

        design = optimize_gate_resistors(gt, board, max_error_tol, use_db=use_db)
        designs[gt] = design

    # Summary
    print()
    print(f"{'Type':<7} {'R1':>8} {'R2':>8} {'R3':>8} "
          f"{'V_H':>7} {'V_L':>7} {'Swing':>6} "
          f"{'MaxErr':>8} {'Power':>8} {'Delay':>8}")
    print("-" * 85)
    for gt, d in designs.items():
        flag = " *" if d.max_error_mV > max_error_tol * 1e3 else ""
        print(f"{gt.value:<7} "
              f"{d.r1/1e3:>7.2f}k {d.r2/1e3:>7.2f}k {d.r3/1e3:>7.2f}k "
              f"{d.v_high:>7.3f} {d.v_low:>7.3f} {d.swing:>6.3f} "
              f"{d.max_error_mV:>7.1f}mV "
              f"{d.power_mW:>7.2f}mW {d.delay_ns:>7.1f}ns{flag}")

    return designs
