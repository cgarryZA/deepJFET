"""Transient delay data generation for the delay NN.

Uses the CPU transient solver (scipy Radau) with multiprocessing.
Each worker runs step-response simulations and extracts propagation delay.

GPU is used for DC solving only (via gpu_solver). Transient integration
stays on CPU where scipy's implicit Radau solver handles stiff circuits well.
"""

import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from .model import COLUMNS_X


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _delay_worker(args):
    """Compute delays for a chunk of R-combos. Runs in a worker process."""
    chunk_x, gate_type_str, board_dict = args

    from model import NChannelJFET, JFETCapacitance, GateType, solve_any_gate
    from transient.engine import simulate, Circuit
    from transient.util import measure_timing

    n_inputs = {
        "INV": 1, "NAND2": 2, "NAND3": 3, "NAND4": 4,
        "NOR2": 2, "NOR3": 3, "NOR4": 4,
    }.get(gate_type_str, 1)

    gt_enum = GateType(gate_type_str)
    v_high = board_dict["v_high"]
    v_low = board_dict["v_low"]

    results = []  # (t_pd_hl_ns, t_pd_lh_ns, converged)

    for row in chunk_x:
        r1, r2, r3 = float(row[0]), float(row[1]), float(row[2])
        v_pos, v_neg, temp_c = float(row[3]), float(row[4]), float(row[5])

        jfet = NChannelJFET(**board_dict["jfet_params"]).at_temp(temp_c)
        caps = JFETCapacitance(**board_dict["caps_params"])
        n_fanout = board_dict.get("n_fanout", 4)

        # Estimate simulation time from RC constants
        cgd0 = board_dict["caps_params"]["cgd0"]
        cgs0 = board_dict["caps_params"]["cgs0"]
        c_gate = cgd0 + cgs0
        r_out = (r2 * r3) / (r2 + r3) if (r2 + r3) > 0 else r2
        tau_max = max(r1 * c_gate, r_out * n_fanout * c_gate)
        t_settle = max(tau_max * 5, 0.5e-6)  # settle before step
        t_sim = t_settle + max(tau_max * 12, 2e-6)
        dt_max = max(tau_max / 20, 5e-9)  # coarser stepping for speed

        t_pd_hl = 0.0
        t_pd_lh = 0.0
        ok = True

        try:
            # --- Step LOW -> HIGH: measure output fall delay ---
            def make_step_up(t_s=t_settle):
                def fn(t):
                    return v_low if t < t_s else v_high
                return fn

            v_in_funcs_up = [make_step_up() for _ in range(n_inputs)]
            circuit_up = Circuit(
                v_pos=v_pos, v_neg=v_neg, r1=r1, r2=r2, r3=r3,
                j1=jfet, j2=jfet, caps=caps,
                v_in_funcs=v_in_funcs_up, gate_type_enum=gt_enum,
                n_fanout=n_fanout, temp_c=temp_c,
            )
            n_eval = min(500, max(100, int(t_sim / dt_max)))
            t_eval = np.linspace(0, t_sim, n_eval)
            res_up = simulate(circuit_up, t_span=(0, t_sim),
                              t_eval=t_eval, max_step=dt_max)
            timing_up = measure_timing(res_up)
            t_pd_hl = timing_up.get("tpd_hl_ns") or 0.0

            # --- Step HIGH -> LOW: measure output rise delay ---
            def make_step_down(t_s=t_settle):
                def fn(t):
                    return v_high if t < t_s else v_low
                return fn

            v_in_funcs_down = [make_step_down() for _ in range(n_inputs)]
            circuit_down = Circuit(
                v_pos=v_pos, v_neg=v_neg, r1=r1, r2=r2, r3=r3,
                j1=jfet, j2=jfet, caps=caps,
                v_in_funcs=v_in_funcs_down, gate_type_enum=gt_enum,
                n_fanout=n_fanout, temp_c=temp_c,
            )
            t_eval2 = np.linspace(0, t_sim, n_eval)
            res_down = simulate(circuit_down, t_span=(0, t_sim),
                                t_eval=t_eval2, max_step=dt_max)
            timing_down = measure_timing(res_down)
            t_pd_lh = timing_down.get("tpd_lh_ns") or 0.0

        except Exception:
            ok = False

        converged = ok and t_pd_hl > 0 and t_pd_lh > 0
        results.append((t_pd_hl, t_pd_lh, 1.0 if converged else 0.0))

    return np.array(results, dtype=np.float32)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_delay_dataset(gate_type, board, n_samples=10000,
                           n_workers=4, seed=42,
                           r1_range=(100.0, 500_000.0),
                           r23_range=(100.0, 500_000.0),
                           v_pos_range=None, v_neg_range=None,
                           progress_cb=None):
    """Generate delay training data using CPU transient sims.

    Args:
        gate_type: GateType enum
        board: BoardConfig
        n_samples: number of samples
        n_workers: parallel workers
        seed: RNG seed
        r1_range, r23_range: resistance ranges (log-sampled)
        v_pos_range, v_neg_range: optional voltage ranges

    Returns:
        dict with 'X' (N,6), 'Y' (N,2) [t_pd_hl_ns, t_pd_lh_ns],
        'mask' (N,) bool
    """
    from .data import _board_to_dict

    rng = np.random.default_rng(seed)

    # LHS sampling
    n = n_samples
    X = np.empty((n, 6), dtype=np.float64)

    # R values in log space
    for col, (lo, hi) in [(0, r1_range), (1, r23_range), (2, r23_range)]:
        u = (rng.permutation(n) + rng.uniform(size=n)) / n
        log_lo, log_hi = np.log10(lo), np.log10(hi)
        X[:, col] = 10.0 ** (log_lo + u * (log_hi - log_lo))

    # V+, V-, temp
    if v_pos_range is not None:
        u = (rng.permutation(n) + rng.uniform(size=n)) / n
        X[:, 3] = v_pos_range[0] + u * (v_pos_range[1] - v_pos_range[0])
    else:
        X[:, 3] = board.v_pos
    if v_neg_range is not None:
        u = (rng.permutation(n) + rng.uniform(size=n)) / n
        X[:, 4] = v_neg_range[0] + u * (v_neg_range[1] - v_neg_range[0])
    else:
        X[:, 4] = board.v_neg
    X[:, 5] = board.temp_c

    board_dict = _board_to_dict(board)
    board_dict["n_fanout"] = board.n_fanout
    gt_str = gate_type.value

    n_workers = min(n_workers, n)
    chunks = np.array_split(X, n_workers)
    args_list = [(chunk, gt_str, board_dict) for chunk in chunks]

    t0 = time.time()
    print(f"  Generating {n:,} delay samples for {gt_str} "
          f"using {n_workers} workers...")

    results_list = []
    if n_workers <= 1:
        results_list.append(_delay_worker(args_list[0]))
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_delay_worker, a): i
                       for i, a in enumerate(args_list)}
            done = 0
            ordered = [None] * len(futures)
            for fut in as_completed(futures):
                idx = futures[fut]
                ordered[idx] = fut.result()
                done += len(ordered[idx])
                if progress_cb:
                    progress_cb(done, n)
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (n - done) / rate if rate > 0 else 0
                print(f"    {done:,}/{n:,} ({done/n*100:.0f}%) "
                      f"rate={rate:.0f}/s  ETA={eta:.0f}s", end="\r")
            results_list = ordered

    Y_raw = np.vstack(results_list)  # (N, 3): t_pd_hl, t_pd_lh, converged
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s ({n/elapsed:.0f} samples/s)")

    Y = Y_raw[:, :2]  # t_pd_hl_ns, t_pd_lh_ns
    mask = Y_raw[:, 2].astype(bool)
    print(f"  Converged: {mask.sum():,}/{n:,} ({mask.sum()/n*100:.1f}%)")

    return {
        "X": X.astype(np.float32),
        "Y": Y,
        "mask": mask,
        "columns_X": list(COLUMNS_X),
        "columns_Y": ["t_pd_hl_ns", "t_pd_lh_ns"],
    }


def save_delay_dataset(dataset: dict, path: str):
    """Save delay dataset to .npz."""
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **{k: np.array(v) for k, v in dataset.items()})
    print(f"  Saved {p} ({p.stat().st_size / 1e6:.1f} MB)")


def load_delay_dataset(path: str) -> dict:
    """Load delay dataset from .npz."""
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"], "Y": d["Y"], "mask": d["mask"],
        "columns_X": list(d["columns_X"]),
        "columns_Y": list(d["columns_Y"]),
    }
