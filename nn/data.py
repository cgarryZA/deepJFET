"""Training data generation — sample R-space, solve with the real solver, build dataset.

The heavy lifting happens in ``generate_dataset``:
  1. Latin-Hypercube (or log-uniform) sample over (R1, R2, R3) and optionally
     (V+, V-, temp).
  2. For each sample, run ``_evaluate_combo`` from the existing optimizer to get
     (V_out_high, V_out_low, avg_power, max_error).
  3. Pack into NumPy arrays and save to disk.

Parallelised with multiprocessing — each worker gets a slice of samples.
"""

import os
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import SamplingConfig


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _latin_hypercube(n: int, dims: int, rng: np.random.Generator) -> np.ndarray:
    """Simple LHS: stratified random in [0,1]^dims."""
    result = np.empty((n, dims))
    for d in range(dims):
        perm = rng.permutation(n)
        result[:, d] = (perm + rng.uniform(size=n)) / n
    return result


def _sample_r_values(cfg: SamplingConfig, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_samples, 3) array of R1, R2, R3 values."""
    lhs = _latin_hypercube(cfg.n_samples, 3, rng)

    ranges = [cfg.r1_range, cfg.r23_range, cfg.r23_range]
    result = np.empty_like(lhs)

    for col, (lo, hi) in enumerate(ranges):
        if cfg.log_r:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            result[:, col] = 10.0 ** (log_lo + lhs[:, col] * (log_hi - log_lo))
        else:
            result[:, col] = lo + lhs[:, col] * (hi - lo)

    return result


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _solve_chunk(args):
    """Solve a chunk of R-combos.  Runs in a worker process."""
    chunk_r, gate_type_str, board_dict = args

    # Reconstruct objects inside the worker (can't pickle JFET/Board directly)
    from model import NChannelJFET, JFETCapacitance, GateType
    from simulator.optimize import BoardConfig
    from simulator.gate_models import truth_table
    from model import solve_any_gate

    gate_type = GateType(gate_type_str)

    jfet = NChannelJFET(**board_dict["jfet_params"]).at_temp(board_dict["temp_c"])
    caps = JFETCapacitance(**board_dict["caps_params"])
    board = BoardConfig(
        v_high=board_dict["v_high"], v_low=board_dict["v_low"],
        v_pos=board_dict["v_pos"], v_neg=board_dict["v_neg"],
        jfet=jfet, caps=caps, temp_c=board_dict["temp_c"],
        f_target=board_dict["f_target"], n_fanout=board_dict["n_fanout"],
    )

    table = truth_table(gate_type)
    v_map = {False: board.v_low, True: board.v_high}

    outputs = []  # (v_out_high, v_out_low, avg_power, max_error, converged)

    for row in chunk_r:
        r1, r2, r3 = float(row[0]), float(row[1]), float(row[2])
        max_err = 0.0
        total_power = 0.0
        v_out_high = 0.0
        v_out_low = 0.0
        ok = True

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

                if all(not b for b in combo):
                    v_out_high = res["v_out"]
                if all(b for b in combo):
                    v_out_low = res["v_out"]
        except Exception:
            ok = False

        n_states = len(table)
        avg_power = total_power / n_states if ok else 0.0

        outputs.append((v_out_high, v_out_low, avg_power, max_err, 1.0 if ok else 0.0))

    return np.array(outputs)


# ---------------------------------------------------------------------------
# Serialise board config for pickling into workers
# ---------------------------------------------------------------------------

def _board_to_dict(board) -> dict:
    """Flatten a BoardConfig + JFET into a plain dict for multiprocessing."""
    j = board.jfet
    return {
        "v_high": board.v_high, "v_low": board.v_low,
        "v_pos": board.v_pos, "v_neg": board.v_neg,
        "temp_c": board.temp_c,
        "f_target": board.f_target, "n_fanout": board.n_fanout,
        "jfet_params": {
            "beta": j.beta, "vto": j.vto, "lmbda": j.lmbda,
            "is_": j.is_, "n": j.n, "isr": j.isr, "nr": j.nr,
            "alpha": j.alpha, "vk": j.vk, "rd": j.rd, "rs": j.rs,
            "betatce": j.betatce, "vtotc": j.vtotc, "xti": j.xti, "eg": j.eg,
        },
        "caps_params": {
            "cgs0": board.caps.cgs0, "cgd0": board.caps.cgd0,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_dataset(gate_type, board, cfg: SamplingConfig = None,
                     progress_cb=None):
    """Generate training data for a single gate type.

    Parameters
    ----------
    gate_type : GateType
        Which gate to generate data for.
    board : BoardConfig
        Board-level configuration (rails, JFET, caps, temp, freq).
    cfg : SamplingConfig, optional
        Sampling parameters.  Defaults to SamplingConfig().
    progress_cb : callable, optional
        Called with (n_done, n_total) periodically.

    Returns
    -------
    dict with keys:
        'X'       : (N, 8) float32 — input features
        'Y'       : (N, 4) float32 — output targets
        'mask'    : (N,)   bool    — True if solver converged
        'columns_X' : list of str
        'columns_Y' : list of str
    """
    if cfg is None:
        cfg = SamplingConfig()

    rng = np.random.default_rng(cfg.seed)
    r_samples = _sample_r_values(cfg, rng)

    # Build full input feature matrix:
    # [R1, R2, R3, V+, V-, V_HIGH, V_LOW, temp]
    n = cfg.n_samples
    X = np.empty((n, 8), dtype=np.float64)
    X[:, 0:3] = r_samples
    X[:, 3] = board.v_pos
    X[:, 4] = board.v_neg
    X[:, 5] = board.v_high
    X[:, 6] = board.v_low
    X[:, 7] = board.temp_c

    board_dict = _board_to_dict(board)
    gt_str = gate_type.value

    # Split into chunks for workers
    n_workers = min(cfg.n_workers, n)
    chunks = np.array_split(r_samples, n_workers)
    args_list = [(chunk, gt_str, board_dict) for chunk in chunks]

    t0 = time.time()
    print(f"  Generating {n:,} samples for {gt_str} "
          f"using {n_workers} workers...")

    results_list = []

    if n_workers <= 1:
        # Single-process (easier to debug)
        result = _solve_chunk(args_list[0])
        results_list.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(_solve_chunk, a): i
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

    Y_raw = np.vstack(results_list)  # (N, 5): v_out_h, v_out_l, power, error, converged

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s "
          f"({n/elapsed:.0f} samples/s)")

    Y = Y_raw[:, :4].astype(np.float32)  # v_out_h, v_out_l, power, error
    mask = Y_raw[:, 4].astype(bool)

    converged_pct = mask.sum() / n * 100
    print(f"  Converged: {mask.sum():,}/{n:,} ({converged_pct:.1f}%)")

    return {
        "X": X.astype(np.float32),
        "Y": Y,
        "mask": mask,
        "columns_X": ["R1", "R2", "R3", "V+", "V-", "V_HIGH", "V_LOW", "temp"],
        "columns_Y": ["V_out_high", "V_out_low", "avg_power", "max_error"],
    }


def save_dataset(dataset: dict, path: str):
    """Save dataset to a .npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=dataset["X"], Y=dataset["Y"], mask=dataset["mask"],
        columns_X=dataset["columns_X"], columns_Y=dataset["columns_Y"],
    )
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {path} ({size_mb:.1f} MB)")


def load_dataset(path: str) -> dict:
    """Load dataset from a .npz file."""
    d = np.load(path, allow_pickle=True)
    return {
        "X": d["X"], "Y": d["Y"], "mask": d["mask"],
        "columns_X": list(d["columns_X"]),
        "columns_Y": list(d["columns_Y"]),
    }
