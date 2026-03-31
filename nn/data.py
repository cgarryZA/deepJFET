"""Training data generation -- sample R-space, solve with the real solver, build dataset.

The heavy lifting happens in ``generate_dataset``:
  1. Latin-Hypercube sample over (R1, R2, R3) and optionally (V+, V-, temp).
  2. For each sample, run the solver to get (V_out_high, V_out_low, avg_power).
  3. Pack into NumPy arrays and save to disk.

Inputs  (6): R1, R2, R3, V+, V-, temp
Outputs (3): V_out_high, V_out_low, avg_power
"""

import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import SamplingConfig
from .model import COLUMNS_X, COLUMNS_Y


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


def _build_samples(cfg: SamplingConfig, board, rng: np.random.Generator) -> np.ndarray:
    """Generate (n_samples, 6) array: R1, R2, R3, V+, V-, temp.

    R values are always sampled. V+, V-, temp are sampled if ranges are set,
    otherwise use fixed board values.
    """
    n = cfg.n_samples

    # Determine how many dimensions to LHS over
    varying = []  # (col_index, lo, hi, log_scale)
    # R1, R2, R3 always vary
    for col, rng_vals in [(0, cfg.r1_range), (1, cfg.r23_range), (2, cfg.r23_range)]:
        varying.append((col, rng_vals[0], rng_vals[1], cfg.log_r))
    # V+, V-, temp vary if ranges given
    if cfg.v_pos_range is not None:
        varying.append((3, cfg.v_pos_range[0], cfg.v_pos_range[1], False))
    if cfg.v_neg_range is not None:
        varying.append((4, cfg.v_neg_range[0], cfg.v_neg_range[1], False))
    if cfg.temp_range is not None:
        varying.append((5, cfg.temp_range[0], cfg.temp_range[1], False))

    lhs = _latin_hypercube(n, len(varying), rng)

    X = np.empty((n, 6), dtype=np.float64)
    # Fill fixed values first
    X[:, 3] = board.v_pos
    X[:, 4] = board.v_neg
    X[:, 5] = board.temp_c

    # Fill varying columns from LHS
    for i, (col, lo, hi, log_scale) in enumerate(varying):
        if log_scale:
            log_lo, log_hi = np.log10(lo), np.log10(hi)
            X[:, col] = 10.0 ** (log_lo + lhs[:, i] * (log_hi - log_lo))
        else:
            X[:, col] = lo + lhs[:, i] * (hi - lo)

    return X


# ---------------------------------------------------------------------------
# Worker function (runs in a subprocess)
# ---------------------------------------------------------------------------

def _solve_chunk(args):
    """Solve a chunk of samples. Runs in a worker process."""
    chunk_x, gate_type_str, board_dict = args

    # Reconstruct objects inside the worker (can't pickle JFET/Board directly)
    from model import NChannelJFET, JFETCapacitance, GateType
    from simulator.optimize import BoardConfig
    from simulator.gate_models import truth_table
    from model import solve_any_gate

    gate_type = GateType(gate_type_str)

    # Outputs: v_out_high, v_out_low, avg_power, converged
    outputs = []

    for row in chunk_x:
        r1, r2, r3 = float(row[0]), float(row[1]), float(row[2])
        v_pos, v_neg, temp_c = float(row[3]), float(row[4]), float(row[5])

        # Reconstruct JFET at this temperature
        jfet = NChannelJFET(**board_dict["jfet_params"]).at_temp(temp_c)
        caps = JFETCapacitance(**board_dict["caps_params"])

        # Determine logic levels from the board config
        # (these are the target levels, used to define "all-off" and "all-on")
        table = truth_table(gate_type)

        v_out_high = 0.0
        v_out_low = 0.0
        total_power = 0.0
        ok = True

        try:
            for combo, out_high in table:
                # Map boolean inputs to voltage levels
                # Use board's v_high/v_low for input mapping
                v_map = {False: board_dict["v_low"], True: board_dict["v_high"]}
                v_ins = [v_map[b] for b in combo]

                res = solve_any_gate(gate_type, v_ins,
                                     v_pos, v_neg, r1, r2, r3,
                                     jfet, jfet, temp_c)

                i_r1 = res["i_r1_mA"] * 1e-3
                i_j2 = res["i_j2_mA"] * 1e-3
                i_load = res["i_load_mA"] * 1e-3
                total_power += v_pos * (i_r1 + i_j2) + (-v_neg) * i_load

                if all(not b for b in combo):
                    v_out_high = res["v_out"]
                if all(b for b in combo):
                    v_out_low = res["v_out"]
        except Exception:
            ok = False

        n_states = len(table)
        avg_power = total_power / n_states if ok else 0.0

        outputs.append((v_out_high, v_out_low, avg_power, 1.0 if ok else 0.0))

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
    board : BoardConfig
    cfg : SamplingConfig, optional
    progress_cb : callable, optional

    Returns
    -------
    dict with keys:
        'X'       : (N, 6) float32 -- R1, R2, R3, V+, V-, temp
        'Y'       : (N, 3) float32 -- V_out_high, V_out_low, avg_power
        'mask'    : (N,)   bool    -- True if solver converged
        'columns_X' : list of str
        'columns_Y' : list of str
    """
    if cfg is None:
        cfg = SamplingConfig()

    rng = np.random.default_rng(cfg.seed)
    X = _build_samples(cfg, board, rng)

    n = cfg.n_samples
    board_dict = _board_to_dict(board)
    gt_str = gate_type.value

    # Split into chunks for workers
    n_workers = min(cfg.n_workers, n)
    chunks = np.array_split(X, n_workers)
    args_list = [(chunk, gt_str, board_dict) for chunk in chunks]

    t0 = time.time()

    # Try GPU backend first
    _has_cuda = False
    try:
        import torch
        _has_cuda = torch.cuda.is_available()
    except ImportError:
        pass

    if cfg.use_gpu and _has_cuda:
        try:
            device = "cuda"
            from .gpu_solver import gpu_solve_batch
            print(f"  Generating {n:,} samples for {gt_str} "
                  f"using GPU solver (device={device})...")

            varying = ["R1", "R2", "R3"]
            if cfg.v_pos_range is not None:
                varying.append(f"V+=[{cfg.v_pos_range[0]:.0f},{cfg.v_pos_range[1]:.0f}]")
            if cfg.v_neg_range is not None:
                varying.append(f"V-=[{cfg.v_neg_range[0]:.0f},{cfg.v_neg_range[1]:.0f}]")
            if cfg.temp_range is not None:
                varying.append(f"T=[{cfg.temp_range[0]:.0f},{cfg.temp_range[1]:.0f}]C")
            print(f"    Varying: {', '.join(varying)}")

            Y_raw = gpu_solve_batch(gt_str, X, board_dict, device=device)

            elapsed = time.time() - t0
            Y = Y_raw[:, :3].astype(np.float32)
            mask = Y_raw[:, 3].astype(bool)
            converged_pct = mask.sum() / n * 100
            print(f"  Done in {elapsed:.1f}s "
                  f"({n/elapsed:.0f} samples/s, GPU)")
            print(f"  Converged: {mask.sum():,}/{n:,} ({converged_pct:.1f}%)")

            return {
                "X": X[:, :6].astype(np.float32),
                "Y": Y,
                "mask": mask,
                "columns_X": list(COLUMNS_X),
                "columns_Y": list(COLUMNS_Y),
            }
        except Exception as e:
            print(f"  GPU solver failed ({e}), falling back to CPU...")

    # CPU fallback
    n_workers = min(cfg.n_workers, n)
    chunks = np.array_split(X, n_workers)
    args_list = [(chunk, gt_str, board_dict) for chunk in chunks]

    print(f"  Generating {n:,} samples for {gt_str} "
          f"using {n_workers} CPU workers...")

    # Report which dimensions are varying
    varying = ["R1", "R2", "R3"]
    if cfg.v_pos_range is not None:
        varying.append(f"V+=[{cfg.v_pos_range[0]:.0f},{cfg.v_pos_range[1]:.0f}]")
    if cfg.v_neg_range is not None:
        varying.append(f"V-=[{cfg.v_neg_range[0]:.0f},{cfg.v_neg_range[1]:.0f}]")
    if cfg.temp_range is not None:
        varying.append(f"T=[{cfg.temp_range[0]:.0f},{cfg.temp_range[1]:.0f}]C")
    print(f"    Varying: {', '.join(varying)}")

    results_list = []

    if n_workers <= 1:
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

    Y_raw = np.vstack(results_list)  # (N, 4): v_out_h, v_out_l, power, converged

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s "
          f"({n/elapsed:.0f} samples/s)")

    Y = Y_raw[:, :3].astype(np.float32)  # v_out_h, v_out_l, power
    mask = Y_raw[:, 3].astype(bool)

    converged_pct = mask.sum() / n * 100
    print(f"  Converged: {mask.sum():,}/{n:,} ({converged_pct:.1f}%)")

    return {
        "X": X[:, :6].astype(np.float32),  # R1, R2, R3, V+, V-, temp
        "Y": Y,
        "mask": mask,
        "columns_X": list(COLUMNS_X),
        "columns_Y": list(COLUMNS_Y),
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
