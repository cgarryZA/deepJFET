"""Surrogate-based optimizer — replaces the grid-search fsolve loop with NN inference.

Given a trained GateSurrogateNet, sweeps the full E-series grid in a single
batched forward pass, then verifies top candidates with the real solver.
"""

import time
import numpy as np
import torch

from model import (
    GateType, solve_any_gate, e_series_values,
    estimate_prop_delay, max_r_out_for_freq,
)
from simulator.optimize import BoardConfig, GateDesign
from simulator.gate_models import truth_table
from .model import GateSurrogateNet


class SurrogateOptimizer:
    """Fast optimizer using a trained NN surrogate.

    Usage::

        opt = SurrogateOptimizer(model, board)
        design = opt.optimize(GateType.NAND2, top_n_verify=20)
    """

    def __init__(self, model: GateSurrogateNet, board: BoardConfig,
                 device: str = None):
        self.model = model
        self.board = board
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _build_grid(self, series: str, r1_range: tuple, r23_range: tuple):
        """Build the full E-series grid as a numpy array."""
        r1_vals = e_series_values(series, r1_range[0], r1_range[1])
        r2_vals = e_series_values(series, r23_range[0], r23_range[1])
        r3_vals = e_series_values(series, r23_range[0], r23_range[1])

        # Speed pre-filter
        cgd = self.board.caps.cgd0
        cgs = self.board.caps.cgs0
        r_out_max = max_r_out_for_freq(self.board.f_target, cgd, cgs,
                                        self.board.n_fanout)
        max_hp = 1.0 / (2.0 * self.board.f_target)

        grid = []
        for r1 in r1_vals:
            if 0.7 * r1 * (cgd + cgs) > max_hp:
                continue
            for r2 in r2_vals:
                for r3 in r3_vals:
                    r_out = (r2 * r3) / (r2 + r3)
                    if r_out > r_out_max:
                        continue
                    d = estimate_prop_delay(r1, r2, r3, cgd, cgs,
                                            self.board.n_fanout)
                    if d > max_hp:
                        continue
                    grid.append((r1, r2, r3))

        return np.array(grid, dtype=np.float32)

    def predict_grid(self, grid: np.ndarray) -> np.ndarray:
        """Run the NN on a grid of (R1, R2, R3) values.

        Returns (N, 4): V_out_high, V_out_low, avg_power, max_error
        """
        n = len(grid)
        # Build full feature matrix: [R1, R2, R3, V+, V-, V_HIGH, V_LOW, temp]
        X = np.empty((n, 8), dtype=np.float32)
        X[:, 0:3] = grid
        X[:, 3] = self.board.v_pos
        X[:, 4] = self.board.v_neg
        X[:, 5] = self.board.v_high
        X[:, 6] = self.board.v_low
        X[:, 7] = self.board.temp_c

        # Batched inference
        batch_size = 65536
        preds = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.tensor(X[i:i + batch_size],
                                  dtype=torch.float32,
                                  device=self.device)
                preds.append(self.model(xb).cpu().numpy())

        return np.vstack(preds)

    def _verify_with_solver(self, gate_type: GateType,
                            candidates: np.ndarray) -> list:
        """Run the real solver on candidate (R1, R2, R3) combos.

        Returns list of (max_err, power, r1, r2, r3, v_high, v_low, delay)
        matching the grid-search result format.
        """
        table = truth_table(gate_type)
        v_map = {False: self.board.v_low, True: self.board.v_high}
        results = []

        for row in candidates:
            r1, r2, r3 = float(row[0]), float(row[1]), float(row[2])
            max_err = 0.0
            total_power = 0.0
            v_out_high = None
            v_out_low = None
            ok = True

            try:
                for combo, out_high in table:
                    v_ins = [v_map[b] for b in combo]
                    target = self.board.v_high if out_high else self.board.v_low
                    res = solve_any_gate(gate_type, v_ins,
                                         self.board.v_pos, self.board.v_neg,
                                         r1, r2, r3,
                                         self.board.jfet, self.board.jfet,
                                         self.board.temp_c)
                    err = abs(res["v_out"] - target)
                    max_err = max(max_err, err)

                    i_r1 = res["i_r1_mA"] * 1e-3
                    i_j2 = res["i_j2_mA"] * 1e-3
                    i_load = res["i_load_mA"] * 1e-3
                    total_power += (self.board.v_pos * (i_r1 + i_j2)
                                    + (-self.board.v_neg) * i_load)

                    if all(not b for b in combo):
                        v_out_high = res["v_out"]
                    if all(b for b in combo):
                        v_out_low = res["v_out"]
            except Exception:
                ok = False

            if not ok:
                continue

            n_states = len(table)
            avg_power = total_power / n_states
            delay = estimate_prop_delay(r1, r2, r3,
                                        self.board.caps.cgd0,
                                        self.board.caps.cgs0,
                                        self.board.n_fanout)
            results.append((max_err, avg_power, r1, r2, r3,
                            v_out_high, v_out_low, delay))

        return results

    def optimize(self, gate_type: GateType,
                 series: str = "E96",
                 r1_range: tuple = None,
                 r23_range: tuple = None,
                 max_error_tol: float = 0.2,
                 top_n_verify: int = 20) -> GateDesign:
        """Full optimization: NN sweep → verify top candidates → pick best.

        Parameters
        ----------
        gate_type : GateType
        series : str
            E-series for the grid ('E12', 'E24', 'E96').
        r1_range, r23_range : tuple, optional
            Override auto-computed resistance ranges.
        max_error_tol : float
            Max acceptable output error in volts.
        top_n_verify : int
            Number of top NN predictions to verify with real solver.

        Returns
        -------
        GateDesign
        """
        # Auto-compute R ranges from board constraints
        cgd = self.board.caps.cgd0
        cgs = self.board.caps.cgs0
        r_out_max = max_r_out_for_freq(self.board.f_target, cgd, cgs,
                                        self.board.n_fanout)
        max_hp = 1.0 / (2.0 * self.board.f_target)
        r1_max = max_hp / (10.0 * (cgd + cgs))

        if r1_range is None:
            r1_range = (100, min(r1_max, 500_000))
        if r23_range is None:
            r23_range = (100, min(2 * r_out_max, 500_000))

        # Step 1: Build grid
        t0 = time.time()
        grid = self._build_grid(series, r1_range, r23_range)
        t1 = time.time()
        print(f"  {gate_type.value}: {series} grid = {len(grid):,} combos "
              f"({t1-t0:.2f}s)")

        if len(grid) == 0:
            print(f"    WARNING: empty grid for {gate_type.value}")
            return GateDesign(
                gate_type=gate_type, r1=1e3, r2=1e3, r3=1e3,
                v_pos=self.board.v_pos, v_neg=self.board.v_neg,
                v_high=0, v_low=0, swing=0, power_mW=0,
                delay_ns=0, max_error_mV=9999, converged=False,
            )

        # Step 2: NN prediction
        preds = self.predict_grid(grid)
        t2 = time.time()
        print(f"    NN inference: {t2-t1:.3f}s "
              f"({len(grid)/(t2-t1):.0f} combos/s)")

        # Step 3: Rank by predicted error, take top candidates
        pred_errors = preds[:, 3]  # max_error column
        best_idx = np.argsort(pred_errors)[:top_n_verify]
        top_grid = grid[best_idx]
        top_preds = preds[best_idx]

        print(f"    Top {top_n_verify} predicted errors: "
              f"{pred_errors[best_idx[0]]*1e3:.1f}mV — "
              f"{pred_errors[best_idx[-1]]*1e3:.1f}mV")

        # Step 4: Verify with real solver
        verified = self._verify_with_solver(gate_type, top_grid)
        t3 = time.time()
        print(f"    Verified {len(verified)}/{top_n_verify} candidates "
              f"({t3-t2:.3f}s)")

        if not verified:
            # Fall back: take NN's best guess unverified
            i = best_idx[0]
            r1, r2, r3 = float(grid[i, 0]), float(grid[i, 1]), float(grid[i, 2])
            return GateDesign(
                gate_type=gate_type, r1=r1, r2=r2, r3=r3,
                v_pos=self.board.v_pos, v_neg=self.board.v_neg,
                v_high=float(preds[i, 0]), v_low=float(preds[i, 1]),
                swing=float(preds[i, 0] - preds[i, 1]),
                power_mW=float(preds[i, 2] * 1e3),
                delay_ns=0, max_error_mV=float(preds[i, 3] * 1e3),
                converged=False,
            )

        # Step 5: Among verified results within tolerance, pick lowest power
        within_tol = [r for r in verified if r[0] <= max_error_tol]
        if within_tol:
            within_tol.sort(key=lambda x: x[1])  # sort by power
            pick = within_tol[0]
            print(f"    Best (min power within {max_error_tol*1e3:.0f}mV): "
                  f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                  f"R3={pick[4]/1e3:.2f}k "
                  f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW")
        else:
            verified.sort(key=lambda x: x[0])  # sort by error
            pick = verified[0]
            print(f"    Best (closest, outside tol): "
                  f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                  f"R3={pick[4]/1e3:.2f}k "
                  f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW")

        max_err, power, r1, r2, r3, v_high, v_low, delay = pick

        total_time = time.time() - t0
        print(f"    Total: {total_time:.2f}s")

        return GateDesign(
            gate_type=gate_type, r1=r1, r2=r2, r3=r3,
            v_pos=self.board.v_pos, v_neg=self.board.v_neg,
            v_high=float(v_high), v_low=float(v_low),
            swing=float(v_high - v_low),
            power_mW=float(power * 1e3),
            delay_ns=float(delay * 1e9),
            max_error_mV=float(max_err * 1e3),
            converged=bool(max_err <= max_error_tol),
        )
