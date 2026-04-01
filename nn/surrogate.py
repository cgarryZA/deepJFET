"""Surrogate-based optimizer using DC NN + optional Delay NN.

DC NN: predicts V_out_high, V_out_low, power from R1/R2/R3/V+/V-/temp
Delay NN: predicts t_pd_hl, t_pd_lh in nanoseconds (replaces RC estimate)

Error is computed from V_out predictions vs target logic levels.
"""

import time
import numpy as np
import torch

from model import (
    GateType, solve_any_gate, e_series_values,
    estimate_prop_delay,
)
from simulator.optimize import BoardConfig, GateDesign
from simulator.gate_models import truth_table
from .model import GateSurrogateNet


class SurrogateOptimizer:
    """Optimizer using DC NN + optional Delay NN.

    Usage::

        opt = SurrogateOptimizer(dc_model, board, delay_model=delay_model)
        design = opt.optimize(GateType.INV, mode='min_power')
        design = opt.optimize(GateType.INV, mode='max_freq', max_power_mW=100)
    """

    def __init__(self, model: GateSurrogateNet, board: BoardConfig,
                 delay_model: GateSurrogateNet = None,
                 device: str = None):
        self.model = model
        self.delay_model = delay_model
        self.board = board
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        if self.delay_model is not None:
            self.delay_model.to(self.device)
            self.delay_model.eval()

    def _build_grid(self, series: str, r1_range: tuple, r23_range: tuple):
        """Build full E-series grid."""
        r1_vals = e_series_values(series, r1_range[0], r1_range[1])
        r2_vals = e_series_values(series, r23_range[0], r23_range[1])
        r3_vals = e_series_values(series, r23_range[0], r23_range[1])

        grid = []
        for r1 in r1_vals:
            for r2 in r2_vals:
                for r3 in r3_vals:
                    grid.append((r1, r2, r3))

        return np.array(grid, dtype=np.float32)

    def _predict_dc(self, grid: np.ndarray) -> np.ndarray:
        """DC NN prediction. Returns (N, 3): V_out_high, V_out_low, power."""
        n = len(grid)
        X = np.empty((n, 6), dtype=np.float32)
        X[:, 0:3] = grid
        X[:, 3] = self.board.v_pos
        X[:, 4] = self.board.v_neg
        X[:, 5] = self.board.temp_c

        batch_size = 65536
        preds = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = torch.tensor(X[i:i + batch_size],
                                  dtype=torch.float32, device=self.device)
                preds.append(self.model(xb).cpu().numpy())
        return np.vstack(preds)

    def _predict_delay(self, grid: np.ndarray) -> np.ndarray:
        """Delay NN prediction. Returns (N, 2): t_pd_hl_ns, t_pd_lh_ns."""
        if self.delay_model is None:
            # Fallback to RC estimate
            delays = np.zeros((len(grid), 2), dtype=np.float32)
            cgd = self.board.caps.cgd0
            cgs = self.board.caps.cgs0
            for i, (r1, r2, r3) in enumerate(grid):
                d = estimate_prop_delay(r1, r2, r3, cgd, cgs,
                                        self.board.n_fanout)
                delays[i] = [d * 1e9, d * 1e9]  # same for both directions
            return delays

        from .delay_model import predict_delay
        n = len(grid)
        X = np.empty((n, 6), dtype=np.float32)
        X[:, 0:3] = grid
        X[:, 3] = self.board.v_pos
        X[:, 4] = self.board.v_neg
        X[:, 5] = self.board.temp_c
        return predict_delay(self.delay_model, X, device=self.device)

    def _compute_error(self, preds: np.ndarray) -> np.ndarray:
        """Max logic-level error from DC predictions."""
        err_high = np.abs(preds[:, 0] - self.board.v_high)
        err_low = np.abs(preds[:, 1] - self.board.v_low)
        return np.maximum(err_high, err_low)

    def _verify_with_solver(self, gate_type: GateType,
                            candidates: np.ndarray,
                            candidate_delays: np.ndarray = None) -> list:
        """Run DC solver on candidates. Use delay NN predictions if available."""
        table = truth_table(gate_type)
        v_map = {False: self.board.v_low, True: self.board.v_high}
        results = []

        for idx, row in enumerate(candidates):
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

            # Use delay NN prediction if available, else RC estimate
            if candidate_delays is not None:
                delay_ns = float(max(candidate_delays[idx, 0],
                                     candidate_delays[idx, 1]))
            else:
                delay_ns = estimate_prop_delay(
                    r1, r2, r3, self.board.caps.cgd0,
                    self.board.caps.cgs0, self.board.n_fanout) * 1e9

            results.append((max_err, avg_power, r1, r2, r3,
                            v_out_high, v_out_low, delay_ns))

        return results

    def optimize(self, gate_type: GateType,
                 series: str = "E96",
                 r1_range: tuple = None,
                 r23_range: tuple = None,
                 max_error_tol: float = 0.2,
                 top_n_verify: int = 50,
                 mode: str = "min_power",
                 max_power_mW: float = 100.0) -> GateDesign:
        """Optimize gate resistors using DC NN + Delay NN.

        Modes:
            'min_power': minimize power subject to error < tol and delay < budget
            'max_freq':  minimize delay subject to error < tol and power < cap
        """
        t0 = time.time()
        max_delay_budget = self.board.max_gate_delay * 1e9  # ns

        if r1_range is None:
            r1_range = (100, 500_000)
        if r23_range is None:
            r23_range = (100, 500_000)

        # Step 1: Build grid
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
                f_target=self.board.f_target,
                max_logic_depth=self.board.max_logic_depth,
                temp_c=self.board.temp_c,
            )

        # Step 2: DC NN prediction
        preds = self._predict_dc(grid)
        t2 = time.time()
        print(f"    DC inference: {t2-t1:.3f}s "
              f"({len(grid)/(t2-t1):.0f} combos/s)")

        # Step 3: Error from V_out + select top candidates
        computed_errors = self._compute_error(preds)
        best_idx = np.argsort(computed_errors)[:top_n_verify]
        top_grid = grid[best_idx]

        print(f"    Top {top_n_verify} errors: "
              f"{computed_errors[best_idx[0]]*1e3:.1f}mV - "
              f"{computed_errors[best_idx[-1]]*1e3:.1f}mV")

        # Step 4: Delay NN on top candidates
        top_delays = self._predict_delay(top_grid)
        t3 = time.time()
        has_delay_nn = self.delay_model is not None
        if has_delay_nn:
            print(f"    Delay inference: {t3-t2:.3f}s  "
                  f"range={top_delays.max(axis=1).min():.0f}-"
                  f"{top_delays.max(axis=1).max():.0f}ns")

        # Step 5: Verify with real DC solver
        verified = self._verify_with_solver(gate_type, top_grid, top_delays)
        t4 = time.time()
        print(f"    Verified {len(verified)}/{top_n_verify} candidates "
              f"({t4-t3:.3f}s)")

        if not verified:
            i = best_idx[0]
            r1, r2, r3 = float(grid[i, 0]), float(grid[i, 1]), float(grid[i, 2])
            return GateDesign(
                gate_type=gate_type, r1=r1, r2=r2, r3=r3,
                v_pos=self.board.v_pos, v_neg=self.board.v_neg,
                v_high=float(preds[i, 0]), v_low=float(preds[i, 1]),
                swing=float(preds[i, 0] - preds[i, 1]),
                power_mW=float(preds[i, 2] * 1e3),
                delay_ns=0, max_error_mV=float(computed_errors[i] * 1e3),
                converged=False,
                f_target=self.board.f_target,
                max_logic_depth=self.board.max_logic_depth,
                temp_c=self.board.temp_c,
            )

        # Step 6: Pick best based on mode
        # verified tuples: (err, power, r1, r2, r3, v_h, v_l, delay_ns)
        within_tol = [r for r in verified if r[0] <= max_error_tol]

        if mode == "max_freq":
            power_cap = max_power_mW / 1e3
            feasible = [r for r in within_tol if r[1] <= power_cap]
            if feasible:
                feasible.sort(key=lambda x: x[7])  # min delay
                pick = feasible[0]
                max_f = 1e9 / (pick[7] * self.board.max_logic_depth) if pick[7] > 0 else 0
                print(f"    Best (max freq, P<{max_power_mW:.0f}mW): "
                      f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                      f"R3={pick[4]/1e3:.2f}k "
                      f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW "
                      f"delay={pick[7]:.0f}ns "
                      f"max_f={max_f/1e3:.0f}kHz")
            elif within_tol:
                within_tol.sort(key=lambda x: x[7])
                pick = within_tol[0]
                print(f"    Best (min delay, power uncapped): "
                      f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                      f"R3={pick[4]/1e3:.2f}k "
                      f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW "
                      f"delay={pick[7]:.0f}ns")
            else:
                verified.sort(key=lambda x: x[7])
                pick = verified[0]
                print(f"    Best (closest, outside tol): "
                      f"err={pick[0]*1e3:.1f}mV delay={pick[7]:.0f}ns")
        else:
            # min_power: must also meet delay budget
            if has_delay_nn:
                feasible = [r for r in within_tol if r[7] <= max_delay_budget]
            else:
                feasible = within_tol

            if feasible:
                feasible.sort(key=lambda x: x[1])  # min power
                pick = feasible[0]
                print(f"    Best (min power, err<{max_error_tol*1e3:.0f}mV, "
                      f"delay<{max_delay_budget:.0f}ns): "
                      f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                      f"R3={pick[4]/1e3:.2f}k "
                      f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW "
                      f"delay={pick[7]:.0f}ns")
            elif within_tol:
                within_tol.sort(key=lambda x: x[1])
                pick = within_tol[0]
                print(f"    Best (min power, delay unchecked): "
                      f"R1={pick[2]/1e3:.2f}k R2={pick[3]/1e3:.2f}k "
                      f"R3={pick[4]/1e3:.2f}k "
                      f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW "
                      f"delay={pick[7]:.0f}ns")
            else:
                verified.sort(key=lambda x: x[0])
                pick = verified[0]
                print(f"    Best (closest, outside tol): "
                      f"err={pick[0]*1e3:.1f}mV P={pick[1]*1e3:.2f}mW")

        max_err, power, r1, r2, r3, v_high, v_low, delay_ns = pick

        total_time = time.time() - t0
        print(f"    Total: {total_time:.2f}s")

        return GateDesign(
            gate_type=gate_type, r1=r1, r2=r2, r3=r3,
            v_pos=self.board.v_pos, v_neg=self.board.v_neg,
            v_high=float(v_high), v_low=float(v_low),
            swing=float(v_high - v_low),
            power_mW=float(power * 1e3),
            delay_ns=float(delay_ns),
            max_error_mV=float(max_err * 1e3),
            converged=bool(max_err <= max_error_tol),
            f_target=self.board.f_target,
            max_logic_depth=self.board.max_logic_depth,
            temp_c=self.board.temp_c,
        )
