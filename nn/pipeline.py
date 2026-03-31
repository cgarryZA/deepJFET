"""End-to-end pipeline: generate data -> train -> optimize -> verify.

Usage::

    from model import GateType, NChannelJFET, JFETCapacitance
    from simulator.optimize import BoardConfig
    from nn import SurrogateGatePipeline, PipelineConfig

    board = BoardConfig(...)
    pipe = SurrogateGatePipeline(board)
    design = pipe.run(GateType.NAND2)
"""

import time
from pathlib import Path

from model import GateType
from simulator.optimize import BoardConfig, GateDesign
from .config import PipelineConfig
from .data import generate_dataset, save_dataset, load_dataset
from .train import train_surrogate, load_surrogate
from .surrogate import SurrogateOptimizer
from .registry import find_model, register_model, jfet_hash


class SurrogateGatePipeline:
    """One-click pipeline: new gate -> data -> train -> optimise.

    Each step checks for cached results (dataset on disk, model registry)
    so re-runs skip completed work.
    """

    def __init__(self, board: BoardConfig, cfg: PipelineConfig = None):
        self.board = board
        self.cfg = cfg or PipelineConfig()
        self.designs = {}
        self._models = {}

    def _data_path(self, gate_type: GateType) -> str:
        jh = jfet_hash(self.board.jfet)
        return str(Path(self.cfg.data_dir) / f"{gate_type.value}_{jh}.npz")

    def _model_path(self, gate_type: GateType) -> str:
        jh = jfet_hash(self.board.jfet)
        return str(Path(self.cfg.model_dir) / f"{gate_type.value}_{jh}.pt")

    # ----- Step 1: Generate data -----

    def generate(self, gate_type: GateType, force: bool = False) -> dict:
        """Generate training data (or load from cache)."""
        path = self._data_path(gate_type)

        if not force and Path(path).exists():
            print(f"[1/3] Loading cached dataset: {path}")
            return load_dataset(path)

        print(f"[1/3] Generating training data for {gate_type.value}...")
        dataset = generate_dataset(
            gate_type, self.board, self.cfg.sampling,
        )
        save_dataset(dataset, path)
        return dataset

    # ----- Step 2: Train -----

    def train(self, gate_type: GateType, dataset: dict = None,
              force: bool = False):
        """Train the surrogate (or load from registry/cache)."""
        # Check registry first
        if not force:
            entry = find_model(self.cfg.model_dir, gate_type,
                               self.board.jfet, self.board)
            if entry is not None:
                model_path = entry["model_path"]
                if Path(model_path).exists():
                    print(f"[2/3] Loading registered model: {model_path}")
                    model = load_surrogate(model_path)
                    self._models[gate_type] = model
                    return model

        # Fallback to file path check
        path = self._model_path(gate_type)
        if not force and Path(path).exists():
            print(f"[2/3] Loading cached model: {path}")
            model = load_surrogate(path)
            self._models[gate_type] = model
            return model

        if dataset is None:
            dataset = self.generate(gate_type)

        print(f"[2/3] Training surrogate for {gate_type.value}...")
        model, history = train_surrogate(
            dataset, self.cfg.training, save_path=path,
        )
        self._models[gate_type] = model

        # Register the model
        best_mae = min(history["val_mae"]) if history["val_mae"] else None
        register_model(
            self.cfg.model_dir,
            gate_type,
            self.board.jfet,
            path,
            sampling_cfg=self.cfg.sampling,
            metrics={"val_mae": best_mae},
        )

        return model

    # ----- Step 3: Optimize -----

    def optimize(self, gate_type: GateType, model=None) -> GateDesign:
        """Run surrogate-based optimization."""
        if model is None:
            model = self._models.get(gate_type)
        if model is None:
            model = self.train(gate_type)

        print(f"[3/3] Optimizing {gate_type.value} with NN surrogate...")
        optimizer = SurrogateOptimizer(model, self.board)
        design = optimizer.optimize(
            gate_type,
            series=self.cfg.e_series,
            r1_range=self.cfg.r1_range,
            r23_range=self.cfg.r23_range,
            top_n_verify=self.cfg.top_n_verify,
        )
        self.designs[gate_type] = design
        return design

    # ----- All-in-one -----

    def run(self, gate_type: GateType, force_data: bool = False,
            force_train: bool = False) -> GateDesign:
        """Full pipeline: generate -> train -> optimize."""
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"Pipeline: {gate_type.value}")
        print(f"  Board: V+={self.board.v_pos:.0f}V  "
              f"V-={self.board.v_neg:.0f}V  "
              f"V_HIGH={self.board.v_high:.2f}V  "
              f"V_LOW={self.board.v_low:.2f}V")
        max_delay_ns = self.board.max_gate_delay * 1e9
        print(f"  f={self.board.f_target/1e3:.0f}kHz  "
              f"depth={self.board.max_logic_depth}  "
              f"budget={max_delay_ns:.0f}ns/gate  "
              f"T={self.board.temp_c:.0f}C  "
              f"fanout={self.board.n_fanout}")
        print(f"{'='*60}")

        dataset = self.generate(gate_type, force=force_data)
        model = self.train(gate_type, dataset, force=force_train)
        design = self.optimize(gate_type, model)

        elapsed = time.time() - t0
        print(f"\nPipeline complete in {elapsed:.1f}s")
        print(f"  Result: R1={design.r1/1e3:.2f}k  "
              f"R2={design.r2/1e3:.2f}k  R3={design.r3/1e3:.2f}k")
        print(f"  V_HIGH={design.v_high:.3f}V  V_LOW={design.v_low:.3f}V  "
              f"swing={design.swing:.3f}V")
        print(f"  Power={design.power_mW:.2f}mW  "
              f"Delay={design.delay_ns:.1f}ns  "
              f"Error={design.max_error_mV:.1f}mV  "
              f"{'PASS' if design.converged else 'FAIL'}")

        return design

    def run_all(self, gate_types: list, **kwargs) -> dict:
        """Run the pipeline for multiple gate types."""
        for gt in gate_types:
            self.run(gt, **kwargs)

        # Summary table
        print(f"\n{'='*60}")
        print(f"{'Type':<7} {'R1':>8} {'R2':>8} {'R3':>8} "
              f"{'V_H':>7} {'V_L':>7} {'Err':>8} {'Power':>8}")
        print("-" * 70)
        for gt, d in self.designs.items():
            print(f"{gt.value:<7} "
                  f"{d.r1/1e3:>7.2f}k {d.r2/1e3:>7.2f}k "
                  f"{d.r3/1e3:>7.2f}k "
                  f"{d.v_high:>7.3f} {d.v_low:>7.3f} "
                  f"{d.max_error_mV:>7.1f}mV "
                  f"{d.power_mW:>7.2f}mW")

        return self.designs
