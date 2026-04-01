"""End-to-end pipeline: generate data -> train -> optimize -> verify.

Accepts either GateType enums or PulldownNetwork topologies.

Usage::

    from model import GateType, Leaf, Series, Parallel
    from nn import SurrogateGatePipeline, PipelineConfig

    board = BoardConfig(...)
    pipe = SurrogateGatePipeline(board)

    # Standard gate types
    pipe.run(GateType.INV)

    # Arbitrary topology
    net = Parallel((Leaf("A"), Series((Leaf("B"), Leaf("C")))))
    pipe.run(net)
"""

import time
import hashlib
from pathlib import Path

from model import GateType
from simulator.optimize import BoardConfig, GateDesign
from .config import PipelineConfig
from .data import generate_dataset, save_dataset, load_dataset
from .train import train_surrogate, load_surrogate
from .surrogate import SurrogateOptimizer
from .registry import find_model, register_model, jfet_hash


def _topo_key(gate_type_or_network) -> str:
    """Get a string key for any gate type or network topology."""
    from model.network import PulldownNetwork, canonical_str
    if isinstance(gate_type_or_network, PulldownNetwork):
        cs = canonical_str(gate_type_or_network)
        # Hash long canonical strings for filesystem safety
        if len(cs) > 30:
            h = hashlib.md5(cs.encode()).hexdigest()[:8]
            return f"net_{h}"
        return cs.replace("(", "_").replace(")", "").replace(",", "_")
    return gate_type_or_network.value


class SurrogateGatePipeline:
    """One-click pipeline for any gate topology."""

    def __init__(self, board: BoardConfig, cfg: PipelineConfig = None):
        self.board = board
        self.cfg = cfg or PipelineConfig()
        self.designs = {}
        self._models = {}

    def _data_path(self, topo) -> str:
        jh = jfet_hash(self.board.jfet)
        key = _topo_key(topo)
        return str(Path(self.cfg.data_dir) / f"{key}_{jh}.npz")

    def _model_path(self, topo) -> str:
        jh = jfet_hash(self.board.jfet)
        key = _topo_key(topo)
        return str(Path(self.cfg.model_dir) / f"{key}_{jh}.pt")

    def generate(self, topo, force: bool = False) -> dict:
        path = self._data_path(topo)
        if not force and Path(path).exists():
            print(f"[1/3] Loading cached dataset: {path}")
            return load_dataset(path)

        key = _topo_key(topo)
        print(f"[1/3] Generating training data for {key}...")
        dataset = generate_dataset(topo, self.board, self.cfg.sampling)
        save_dataset(dataset, path)
        return dataset

    def train(self, topo, dataset: dict = None, force: bool = False):
        key = _topo_key(topo)
        path = self._model_path(topo)

        if not force and Path(path).exists():
            print(f"[2/3] Loading cached model: {path}")
            model = load_surrogate(path)
            self._models[key] = model
            return model

        if dataset is None:
            dataset = self.generate(topo)

        print(f"[2/3] Training surrogate for {key}...")
        model, history = train_surrogate(
            dataset, self.cfg.training, save_path=path,
        )
        self._models[key] = model
        return model

    def optimize(self, topo, model=None, **kwargs) -> GateDesign:
        key = _topo_key(topo)
        if model is None:
            model = self._models.get(key)
        if model is None:
            model = self.train(topo)

        print(f"[3/3] Optimizing {key} with NN surrogate...")
        optimizer = SurrogateOptimizer(model, self.board)

        # For verification, need gate_type for solve_any_gate or network for solve_network
        from model.network import PulldownNetwork
        if isinstance(topo, PulldownNetwork):
            design = optimizer.optimize_network(
                topo,
                series=self.cfg.e_series,
                r1_range=self.cfg.r1_range,
                r23_range=self.cfg.r23_range,
                top_n_verify=self.cfg.top_n_verify,
                **kwargs,
            )
        else:
            design = optimizer.optimize(
                topo,
                series=self.cfg.e_series,
                r1_range=self.cfg.r1_range,
                r23_range=self.cfg.r23_range,
                top_n_verify=self.cfg.top_n_verify,
                **kwargs,
            )
        self.designs[key] = design
        return design

    def run(self, topo, force_data: bool = False,
            force_train: bool = False, **kwargs) -> GateDesign:
        t0 = time.time()
        key = _topo_key(topo)
        print(f"\n{'='*60}")
        print(f"Pipeline: {key}")
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

        dataset = self.generate(topo, force=force_data)
        model = self.train(topo, dataset, force=force_train)
        design = self.optimize(topo, model, **kwargs)

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

    def run_all(self, topos: list, **kwargs) -> dict:
        for topo in topos:
            self.run(topo, **kwargs)

        print(f"\n{'='*60}")
        print(f"{'Type':<20} {'R1':>8} {'R2':>8} {'R3':>8} "
              f"{'V_H':>7} {'V_L':>7} {'Err':>8} {'Power':>8}")
        print("-" * 75)
        for key, d in self.designs.items():
            print(f"{key:<20} "
                  f"{d.r1/1e3:>7.2f}k {d.r2/1e3:>7.2f}k "
                  f"{d.r3/1e3:>7.2f}k "
                  f"{d.v_high:>7.3f} {d.v_low:>7.3f} "
                  f"{d.max_error_mV:>7.1f}mV "
                  f"{d.power_mW:>7.2f}mW")

        return self.designs
