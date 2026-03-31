"""Demo: NN surrogate pipeline for gate optimization.

Usage:
    python nn/examples/optimize_with_nn.py
"""

import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)

from model import NChannelJFET, JFETCapacitance, GateType
from simulator.optimize import BoardConfig
from nn import SurrogateGatePipeline, PipelineConfig, SamplingConfig, TrainConfig

# --- Board config (same as simulator example) ---

JFET_NOM = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
)
jfet = JFET_NOM.at_temp(27.0)
caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

board = BoardConfig(
    v_high=-0.8, v_low=-4.0,
    v_pos=24.0, v_neg=-20.0,
    jfet=jfet, caps=caps, temp_c=27.0,
    f_target=100e3, n_fanout=4,
)

# --- Pipeline config ---

cfg = PipelineConfig(
    sampling=SamplingConfig(
        n_samples=50_000,      # 50k points — enough for a good model
        n_workers=4,
        seed=42,
    ),
    training=TrainConfig(
        hidden_dims=[256, 256, 128],
        activation="silu",
        lr=1e-3,
        batch_size=2048,
        max_epochs=300,
        patience=25,
    ),
    top_n_verify=20,
    e_series="E96",
)

# --- Run ---

pipe = SurrogateGatePipeline(board, cfg)

# Single gate:
# design = pipe.run(GateType.INV)

# All gates:
designs = pipe.run_all([GateType.INV, GateType.NAND2, GateType.NOR2])
