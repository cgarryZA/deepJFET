"""Demo: NN surrogate pipeline for gate optimization.

Default board: V+=10V, V-=-10V, 100kHz, 4-gate logic depth.

Usage:
    python nn/examples/optimize_with_nn.py
"""

import sys, os, multiprocessing
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)


def main():
    from model import NChannelJFET, JFETCapacitance, GateType
    from simulator.optimize import BoardConfig
    from nn import SurrogateGatePipeline, PipelineConfig, SamplingConfig, TrainConfig

    # --- JFET model ---
    JFET_NOM = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    )
    jfet = JFET_NOM.at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

    # --- Board: +/-10V rails ---
    board = BoardConfig(
        v_high=-0.8, v_low=-4.0,
        v_pos=10.0, v_neg=-10.0,
        jfet=jfet, caps=caps, temp_c=27.0,
        f_target=100e3, n_fanout=4,
        max_logic_depth=4,
    )

    # --- Pipeline config ---
    cfg = PipelineConfig(
        sampling=SamplingConfig(
            n_samples=50_000,
            n_workers=4,
            seed=42,
            v_pos_range=(8.0, 18.0),
            v_neg_range=(-18.0, -8.0),
        ),
        training=TrainConfig(
            hidden_dims=[256, 256, 128],
            activation="silu",
            lr=1e-3,
            batch_size=2048,
            max_epochs=300,
            patience=30,
        ),
        top_n_verify=50,
        e_series="E96",
    )

    # --- Run all gate types ---
    pipe = SurrogateGatePipeline(board, cfg)
    designs = pipe.run_all([GateType.INV, GateType.NAND2, GateType.NOR2])


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
