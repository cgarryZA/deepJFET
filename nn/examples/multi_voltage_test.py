"""Test: Train one model with variable V_POS/V_NEG, optimize for multiple rail configs.

Demonstrates that a single training run covers all voltage configurations.

Usage:
    python nn/examples/multi_voltage_test.py
"""

import sys, os, multiprocessing
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _root)


def main():
    from model import NChannelJFET, JFETCapacitance, GateType
    from simulator.optimize import BoardConfig
    from nn import SurrogateGatePipeline, PipelineConfig, SamplingConfig, TrainConfig
    from nn.surrogate import SurrogateOptimizer

    # --- JFET model ---
    JFET_NOM = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    )
    jfet = JFET_NOM.at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)

    # --- Train with variable voltage rails ---
    train_board = BoardConfig(
        v_high=-0.8, v_low=-4.0,
        v_pos=12.0, v_neg=-12.0,
        jfet=jfet, caps=caps, temp_c=27.0,
        f_target=100e3, n_fanout=4,
        max_logic_depth=4,
    )

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

    gate_types = [GateType.INV, GateType.NAND2, GateType.NOR2]

    # Train models
    pipe = SurrogateGatePipeline(train_board, cfg)
    print("=" * 70)
    print("STEP 1: Train models (one-time cost per gate type)")
    print("=" * 70)
    models = {}
    for gt in gate_types:
        dataset = pipe.generate(gt, force=True)
        models[gt] = pipe.train(gt, dataset, force=True)

    # Optimize for each voltage config
    print("\n" + "=" * 70)
    print("STEP 2: Optimize for each voltage configuration")
    print("=" * 70)

    voltage_configs = [
        (10.0, -10.0),
        (15.0, -15.0),
    ]

    all_results = []
    for v_pos, v_neg in voltage_configs:
        board = BoardConfig(
            v_high=-0.8, v_low=-4.0,
            v_pos=v_pos, v_neg=v_neg,
            jfet=jfet, caps=caps, temp_c=27.0,
            f_target=100e3, n_fanout=4,
            max_logic_depth=4,
        )
        for gt in gate_types:
            opt = SurrogateOptimizer(models[gt], board)
            print(f"\n--- {gt.value}  V+={v_pos:.0f}V  V-={v_neg:.0f}V  "
                  f"(budget={board.max_gate_delay*1e9:.0f}ns/gate) ---")
            design = opt.optimize(gt, top_n_verify=50)
            all_results.append(design)

    # Summary table
    print(f"\n{'='*85}")
    print(f"{'Gate':<6} {'V+':>4} {'V-':>4} | {'R1':>8} {'R2':>8} {'R3':>8} | "
          f"{'V_H':>7} {'V_L':>7} {'Err':>7} | "
          f"{'Power':>8} {'Delay':>8} {'OK':>4}")
    print("-" * 85)
    for d in all_results:
        print(f"{d.gate_type.value:<6} {d.v_pos:>4.0f} {d.v_neg:>4.0f} | "
              f"{d.r1/1e3:>7.2f}k {d.r2/1e3:>7.2f}k {d.r3/1e3:>7.2f}k | "
              f"{d.v_high:>7.3f} {d.v_low:>7.3f} {d.max_error_mV:>6.0f}mV | "
              f"{d.power_mW:>7.2f}mW {d.delay_ns:>7.1f}ns "
              f"{'PASS' if d.converged else 'FAIL':>4}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
