"""NN Audit: generate data, train, produce diagnostic plots."""
import sys
import os
import multiprocessing

sys.path.insert(0, os.path.dirname(__file__))


def main():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    import torch

    from model import GateType, NChannelJFET, JFETCapacitance
    from simulator.optimize import BoardConfig
    from nn.data import generate_dataset, save_dataset, load_dataset
    from nn.train import train_surrogate
    from nn.config import SamplingConfig, TrainConfig

    PLOT_DIR = Path("nn_plots")
    PLOT_DIR.mkdir(exist_ok=True)

    # ── Board config (same as working examples) ──
    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    board = BoardConfig(
        v_high=-0.8, v_low=-4.0,
        v_pos=24.0, v_neg=-20.0,
        jfet=jfet, caps=caps, temp_c=27.0,
        f_target=100e3, n_fanout=4,
    )

    gate_type = GateType.INV

    # ── Step 1: Generate data ──
    data_path = Path("nn/data/INV_audit_v2.npz")
    data_path.parent.mkdir(parents=True, exist_ok=True)

    if data_path.exists():
        print("Loading cached dataset...")
        dataset = load_dataset(str(data_path))
    else:
        print("Generating 50k samples...")
        cfg = SamplingConfig(n_samples=50_000, n_workers=4, seed=42)
        dataset = generate_dataset(gate_type, board, cfg)
        save_dataset(dataset, str(data_path))

    X = dataset["X"]
    Y = dataset["Y"]
    mask = dataset["mask"]
    cols_x = list(dataset["columns_X"])
    cols_y = list(dataset["columns_Y"])

    print(f"\nDataset: {X.shape[0]} samples, {mask.sum()} converged "
          f"({mask.sum()/len(mask)*100:.1f}%)")

    Xc = X[mask]
    Yc = Y[mask]

    # ── Data distribution plots ──
    print("\n=== Y stats (converged) ===")
    for i, c in enumerate(cols_y):
        v = Yc[:, i]
        print(f"  {c:>12}: min={v.min():.4f}  max={v.max():.4f}  "
              f"mean={v.mean():.4f}  std={v.std():.4f}")

    # Plot 1: Output distributions
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, (ax, name) in enumerate(zip(axes.flat, cols_y)):
        v = Yc[:, i]
        ax.hist(v, bins=100, alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_title(f"{name} distribution (converged)")
        ax.set_xlabel(name)
        ax.set_ylabel("Count")
        ax.axvline(v.mean(), color="red", linestyle="--",
                   label=f"mean={v.mean():.3f}")
        ax.axvline(np.median(v), color="orange", linestyle="--",
                   label=f"median={np.median(v):.3f}")
        ax.legend()
    fig.suptitle("Training data output distributions", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "01_output_distributions.png", dpi=150)
    plt.close(fig)
    print(f"Saved 01_output_distributions.png")

    # Plot 2: max_error distribution (zoomed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    errs = Yc[:, 3]
    ax1.hist(errs, bins=100, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax1.set_title("max_error - full range")
    ax1.set_xlabel("max_error (V)")

    low_errs = errs[errs < 1.0]
    if len(low_errs) > 0:
        ax2.hist(low_errs, bins=100, alpha=0.7, color="green",
                 edgecolor="black", linewidth=0.5)
    ax2.set_title("max_error < 1V (the region we care about)")
    ax2.set_xlabel("max_error (V)")
    for thresh in [0.1, 0.2, 0.5]:
        n = (errs < thresh).sum()
        ax2.axvline(thresh, color="red", linestyle="--", alpha=0.7)
        ax2.text(thresh + 0.01, ax2.get_ylim()[1] * 0.9,
                 f"{n} ({n/len(errs)*100:.1f}%)", fontsize=8, color="red")
    fig.suptitle("max_error distribution - most samples are FAR from the good region",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "02_error_distribution.png", dpi=150)
    plt.close(fig)
    print(f"Saved 02_error_distribution.png")

    # ── Step 2: Train the model (current architecture) ──
    print("\n=== Training current model ===")
    train_cfg = TrainConfig(
        hidden_dims=[256, 256, 128],
        activation="silu",
        max_epochs=300,
        patience=30,
        lr=1e-3,
        batch_size=2048,
    )
    model_path = "nn/models/INV_audit_v2.pt"
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model, history = train_surrogate(dataset, train_cfg, save_path=model_path)

    # Plot 3: Learning curves
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    ax1.plot(history["train_loss"], label="train", alpha=0.8)
    ax1.plot(history["val_loss"], label="val", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE Loss (normalised)")
    ax1.set_title("Loss curves")
    ax1.set_yscale("log")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["val_mae"], label="val MAE", color="green", alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MAE (real units, avg over outputs)")
    ax2.set_title("Validation MAE")
    ax2.grid(True, alpha=0.3)

    ax3.plot(history["lr"], label="LR", color="orange")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("LR Schedule")
    ax3.set_yscale("log")
    ax3.grid(True, alpha=0.3)

    fig.suptitle("Training diagnostics", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "03_learning_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved 03_learning_curves.png")

    # ── Step 3: Prediction vs Truth plots ──
    model.eval()
    model.to("cpu")

    # Use validation set (last 15%)
    n_val = int(len(Xc) * 0.15)
    idx = np.random.RandomState(42).permutation(len(Xc))
    X_val = Xc[idx[-n_val:]]
    Y_val = Yc[idx[-n_val:]]

    with torch.no_grad():
        X_t = torch.tensor(X_val, dtype=torch.float32)
        Y_pred = model(X_t).numpy()

    # Plot 4: Pred vs True scatter
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for i, (ax, name) in enumerate(zip(axes.flat, cols_y)):
        true = Y_val[:, i]
        pred = Y_pred[:, i]
        n_plot = min(5000, len(true))
        idx_plot = np.random.choice(len(true), n_plot, replace=False)

        ax.scatter(true[idx_plot], pred[idx_plot], s=1, alpha=0.3, color="blue")
        lo = min(true.min(), pred.min())
        hi = max(true.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="perfect")

        mae = np.abs(true - pred).mean()
        ss_res = np.sum((true - pred)**2)
        ss_tot = np.sum((true - true.mean())**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        ax.set_title(f"{name}  |  MAE={mae:.4f}  R2={r2:.4f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs True (validation set)", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "04_pred_vs_true.png", dpi=150)
    plt.close(fig)
    print(f"Saved 04_pred_vs_true.png")

    # Plot 5: Residuals vs R values
    n_plot = min(5000, len(X_val))
    idx_plot = np.random.choice(len(X_val), n_plot, replace=False)
    fig, axes = plt.subplots(3, 4, figsize=(20, 12))
    r_names = ["R1", "R2", "R3"]
    for row, (r_idx, r_name) in enumerate(zip([0, 1, 2], r_names)):
        for col, (y_idx, y_name) in enumerate(zip(range(4), cols_y)):
            ax = axes[row, col]
            r_vals = X_val[idx_plot, r_idx]
            residual = Y_pred[idx_plot, y_idx] - Y_val[idx_plot, y_idx]
            ax.scatter(r_vals, residual, s=1, alpha=0.2, color="blue")
            ax.axhline(0, color="red", linestyle="--")
            ax.set_xlabel(r_name)
            ax.set_ylabel(f"Residual ({y_name})")
            ax.set_xscale("log")
            if row == 0:
                ax.set_title(y_name)

    fig.suptitle("Residuals vs R values - looking for systematic bias", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "05_residuals_vs_R.png", dpi=150)
    plt.close(fig)
    print(f"Saved 05_residuals_vs_R.png")

    # Plot 6: Direct vs computed error comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    true_err = Y_val[:, 3]
    pred_err = Y_pred[:, 3]

    # Computed error from V_out predictions
    v_high_target = board.v_high
    v_low_target = board.v_low
    computed_err = np.maximum(
        np.abs(Y_pred[:, 0] - v_high_target),
        np.abs(Y_pred[:, 1] - v_low_target)
    )

    n_s = min(5000, len(true_err))
    idx_s = np.random.choice(len(true_err), n_s, replace=False)

    ax1.scatter(true_err[idx_s], pred_err[idx_s], s=1, alpha=0.3,
                color="blue", label="NN predicted max_error")
    ax1.scatter(true_err[idx_s], computed_err[idx_s], s=1, alpha=0.3,
                color="green", label="Computed from NN V_out")
    ax1.plot([0, 5], [0, 5], "r--", linewidth=2)
    ax1.set_xlabel("True max_error (V)")
    ax1.set_ylabel("Predicted/Computed max_error (V)")
    ax1.set_title("Error prediction: direct vs computed from V_out")
    ax1.legend(markerscale=10)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 5)
    ax1.grid(True, alpha=0.3)

    # Zoom into low-error region
    low_mask = true_err < 1.0
    if low_mask.sum() > 10:
        ax2.scatter(true_err[low_mask], pred_err[low_mask], s=5, alpha=0.5,
                    color="blue", label="NN max_error")
        ax2.scatter(true_err[low_mask], computed_err[low_mask], s=5, alpha=0.5,
                    color="green", label="Computed from V_out")
        ax2.plot([0, 1], [0, 1], "r--", linewidth=2)
        ax2.set_xlabel("True max_error (V)")
        ax2.set_ylabel("Predicted error (V)")
        ax2.set_title(f"Low-error region (<1V) - {low_mask.sum()} samples")
        ax2.legend(markerscale=3)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, f"Only {low_mask.sum()} low-error samples",
                 ha="center", va="center", transform=ax2.transAxes)

    fig.suptitle("KEY PLOT: Can the NN find the good region?", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "06_error_ranking.png", dpi=150)
    plt.close(fig)
    print(f"Saved 06_error_ranking.png")

    # Plot 7: Ranking test
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    sort_by_pred = np.argsort(pred_err)
    sort_by_computed = np.argsort(computed_err)
    sort_by_true = np.argsort(true_err)

    n_top = min(500, len(true_err))
    ax1.plot(true_err[sort_by_true[:n_top]], "g-", alpha=0.8,
             label="Oracle (sort by true)")
    ax1.plot(true_err[sort_by_pred[:n_top]], "b-", alpha=0.8,
             label="Sort by NN max_error")
    ax1.plot(true_err[sort_by_computed[:n_top]], "r-", alpha=0.8,
             label="Sort by computed V_out error")
    ax1.set_xlabel("Rank")
    ax1.set_ylabel("True max_error (V)")
    ax1.set_title("Top-500 ranking quality")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(["True best", "NN max_error\n#1 pick", "Computed V_out\n#1 pick"],
            [true_err[sort_by_true[0]],
             true_err[sort_by_pred[0]],
             true_err[sort_by_computed[0]]],
            color=["green", "blue", "red"], alpha=0.7)
    ax2.set_ylabel("True max_error (V)")
    ax2.set_title("Best pick comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Ranking quality: does NN find the best candidates?", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "07_ranking_quality.png", dpi=150)
    plt.close(fig)
    print(f"Saved 07_ranking_quality.png")

    # ── Summary ──
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"Samples: {len(X)} total, {mask.sum()} converged "
          f"({mask.sum()/len(mask)*100:.1f}%)")
    print(f"Low-error samples (< 0.2V): {(Yc[:,3] < 0.2).sum()} "
          f"({(Yc[:,3] < 0.2).sum()/len(Yc)*100:.1f}%)")
    print(f"Low-error samples (< 0.5V): {(Yc[:,3] < 0.5).sum()} "
          f"({(Yc[:,3] < 0.5).sum()/len(Yc)*100:.1f}%)")
    print()
    per_output_mae = np.abs(Y_pred - Y_val).mean(axis=0)
    for i, name in enumerate(cols_y):
        print(f"  {name:>12} MAE: {per_output_mae[i]:.4f}")
    print()
    print("NN's best pick (by predicted max_error):")
    best_nn = sort_by_pred[0]
    print(f"  True error: {true_err[best_nn]:.4f}V  "
          f"Pred error: {pred_err[best_nn]:.4f}V")
    print(f"  R1={X_val[best_nn, 0]:.0f}  R2={X_val[best_nn, 1]:.0f}  "
          f"R3={X_val[best_nn, 2]:.0f}")
    print()
    print("Actual best (by true max_error):")
    best_true = sort_by_true[0]
    print(f"  True error: {true_err[best_true]:.4f}V  "
          f"Pred error: {pred_err[best_true]:.4f}V")
    print(f"  R1={X_val[best_true, 0]:.0f}  R2={X_val[best_true, 1]:.0f}  "
          f"R3={X_val[best_true, 2]:.0f}")
    print()
    print("Computed-error best pick (from V_out predictions):")
    best_comp = sort_by_computed[0]
    print(f"  True error: {true_err[best_comp]:.4f}V  "
          f"Computed: {computed_err[best_comp]:.4f}V")
    print(f"  R1={X_val[best_comp, 0]:.0f}  R2={X_val[best_comp, 1]:.0f}  "
          f"R3={X_val[best_comp, 2]:.0f}")

    print(f"\nPlots saved to {PLOT_DIR}/")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
