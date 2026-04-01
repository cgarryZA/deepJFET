"""Delay NN — predicts propagation delay from circuit parameters.

Separate from the DC NN. Trained on transient simulation data.
Smaller model (delay is a smoother function of R than DC voltages).

Input:  (R1, R2, R3, V+, V-, temp) — same 6 features as DC NN
Output: (t_pd_hl_ns, t_pd_lh_ns) — 2 delay values in nanoseconds
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from .model import InputNormaliser, OutputDenormaliser, GateSurrogateNet


def train_delay_model(dataset: dict, hidden_dims=None, max_epochs=300,
                      patience=30, lr=1e-3, batch_size=1024,
                      save_path=None, device=None):
    """Train a delay prediction model.

    Args:
        dataset: from generate_delay_dataset — has X, Y, mask
        hidden_dims: MLP hidden layer sizes (default [128, 128])
        save_path: where to save the model

    Returns:
        (model, history)
    """
    if hidden_dims is None:
        hidden_dims = [128, 128]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(42)
    np.random.seed(42)

    X = dataset["X"][dataset["mask"]]
    Y = dataset["Y"][dataset["mask"]]

    # Log-transform delays (they span orders of magnitude: 10ns to 10000ns)
    Y_log = np.log10(np.clip(Y, 1.0, None)).astype(np.float32)

    n_total = len(X)
    n_val = int(n_total * 0.15)
    n_train = n_total - n_val
    idx = np.random.permutation(n_total)
    X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
    Y_train, Y_val = Y_log[idx[:n_train]], Y_log[idx[n_train:]]
    Y_val_real = Y[idx[n_train:]]  # real ns for MAE reporting

    print(f"  Training delay model on {n_total:,} samples (device={device})")
    print(f"  Delay range: {Y[Y > 0].min():.0f}ns - {Y.max():.0f}ns")

    normaliser = InputNormaliser.from_data(X_train)
    output_denorm = OutputDenormaliser.from_data(Y_train)

    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    model = GateSurrogateNet(
        normaliser=normaliser,
        output_denorm=output_denorm,
        n_inputs=6,
        n_outputs=2,
        hidden_dims=hidden_dims,
        activation="silu",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {hidden_dims} -> {n_params:,} params")

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train_norm, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val_norm, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size * 2)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": [], "val_mae_ns": [], "lr": []}
    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    t0 = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model.forward_normalised(xb)
            loss = criterion(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss_sum += loss.item() * len(xb)
        train_loss = train_loss_sum / n_train

        model.eval()
        val_loss_sum = 0.0
        val_preds = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred_norm = model.forward_normalised(xb)
                val_loss_sum += criterion(pred_norm, yb).item() * len(xb)
                pred_real = model.output_denorm(pred_norm)
                val_preds.append(pred_real.cpu().numpy())

        val_loss = val_loss_sum / n_val

        # MAE in real ns (convert from log10)
        val_preds_log = np.vstack(val_preds)
        val_preds_ns = 10.0 ** val_preds_log
        val_mae_ns = np.abs(val_preds_ns - Y_val_real).mean()

        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae_ns"].append(val_mae_ns)
        history["lr"].append(current_lr)

        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or epoch == 1 or patience_ctr == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:4d}  train={train_loss:.6f}  "
                  f"val={val_loss:.6f}  mae={val_mae_ns:.1f}ns  "
                  f"lr={current_lr:.2e}  "
                  f"{'*' if patience_ctr == 0 else ' '} [{elapsed:.0f}s]")

        if patience_ctr >= patience:
            print(f"    Early stop at epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s  best_val={best_val:.6f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"  Saved to {save_path}")

    return model, history


def predict_delay(model, X: np.ndarray, device="cpu") -> np.ndarray:
    """Predict delays from circuit parameters.

    Args:
        model: trained delay model
        X: (N, 6) — [R1, R2, R3, V+, V-, temp]

    Returns:
        (N, 2) — [t_pd_hl_ns, t_pd_lh_ns] in nanoseconds
    """
    model.eval()
    model.to(device)
    with torch.no_grad():
        x = torch.tensor(X, dtype=torch.float32, device=device)
        pred_log = model(x).cpu().numpy()
    # Convert from log10(ns) back to ns
    return 10.0 ** pred_log
