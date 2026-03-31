"""Training loop for gate surrogate models.

Handles train/val split, early stopping, LR scheduling, and checkpointing.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from .config import TrainConfig
from .model import GateSurrogateNet, InputNormaliser, OutputDenormaliser, N_OUTPUTS


def train_surrogate(
    dataset: dict,
    cfg: TrainConfig = None,
    save_path: str = None,
    device: str = None,
) -> tuple:
    """Train a surrogate model from a generated dataset.

    Parameters
    ----------
    dataset : dict
        From ``generate_dataset`` -- has keys X, Y, mask.
    cfg : TrainConfig
    save_path : str, optional
    device : str, optional

    Returns
    -------
    (model, history)
    """
    if cfg is None:
        cfg = TrainConfig()

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # --- Filter to converged samples only ---
    X = dataset["X"][dataset["mask"]]
    Y = dataset["Y"][dataset["mask"]]
    n_total = len(X)
    n_inputs = X.shape[1]
    n_outputs = Y.shape[1]
    print(f"  Training on {n_total:,} converged samples "
          f"({n_inputs} inputs -> {n_outputs} outputs, device={device})")

    # --- Train/val split ---
    n_val = int(n_total * cfg.val_fraction)
    n_train = n_total - n_val
    idx = np.random.permutation(n_total)
    X_train, X_val = X[idx[:n_train]], X[idx[n_train:]]
    Y_train, Y_val = Y[idx[:n_train]], Y[idx[n_train:]]

    # --- Build normalisers from training data ---
    normaliser = InputNormaliser.from_data(X_train)
    output_denorm = OutputDenormaliser.from_data(Y_train)

    # --- Normalise targets for training ---
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_std[Y_std < 1e-8] = 1.0
    Y_train_norm = (Y_train - Y_mean) / Y_std
    Y_val_norm = (Y_val - Y_mean) / Y_std

    # --- Build model ---
    model = GateSurrogateNet(
        normaliser=normaliser,
        output_denorm=output_denorm,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {cfg.hidden_dims} -> {n_params:,} params")

    # --- Data loaders ---
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(Y_train_norm, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(Y_val_norm, dtype=torch.float32),
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                          pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2,
                        pin_memory=(device == "cuda"))

    # --- Optimiser & scheduler ---
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)

    if cfg.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=cfg.max_epochs)
    elif cfg.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, factor=0.5, patience=10)
    else:
        sched = None

    criterion = nn.MSELoss()

    # --- Training loop ---
    history = {"train_loss": [], "val_loss": [], "val_mae": [], "lr": []}
    best_val = float("inf")
    best_state = None
    patience_ctr = 0
    t0 = time.time()

    for epoch in range(1, cfg.max_epochs + 1):
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
        val_mae_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred_norm = model.forward_normalised(xb)
                val_loss_sum += criterion(pred_norm, yb).item() * len(xb)
                pred_real = model.output_denorm(pred_norm)
                yb_real = model.output_denorm(yb)
                val_mae_sum += (pred_real - yb_real).abs().sum().item()
        val_loss = val_loss_sum / n_val
        val_mae = val_mae_sum / (n_val * n_outputs)

        current_lr = opt.param_groups[0]["lr"]
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(current_lr)

        if cfg.scheduler == "plateau" and sched:
            sched.step(val_loss)
        elif sched:
            sched.step()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 20 == 0 or epoch == 1 or patience_ctr == 0:
            elapsed = time.time() - t0
            print(f"    Epoch {epoch:4d}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  "
                  f"mae={val_mae:.4f}  lr={current_lr:.2e}  "
                  f"{'*' if patience_ctr == 0 else ' '} "
                  f"[{elapsed:.0f}s]")

        if patience_ctr >= cfg.patience:
            print(f"    Early stop at epoch {epoch} "
                  f"(no improvement for {cfg.patience} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.1f}s  "
          f"best_val_loss={best_val:.6f}")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(save_path)
        print(f"  Saved model to {save_path}")

    return model, history


def load_surrogate(path: str, device: str = "cpu") -> GateSurrogateNet:
    """Load a trained surrogate model."""
    return GateSurrogateNet.load(path, device=device)
