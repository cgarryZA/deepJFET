"""Neural-network architecture for gate surrogate models.

A simple MLP that maps normalised circuit parameters to predicted outputs.
Input normalisation (log-R, standardisation) is baked into the model so
inference is a single ``model(raw_tensor)`` call with no external preprocessing.
"""

import torch
import torch.nn as nn
import numpy as np


class InputNormaliser(nn.Module):
    """Learnable-free normalisation layer.

    Stores per-feature mean/std computed from training data.
    R-columns (indices 0-2) are log10-transformed before standardising.
    """

    def __init__(self, mean: torch.Tensor, std: torch.Tensor,
                 log_cols: tuple = (0, 1, 2)):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.log_cols = log_cols

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        for c in self.log_cols:
            x[:, c] = torch.log10(x[:, c].clamp(min=1.0))
        return (x - self.mean) / self.std.clamp(min=1e-8)

    @classmethod
    def from_data(cls, X: np.ndarray, log_cols=(0, 1, 2)):
        """Compute normalisation stats from a numpy array."""
        X_t = X.copy().astype(np.float32)
        for c in log_cols:
            X_t[:, c] = np.log10(np.clip(X_t[:, c], 1.0, None))
        mean = torch.tensor(X_t.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(X_t.std(axis=0), dtype=torch.float32)
        return cls(mean, std, log_cols)


class OutputDenormaliser(nn.Module):
    """Stores output mean/std to map normalised predictions back to real scale."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, y_norm: torch.Tensor) -> torch.Tensor:
        return y_norm * self.std + self.mean

    def normalise(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std.clamp(min=1e-8)

    @classmethod
    def from_data(cls, Y: np.ndarray):
        mean = torch.tensor(Y.mean(axis=0), dtype=torch.float32)
        std = torch.tensor(Y.std(axis=0), dtype=torch.float32)
        return cls(mean, std)


class GateSurrogateNet(nn.Module):
    """MLP surrogate for a single gate type.

    Architecture:  InputNormaliser → [Linear → Act → (Dropout)] × N → Linear → OutputDenormaliser

    Inputs  (8): R1, R2, R3, V+, V-, V_HIGH, V_LOW, temp
    Outputs (4): V_out_high, V_out_low, avg_power, max_error
    """

    def __init__(self, normaliser: InputNormaliser,
                 output_denorm: OutputDenormaliser = None,
                 hidden_dims: list = None,
                 activation: str = "silu",
                 dropout: float = 0.0):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256, 128]

        self.normaliser = normaliser
        self.output_denorm = output_denorm

        act_map = {
            "relu": nn.ReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
        }
        Act = act_map.get(activation, nn.SiLU)

        layers = []
        in_dim = 8  # input features
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(Act())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 4))  # 4 outputs

        self.trunk = nn.Sequential(*layers)

        # Store metadata for serialisation
        self.meta = {
            "hidden_dims": hidden_dims,
            "activation": activation,
            "dropout": dropout,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw inputs → denormalised predictions."""
        y_norm = self.trunk(self.normaliser(x))
        if self.output_denorm is not None:
            return self.output_denorm(y_norm)
        return y_norm

    def forward_normalised(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning normalised outputs (for training loss)."""
        return self.trunk(self.normaliser(x))

    def predict_numpy(self, X: np.ndarray) -> np.ndarray:
        """Convenience: numpy in → numpy out, no grad."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(X, dtype=torch.float32, device=self._device())
            return self(x).cpu().numpy()

    def _device(self):
        return next(self.parameters()).device

    def save(self, path: str):
        """Save model + normaliser + meta to a .pt file."""
        save_dict = {
            "state_dict": self.state_dict(),
            "meta": self.meta,
            "norm_mean": self.normaliser.mean,
            "norm_std": self.normaliser.std,
            "norm_log_cols": self.normaliser.log_cols,
        }
        if self.output_denorm is not None:
            save_dict["out_mean"] = self.output_denorm.mean
            save_dict["out_std"] = self.output_denorm.std
        torch.save(save_dict, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "GateSurrogateNet":
        """Load a saved model."""
        ckpt = torch.load(path, map_location=device, weights_only=True)
        norm = InputNormaliser(
            ckpt["norm_mean"], ckpt["norm_std"], ckpt["norm_log_cols"],
        )
        out_denorm = None
        if "out_mean" in ckpt:
            out_denorm = OutputDenormaliser(ckpt["out_mean"], ckpt["out_std"])
        model = cls(
            normaliser=norm,
            output_denorm=out_denorm,
            hidden_dims=ckpt["meta"]["hidden_dims"],
            activation=ckpt["meta"]["activation"],
            dropout=ckpt["meta"].get("dropout", 0.0),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model
