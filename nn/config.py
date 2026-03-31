"""Configuration dataclasses for the NN surrogate pipeline."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SamplingConfig:
    """Controls training-data generation."""

    n_samples: int = 100_000
    """Total number of (R1, R2, R3) sample points."""

    r1_range: tuple = (100.0, 500_000.0)
    """Ohm range for R1 (pull-down resistor)."""

    r23_range: tuple = (100.0, 500_000.0)
    """Ohm range for R2 and R3 (output divider)."""

    v_pos_range: tuple = None
    """If set, sample V+ uniformly in this range; else use board default."""

    v_neg_range: tuple = None
    """If set, sample V- uniformly in this range; else use board default."""

    temp_range: tuple = None
    """If set, sample temperature uniformly (°C); else use board default."""

    log_r: bool = True
    """Sample resistor values in log-space (True) or linear (False)."""

    n_workers: int = 4
    """Number of parallel worker processes for data generation."""

    seed: int = 42
    """RNG seed for reproducibility."""


@dataclass
class TrainConfig:
    """Controls neural-network training."""

    hidden_dims: list = field(default_factory=lambda: [256, 256, 128])
    """Hidden layer sizes for the MLP."""

    activation: str = "silu"
    """Activation function: 'relu', 'silu', 'gelu'."""

    lr: float = 1e-3
    """Initial learning rate."""

    batch_size: int = 2048
    """Mini-batch size."""

    max_epochs: int = 500
    """Maximum training epochs."""

    patience: int = 30
    """Early-stopping patience (epochs without val improvement)."""

    val_fraction: float = 0.15
    """Fraction of data held out for validation."""

    weight_decay: float = 1e-5
    """L2 regularisation strength."""

    scheduler: str = "cosine"
    """LR schedule: 'cosine', 'plateau', 'none'."""

    seed: int = 42
    """RNG seed for reproducibility."""


@dataclass
class PipelineConfig:
    """Top-level orchestration config."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    training: TrainConfig = field(default_factory=TrainConfig)

    data_dir: str = "nn/data"
    """Directory for generated datasets."""

    model_dir: str = "nn/models"
    """Directory for saved model checkpoints."""

    top_n_verify: int = 20
    """Number of top NN predictions to verify with the real solver."""

    e_series: str = "E96"
    """E-series for the final grid sweep through the NN."""

    r1_range: tuple = None
    """Override R1 range for grid sweep (auto-computed from board if None)."""

    r23_range: tuple = None
    """Override R23 range for grid sweep (auto-computed from board if None)."""
