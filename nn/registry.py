"""Model registry -- tracks trained models and their metadata.

Simple JSON-based registry that maps (gate_type, jfet_hash) to model files.
Stores training ranges, metrics, and timestamps so the pipeline can decide
whether an existing model covers the current board config.
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime


def jfet_hash(jfet) -> str:
    """Create a short hash from JFET parameters for model lookup."""
    params = (
        jfet.beta, jfet.vto, jfet.lmbda,
        jfet.is_, jfet.n, jfet.isr, jfet.nr,
        jfet.alpha, jfet.vk, jfet.rd, jfet.rs,
        jfet.betatce, jfet.vtotc, jfet.xti, jfet.eg,
    )
    key = "|".join(f"{p:.8g}" for p in params)
    return hashlib.md5(key.encode()).hexdigest()[:8]


def _registry_path(model_dir: str) -> Path:
    return Path(model_dir) / "registry.json"


def load_registry(model_dir: str) -> dict:
    """Load the registry from disk, or return empty dict."""
    path = _registry_path(model_dir)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def save_registry(registry: dict, model_dir: str):
    """Save registry to disk."""
    path = _registry_path(model_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)


def _entry_key(gate_type_str: str, jfet_h: str) -> str:
    return f"{gate_type_str}_{jfet_h}"


def find_model(model_dir: str, gate_type, jfet, board=None) -> dict | None:
    """Look up a model that covers the given config.

    Returns the registry entry dict if found, None otherwise.
    If board is provided, also checks that the model's training ranges
    cover the board's V_POS, V_NEG, temp.
    """
    registry = load_registry(model_dir)
    jh = jfet_hash(jfet)
    key = _entry_key(gate_type.value, jh)
    entry = registry.get(key)

    if entry is None:
        return None

    if board is not None:
        ranges = entry.get("training_ranges", {})
        # Check voltage/temp coverage
        v_pos_r = ranges.get("v_pos_range")
        if v_pos_r and not (v_pos_r[0] <= board.v_pos <= v_pos_r[1]):
            return None
        v_neg_r = ranges.get("v_neg_range")
        if v_neg_r and not (v_neg_r[0] <= board.v_neg <= v_neg_r[1]):
            return None
        temp_r = ranges.get("temp_range")
        if temp_r and not (temp_r[0] <= board.temp_c <= temp_r[1]):
            return None

    return entry


def register_model(
    model_dir: str,
    gate_type,
    jfet,
    model_path: str,
    sampling_cfg=None,
    metrics: dict = None,
):
    """Register a trained model in the registry."""
    registry = load_registry(model_dir)
    jh = jfet_hash(jfet)
    key = _entry_key(gate_type.value, jh)

    entry = {
        "gate_type": gate_type.value,
        "jfet_hash": jh,
        "model_path": str(model_path),
        "trained_at": datetime.now().isoformat(),
    }

    if sampling_cfg is not None:
        entry["training_ranges"] = {
            "r1_range": list(sampling_cfg.r1_range),
            "r23_range": list(sampling_cfg.r23_range),
            "v_pos_range": list(sampling_cfg.v_pos_range) if sampling_cfg.v_pos_range else None,
            "v_neg_range": list(sampling_cfg.v_neg_range) if sampling_cfg.v_neg_range else None,
            "temp_range": list(sampling_cfg.temp_range) if sampling_cfg.temp_range else None,
            "n_samples": sampling_cfg.n_samples,
        }

    if metrics is not None:
        entry["metrics"] = metrics

    registry[key] = entry
    save_registry(registry, model_dir)
    return entry
