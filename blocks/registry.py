"""Block registry: discover, list, and instantiate building blocks."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from blocks.dff import dff
from blocks.register import register
from blocks.mux import mux2to1
from blocks.decoder import decoder

# Registry maps block name -> (constructor, description, default_kwargs)
_BLOCKS = {
    "dff":      (dff,      "D flip-flop (master-slave, 10 gates)", {}),
    "register": (register, "N-bit parallel-load register",         {"n_bits": 4}),
    "mux2to1":  (mux2to1,  "2:1 multiplexer, N-bit bus",          {"n_bits": 1}),
    "decoder":  (decoder,  "N-to-2^N line decoder",                {"n_bits": 2}),
}


def list_blocks() -> list:
    """Return list of (name, description) for all available blocks."""
    return [(name, desc) for name, (_, desc, _) in _BLOCKS.items()]


def get_block(name: str, **kwargs):
    """Instantiate a block by name. kwargs override defaults.

    Examples:
        get_block("dff")              -> 1-bit DFF Module
        get_block("register", n_bits=8) -> 8-bit register Module
        get_block("decoder", n_bits=3)  -> 3-to-8 decoder Module
    """
    if name not in _BLOCKS:
        available = ", ".join(_BLOCKS.keys())
        raise ValueError(f"Unknown block '{name}'. Available: {available}")

    constructor, _, defaults = _BLOCKS[name]
    merged = {**defaults, **kwargs}
    return constructor(**merged)
