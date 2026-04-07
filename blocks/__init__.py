"""Reusable parameterized building blocks for CPU design.

Each block is a function returning a simulator.Module that can be
flattened, simulated, or exported to LTSpice.
"""

from .dff import dff
from .register import register
from .mux import mux2to1
from .decoder import decoder
from .registry import list_blocks, get_block
