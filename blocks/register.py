"""N-bit parallel-load register built by tiling D flip-flops.

Tiling pattern:
    1-bit:  1 DFF                  -> ports: clk, d0, q0
    2-bit:  2 DFFs, shared clk     -> ports: clk, d0, d1, q0, q1
    N-bit:  N DFFs, shared clk     -> ports: clk, d0..d(N-1), q0..q(N-1)

    Each DFF instance "bit_i" connects:
        clk  -> shared "clk" port
        d    -> "d{i}" port
        q    -> "q{i}" port
        q_bar is left as internal net (bit_i.q_bar)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.module import Module, ModuleInstance, Port, PortDir
from model import GateType
from blocks.dff import dff

IN, OUT = PortDir.IN, PortDir.OUT


def register(n_bits: int) -> Module:
    """N-bit parallel-load register.

    Ports:
        clk       (IN)  — clock (shared by all bits)
        d0..d{N-1} (IN) — data inputs
        q0..q{N-1} (OUT) — data outputs
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")

    dff_mod = dff()

    ports = [Port("clk", IN)]
    for i in range(n_bits):
        ports.append(Port(f"d{i}", IN))
    for i in range(n_bits):
        ports.append(Port(f"q{i}", OUT))

    submodules = []
    for i in range(n_bits):
        inst = ModuleInstance(
            name=f"bit_{i}",
            module=dff_mod,
            connections={
                "clk": "clk",
                "d": f"d{i}",
                "q": f"q{i}",
                # q_bar left unconnected (becomes internal net)
            },
        )
        submodules.append(inst)

    return Module(
        name=f"register_{n_bits}b",
        ports=ports,
        gates=[],
        submodules=submodules,
    )
