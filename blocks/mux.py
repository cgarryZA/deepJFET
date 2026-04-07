"""2-to-1 multiplexer, scalable to N-bit bus width.

Tiling pattern:
    1-bit mux:  sel=0 -> out=a, sel=1 -> out=b
    N-bit mux:  N copies sharing the same sel line
                Each bit i: a{i}, b{i} -> out{i}

Implementation (1-bit, NAND-only):
    i1  = INV(sel)           -> sel_bar
    n1  = NAND2(a, sel_bar)  -> p
    n2  = NAND2(b, sel)      -> q
    n3  = NAND2(p, q)        -> out

    3 NAND2 + 1 INV = 4 gates per bit.
    The INV on sel is shared across all bits in the N-bit version.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.module import Module, Port, PortDir
from simulator.netlist import Gate
from model import GateType

IN, OUT = PortDir.IN, PortDir.OUT


def mux2to1(n_bits: int = 1) -> Module:
    """2-to-1 multiplexer. sel=0 selects a, sel=1 selects b.

    Ports:
        sel        (IN)  — select line
        a0..a{N-1} (IN)  — input A bus
        b0..b{N-1} (IN)  — input B bus
        out0..out{N-1} (OUT) — output bus
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")

    ports = [Port("sel", IN)]
    for i in range(n_bits):
        ports.append(Port(f"a{i}", IN))
    for i in range(n_bits):
        ports.append(Port(f"b{i}", IN))
    for i in range(n_bits):
        ports.append(Port(f"out{i}", OUT))

    gates = [
        # Shared sel inverter
        Gate("i_sel", GateType.INV, ["sel"], "sel_bar"),
    ]

    for i in range(n_bits):
        gates.extend([
            Gate(f"n1_{i}", GateType.NAND2, [f"a{i}", "sel_bar"], f"p{i}"),
            Gate(f"n2_{i}", GateType.NAND2, [f"b{i}", "sel"], f"q{i}"),
            Gate(f"n3_{i}", GateType.NAND2, [f"p{i}", f"q{i}"], f"out{i}"),
        ])

    return Module(
        name=f"mux2to1_{n_bits}b",
        ports=ports,
        gates=gates,
    )
