"""D flip-flop built from NAND gates.

Tiling pattern:
    A 1-bit DFF has ports: clk, d, q, q_bar.
    To make an N-bit register, instantiate N DFFs and share the clk line.
    Each bit gets its own d[i] and q[i] ports.
    See blocks/register.py for the tiled version.

Implementation:
    Edge-triggered master-slave DFF using 8 NAND2 + 2 INV = 10 gates.

    Inverters:
        i1 = INV(d)   -> d_bar
        i2 = INV(clk) -> clk_bar

    Master latch (transparent when clk=HIGH):
        n1 = NAND2(d, clk)          -> s_bar
        n2 = NAND2(d_bar, clk)      -> r_bar
        n3 = NAND2(s_bar, mq_bar)   -> mq        (cross-coupled)
        n4 = NAND2(r_bar, mq)       -> mq_bar    (cross-coupled)

    Slave latch (transparent when clk_bar=HIGH, i.e. clk=LOW):
        n5 = NAND2(mq, clk_bar)     -> s2_bar
        n6 = NAND2(mq_bar, clk_bar) -> r2_bar
        n7 = NAND2(s2_bar, q_bar)   -> q         (cross-coupled)
        n8 = NAND2(r2_bar, q)       -> q_bar     (cross-coupled)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.module import Module, Port, PortDir
from simulator.netlist import Gate
from model import GateType

IN, OUT = PortDir.IN, PortDir.OUT


def dff() -> Module:
    """Edge-triggered D flip-flop (master-slave, rising-edge triggered).

    Ports:
        clk   (IN)  — clock, latches on rising edge
        d     (IN)  — data input
        q     (OUT) — output
        q_bar (OUT) — inverted output
    """
    return Module(
        name="dff",
        ports=[
            Port("clk", IN),
            Port("d", IN),
            Port("q", OUT),
            Port("q_bar", OUT),
        ],
        gates=[
            # Inverters
            Gate("i1", GateType.INV, ["d"], "d_bar"),
            Gate("i2", GateType.INV, ["clk"], "clk_bar"),

            # Master latch (transparent when clk=HIGH)
            Gate("n1", GateType.NAND2, ["d", "clk"], "s_bar"),
            Gate("n2", GateType.NAND2, ["d_bar", "clk"], "r_bar"),
            Gate("n3", GateType.NAND2, ["s_bar", "mq_bar"], "mq"),
            Gate("n4", GateType.NAND2, ["r_bar", "mq"], "mq_bar"),

            # Slave latch (transparent when clk=LOW)
            Gate("n5", GateType.NAND2, ["mq", "clk_bar"], "s2_bar"),
            Gate("n6", GateType.NAND2, ["mq_bar", "clk_bar"], "r2_bar"),
            Gate("n7", GateType.NAND2, ["s2_bar", "q_bar"], "q"),
            Gate("n8", GateType.NAND2, ["r2_bar", "q"], "q_bar"),
        ],
    )
