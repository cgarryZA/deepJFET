"""Register builder — generates D latch netlists for N-bit registers.

A 1-bit register = 7 gates (NAND_CE + INV_CE + INV_D + NAND_S + NAND_R + NAND_Q + NAND_QB)
An N-bit register = N copies sharing CLK and EN signals.

Naming convention:
  Input:  {name}_{bit}_In
  Output: {name}_{bit}
  Enable: {name}_Enable
  Clock:  CLK (shared)

Bit 0 = LSB, Bit N-1 = MSB.
"""

from model import GateType
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams


def make_register(name: str, n_bits: int) -> tuple:
    """Create an N-bit register from D latches.

    Args:
        name: Register name (e.g. "R0", "ACC")
        n_bits: Number of bits

    Returns:
        (gates, input_nets, output_nets, control_nets)
        gates: list of Gate objects
        input_nets: list of input net names [{name}_0_In, ...]
        output_nets: list of output net names [{name}_0, ...]
        control_nets: dict with 'clk' and 'enable' net names
    """
    gates = []
    input_nets = []
    output_nets = []

    clk_net = "CLK"
    en_net = f"{name}_Enable"

    for bit in range(n_bits):
        prefix = f"{name}_{bit}"
        d_in = f"{prefix}_In"
        q_out = f"{prefix}"
        q_bar = f"{prefix}_bar"
        d_bar = f"{prefix}_D_bar"
        clk_en = f"{prefix}_CLK_EN"
        clk_en_bar = f"{prefix}_CLK_EN_bar"
        s_bar = f"{prefix}_S_bar"
        r_bar = f"{prefix}_R_bar"

        gates.extend([
            Gate(f"{prefix}_nand_ce", GateType.NAND2, [clk_net, en_net], clk_en_bar),
            Gate(f"{prefix}_inv_ce",  GateType.INV,   [clk_en_bar],      clk_en),
            Gate(f"{prefix}_inv_d",   GateType.INV,   [d_in],            d_bar),
            Gate(f"{prefix}_nand_s",  GateType.NAND2, [d_in, clk_en],    s_bar),
            Gate(f"{prefix}_nand_r",  GateType.NAND2, [d_bar, clk_en],   r_bar),
            Gate(f"{prefix}_nand_q",  GateType.NAND2, [s_bar, q_bar],    q_out),
            Gate(f"{prefix}_nand_qb", GateType.NAND2, [r_bar, q_out],    q_bar),
        ])

        input_nets.append(d_in)
        output_nets.append(q_out)

    control_nets = {"clk": clk_net, "enable": en_net}
    return gates, input_nets, output_nets, control_nets
