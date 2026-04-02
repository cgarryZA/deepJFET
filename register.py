"""Register builder — generates master-slave edge-triggered flip-flop netlists.

A 1-bit register = master D-latch + slave D-latch = 14 gates + 2 for CLK/EN gating
An N-bit register = N copies sharing CLK and EN signals.

Master-slave eliminates transparent-latch feedback race:
  - Master captures D on CLK falling edge (transparent when CLK=LOW)
  - Slave outputs on CLK rising edge (transparent when CLK=HIGH)
  - Data sampled on rising edge of (CLK AND EN)

Each D-latch = 4 NAND + 1 INV = 5 gates
Master + Slave = 10 gates per bit
CLK gating (NAND + INV for CLK_bar) = shared
EN gating = shared

Naming convention:
  Input:  {name}_{bit}_In
  Output: {name}_{bit}
  Enable: {name}_Enable
  Clock:  CLK (shared)
"""

from model import GateType
from simulator.netlist import Gate


def make_register(name: str, n_bits: int) -> tuple:
    """Create an N-bit edge-triggered register (master-slave flip-flops).

    Args:
        name: Register name (e.g. "R0", "ACC")
        n_bits: Number of bits

    Returns:
        (gates, input_nets, output_nets, control_nets)
    """
    gates = []
    input_nets = []
    output_nets = []

    clk_net = "CLK"
    en_net = f"{name}_Enable"

    # Shared: CLK AND EN -> gated_clk, and its inverse
    gated_clk = f"{name}_GCLK"
    gated_clk_bar = f"{name}_GCLK_bar"

    # gated_clk = CLK AND EN = NOT(NAND(CLK, EN))
    gates.append(Gate(f"{name}_gclk_nand", GateType.NAND2,
                      [clk_net, en_net], f"{name}_gclk_n"))
    gates.append(Gate(f"{name}_gclk_inv", GateType.INV,
                      [f"{name}_gclk_n"], gated_clk))
    # gated_clk_bar = NOT(gated_clk)
    gates.append(Gate(f"{name}_gclk_bar", GateType.INV,
                      [gated_clk], gated_clk_bar))

    for bit in range(n_bits):
        prefix = f"{name}_{bit}"
        d_in = f"{prefix}_In"
        q_out = f"{prefix}"

        # Internal nets
        m_q = f"{prefix}_M_Q"        # master Q
        m_qbar = f"{prefix}_M_Qbar"  # master Q_bar
        s_q = q_out                   # slave Q = register output
        s_qbar = f"{prefix}_bar"      # slave Q_bar

        # === Master latch (transparent when gated_clk = LOW, i.e. gated_clk_bar = HIGH) ===
        # D_bar = NOT(D)
        m_d_bar = f"{prefix}_M_Dbar"
        gates.append(Gate(f"{prefix}_m_inv_d", GateType.INV,
                          [d_in], m_d_bar))

        # S_bar = NAND(D, gated_clk_bar)  -- pass D when CLK_bar=HIGH (CLK=LOW)
        m_sbar = f"{prefix}_M_Sbar"
        gates.append(Gate(f"{prefix}_m_nand_s", GateType.NAND2,
                          [d_in, gated_clk_bar], m_sbar))

        # R_bar = NAND(D_bar, gated_clk_bar)
        m_rbar = f"{prefix}_M_Rbar"
        gates.append(Gate(f"{prefix}_m_nand_r", GateType.NAND2,
                          [m_d_bar, gated_clk_bar], m_rbar))

        # Q = NAND(S_bar, Q_bar)
        gates.append(Gate(f"{prefix}_m_nand_q", GateType.NAND2,
                          [m_sbar, m_qbar], m_q))

        # Q_bar = NAND(R_bar, Q)
        gates.append(Gate(f"{prefix}_m_nand_qb", GateType.NAND2,
                          [m_rbar, m_q], m_qbar))

        # === Slave latch (transparent when gated_clk = HIGH) ===
        # S_bar = NAND(master_Q, gated_clk)
        s_sbar = f"{prefix}_S_Sbar"
        gates.append(Gate(f"{prefix}_s_nand_s", GateType.NAND2,
                          [m_q, gated_clk], s_sbar))

        # R_bar = NAND(master_Q_bar, gated_clk)
        s_rbar = f"{prefix}_S_Rbar"
        gates.append(Gate(f"{prefix}_s_nand_r", GateType.NAND2,
                          [m_qbar, gated_clk], s_rbar))

        # Q = NAND(S_bar, Q_bar)
        gates.append(Gate(f"{prefix}_s_nand_q", GateType.NAND2,
                          [s_sbar, s_qbar], s_q))

        # Q_bar = NAND(R_bar, Q)
        gates.append(Gate(f"{prefix}_s_nand_qb", GateType.NAND2,
                          [s_rbar, s_q], s_qbar))

        input_nets.append(d_in)
        output_nets.append(q_out)

    control_nets = {"clk": clk_net, "enable": en_net}
    return gates, input_nets, output_nets, control_nets
