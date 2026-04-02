"""Program Counter — N-bit register with dedicated incrementer.

The PC is a register that can:
  1. Hold its current value (default)
  2. Increment by 1 (INC signal)
  3. Load a value from external input (LOAD signal, for jumps)

The incrementer is a half-adder chain (simpler than full adder — no B input).
Half adder: SUM = A XOR 1 = NOT A (when carry_in=1), COUT = A AND carry_in.

When INC is asserted, the incremented value feeds back into the register.
When LOAD is asserted, the external input feeds into the register.
When neither, the register holds.

Structure:
  - N-bit register (from register.py)
  - N-bit half-adder chain (incrementer)
  - Mux: selects between hold (current Q), increment (Q+1), or load (external)
  - The register's D inputs come from the mux output
  - CLK gates when the register latches

Naming:
  PC_{bit}       — current counter value (register Q output)
  PC_{bit}_In    — register D input (from mux)
  PC_Load_{bit}  — external load input (for jumps)
  INC            — increment signal
  LOAD           — load signal (overrides increment)
  CLK            — clock
"""

from model import GateType
from simulator.netlist import Gate
from register import make_register


def make_program_counter(name: str, n_bits: int) -> tuple:
    """Create an N-bit program counter with incrementer.

    Args:
        name: PC name (e.g. "PC")
        n_bits: bit width

    Returns:
        (gates, output_nets, control_nets, load_input_nets)
    """
    gates = []

    # --- Incrementer: half-adder chain on current Q values ---
    # When INC=1, carry starts at 1 and ripples through
    # Half adder bit k: SUM_k = Q_k XOR CARRY_k, CARRY_{k+1} = Q_k AND CARRY_k
    # Initial carry = INC signal

    for bit in range(n_bits):
        c_in = 'INC' if bit == 0 else f'{name}_CARRY_{bit}'
        q_net = f'{name}_{bit}'  # current register output

        # XOR for sum: Q XOR Cin (4 NAND gates)
        prefix = f'{name}_INC_{bit}'
        _build_xor(gates, prefix, q_net, c_in, f'{name}_INCR_{bit}')

        # AND for carry: Q AND Cin = NOT(NAND(Q, Cin))
        gates.append(Gate(f'{prefix}_carry_nand', GateType.NAND2,
                          [q_net, c_in], f'{prefix}_carry_n'))
        gates.append(Gate(f'{prefix}_carry_inv', GateType.INV,
                          [f'{prefix}_carry_n'], f'{name}_CARRY_{bit+1}'))

    # --- Mux: select between HOLD (Q), INCREMENT (Q+1), or LOAD (external) ---
    # Priority: LOAD > INC > HOLD
    # When LOAD=1: D = Load_input
    # When INC=1 (and LOAD=0): D = incremented value
    # Otherwise: D = current Q (hold)
    #
    # Implementation: two muxes per bit
    #   Mux1: INC ? INCR : Q  (select increment vs hold)
    #   Mux2: LOAD ? LOAD_IN : Mux1  (select load vs mux1 result)

    for bit in range(n_bits):
        q_net = f'{name}_{bit}'
        incr_net = f'{name}_INCR_{bit}'
        load_net = f'{name}_Load_{bit}'
        d_net = f'{name}_{bit}_In'  # register D input

        # Mux1: INC ? INCR : Q
        _build_mux2(gates, f'{name}_MUX1_{bit}', 'INC',
                     q_net, incr_net, f'{name}_MUX1_{bit}_out')

        # Mux2: LOAD ? LOAD_IN : MUX1_out
        _build_mux2(gates, f'{name}_MUX2_{bit}', 'LOAD',
                     f'{name}_MUX1_{bit}_out', load_net, d_net)

    # --- Register ---
    # The register enable = INC OR LOAD
    # This way the register only latches when we actually want to update.
    # When neither INC nor LOAD, enable=LOW, register holds, no feedback race.
    #
    # OR(INC, LOAD) = NAND(NOT(INC), NOT(LOAD))
    gates.append(Gate(f'{name}_en_inv_inc', GateType.INV,
                      ['INC'], f'{name}_not_inc'))
    gates.append(Gate(f'{name}_en_inv_load', GateType.INV,
                      ['LOAD'], f'{name}_not_load'))
    gates.append(Gate(f'{name}_en_or', GateType.NAND2,
                      [f'{name}_not_inc', f'{name}_not_load'],
                      f'{name}_Enable'))

    reg_gates, reg_ins, reg_outs, reg_ctrl = make_register(name, n_bits)
    gates.extend(reg_gates)

    output_nets = reg_outs  # PC_{bit} values
    load_input_nets = [f'{name}_Load_{bit}' for bit in range(n_bits)]
    control_nets = {
        'clk': 'CLK',
        'inc': 'INC',
        'load': 'LOAD',
        'enable': f'{name}_Enable',
    }

    return gates, output_nets, control_nets, load_input_nets


def _build_xor(gates, prefix, a_net, b_net, out_net):
    """XOR from 4 NAND gates."""
    gates.append(Gate(f'{prefix}_xn1', GateType.NAND2,
                      [a_net, b_net], f'{prefix}_nab'))
    gates.append(Gate(f'{prefix}_xn2', GateType.NAND2,
                      [a_net, f'{prefix}_nab'], f'{prefix}_na'))
    gates.append(Gate(f'{prefix}_xn3', GateType.NAND2,
                      [b_net, f'{prefix}_nab'], f'{prefix}_nb'))
    gates.append(Gate(f'{prefix}_xn4', GateType.NAND2,
                      [f'{prefix}_na', f'{prefix}_nb'], out_net))


def _build_mux2(gates, prefix, sel, a_net, b_net, out_net):
    """2:1 mux: out = sel ? b : a. From 3 NAND + 1 INV."""
    gates.append(Gate(f'{prefix}_inv_sel', GateType.INV,
                      [sel], f'{prefix}_nsel'))
    gates.append(Gate(f'{prefix}_na', GateType.NAND2,
                      [a_net, f'{prefix}_nsel'], f'{prefix}_an'))
    gates.append(Gate(f'{prefix}_nb', GateType.NAND2,
                      [b_net, sel], f'{prefix}_bn'))
    gates.append(Gate(f'{prefix}_out', GateType.NAND2,
                      [f'{prefix}_an', f'{prefix}_bn'], out_net))


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from simulator.netlist import Netlist
    from simulator.precompute import CircuitParams, precompute_gate
    from simulator.engine import SimulationEngine, Stimulus
    from model import NChannelJFET, JFETCapacitance

    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    params = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)
    V_HIGH, V_LOW = -0.8, -4.0

    N = 4
    gates, outs, ctrl, load_ins = make_program_counter("PC", N)
    print(f"4-bit Program Counter: {len(gates)} gates")
    print(f"  Outputs: {outs}")
    print(f"  Controls: {ctrl}")
    print(f"  Load inputs: {load_ins}")

    netlist = Netlist.from_gates(gates, primary_outputs=set(outs))
    ordered, feedback = netlist.topological_sort()
    print(f"  Ordered: {len(ordered)}, Feedback: {len(feedback)}")

    # Test: increment from 0 to 5
    profiles = {}
    for gt in [GateType.INV, GateType.NAND2]:
        profiles[gt] = precompute_gate(gt, params, V_HIGH, V_LOW)

    netlist = Netlist.from_gates(gates, primary_outputs=set(outs))
    engine = SimulationEngine(netlist, profiles, v_high=V_HIGH, v_low=V_LOW,
                              auto_precompute_params=params)

    T = 10e-6  # clock period
    t_settle = 2e-6
    n_increments = 18  # count 0 through 17 (wraps at 16)

    # CLK: 100kHz square wave
    clk_times = [0]
    clk_vals = [True]  # start high for initial eval
    for i in range(n_increments + 2):
        t = t_settle + i * T
        clk_times.extend([t, t + T/2])
        clk_vals.extend([True, False])
    engine.add_stimulus(Stimulus('CLK', clk_times, clk_vals))

    # INC: pulse high only during CLK low phase (when register is NOT transparent)
    # This way the incrementer settles while the register holds,
    # then CLK goes high and latches the settled increment value.
    inc_times = [0]
    inc_vals = [True]  # initial toggle
    for i in range(n_increments + 2):
        t = t_settle + i * T
        # INC high during CLK-low phase (t + T/2 to t + T)
        inc_times.extend([t, t + T/2 - 0.1e-6, t + T/2, t + T - 0.1e-6])
        inc_vals.extend([False, False, True, True])
    engine.add_stimulus(Stimulus('INC', inc_times, inc_vals))

    # LOAD: always low
    engine.add_stimulus(Stimulus('LOAD', [0, t_settle], [True, False]))

    # Enable: always high
    engine.add_stimulus(Stimulus('PC_Enable', [0, t_settle], [True, True]))

    # Load inputs: all low (not used)
    for bit in range(N):
        engine.add_stimulus(Stimulus(f'PC_Load_{bit}', [0, t_settle], [True, False]))

    result = engine.run(t_settle + (n_increments + 1) * T)
    print(f"\n  Events: {result.events_processed}")

    # Read PC value at each clock cycle
    print(f"\n  Cycle  PC Value")
    for i in range(n_increments + 1):
        # Sample at 80% through each cycle
        t_sample = t_settle + (i + 0.8) * T
        val = 0
        for bit in range(N):
            ns = result.net_states.get(f'PC_{bit}')
            if ns:
                # Check history for value at t_sample
                v = ns.value  # final value as fallback
                for t_ev, ev_val, ev_v in ns.history:
                    if t_ev <= t_sample:
                        v = ev_val
                if v:
                    val |= (1 << bit)
        print(f"  {i:5d}  {val:4d}  (0b{val:04b})")
