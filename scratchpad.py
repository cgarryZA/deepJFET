"""Scratchpad memory — N addressable registers with bus interface.

Architecture:
  - N registers (each M bits wide)
  - Address register (log2(N) bits) selects which register is active
  - Address decoder (log2(N) to N one-hot) drives per-register select
  - Output gating: AND(register_bit, select, output_enable) per bit per register
    All gated outputs OR'd together onto the bus (MI-OR style)
  - Input gating: register_enable = AND(select, write_enable)

Usage:
    sp = make_scratchpad("SP", n_registers=16, data_width=4)
    # sp.gates: all gate objects
    # sp.data_bus_out: ['SP_BUS_OUT_0', ...] — read selected register
    # sp.data_bus_in: uses bus nets directly when write_enable + select active
    # sp.addr_inputs: ['SP_ADDR_0', ...] — address bits
    # sp.read_enable: 'SP_READ' — gates output onto bus
    # sp.write_enable: 'SP_WRITE' — gates bus input into selected register
"""

import math
from model import GateType
from simulator.netlist import Gate
from register import make_register


class Scratchpad:
    """Parameterised scratchpad memory block."""

    def __init__(self, name, n_registers, data_width):
        self.name = name
        self.n_registers = n_registers
        self.data_width = data_width
        self.addr_width = max(1, math.ceil(math.log2(n_registers))) if n_registers > 1 else 0

        self.gates = []
        self.addr_inputs = [f'{name}_ADDR_{i}' for i in range(self.addr_width)]
        self.read_enable = f'{name}_READ'
        self.write_enable = f'{name}_WRITE'
        self.bus_out_nets = [f'{name}_DOUT_{bit}' for bit in range(data_width)]

        self._build()

    def _build(self):
        n = self.n_registers
        w = self.data_width
        name = self.name

        # --- Registers ---
        self.reg_outputs = {}  # reg_idx -> [net_names]
        for reg in range(n):
            reg_name = f'{name}_R{reg}'
            g, ins, outs, ctrl = make_register(reg_name, w)
            self.gates.extend(g)
            self.reg_outputs[reg] = outs

        # --- Address decoder: addr_width bits -> n one-hot select lines ---
        if self.addr_width > 0:
            self._build_decoder()
        else:
            # Only 1 register — always selected
            self.select_nets = [f'{name}_SEL_0']
            # Tie high via double inverter
            self.gates.append(Gate(f'{name}_sel0_inv', GateType.INV,
                                   ['CLK'], f'{name}_sel0_tmp'))  # dummy input
            self.gates.append(Gate(f'{name}_sel0_buf', GateType.INV,
                                   [f'{name}_sel0_tmp'], f'{name}_SEL_0'))

        # --- Output gating: AND(reg_bit, select, read_enable) per register ---
        for bit in range(w):
            writer_outs = []
            for reg in range(n):
                sel = self.select_nets[reg]
                reg_bit = self.reg_outputs[reg][bit]
                prefix = f'{name}_ROUT_R{reg}_B{bit}'

                # AND(reg_bit, select) first
                self.gates.append(Gate(f'{prefix}_and1_nand', GateType.NAND2,
                                       [reg_bit, sel], f'{prefix}_and1_n'))
                self.gates.append(Gate(f'{prefix}_and1_inv', GateType.INV,
                                       [f'{prefix}_and1_n'], f'{prefix}_and1'))

                # AND(result, read_enable)
                self.gates.append(Gate(f'{prefix}_and2_nand', GateType.NAND2,
                                       [f'{prefix}_and1', self.read_enable],
                                       f'{prefix}_and2_n'))
                self.gates.append(Gate(f'{prefix}_and2_inv', GateType.INV,
                                       [f'{prefix}_and2_n'], f'{prefix}_out'))

                writer_outs.append(f'{prefix}_out')

            # OR all gated outputs together for this bus bit
            self._or_tree(writer_outs, self.bus_out_nets[bit],
                          f'{name}_DOUT_OR_{bit}')

        # --- Input gating: register_enable = AND(select, write_enable) ---
        for reg in range(n):
            sel = self.select_nets[reg]
            reg_name = f'{name}_R{reg}'
            en_net = f'{reg_name}_Enable'
            prefix = f'{name}_WEN_R{reg}'

            # AND(select, write_enable)
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [sel, self.write_enable], f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [f'{prefix}_n'], en_net))

        # --- Wire bus input to all register D inputs ---
        # All registers share the same bus input; only the enabled one latches
        for bit in range(w):
            bus_in = f'{name}_DIN_{bit}'
            for reg in range(n):
                reg_d = f'{name}_R{reg}_{bit}_In'
                # Buffer from bus to register input
                self.gates.append(Gate(f'{name}_DIN_R{reg}_B{bit}_inv', GateType.INV,
                                       [bus_in], f'{name}_DIN_R{reg}_B{bit}_n'))
                self.gates.append(Gate(f'{name}_DIN_R{reg}_B{bit}_buf', GateType.INV,
                                       [f'{name}_DIN_R{reg}_B{bit}_n'], reg_d))

    def _build_decoder(self):
        """Build address decoder: addr_width -> n_registers one-hot."""
        name = self.name
        n = self.n_registers
        aw = self.addr_width

        # Generate inverted address bits
        for bit in range(aw):
            self.gates.append(Gate(f'{name}_ADDR_INV_{bit}', GateType.INV,
                                   [f'{name}_ADDR_{bit}'],
                                   f'{name}_ADDR_{bit}_bar'))

        # For each register, AND the appropriate address bit combination
        self.select_nets = []
        for reg in range(n):
            sel_net = f'{name}_SEL_{reg}'
            self.select_nets.append(sel_net)

            # Which address bits should be true/false for this register
            and_inputs = []
            for bit in range(aw):
                if (reg >> bit) & 1:
                    and_inputs.append(f'{name}_ADDR_{bit}')
                else:
                    and_inputs.append(f'{name}_ADDR_{bit}_bar')

            # Build AND tree
            self._and_tree(and_inputs, sel_net, f'{name}_DEC_{reg}')

    def _and_tree(self, inputs, output, prefix):
        if len(inputs) == 1:
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [inputs[0]], f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_buf', GateType.INV,
                                   [f'{prefix}_n'], output))
        elif len(inputs) == 2:
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   inputs, f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [f'{prefix}_n'], output))
        else:
            mid = len(inputs) // 2
            self._and_tree(inputs[:mid], f'{prefix}_L', f'{prefix}_L')
            self._and_tree(inputs[mid:], f'{prefix}_R', f'{prefix}_R')
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [f'{prefix}_L', f'{prefix}_R'], f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [f'{prefix}_n'], output))

    def _or_tree(self, inputs, output, prefix):
        if len(inputs) == 1:
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [inputs[0]], f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_buf', GateType.INV,
                                   [f'{prefix}_n'], output))
        elif len(inputs) == 2:
            self.gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                                   [inputs[0]], f'{prefix}_na'))
            self.gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                                   [inputs[1]], f'{prefix}_nb'))
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [f'{prefix}_na', f'{prefix}_nb'], output))
        else:
            mid = len(inputs) // 2
            self._or_tree(inputs[:mid], f'{prefix}_L', f'{prefix}_L')
            self._or_tree(inputs[mid:], f'{prefix}_R', f'{prefix}_R')
            self.gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                                   [f'{prefix}_L'], f'{prefix}_na'))
            self.gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                                   [f'{prefix}_R'], f'{prefix}_nb'))
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [f'{prefix}_na', f'{prefix}_nb'], output))

    def summary(self):
        print(f"Scratchpad '{self.name}': {self.n_registers} x {self.data_width}-bit")
        print(f"  Address width: {self.addr_width} bits")
        print(f"  Gates: {len(self.gates)}")
        print(f"  Addr inputs: {self.addr_inputs}")
        print(f"  Read enable: {self.read_enable}")
        print(f"  Write enable: {self.write_enable}")
        print(f"  Data out: {self.bus_out_nets}")
        print(f"  Data in: {[f'{self.name}_DIN_{b}' for b in range(self.data_width)]}")


def make_scratchpad(name, n_registers, data_width):
    """Create a scratchpad memory block."""
    return Scratchpad(name, n_registers, data_width)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))

    from simulator.netlist import Netlist
    from simulator.precompute import CircuitParams, precompute_gate
    from simulator.engine import SimulationEngine, Stimulus
    from model import NChannelJFET, JFETCapacitance

    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005, is_=205.2e-15, n=3.0,
        isr=1988e-15, nr=4.0, alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    params = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)
    V_HIGH, V_LOW = -0.8, -4.0

    # Test different sizes
    for n_reg in [1, 2, 4, 8, 16]:
        sp = make_scratchpad("SP", n_reg, 4)
        sp.summary()
        print()

    # Functional test: 4 registers x 4 bits
    print("=" * 50)
    print("Functional test: 4 registers x 4 bits")
    print("=" * 50)

    sp = make_scratchpad("SP", 4, 4)
    sp.summary()

    netlist = Netlist.from_gates(sp.gates, primary_outputs=set(sp.bus_out_nets))
    profiles = {gt: precompute_gate(gt, params, V_HIGH, V_LOW)
                for gt in [GateType.INV, GateType.NAND2]}
    engine = SimulationEngine(netlist, profiles, v_high=V_HIGH, v_low=V_LOW)

    # Init: force all registers to 0
    for reg in range(4):
        for bit in range(4):
            for suf, val in [('', False), ('_bar', True),
                             ('_M_Q', False), ('_M_Qbar', True)]:
                net = f'SP_R{reg}_{bit}{suf}'
                if net in engine._nets:
                    engine._nets[net].value = val
                    engine._nets[net].voltage = V_HIGH if val else V_LOW

    engine.force_evaluate_all()

    # Set all inputs to initial state
    for bit in range(2):  # 2-bit address
        engine.add_stimulus(Stimulus(f'SP_ADDR_{bit}', [0], [False]))
    for bit in range(4):  # 4-bit data
        engine.add_stimulus(Stimulus(f'SP_DIN_{bit}', [0], [False]))
    engine.add_stimulus(Stimulus('SP_READ', [0], [False]))
    engine.add_stimulus(Stimulus('SP_WRITE', [0], [False]))
    engine.add_stimulus(Stimulus('CLK', [0], [False]))
    engine.run(1e-6)

    T = 50e-6  # step time
    t = 5e-6

    def write_reg(addr, value, t):
        """Write value to register at addr."""
        for bit in range(2):
            engine.add_stimulus(Stimulus(f'SP_ADDR_{bit}', [t], [bool((addr >> bit) & 1)]))
        for bit in range(4):
            engine.add_stimulus(Stimulus(f'SP_DIN_{bit}', [t], [bool((value >> bit) & 1)]))
        engine.add_stimulus(Stimulus('SP_WRITE', [t], [True]))
        engine.add_stimulus(Stimulus('SP_READ', [t], [False]))
        # Clock: setup, rise, fall
        engine.add_stimulus(Stimulus('CLK', [t + T/4, t + T/2], [True, False]))
        engine.add_stimulus(Stimulus('SP_WRITE', [t + T*3/4], [False]))

    def read_reg(addr, t):
        """Read register at addr. Returns value after running."""
        for bit in range(2):
            engine.add_stimulus(Stimulus(f'SP_ADDR_{bit}', [t], [bool((addr >> bit) & 1)]))
        engine.add_stimulus(Stimulus('SP_READ', [t], [True]))
        engine.add_stimulus(Stimulus('SP_WRITE', [t], [False]))
        engine.add_stimulus(Stimulus('CLK', [t], [False]))  # no clock needed for read

    # Write different values to each register
    test_data = {0: 5, 1: 10, 2: 3, 3: 15}
    for addr, value in test_data.items():
        write_reg(addr, value, t)
        t += T
        engine.run(t)
        print(f"  Write R{addr} = {value}")

    # Read back each register
    print()
    for addr in range(4):
        read_reg(addr, t)
        t += T
        engine.run(t)

        val = 0
        for bit in range(4):
            ns = engine._nets.get(f'SP_DOUT_{bit}')
            if ns and ns.value:
                val |= (1 << bit)
        expected = test_data.get(addr, 0)
        match = "OK" if val == expected else "FAIL"
        print(f"  Read R{addr} = {val} (expected {expected}) {match}")
