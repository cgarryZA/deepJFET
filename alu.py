"""Parameterizable ALU builder.

Generates a gate-level netlist for an N-bit ALU with configurable operations.

Subcomponents (each optional):
  - Arithmetic: ripple-carry adder (ADD/SUB via carry-in + invert)
  - Logic: AND, OR, XOR (any combination)
  - Shift: SHL, SHR

Flags (configurable):
  - Zero: NOR of all output bits
  - Carry: carry-out from adder
  - Negative: MSB of result
  - Custom: comparator (e.g. result > threshold)

The opcode selects which subcomponent feeds the output via a mux tree.

Usage:
    alu = ALUBuilder(n_bits=4, arithmetic=True, logic=['AND', 'OR', 'XOR'],
                     shift=True, flags=['zero', 'carry', 'negative'])
    gates = alu.build()
    netlist = Netlist.from_gates(gates, primary_outputs=alu.outputs)
"""

from model import GateType
from simulator.netlist import Gate


class ALUBuilder:
    def __init__(self, n_bits=4, arithmetic=True, logic=None, shift=False,
                 flags=None):
        """
        Args:
            n_bits: bit width
            arithmetic: include ADD/SUB (ripple-carry adder)
            logic: list of logic ops to include, e.g. ['AND', 'OR', 'XOR']
            shift: include SHL/SHR
            flags: list of flag names, e.g. ['zero', 'carry', 'negative']
        """
        self.n_bits = n_bits
        self.arithmetic = arithmetic
        self.logic = logic or []
        self.shift = shift
        self.flags = flags or ['zero', 'carry']

        self.gates = []
        self.outputs = set()
        self.flag_nets = {}

        # Opcode mapping: which operation index corresponds to which op
        self.ops = []
        if self.arithmetic:
            self.ops.append('ADD')  # SUB = ADD with B inverted + carry_in=1
        for op in self.logic:
            self.ops.append(op)
        if self.shift:
            self.ops.append('SHL')
            self.ops.append('SHR')

        self.n_ops = len(self.ops)

    def build(self):
        """Generate all gates. Returns list of Gate objects."""
        self.gates = []

        # Input nets: A_0..A_{n-1}, B_0..B_{n-1}
        # Control: SUB (invert B and set carry_in), OP_0..OP_k (mux select)
        # Output: OUT_0..OUT_{n-1}, flags

        if self.arithmetic:
            self._build_adder()

        for op in self.logic:
            self._build_logic_op(op)

        if self.shift:
            self._build_shifter()

        # Mux to select output based on opcode
        if self.n_ops > 1:
            self._build_output_mux()
        elif self.n_ops == 1:
            # Only one operation — wire directly to output
            op = self.ops[0]
            for bit in range(self.n_bits):
                src = f'{op}_{bit}'
                out = f'OUT_{bit}'
                # Identity gate (buffer) = double inversion
                self.gates.append(Gate(f'buf_{bit}_inv', GateType.INV,
                                       [src], f'{out}_inv'))
                self.gates.append(Gate(f'buf_{bit}', GateType.INV,
                                       [f'{out}_inv'], out))
                self.outputs.add(out)

        # Flags
        self._build_flags()

        return self.gates

    def _build_adder(self):
        """Ripple-carry adder. SUB control inverts B and sets carry_in."""
        n = self.n_bits

        for bit in range(n):
            b_net = f'B_{bit}'
            # XOR B with SUB control to conditionally invert
            # XOR from NAND: a^b = ((a NAND (a NAND b)) NAND (b NAND (a NAND b)))
            self._build_xor(f'BINV_{bit}', b_net, 'SUB', f'B_eff_{bit}')

            # Carry in: SUB for bit 0, carry from previous for others
            c_in = 'SUB' if bit == 0 else f'CARRY_{bit}'

            # Full adder: SUM and CARRY
            self._build_full_adder(bit, f'A_{bit}', f'B_eff_{bit}', c_in)

        # Final carry out
        self.flag_nets['carry'] = f'CARRY_{n}'

    def _build_full_adder(self, bit, a_net, b_net, c_in):
        """1-bit full adder from NAND gates.

        SUM = A ^ B ^ Cin
        COUT = (A & B) | (A & Cin) | (B & Cin)
              = majority function

        Using 9 NAND gates.
        """
        prefix = f'FA_{bit}'

        # A XOR B
        self._build_xor(f'{prefix}_axb', a_net, b_net, f'{prefix}_AxB')

        # (A XOR B) XOR Cin = SUM
        self._build_xor(f'{prefix}_sum', f'{prefix}_AxB', c_in, f'ADD_{bit}')

        # Carry: NAND-based majority
        # COUT = (A NAND B) NAND ((A NAND B) NAND (Cin NAND (A XOR B)))
        # Simplified: use (A&B) | (Cin & (A^B))
        self.gates.append(Gate(f'{prefix}_ab_nand', GateType.NAND2,
                               [a_net, b_net], f'{prefix}_ab_n'))
        self.gates.append(Gate(f'{prefix}_caxb_nand', GateType.NAND2,
                               [c_in, f'{prefix}_AxB'], f'{prefix}_caxb_n'))
        self.gates.append(Gate(f'{prefix}_cout', GateType.NAND2,
                               [f'{prefix}_ab_n', f'{prefix}_caxb_n'],
                               f'CARRY_{bit + 1}'))

    def _build_xor(self, prefix, a_net, b_net, out_net):
        """XOR from 4 NAND gates: a^b = NAND(NAND(a, NAND(a,b)), NAND(b, NAND(a,b)))"""
        self.gates.append(Gate(f'{prefix}_n1', GateType.NAND2,
                               [a_net, b_net], f'{prefix}_nab'))
        self.gates.append(Gate(f'{prefix}_n2', GateType.NAND2,
                               [a_net, f'{prefix}_nab'], f'{prefix}_na'))
        self.gates.append(Gate(f'{prefix}_n3', GateType.NAND2,
                               [b_net, f'{prefix}_nab'], f'{prefix}_nb'))
        self.gates.append(Gate(f'{prefix}_n4', GateType.NAND2,
                               [f'{prefix}_na', f'{prefix}_nb'], out_net))

    def _build_logic_op(self, op):
        """Build logic operation for all bits."""
        for bit in range(self.n_bits):
            a = f'A_{bit}'
            b = f'B_{bit}'
            out = f'{op}_{bit}'

            if op == 'AND':
                # AND from NAND + INV
                self.gates.append(Gate(f'{op}_{bit}_nand', GateType.NAND2,
                                       [a, b], f'{out}_n'))
                self.gates.append(Gate(f'{op}_{bit}_inv', GateType.INV,
                                       [f'{out}_n'], out))
            elif op == 'OR':
                # OR from NOR + INV... but we have NAND.
                # OR = NAND(NOT(a), NOT(b))
                self.gates.append(Gate(f'{op}_{bit}_inva', GateType.INV,
                                       [a], f'{out}_na'))
                self.gates.append(Gate(f'{op}_{bit}_invb', GateType.INV,
                                       [b], f'{out}_nb'))
                self.gates.append(Gate(f'{op}_{bit}_nand', GateType.NAND2,
                                       [f'{out}_na', f'{out}_nb'], out))
            elif op == 'XOR':
                self._build_xor(f'{op}_{bit}', a, b, out)

    def _build_shifter(self):
        """Simple barrel shifter (1-bit shift only for now)."""
        n = self.n_bits
        for bit in range(n):
            # SHL: bit i gets bit i-1 (bit 0 gets 0)
            if bit == 0:
                # SHL_0 = 0 (ground) — use a tied-low net
                # Buffer through inverters to create a LOW
                self.gates.append(Gate(f'SHL_{bit}_inv', GateType.INV,
                                       ['VDD_LOGIC'], f'SHL_{bit}_h'))
                self.gates.append(Gate(f'SHL_{bit}_buf', GateType.INV,
                                       [f'SHL_{bit}_h'], f'SHL_{bit}'))
            else:
                # SHL_bit = A_{bit-1} (buffer)
                self.gates.append(Gate(f'SHL_{bit}_inv', GateType.INV,
                                       [f'A_{bit-1}'], f'SHL_{bit}_n'))
                self.gates.append(Gate(f'SHL_{bit}_buf', GateType.INV,
                                       [f'SHL_{bit}_n'], f'SHL_{bit}'))

            # SHR: bit i gets bit i+1 (MSB gets 0)
            if bit == n - 1:
                self.gates.append(Gate(f'SHR_{bit}_inv', GateType.INV,
                                       ['VDD_LOGIC'], f'SHR_{bit}_h'))
                self.gates.append(Gate(f'SHR_{bit}_buf', GateType.INV,
                                       [f'SHR_{bit}_h'], f'SHR_{bit}'))
            else:
                self.gates.append(Gate(f'SHR_{bit}_inv', GateType.INV,
                                       [f'A_{bit+1}'], f'SHR_{bit}_n'))
                self.gates.append(Gate(f'SHR_{bit}_buf', GateType.INV,
                                       [f'SHR_{bit}_n'], f'SHR_{bit}'))

    def _build_output_mux(self):
        """Mux tree to select operation output.

        For now: 2-input mux from gates. For >2 ops, chain muxes.
        MUX(sel, a, b) = (a AND NOT sel) OR (b AND sel)
                       = NAND(NAND(a, NOT sel), NAND(b, sel))
        """
        n = self.n_bits

        if self.n_ops == 2:
            # Single select bit: OP_SEL
            for bit in range(n):
                a_net = f'{self.ops[0]}_{bit}'
                b_net = f'{self.ops[1]}_{bit}'
                self._build_mux2(f'MUX_{bit}', 'OP_SEL', a_net, b_net,
                                 f'OUT_{bit}')
                self.outputs.add(f'OUT_{bit}')

        elif self.n_ops <= 4:
            # 2 select bits: OP_SEL_0, OP_SEL_1
            # First level: mux pairs with OP_SEL_0
            for bit in range(n):
                # Mux ops[0] vs ops[1]
                a0 = f'{self.ops[0]}_{bit}' if 0 < len(self.ops) else f'A_{bit}'
                a1 = f'{self.ops[1]}_{bit}' if 1 < len(self.ops) else f'A_{bit}'
                self._build_mux2(f'MUX_L0_{bit}', 'OP_SEL_0', a0, a1,
                                 f'MUX_L0_{bit}_out')

                if len(self.ops) > 2:
                    a2 = f'{self.ops[2]}_{bit}' if 2 < len(self.ops) else f'A_{bit}'
                    a3 = f'{self.ops[3]}_{bit}' if 3 < len(self.ops) else f'A_{bit}'
                    self._build_mux2(f'MUX_L1_{bit}', 'OP_SEL_0', a2, a3,
                                     f'MUX_L1_{bit}_out')

                    # Second level: mux the two results with OP_SEL_1
                    self._build_mux2(f'MUX_OUT_{bit}', 'OP_SEL_1',
                                     f'MUX_L0_{bit}_out', f'MUX_L1_{bit}_out',
                                     f'OUT_{bit}')
                else:
                    # Only 3 ops: second mux input is the third op
                    a2 = f'{self.ops[2]}_{bit}'
                    self._build_mux2(f'MUX_OUT_{bit}', 'OP_SEL_1',
                                     f'MUX_L0_{bit}_out', a2, f'OUT_{bit}')

                self.outputs.add(f'OUT_{bit}')
        else:
            # For >4 ops, extend the mux tree (not implemented yet)
            raise NotImplementedError(f"Mux for {self.n_ops} ops not yet supported")

    def _build_mux2(self, prefix, sel, a_net, b_net, out_net):
        """2:1 mux: out = sel ? b : a. Built from 3 NAND + 1 INV."""
        # NOT sel
        self.gates.append(Gate(f'{prefix}_inv_sel', GateType.INV,
                               [sel], f'{prefix}_nsel'))
        # NAND(a, NOT sel)
        self.gates.append(Gate(f'{prefix}_na', GateType.NAND2,
                               [a_net, f'{prefix}_nsel'], f'{prefix}_an'))
        # NAND(b, sel)
        self.gates.append(Gate(f'{prefix}_nb', GateType.NAND2,
                               [b_net, sel], f'{prefix}_bn'))
        # NAND(an, bn) = out
        self.gates.append(Gate(f'{prefix}_out', GateType.NAND2,
                               [f'{prefix}_an', f'{prefix}_bn'], out_net))

    def _build_flags(self):
        """Build flag logic."""
        n = self.n_bits

        if 'zero' in self.flags:
            # Zero = NOR of all output bits
            # Build OR tree, then invert
            if n <= 4:
                # Direct NOR for small widths
                if n == 1:
                    self.gates.append(Gate('flag_zero', GateType.INV,
                                           ['OUT_0'], 'FLAG_ZERO'))
                else:
                    # OR chain: OR pairs, then OR the results
                    # OR = NAND(INV(a), INV(b))
                    prev = 'OUT_0'
                    for bit in range(1, n):
                        # OR prev with OUT_{bit}
                        self.gates.append(Gate(f'zero_inv_a_{bit}', GateType.INV,
                                               [prev], f'zero_na_{bit}'))
                        self.gates.append(Gate(f'zero_inv_b_{bit}', GateType.INV,
                                               [f'OUT_{bit}'], f'zero_nb_{bit}'))
                        self.gates.append(Gate(f'zero_or_{bit}', GateType.NAND2,
                                               [f'zero_na_{bit}', f'zero_nb_{bit}'],
                                               f'zero_or_{bit}'))
                        prev = f'zero_or_{bit}'
                    # Invert final OR to get NOR (zero flag)
                    self.gates.append(Gate('flag_zero', GateType.INV,
                                           [prev], 'FLAG_ZERO'))
            self.flag_nets['zero'] = 'FLAG_ZERO'
            self.outputs.add('FLAG_ZERO')

        if 'carry' in self.flags and self.arithmetic:
            self.outputs.add(self.flag_nets.get('carry', f'CARRY_{n}'))

        if 'negative' in self.flags:
            # Negative = MSB of output (buffer it)
            self.gates.append(Gate('flag_neg_inv', GateType.INV,
                                   [f'OUT_{n-1}'], 'FLAG_NEG_inv'))
            self.gates.append(Gate('flag_neg', GateType.INV,
                                   ['FLAG_NEG_inv'], 'FLAG_NEG'))
            self.flag_nets['negative'] = 'FLAG_NEG'
            self.outputs.add('FLAG_NEG')

    @property
    def input_nets(self):
        """All primary input net names."""
        nets = []
        for bit in range(self.n_bits):
            nets.extend([f'A_{bit}', f'B_{bit}'])
        if self.arithmetic:
            nets.append('SUB')
        if self.n_ops == 2:
            nets.append('OP_SEL')
        elif self.n_ops > 2:
            nets.extend(['OP_SEL_0', 'OP_SEL_1'])
        return nets

    def summary(self):
        """Print ALU configuration."""
        print(f"ALU: {self.n_bits}-bit")
        print(f"  Operations: {', '.join(self.ops)}")
        print(f"  Flags: {', '.join(self.flags)}")
        print(f"  Gates: {len(self.gates)}")
        print(f"  Inputs: {', '.join(self.input_nets)}")
        print(f"  Outputs: {', '.join(sorted(self.outputs))}")
