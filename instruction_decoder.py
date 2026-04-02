"""Instruction register + decoder.

Parameterizable instruction decoder that generates control signals
from an instruction set definition.

Architecture (based on 4004 format):
  - Instruction Register: N-bit register holding current instruction
  - OPR field (opcode): upper bits, decoded to one-hot control lines
  - OPA field (modifier): lower bits, passed through as data/address

The instruction set is defined as a Python dict mapping opcode patterns
to control signal names. The decoder generates combinational logic
(AND/OR/NOT gates) to produce the control signals.

Usage:
    isa = InstructionSet(opcode_bits=4, modifier_bits=4)
    isa.add("NOP",  "0000", signals=[])
    isa.add("LDM",  "1101", signals=["ACC_LOAD", "USE_IMMEDIATE"])
    isa.add("ADD",  "1000", signals=["ALU_ADD", "ACC_LOAD"])
    isa.add("SUB",  "1001", signals=["ALU_SUB", "ACC_LOAD"])

    gates = isa.build_decoder()
"""

from model import GateType
from simulator.netlist import Gate
from register import make_register


class InstructionSet:
    """Define an instruction set and generate decoder logic."""

    def __init__(self, opcode_bits=4, modifier_bits=4):
        self.opcode_bits = opcode_bits
        self.modifier_bits = modifier_bits
        self.total_bits = opcode_bits + modifier_bits
        self.instructions = {}  # name -> (pattern, signals)
        self.all_signals = set()

    def add(self, name: str, opcode_pattern: str, signals: list = None):
        """Add an instruction.

        Args:
            name: instruction mnemonic (e.g. "ADD")
            opcode_pattern: binary string for opcode bits (e.g. "1000")
                           Can use 'x' for don't-care bits.
            signals: list of control signal names this instruction asserts
        """
        if len(opcode_pattern) != self.opcode_bits:
            raise ValueError(f"{name}: pattern '{opcode_pattern}' must be "
                             f"{self.opcode_bits} bits")
        self.instructions[name] = (opcode_pattern, signals or [])
        self.all_signals.update(signals or [])

    def build(self) -> tuple:
        """Build instruction register + decoder.

        Returns:
            (gates, ir_input_nets, control_signal_nets, modifier_nets)
        """
        gates = []
        n = self.total_bits

        # --- Instruction Register ---
        reg_gates, reg_ins, reg_outs, reg_ctrl = make_register("IR", n)
        gates.extend(reg_gates)

        # IR bit naming: IR_7 IR_6 IR_5 IR_4 | IR_3 IR_2 IR_1 IR_0
        #                 OPR (opcode)        |  OPA (modifier)
        # IR_{total-1} is MSB of opcode

        # --- Generate inverted opcode bits (needed for decoding) ---
        for bit in range(self.opcode_bits):
            ir_bit = self.modifier_bits + bit  # opcode is upper bits
            gates.append(Gate(f'DEC_inv_{bit}', GateType.INV,
                              [f'IR_{ir_bit}'], f'DEC_OPR_{bit}_bar'))

        # --- Decode each instruction to a one-hot line ---
        # For each instruction, AND together the required opcode bits
        # (using the true or inverted form based on the pattern)

        decode_nets = {}  # instruction_name -> decoded net name

        for inst_name, (pattern, signals) in self.instructions.items():
            # Build AND chain for this pattern
            # pattern[0] is MSB of opcode = IR_{modifier_bits + opcode_bits - 1}
            and_inputs = []
            for i, bit_char in enumerate(pattern):
                if bit_char == 'x':
                    continue  # don't care
                ir_bit = self.modifier_bits + (self.opcode_bits - 1 - i)
                dec_bit = i  # position in opcode from MSB
                # Map to decoder inverted bit index
                inv_idx = self.opcode_bits - 1 - i
                if bit_char == '1':
                    and_inputs.append(f'IR_{ir_bit}')
                else:  # '0'
                    and_inputs.append(f'DEC_OPR_{inv_idx}_bar')

            if len(and_inputs) == 0:
                # All don't-care — always active (shouldn't happen for real instructions)
                decode_net = f'DEC_{inst_name}'
                # Tie high via double inverter
                gates.append(Gate(f'DEC_{inst_name}_inv', GateType.INV,
                                  ['IR_0'], f'DEC_{inst_name}_tmp'))
                gates.append(Gate(f'DEC_{inst_name}_buf', GateType.INV,
                                  [f'DEC_{inst_name}_tmp'], decode_net))
            elif len(and_inputs) == 1:
                # Single bit — just buffer it
                decode_net = f'DEC_{inst_name}'
                gates.append(Gate(f'DEC_{inst_name}_inv', GateType.INV,
                                  [and_inputs[0]], f'DEC_{inst_name}_tmp'))
                gates.append(Gate(f'DEC_{inst_name}_buf', GateType.INV,
                                  [f'DEC_{inst_name}_tmp'], decode_net))
            else:
                # Multi-bit AND: chain NANDs
                # AND(a,b,c,d) = NOT(NAND(NAND(a,b), NAND(c,d)))
                # For simplicity: pairwise NAND tree
                decode_net = self._build_and_tree(
                    gates, f'DEC_{inst_name}', and_inputs)

            decode_nets[inst_name] = decode_net

        # --- Generate control signals ---
        # Each control signal is OR of all instructions that assert it
        control_signal_nets = {}

        for signal in sorted(self.all_signals):
            # Find all instructions that assert this signal
            asserting = []
            for inst_name, (pattern, signals) in self.instructions.items():
                if signal in signals:
                    asserting.append(decode_nets[inst_name])

            if len(asserting) == 0:
                continue
            elif len(asserting) == 1:
                # Single source — buffer
                sig_net = f'CTRL_{signal}'
                gates.append(Gate(f'CTRL_{signal}_inv', GateType.INV,
                                  [asserting[0]], f'CTRL_{signal}_tmp'))
                gates.append(Gate(f'CTRL_{signal}_buf', GateType.INV,
                                  [f'CTRL_{signal}_tmp'], sig_net))
            else:
                # OR tree: OR(a,b) = NAND(NOT(a), NOT(b))
                sig_net = self._build_or_tree(
                    gates, f'CTRL_{signal}', asserting)

            control_signal_nets[signal] = sig_net

        # Modifier bits pass through directly
        modifier_nets = [f'IR_{bit}' for bit in range(self.modifier_bits)]

        return (gates, reg_ins, control_signal_nets, modifier_nets,
                decode_nets, reg_ctrl)

    def _build_and_tree(self, gates, prefix, inputs):
        """Build AND from NAND tree. Returns output net name."""
        if len(inputs) == 1:
            gates.append(Gate(f'{prefix}_inv', GateType.INV,
                              [inputs[0]], f'{prefix}_tmp'))
            gates.append(Gate(f'{prefix}_buf', GateType.INV,
                              [f'{prefix}_tmp'], prefix))
            return prefix
        if len(inputs) == 2:
            gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                              inputs, f'{prefix}_n'))
            gates.append(Gate(f'{prefix}_inv', GateType.INV,
                              [f'{prefix}_n'], prefix))
            return prefix

        # Split and recurse
        mid = len(inputs) // 2
        left = self._build_and_tree(gates, f'{prefix}_L', inputs[:mid])
        right = self._build_and_tree(gates, f'{prefix}_R', inputs[mid:])
        gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                          [left, right], f'{prefix}_n'))
        gates.append(Gate(f'{prefix}_inv', GateType.INV,
                          [f'{prefix}_n'], prefix))
        return prefix

    def _build_or_tree(self, gates, prefix, inputs):
        """Build OR from NAND tree. OR(a,b) = NAND(NOT a, NOT b)."""
        if len(inputs) == 1:
            # Buffer
            gates.append(Gate(f'{prefix}_inv', GateType.INV,
                              [inputs[0]], f'{prefix}_tmp'))
            gates.append(Gate(f'{prefix}_buf', GateType.INV,
                              [f'{prefix}_tmp'], prefix))
            return prefix
        if len(inputs) == 2:
            gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                              [inputs[0]], f'{prefix}_na'))
            gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                              [inputs[1]], f'{prefix}_nb'))
            gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                              [f'{prefix}_na', f'{prefix}_nb'], prefix))
            return prefix

        # Split and recurse
        mid = len(inputs) // 2
        left = self._build_or_tree(gates, f'{prefix}_L', inputs[:mid])
        right = self._build_or_tree(gates, f'{prefix}_R', inputs[mid:])
        gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                          [left], f'{prefix}_na'))
        gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                          [right], f'{prefix}_nb'))
        gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                          [f'{prefix}_na', f'{prefix}_nb'], prefix))
        return prefix

    def summary(self):
        print(f"Instruction Set: {self.opcode_bits}-bit opcode, "
              f"{self.modifier_bits}-bit modifier")
        print(f"  Instructions: {len(self.instructions)}")
        print(f"  Control signals: {len(self.all_signals)}")
        for name, (pattern, signals) in sorted(self.instructions.items()):
            sigs = ', '.join(signals) if signals else '(none)'
            print(f"    {name:6s} [{pattern}] -> {sigs}")


def build_4004_subset():
    """Build a subset of the Intel 4004 instruction set for testing."""
    isa = InstructionSet(opcode_bits=4, modifier_bits=4)

    # Core instructions (simplified 4004 subset)
    isa.add("NOP",  "0000", signals=[])
    isa.add("JCN",  "0001", signals=["PC_LOAD_COND", "FETCH_ADDR"])
    isa.add("FIM",  "0010", signals=["REG_LOAD_IMM", "FETCH_DATA"])
    isa.add("JUN",  "0100", signals=["PC_LOAD", "FETCH_ADDR"])
    isa.add("JMS",  "0101", signals=["PC_LOAD", "STACK_PUSH", "FETCH_ADDR"])
    isa.add("INC",  "0110", signals=["REG_INC"])
    isa.add("ISZ",  "0111", signals=["REG_INC", "PC_LOAD_COND", "FETCH_ADDR"])
    isa.add("ADD",  "1000", signals=["ALU_ADD", "ACC_LOAD"])
    isa.add("SUB",  "1001", signals=["ALU_SUB", "ACC_LOAD"])
    isa.add("LD",   "1010", signals=["ACC_LOAD", "REG_TO_BUS"])
    isa.add("XCH",  "1011", signals=["ACC_LOAD", "REG_LOAD", "REG_TO_BUS"])
    isa.add("BBL",  "1100", signals=["STACK_POP", "ACC_LOAD_IMM"])
    isa.add("LDM",  "1101", signals=["ACC_LOAD", "USE_IMMEDIATE"])

    return isa


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
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    params = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)
    V_HIGH, V_LOW = -0.8, -4.0

    isa = build_4004_subset()
    isa.summary()

    gates, ir_ins, ctrl_nets, mod_nets, dec_nets, reg_ctrl = isa.build()
    print(f"\nDecoder: {len(gates)} gates")
    print(f"  Control signals: {list(ctrl_nets.keys())}")
    print(f"  Decode lines: {list(dec_nets.keys())}")

    # Test: load different instructions and check which control signals fire
    all_outputs = set(ctrl_nets.values()) | set(dec_nets.values())
    netlist = Netlist.from_gates(gates, primary_outputs=all_outputs)
    ordered, feedback = netlist.topological_sort()
    print(f"  Ordered: {len(ordered)}, Feedback: {len(feedback)}")

    profiles = {}
    for gt in [GateType.INV, GateType.NAND2]:
        profiles[gt] = precompute_gate(gt, params, V_HIGH, V_LOW)

    # Test each instruction
    test_instructions = [
        ("NOP",  0x00),  # 0000 0000
        ("ADD",  0x83),  # 1000 0011 (ADD register 3)
        ("SUB",  0x95),  # 1001 0101 (SUB register 5)
        ("LDM",  0xD7),  # 1101 0111 (LDM immediate 7)
        ("LD",   0xA2),  # 1010 0010 (LD register 2)
        ("INC",  0x64),  # 0110 0100 (INC register 4)
        ("JUN",  0x4F),  # 0100 1111 (JUN address F)
    ]

    print(f"\n{'Inst':>5} {'Hex':>5} {'Binary':>10}  Active control signals")
    print("-" * 65)

    for inst_name, opcode in test_instructions:
        netlist = Netlist.from_gates(gates, primary_outputs=all_outputs)
        engine = SimulationEngine(netlist, profiles, v_high=V_HIGH, v_low=V_LOW,
                                  auto_precompute_params=params)

        # Load instruction into IR
        for bit in range(8):
            val = bool((opcode >> bit) & 1)
            engine.add_stimulus(Stimulus(f'IR_{bit}_In', [0, 1e-6],
                                         [True, val]))

        engine.add_stimulus(Stimulus('CLK', [0, 1e-6, 6e-6],
                                     [True, True, False]))
        engine.add_stimulus(Stimulus('IR_Enable', [0, 1e-6],
                                     [True, True]))

        result = engine.run(50e-6)

        # Check which control signals are active
        active = []
        for sig_name, net_name in sorted(ctrl_nets.items()):
            ns = result.net_states.get(net_name)
            if ns and ns.value:
                active.append(sig_name)

        # Check which decode line fired
        decoded = []
        for dec_name, net_name in sorted(dec_nets.items()):
            ns = result.net_states.get(net_name)
            if ns and ns.value:
                decoded.append(dec_name)

        binary = format(opcode, '08b')
        active_str = ', '.join(active) if active else '(none)'
        print(f"{inst_name:>5} 0x{opcode:02X}  {binary}  {active_str}")
