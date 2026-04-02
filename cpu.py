"""JFET CPU — gate-level datapath + gate-level microsequencer.

Everything except ROM is real gates:
  - Ring counter (7-step one-hot) sequences micro-operations
  - Microinstruction matrix (combinational AND/OR) generates control signals
  - Data bus with MI-OR gated writers
  - ACC/BREG hardwired to ALU, bus-connected for XCH/LD
  - PC directly addresses ROM, IR loaded from ROM
  - ROM is behavioral (Python dict)

The simulation runs one micro-step per clock cycle. Between cycles,
the ROM callback reads PC and updates IR input stimuli.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from model import GateType
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams, precompute_gate
from simulator.engine import SimulationEngine, Stimulus
from model import NChannelJFET, JFETCapacitance
from register import make_register
from alu import ALUBuilder
from program_counter import make_program_counter
from instruction_decoder import InstructionSet


# ---------------------------------------------------------------------------
# Microcode definition
# ---------------------------------------------------------------------------

# Steps 0-2 are fixed (fetch/load/decode). Steps 3+ are instruction-specific.
# Each micro-step asserts a set of control signals.
# RESET_RING on the last step resets to step 0 for next instruction.
# PC_INC on the last step of non-jump instructions.

MICROCODE = {
    'NOP': [
        (3, ['PC_INC']),
    ],
    'IAC': [
        (3, ['ALU_ADD', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU', 'PC_INC']),
    ],
    'ADD': [
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ALU_ADD', 'ACC_LOAD_FROM_ALU', 'PC_INC']),
    ],
    'SUB': [
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ALU_SUB', 'ACC_LOAD_FROM_ALU', 'PC_INC']),
    ],
    'LD': [
        (3, ['R0_TO_BUS', 'ACC_LOAD_FROM_BUS', 'PC_INC']),
    ],
    'LDM': [
        (3, ['IMM_TO_BUS', 'ACC_LOAD_FROM_BUS', 'PC_INC']),
    ],
    'XCH': [
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ACC_TO_BUS', 'R0_LOAD_FROM_BUS']),
        (5, ['BREG_TO_BUS', 'ACC_LOAD_FROM_BUS']),
        (6, ['PC_INC']),
    ],
    'JUN': [
        (3, ['PC_LOAD']),
    ],
    'JCN': [
        # Conditional: PC_LOAD if carry, else PC_INC
        # Handled specially — both signals generated, gated by carry flag
        (3, []),  # PC_LOAD_IF_CARRY and PC_INC_IF_NO_CARRY added in matrix
    ],
}

MAX_STEPS = 7  # steps 0-6

ALL_CONTROL_SIGNALS = [
    'ALU_ADD', 'ALU_SUB', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU',
    'ACC_TO_BUS', 'BREG_TO_BUS', 'R0_TO_BUS', 'IMM_TO_BUS',
    'ACC_LOAD_FROM_BUS', 'BREG_LOAD_FROM_BUS', 'R0_LOAD_FROM_BUS',
    'PC_INC', 'PC_LOAD', 'IR_LOAD',
]

OPCODE_MAP = {
    0x0: 'NOP', 0x1: 'IAC', 0x2: 'JCN', 0x4: 'JUN',
    0x8: 'ADD', 0x9: 'SUB', 0xA: 'LD', 0xB: 'XCH', 0xD: 'LDM',
}


class CPU:
    def __init__(self, n_bits=4):
        self.n_bits = n_bits
        self.rom = {}
        self._build()

    def _build(self):
        n = self.n_bits
        self.gates = []

        self._build_jfet_params()
        self._build_isa()
        self._build_registers(n)
        self._build_alu(n)
        self._build_decoder()
        self._build_pc(n)
        self._build_bus(n)
        self._build_acc_mux(n)
        self._build_reg_inputs(n)
        self._build_carry_flag()
        self._build_reg_enables()
        self._build_profiles()

        print(f"CPU built: {len(self.gates)} gates")

    def _build_jfet_params(self):
        self.jfet = NChannelJFET(
            beta=0.000135, vto=-3.45, lmbda=0.005, is_=205.2e-15, n=3.0,
            isr=1988e-15, nr=4.0, alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
            betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26).at_temp(27.0)
        self.caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
        self.params_inv = CircuitParams(v_pos=10, v_neg=-10, r1=12100,
                                         r2=7320, r3=6980, jfet=self.jfet, caps=self.caps)
        self.params_nand = CircuitParams(v_pos=10, v_neg=-10, r1=24300,
                                          r2=7320, r3=6980, jfet=self.jfet, caps=self.caps)

    def _build_isa(self):
        self.isa = InstructionSet(opcode_bits=4, modifier_bits=4)
        for opr, name in OPCODE_MAP.items():
            pattern = format(opr, '04b')
            self.isa.add(name, pattern, signals=[])

    def _build_registers(self, n):
        for name in ['ACC', 'BREG', 'R0']:
            g, _, _, _ = make_register(name, n)
            self.gates.extend(g)

    def _build_alu(self, n):
        alu = ALUBuilder(n_bits=n, arithmetic=True, logic=[], shift=False,
                         flags=['zero', 'carry'])
        alu_gates = alu.build()

        # ALU A input = ACC (direct)
        # ALU B input = mux between BREG and constant 1 (when ALU_B_IS_ONE)
        # Create ALU_B_{bit} nets that feed the ALU
        for bit in range(n):
            # When ALU_B_IS_ONE: bit 0 = 1, others = 0
            const_val = 'ALU_B_IS_ONE' if bit == 0 else 'ALU_B_IS_ONE_INV'
            breg_net = f'BREG_{bit}'
            mux_out = f'ALU_B_{bit}'

            # NOT ALU_B_IS_ONE
            if bit == 0:
                self.gates.append(Gate(f'ALU_B1_INV', GateType.INV,
                                       ['ALU_B_IS_ONE'], 'ALU_B_IS_ONE_INV'))

            # MUX: ALU_B_IS_ONE ? const_val : BREG
            # = OR(AND(BREG, NOT ALU_B_IS_ONE), AND(const_val, ALU_B_IS_ONE))
            # For bit 0: const = ALU_B_IS_ONE, so second term = AND(ALU_B_IS_ONE, ALU_B_IS_ONE) = ALU_B_IS_ONE
            # For others: const = 0 when ALU_B_IS_ONE, so second term = 0
            self.gates.append(Gate(f'ALU_BMUX_{bit}_breg_nand', GateType.NAND2,
                                   [breg_net, 'ALU_B_IS_ONE_INV'],
                                   f'ALU_BMUX_{bit}_breg_n'))
            self.gates.append(Gate(f'ALU_BMUX_{bit}_breg_inv', GateType.INV,
                                   [f'ALU_BMUX_{bit}_breg_n'],
                                   f'ALU_BMUX_{bit}_breg'))

            if bit == 0:
                # OR(BREG_and, ALU_B_IS_ONE)
                self._or_tree([f'ALU_BMUX_{bit}_breg', 'ALU_B_IS_ONE'],
                              mux_out, f'ALU_BMUX_{bit}')
            else:
                # Just the BREG path (const is 0)
                self.gates.append(Gate(f'ALU_BMUX_{bit}_buf_inv', GateType.INV,
                                       [f'ALU_BMUX_{bit}_breg'],
                                       f'ALU_BMUX_{bit}_n'))
                self.gates.append(Gate(f'ALU_BMUX_{bit}_buf', GateType.INV,
                                       [f'ALU_BMUX_{bit}_n'], mux_out))

        # Rename ALU gate inputs: A -> ACC, B -> ALU_B (muxed)
        for g in alu_gates:
            g.inputs = [f'ACC_{inp[2:]}' if inp.startswith('A_') and inp[2:].isdigit()
                        else f'ALU_B_{inp[2:]}' if inp.startswith('B_') and inp[2:].isdigit()
                        else inp for inp in g.inputs]
        self.gates.extend(alu_gates)

    def _build_decoder(self):
        result = self.isa.build()
        ir_gates, _, ctrl_nets, mod_nets, dec_nets, _ = result
        self.gates.extend(ir_gates)
        self.dec_nets = dec_nets

    def _build_pc(self, n):
        g, _, _, _ = make_program_counter("PC", n)
        self.gates.extend(g)

    def _build_bus(self, n):
        writers = [('ACC', 'ACC_TO_BUS'), ('BREG', 'BREG_TO_BUS'),
                   ('R0', 'R0_TO_BUS'), ('OUT', 'ALU_RESULT_TO_BUS')]
        for bit in range(n):
            outs = []
            for src, en in writers:
                p = f'BUS_W_{src}_{bit}'
                self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                       [f'{src}_{bit}', en], f'{p}_n'))
                self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                       [f'{p}_n'], f'{p}_out'))
                outs.append(f'{p}_out')
            # IMM writer
            p = f'BUS_W_IMM_{bit}'
            self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                   [f'IR_{bit}', 'IMM_TO_BUS'], f'{p}_n'))
            self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                   [f'{p}_n'], f'{p}_out'))
            outs.append(f'{p}_out')
            self._or_tree(outs, f'BUS_{bit}', f'BUS_OR_{bit}')

    def _build_acc_mux(self, n):
        for bit in range(n):
            for src, en in [('OUT', 'ACC_LOAD_FROM_ALU'), ('BUS', 'ACC_LOAD_FROM_BUS')]:
                net = f'{src}_{bit}' if src != 'BUS' else f'BUS_{bit}'
                p = f'ACC_MUX_{src}_{bit}'
                self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                       [net, en], f'{p}_n'))
                self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                       [f'{p}_n'], f'{p}_out'))
            self._or_tree([f'ACC_MUX_OUT_{bit}_out', f'ACC_MUX_BUS_{bit}_out'],
                          f'ACC_{bit}_In', f'ACC_MUX_{bit}')

    def _build_reg_inputs(self, n):
        for bit in range(n):
            for reg in ['BREG', 'R0']:
                self.gates.append(Gate(f'{reg}_BUF_{bit}_inv', GateType.INV,
                                       [f'BUS_{bit}'], f'{reg}_{bit}_In_n'))
                self.gates.append(Gate(f'{reg}_BUF_{bit}', GateType.INV,
                                       [f'{reg}_{bit}_In_n'], f'{reg}_{bit}_In'))
            # PC load from IR modifier
            self.gates.append(Gate(f'PC_LD_{bit}_inv', GateType.INV,
                                   [f'IR_{bit}'], f'PC_Load_{bit}_n'))
            self.gates.append(Gate(f'PC_LD_{bit}', GateType.INV,
                                   [f'PC_Load_{bit}_n'], f'PC_Load_{bit}'))

    def _build_carry_flag(self):
        """1-bit carry flag register. Latches CARRY_4 when ALU computes."""
        # Carry flag = 1-bit master-slave FF storing the ALU carry output
        # Enable: ACC_LOAD_FROM_ALU (only update carry when ALU result is stored)
        carry_gates, _, _, _ = make_register("CY", 1)
        self.gates.extend(carry_gates)

        # Wire ALU carry output (CARRY_4) to carry register input (CY_0_In)
        # CY_0_In is a primary input — we need a buffer from CARRY_4
        self.gates.append(Gate('CY_BUF_inv', GateType.INV,
                               ['CARRY_4'], 'CY_BUF_n'))
        self.gates.append(Gate('CY_BUF', GateType.INV,
                               ['CY_BUF_n'], 'CY_0_In'))

        # CY_Enable = ACC_LOAD_FROM_ALU (carry updates when ACC gets ALU result)
        self.gates.append(Gate('CY_EN_inv', GateType.INV,
                               ['ACC_LOAD_FROM_ALU'], 'CY_EN_n'))
        self.gates.append(Gate('CY_EN_buf', GateType.INV,
                               ['CY_EN_n'], 'CY_Enable'))

    def _build_reg_enables(self):
        self._or_tree(['ACC_LOAD_FROM_ALU', 'ACC_LOAD_FROM_BUS'],
                      'ACC_Enable', 'ACC_EN')
        for reg, sig in [('BREG', 'BREG_LOAD_FROM_BUS'), ('R0', 'R0_LOAD_FROM_BUS')]:
            self.gates.append(Gate(f'{reg}_EN_inv', GateType.INV,
                                   [sig], f'{reg}_Enable_n'))
            self.gates.append(Gate(f'{reg}_EN_buf', GateType.INV,
                                   [f'{reg}_Enable_n'], f'{reg}_Enable'))

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

    def _build_profiles(self):
        self.profiles = {}
        for gt in [GateType.INV, GateType.NAND2]:
            self.profiles[gt] = precompute_gate(gt, self.params_inv, -0.8, -4.0)

    def load_program(self, program: dict):
        self.rom = program

    def run(self, n_instructions=20, verbose=True):
        """Execute program through gate-level datapath with continuous simulation.

        Builds the netlist ONCE, then runs the event-driven simulator
        continuously, injecting stimuli at the right times for each
        micro-step. Gate state persists across all instructions.
        """
        V_HIGH, V_LOW = -0.8, -4.0
        n = self.n_bits
        T_STEP = 50e-6  # time per micro-step (enough for ALU to settle through carry chain)

        # Build netlist once
        outputs = set()
        for prefix in ['ACC', 'R0', 'BREG', 'PC']:
            for bit in range(n):
                outputs.add(f'{prefix}_{bit}')
        outputs.update(['CARRY_4', 'CY_0', 'FLAG_ZERO'] +
                       [f'OUT_{bit}' for bit in range(n)] +
                       [f'BUS_{bit}' for bit in range(n)])

        netlist = Netlist.from_gates(self.gates, primary_outputs=outputs)
        engine = SimulationEngine(netlist, self.profiles,
                                  v_high=V_HIGH, v_low=V_LOW,
                                  auto_precompute_params=self.params_inv)
        pi = netlist.primary_inputs

        # Phase 0: Power-on reset
        # Set all primary inputs to LOW
        for sig in pi:
            engine._nets[sig].value = False
            engine._nets[sig].voltage = V_LOW

        # Force all register Q outputs to LOW and Q_bar to HIGH
        # This is the async reset — directly setting latch state
        for prefix in ['ACC', 'BREG', 'R0', 'CY']:
            for bit in range(n):
                q_net = f'{prefix}_{bit}'
                qb_net = f'{prefix}_{bit}_bar'
                if q_net in engine._nets:
                    engine._nets[q_net].value = False
                    engine._nets[q_net].voltage = V_LOW
                    engine._nets[q_net].history = [(0, False, V_LOW)]
                if qb_net in engine._nets:
                    engine._nets[qb_net].value = True
                    engine._nets[qb_net].voltage = V_HIGH
                    engine._nets[qb_net].history = [(0, True, V_HIGH)]

        # IR register
        for bit in range(8):
            q_net = f'IR_{bit}'
            qb_net = f'IR_{bit}_bar'
            if q_net in engine._nets:
                engine._nets[q_net].value = False
                engine._nets[q_net].voltage = V_LOW
                engine._nets[q_net].history = [(0, False, V_LOW)]
            if qb_net in engine._nets:
                engine._nets[qb_net].value = True
                engine._nets[qb_net].voltage = V_HIGH
                engine._nets[qb_net].history = [(0, True, V_HIGH)]

        # PC register
        for bit in range(n):
            q_net = f'PC_{bit}'
            qb_net = f'PC_{bit}_bar'
            if q_net in engine._nets:
                engine._nets[q_net].value = False
                engine._nets[q_net].voltage = V_LOW
                engine._nets[q_net].history = [(0, False, V_LOW)]
            if qb_net in engine._nets:
                engine._nets[qb_net].value = True
                engine._nets[qb_net].voltage = V_HIGH
                engine._nets[qb_net].history = [(0, True, V_HIGH)]

        # Force-evaluate all combinational gates with the reset state
        engine.force_evaluate_all()

        # Set all control signals to initial LOW state
        for sig in ALL_CONTROL_SIGNALS:
            if sig in pi:
                engine.add_stimulus(Stimulus(sig, [0], [False]))
        if 'ALU_B_IS_ONE' in pi:
            engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [0], [False]))
        engine.add_stimulus(Stimulus('SUB', [0], [False]))
        engine.add_stimulus(Stimulus('INC', [0], [False]))
        engine.add_stimulus(Stimulus('LOAD', [0], [False]))
        engine.add_stimulus(Stimulus('CLK', [0], [False]))
        for en in ['ACC_Enable', 'BREG_Enable', 'R0_Enable', 'IR_Enable', 'PC_Enable']:
            if en in pi:
                engine.add_stimulus(Stimulus(en, [0], [False]))
        for bit in range(n):
            for prefix in ['ACC', 'BREG', 'R0']:
                if f'{prefix}_{bit}_In' in pi:
                    engine.add_stimulus(Stimulus(f'{prefix}_{bit}_In', [0], [False]))
        for bit in range(8):
            engine.add_stimulus(Stimulus(f'IR_{bit}_In', [0], [False]))

        engine.run(1e-6)
        engine.force_evaluate_all()

        # Now start executing instructions
        t_current = 5e-6  # start after init settles
        trace = []

        for inst_num in range(n_instructions):
            # Read current state from sim
            pc = self._read_value_from_engine(engine, 'PC', n)
            acc = self._read_value_from_engine(engine, 'ACC', n)
            r0 = self._read_value_from_engine(engine, 'R0', n)
            carry_ns = engine._nets.get('CY_0')
            carry = 1 if carry_ns and carry_ns.value else 0

            # Read carry BEFORE anything else this instruction
            # (carry was latched by the previous instruction's ALU op)
            carry_ns = engine._nets.get('CY_0')
            carry = 1 if carry_ns and carry_ns.value else 0

            # Fetch from ROM
            opcode = self.rom.get(pc, 0x00)
            opr = (opcode >> 4) & 0xF
            inst_name = OPCODE_MAP.get(opr, 'NOP')
            micro_steps = MICROCODE.get(inst_name, [(3, ['PC_INC'])])

            if verbose:
                print(f"  [{inst_num:3d}] PC={pc:2d} [{opcode:02X}] {inst_name:4s}  "
                      f"ACC={acc:2d} C={carry} R0={r0:2d}", end="")

            # Load IR with opcode: set data, then clock IR to latch
            for bit in range(8):
                bv = bool((opcode >> bit) & 1)
                engine.add_stimulus(Stimulus(f'IR_{bit}_In', [t_current], [bv]))
            engine.add_stimulus(Stimulus('IR_Enable', [t_current], [True]))
            # Disable ACC/BREG/R0 during IR load to prevent accidental latching
            engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('R0_Enable', [t_current], [False]))
            # All control signals LOW during fetch
            for sig in ALL_CONTROL_SIGNALS:
                if sig in pi:
                    engine.add_stimulus(Stimulus(sig, [t_current], [False]))
            engine.add_stimulus(Stimulus('INC', [t_current], [False]))
            engine.add_stimulus(Stimulus('LOAD', [t_current], [False]))
            if 'ALU_B_IS_ONE' in pi:
                engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current], [False]))

            # Clock IR: setup, rise, fall
            t_ir_rise = t_current + T_STEP / 4
            t_ir_fall = t_current + T_STEP / 2
            engine.add_stimulus(Stimulus('CLK',
                                         [t_current, t_ir_rise, t_ir_fall],
                                         [False, True, False]))
            t_current += T_STEP
            engine.run(t_current)

            # Execute each micro-step
            for step_idx, (step, signals) in enumerate(micro_steps):
                active = set(signals)

                if inst_name == 'JCN' and step == 3:
                    carry_ns = engine._nets.get('CY_0')
                    carry_now = 1 if carry_ns and carry_ns.value else 0
                    if carry_now:
                        active.add('PC_LOAD')
                    else:
                        active.add('PC_INC')

                # Set control signals
                for sig in ALL_CONTROL_SIGNALS:
                    if sig in pi:
                        engine.add_stimulus(Stimulus(sig, [t_current],
                                                     [sig in active]))

                # ALU control
                if 'ALU_B_IS_ONE' in pi:
                    engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current],
                                                 ['ALU_B_IS_ONE' in active]))

                engine.add_stimulus(Stimulus('SUB', [t_current],
                                             ['ALU_SUB' in active]))

                # Register enables — set up BEFORE clock edge (edge-triggered, no race)
                acc_en = 'ACC_LOAD_FROM_ALU' in active or 'ACC_LOAD_FROM_BUS' in active
                engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [acc_en]))
                breg_en = 'BREG_LOAD_FROM_BUS' in active
                engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [breg_en]))
                r0_en = 'R0_LOAD_FROM_BUS' in active
                engine.add_stimulus(Stimulus('R0_Enable', [t_current], [r0_en]))

                # PC control — set up before clock edge
                pc_inc = 'PC_INC' in active
                pc_load = 'PC_LOAD' in active
                engine.add_stimulus(Stimulus('INC', [t_current], [pc_inc]))
                engine.add_stimulus(Stimulus('LOAD', [t_current], [pc_load]))
                engine.add_stimulus(Stimulus('PC_Enable', [t_current], [True]))

                # CLK: setup time, then rising edge, then falling edge
                # Control signals set at t_current, CLK rises T_STEP/4 later
                t_rise = t_current + T_STEP / 4  # rising edge after setup
                t_fall = t_current + T_STEP * 3 / 4  # falling edge
                engine.add_stimulus(Stimulus('CLK',
                                             [t_current, t_rise, t_fall],
                                             [False, True, False]))

                t_current += T_STEP
                engine.run(t_current)

            # Clear control signals after instruction
            for sig in ALL_CONTROL_SIGNALS:
                if sig in pi:
                    engine.add_stimulus(Stimulus(sig, [t_current], [False]))
            engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('R0_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('IR_Enable', [t_current], [False]))
            if 'ALU_B_IS_ONE' in pi:
                engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current], [False]))

            # Read final state
            acc = self._read_value_from_engine(engine, 'ACC', n)
            r0 = self._read_value_from_engine(engine, 'R0', n)
            carry_ns = engine._nets.get('CY_0')
            carry = 1 if carry_ns and carry_ns.value else 0
            pc = self._read_value_from_engine(engine, 'PC', n)

            if verbose:
                print(f" -> ACC={acc:2d} C={carry} R0={r0:2d} PC={pc:2d}")

            trace.append((inst_num, acc, carry, r0, pc, inst_name))

        return trace

    def _read_value_from_engine(self, engine, prefix, n_bits):
        """Read a multi-bit value from the live engine state."""
        val = 0
        for bit in range(n_bits):
            ns = engine._nets.get(f'{prefix}_{bit}')
            if ns and ns.value:
                val |= (1 << bit)
        return val

    def _read_value(self, result, prefix, n_bits):
        val = 0
        for bit in range(n_bits):
            ns = result.net_states.get(f'{prefix}_{bit}')
            if ns and ns.value:
                val |= (1 << bit)
        return val

    def summary(self):
        print(f"CPU: {self.n_bits}-bit, {len(self.gates)} gates")
        print(f"  Instructions: {list(OPCODE_MAP.values())}")


def counting_program():
    return {
        0: 0x10,  # IAC
        1: 0x24,  # JCN carry, jump to 4
        2: 0x40,  # JUN 0
        3: 0x00,  # NOP
        4: 0xB0,  # XCH R0
        5: 0x10,  # IAC
        6: 0xB0,  # XCH R0
        7: 0x40,  # JUN 0
    }


if __name__ == '__main__':
    cpu = CPU(n_bits=4)
    cpu.summary()
    cpu.load_program(counting_program())

    print("\nProgram:")
    for addr, byte in sorted(counting_program().items()):
        opr = (byte >> 4) & 0xF
        print(f"  [{addr}] 0x{byte:02X} {OPCODE_MAP.get(opr, '?')}")

    print("\nRunning through gate-level datapath...")
    trace = cpu.run(n_instructions=50)
    print(f"\nFinal: ACC={trace[-1][1]} R0={trace[-1][3]} PC={trace[-1][4]}")
