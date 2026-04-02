"""Declarative CPU Builder — specify blocks, get a wired CPU.

Usage:
    builder = CPUBuilder(data_width=4, addr_width=4)
    builder.add_scratchpad(n_registers=16)
    builder.add_alu(ops=['ADD'])
    builder.add_flags(['carry', 'zero'])

    builder.add_instruction('NOP', '0000', [])
    builder.add_instruction('IAC', '0001', [
        (3, ['ALU_ADD', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU']),
    ])
    builder.add_instruction('LD', '1010', [
        (3, ['SCRATCH_TO_BUS', 'ACC_LOAD_FROM_BUS']),
    ])
    # etc.

    cpu = builder.build()
    cpu.load_program({0: 0x10, ...})
    trace = cpu.run(50)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import math
from model import GateType
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams, precompute_gate
from simulator.engine import SimulationEngine, Stimulus
from model import NChannelJFET, JFETCapacitance
from register import make_register
from alu import ALUBuilder
from program_counter import make_program_counter
from instruction_decoder import InstructionSet
from scratchpad import Scratchpad


class CPUBuilder:
    """Declarative CPU builder."""

    def __init__(self, data_width=4, addr_width=4):
        self.data_width = data_width
        self.addr_width = addr_width  # PC / ROM address width
        self.scratchpad_size = 0
        self.alu_ops = []
        self.flags = ['carry']
        self.instructions = []  # (name, opcode, micro_steps)
        self.opcode_bits = 4
        self.modifier_bits = 4

    def add_scratchpad(self, n_registers=16):
        """Add addressable scratchpad memory."""
        self.scratchpad_size = n_registers
        return self

    def add_alu(self, ops=None):
        """Add ALU with specified operations."""
        self.alu_ops = ops or ['ADD']
        return self

    def add_flags(self, flags):
        """Add CPU flags."""
        self.flags = flags
        return self

    def add_instruction(self, name, opcode, micro_steps):
        """Add an instruction.

        Args:
            name: mnemonic (e.g. 'IAC')
            opcode: 4-bit binary string (e.g. '0001')
            micro_steps: list of (step_number, [control_signals])
                Step 3+ are execute phase. PC_INC auto-added to last
                step of non-jump instructions.
        """
        self.instructions.append((name, opcode, micro_steps))
        return self

    def build(self):
        """Generate the complete CPU. Returns a CPU instance."""
        return CPU(self)


class CPU:
    """A built CPU ready to execute programs."""

    def __init__(self, spec: CPUBuilder):
        self.spec = spec
        self.n = spec.data_width
        self.rom = {}
        self.gates = []
        self.opcode_map = {}

        self._build_jfet()
        self._build_all()
        self._build_profiles()

    def _build_jfet(self):
        self.jfet = NChannelJFET(
            beta=0.000135, vto=-3.45, lmbda=0.005, is_=205.2e-15, n=3.0,
            isr=1988e-15, nr=4.0, alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
            betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26).at_temp(27.0)
        self.caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
        self.params = CircuitParams(v_pos=10, v_neg=-10, r1=12100,
                                     r2=7320, r3=6980, jfet=self.jfet, caps=self.caps)

    def _build_all(self):
        n = self.n
        self.gates = []

        # --- Core registers (hardwired to ALU) ---
        for name in ['ACC', 'BREG']:
            g, _, _, _ = make_register(name, n)
            self.gates.extend(g)

        # --- Carry flag register ---
        cy_g, _, _, _ = make_register("CY", 1)
        self.gates.extend(cy_g)
        self.gates.append(Gate('CY_BUF_inv', GateType.INV, ['CARRY_4'], 'CY_BUF_n'))
        self.gates.append(Gate('CY_BUF', GateType.INV, ['CY_BUF_n'], 'CY_0_In'))
        self.gates.append(Gate('CY_EN_inv', GateType.INV,
                               ['ACC_LOAD_FROM_ALU'], 'CY_EN_n'))
        self.gates.append(Gate('CY_EN_buf', GateType.INV, ['CY_EN_n'], 'CY_Enable'))

        # --- ALU with B-input mux ---
        alu = ALUBuilder(n_bits=n, arithmetic=True, logic=[], shift=False,
                         flags=['zero', 'carry'])
        alu_gates = alu.build()

        # B-input mux (BREG or constant 1)
        self.gates.append(Gate('ALU_B1_INV', GateType.INV,
                               ['ALU_B_IS_ONE'], 'ALU_B_IS_ONE_INV'))
        for bit in range(n):
            breg_net = f'BREG_{bit}'
            mux_out = f'ALU_B_{bit}'
            prefix = f'ALU_BMUX_{bit}'
            self.gates.append(Gate(f'{prefix}_breg_nand', GateType.NAND2,
                                   [breg_net, 'ALU_B_IS_ONE_INV'], f'{prefix}_breg_n'))
            self.gates.append(Gate(f'{prefix}_breg_inv', GateType.INV,
                                   [f'{prefix}_breg_n'], f'{prefix}_breg'))
            if bit == 0:
                self._or_tree([f'{prefix}_breg', 'ALU_B_IS_ONE'],
                              mux_out, f'{prefix}_or')
            else:
                self.gates.append(Gate(f'{prefix}_buf_inv', GateType.INV,
                                       [f'{prefix}_breg'], f'{prefix}_n'))
                self.gates.append(Gate(f'{prefix}_buf', GateType.INV,
                                       [f'{prefix}_n'], mux_out))

        for g in alu_gates:
            g.inputs = [f'ACC_{inp[2:]}' if inp.startswith('A_') and inp[2:].isdigit()
                        else f'ALU_B_{inp[2:]}' if inp.startswith('B_') and inp[2:].isdigit()
                        else inp for inp in g.inputs]
        self.gates.extend(alu_gates)

        # --- Scratchpad ---
        if self.spec.scratchpad_size > 0:
            self.scratchpad = Scratchpad("SP", self.spec.scratchpad_size, n)
            self.gates.extend(self.scratchpad.gates)

        # --- Instruction decoder ---
        isa = InstructionSet(opcode_bits=self.spec.opcode_bits,
                             modifier_bits=self.spec.modifier_bits)
        for name, opcode, _ in self.spec.instructions:
            isa.add(name, opcode, signals=[])
            opr = int(opcode, 2)
            self.opcode_map[opr] = name
        ir_result = isa.build()
        ir_gates = ir_result[0]
        self.dec_nets = ir_result[4]
        self.gates.extend(ir_gates)

        # --- Program counter ---
        pc_g, _, _, _ = make_program_counter("PC", self.spec.addr_width)
        self.gates.extend(pc_g)

        # --- Data bus with gated writers ---
        writers = [('ACC', 'ACC_TO_BUS'), ('BREG', 'BREG_TO_BUS')]
        if self.spec.scratchpad_size > 0:
            # Scratchpad output already gated; connect to bus
            pass  # handled below

        for bit in range(n):
            outs = []
            for src, en in writers:
                p = f'BUS_W_{src}_{bit}'
                self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                       [f'{src}_{bit}', en], f'{p}_n'))
                self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                       [f'{p}_n'], f'{p}_out'))
                outs.append(f'{p}_out')

            # ALU result to bus
            p = f'BUS_W_ALU_{bit}'
            self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                   [f'OUT_{bit}', 'ALU_RESULT_TO_BUS'], f'{p}_n'))
            self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                   [f'{p}_n'], f'{p}_out'))
            outs.append(f'{p}_out')

            # IMM (IR modifier) to bus
            p = f'BUS_W_IMM_{bit}'
            self.gates.append(Gate(f'{p}_nand', GateType.NAND2,
                                   [f'IR_{bit}', 'IMM_TO_BUS'], f'{p}_n'))
            self.gates.append(Gate(f'{p}_inv', GateType.INV,
                                   [f'{p}_n'], f'{p}_out'))
            outs.append(f'{p}_out')

            # Scratchpad output to bus (already gated by SP_READ)
            if self.spec.scratchpad_size > 0:
                outs.append(f'SP_DOUT_{bit}')

            self._or_tree(outs, f'BUS_{bit}', f'BUS_OR_{bit}')

        # --- ACC input mux (ALU or bus) ---
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

        # --- BREG input from bus ---
        for bit in range(n):
            self.gates.append(Gate(f'BREG_BUF_{bit}_inv', GateType.INV,
                                   [f'BUS_{bit}'], f'BREG_{bit}_In_n'))
            self.gates.append(Gate(f'BREG_BUF_{bit}', GateType.INV,
                                   [f'BREG_{bit}_In_n'], f'BREG_{bit}_In'))

        # --- Scratchpad input from bus ---
        if self.spec.scratchpad_size > 0:
            for bit in range(n):
                self.gates.append(Gate(f'SP_DIN_BUF_{bit}_inv', GateType.INV,
                                       [f'BUS_{bit}'], f'SP_DIN_{bit}_n'))
                self.gates.append(Gate(f'SP_DIN_BUF_{bit}', GateType.INV,
                                       [f'SP_DIN_{bit}_n'], f'SP_DIN_{bit}'))

            # Scratchpad address from IR modifier (lower bits)
            sp_aw = self.scratchpad.addr_width
            for bit in range(sp_aw):
                self.gates.append(Gate(f'SP_ADDR_BUF_{bit}_inv', GateType.INV,
                                       [f'IR_{bit}'], f'SP_ADDR_{bit}_n'))
                self.gates.append(Gate(f'SP_ADDR_BUF_{bit}', GateType.INV,
                                       [f'SP_ADDR_{bit}_n'], f'SP_ADDR_{bit}'))

        # --- PC load from IR modifier ---
        for bit in range(self.spec.addr_width):
            if bit < self.spec.modifier_bits:
                self.gates.append(Gate(f'PC_LD_{bit}_inv', GateType.INV,
                                       [f'IR_{bit}'], f'PC_Load_{bit}_n'))
                self.gates.append(Gate(f'PC_LD_{bit}', GateType.INV,
                                       [f'PC_Load_{bit}_n'], f'PC_Load_{bit}'))

        # --- Register enables ---
        self._or_tree(['ACC_LOAD_FROM_ALU', 'ACC_LOAD_FROM_BUS'],
                      'ACC_Enable', 'ACC_EN')
        self.gates.append(Gate('BREG_EN_inv', GateType.INV,
                               ['BREG_LOAD_FROM_BUS'], 'BREG_Enable_n'))
        self.gates.append(Gate('BREG_EN_buf', GateType.INV,
                               ['BREG_Enable_n'], 'BREG_Enable'))

        print(f"CPU built: {len(self.gates)} gates")

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
            self.profiles[gt] = precompute_gate(gt, self.params, -0.8, -4.0)

    def load_program(self, program: dict):
        self.rom = program

    def run(self, n_instructions=20, verbose=True):
        """Execute program through gate-level datapath."""
        V_HIGH, V_LOW = -0.8, -4.0
        n = self.n
        T_STEP = 50e-6

        # Build netlist
        outputs = set()
        for prefix in ['ACC', 'BREG', 'PC']:
            for bit in range(n):
                outputs.add(f'{prefix}_{bit}')
        outputs.update(['CY_0', 'CARRY_4', 'FLAG_ZERO'])
        for bit in range(n):
            outputs.update([f'OUT_{bit}', f'BUS_{bit}'])
        if self.spec.scratchpad_size > 0:
            for bit in range(n):
                outputs.add(f'SP_DOUT_{bit}')
            for reg in range(self.spec.scratchpad_size):
                for bit in range(n):
                    outputs.add(f'SP_R{reg}_{bit}')

        netlist = Netlist.from_gates(self.gates, primary_outputs=outputs)
        engine = SimulationEngine(netlist, self.profiles,
                                  v_high=V_HIGH, v_low=V_LOW,
                                  auto_precompute_params=self.params)
        pi = netlist.primary_inputs

        # --- Power-on reset ---
        for sig in pi:
            engine._nets[sig].value = False
            engine._nets[sig].voltage = V_LOW

        # Force all register latches to 0
        for net_name, ns in engine._nets.items():
            if '_bar' in net_name or '_Qbar' in net_name or '_M_Qbar' in net_name:
                ns.value = True
                ns.voltage = V_HIGH
            elif any(net_name.startswith(f'{p}_') and net_name.split('_')[-1].isdigit()
                     for p in ['ACC', 'BREG', 'CY', 'PC']):
                ns.value = False
                ns.voltage = V_LOW
            # Scratchpad registers
            if self.spec.scratchpad_size > 0:
                for reg in range(self.spec.scratchpad_size):
                    if net_name.startswith(f'SP_R{reg}_'):
                        if '_bar' in net_name or '_Qbar' in net_name:
                            ns.value = True
                            ns.voltage = V_HIGH
                        elif net_name.split('_')[-1].isdigit():
                            ns.value = False
                            ns.voltage = V_LOW

        engine.force_evaluate_all()

        # Set all control signals LOW
        all_control = [
            'ALU_ADD', 'ALU_SUB', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU',
            'ACC_TO_BUS', 'BREG_TO_BUS', 'IMM_TO_BUS', 'ALU_RESULT_TO_BUS',
            'ACC_LOAD_FROM_BUS', 'BREG_LOAD_FROM_BUS',
            'SP_READ', 'SP_WRITE',
            'PC_INC', 'PC_LOAD', 'IR_LOAD',
        ]
        for sig in all_control:
            if sig in pi:
                engine.add_stimulus(Stimulus(sig, [0], [False]))
        if 'ALU_B_IS_ONE' in pi:
            engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [0], [False]))
        engine.add_stimulus(Stimulus('SUB', [0], [False]))
        engine.add_stimulus(Stimulus('INC', [0], [False]))
        engine.add_stimulus(Stimulus('LOAD', [0], [False]))
        engine.add_stimulus(Stimulus('CLK', [0], [False]))
        for en in ['ACC_Enable', 'BREG_Enable', 'IR_Enable', 'PC_Enable', 'CY_Enable']:
            if en in pi:
                engine.add_stimulus(Stimulus(en, [0], [False]))

        engine.run(1e-6)
        engine.force_evaluate_all()

        # --- Execute ---
        t_current = 5e-6
        trace = []

        for inst_num in range(n_instructions):
            pc = self._read_val(engine, 'PC', self.spec.addr_width)
            acc = self._read_val(engine, 'ACC', n)
            carry = 1 if engine._nets.get('CY_0') and engine._nets['CY_0'].value else 0

            opcode = self.rom.get(pc, 0x00)
            opr = (opcode >> 4) & 0xF
            inst_name = self.opcode_map.get(opr, 'NOP')

            # Find micro-steps for this instruction
            micro_steps = [(3, ['PC_INC'])]  # default NOP
            for iname, iopc, isteps in self.spec.instructions:
                if iname == inst_name and isteps:
                    micro_steps = isteps
                    break

            # Read scratchpad state for display
            sp_vals = {}
            if self.spec.scratchpad_size > 0:
                for reg in range(min(self.spec.scratchpad_size, 4)):
                    sp_vals[reg] = self._read_val(engine, f'SP_R{reg}', n)

            if verbose:
                sp_str = ' '.join(f'R{k}={v}' for k, v in sp_vals.items())
                print(f"  [{inst_num:3d}] PC={pc:2d} [{opcode:02X}] {inst_name:4s}  "
                      f"ACC={acc:2d} C={carry} {sp_str}", end="")

            # --- IR load phase ---
            for bit in range(8):
                engine.add_stimulus(Stimulus(f'IR_{bit}_In', [t_current],
                                             [bool((opcode >> bit) & 1)]))
            engine.add_stimulus(Stimulus('IR_Enable', [t_current], [True]))
            for sig in all_control:
                if sig in pi:
                    engine.add_stimulus(Stimulus(sig, [t_current], [False]))
            engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('INC', [t_current], [False]))
            engine.add_stimulus(Stimulus('LOAD', [t_current], [False]))
            if 'ALU_B_IS_ONE' in pi:
                engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current], [False]))

            t_rise = t_current + T_STEP / 4
            t_fall = t_current + T_STEP / 2
            engine.add_stimulus(Stimulus('CLK', [t_current, t_rise, t_fall],
                                         [False, True, False]))
            t_current += T_STEP
            engine.run(t_current)

            # --- Execute micro-steps ---
            for step, signals in micro_steps:
                active = set(signals)

                # Handle JCN
                if inst_name == 'JCN' and step == 3:
                    cy = engine._nets.get('CY_0')
                    if cy and cy.value:
                        active.add('PC_LOAD')
                    else:
                        active.add('PC_INC')

                # Map SCRATCH_TO_BUS / SCRATCH_LOAD to SP signals
                if 'SCRATCH_TO_BUS' in active:
                    active.discard('SCRATCH_TO_BUS')
                    active.add('SP_READ')
                if 'SCRATCH_LOAD' in active:
                    active.discard('SCRATCH_LOAD')
                    active.add('SP_WRITE')

                # Auto-add PC_INC to last step of non-jump instructions
                if step == micro_steps[-1][0]:
                    if 'PC_LOAD' not in active and 'PC_INC' not in signals:
                        active.add('PC_INC')

                # Set control signals
                for sig in all_control:
                    if sig in pi:
                        engine.add_stimulus(Stimulus(sig, [t_current],
                                                     [sig in active]))
                if 'ALU_B_IS_ONE' in pi:
                    engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current],
                                                 ['ALU_B_IS_ONE' in active]))
                engine.add_stimulus(Stimulus('SUB', [t_current],
                                             ['ALU_SUB' in active]))

                acc_en = 'ACC_LOAD_FROM_ALU' in active or 'ACC_LOAD_FROM_BUS' in active
                engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [acc_en]))
                breg_en = 'BREG_LOAD_FROM_BUS' in active
                engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [breg_en]))

                engine.add_stimulus(Stimulus('INC', [t_current],
                                             ['PC_INC' in active]))
                engine.add_stimulus(Stimulus('LOAD', [t_current],
                                             ['PC_LOAD' in active]))
                engine.add_stimulus(Stimulus('PC_Enable', [t_current], [True]))

                t_rise = t_current + T_STEP / 4
                t_fall = t_current + T_STEP / 2
                engine.add_stimulus(Stimulus('CLK', [t_current, t_rise, t_fall],
                                             [False, True, False]))

                t_current += T_STEP
                engine.run(t_current)

            # Clear controls
            for sig in all_control:
                if sig in pi:
                    engine.add_stimulus(Stimulus(sig, [t_current], [False]))
            engine.add_stimulus(Stimulus('ACC_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('BREG_Enable', [t_current], [False]))
            engine.add_stimulus(Stimulus('IR_Enable', [t_current], [False]))
            if 'ALU_B_IS_ONE' in pi:
                engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t_current], [False]))

            # Read final state
            acc = self._read_val(engine, 'ACC', n)
            carry = 1 if engine._nets.get('CY_0') and engine._nets['CY_0'].value else 0
            pc = self._read_val(engine, 'PC', self.spec.addr_width)

            if verbose:
                sp_str2 = ' '.join(f'R{k}={self._read_val(engine, f"SP_R{k}", n)}'
                                   for k in range(min(self.spec.scratchpad_size, 4)))
                print(f" -> ACC={acc:2d} C={carry} {sp_str2} PC={pc:2d}")

            trace.append((inst_num, acc, carry, pc, inst_name))

        return trace

    def _read_val(self, engine, prefix, n_bits):
        val = 0
        for bit in range(n_bits):
            ns = engine._nets.get(f'{prefix}_{bit}')
            if ns and ns.value:
                val |= (1 << bit)
        return val

    def summary(self):
        print(f"CPU: {self.n}-bit, {len(self.gates)} gates")
        print(f"  Scratchpad: {self.spec.scratchpad_size} registers")
        print(f"  Instructions: {[name for name, _, _ in self.spec.instructions]}")
        print(f"  Flags: {self.spec.flags}")


# ---------------------------------------------------------------------------
# Example: 4-bit CPU with 4 scratchpad registers
# ---------------------------------------------------------------------------

def build_test_cpu():
    b = CPUBuilder(data_width=4, addr_width=4)
    b.add_scratchpad(n_registers=4)
    b.add_alu(ops=['ADD'])
    b.add_flags(['carry', 'zero'])

    # Instructions
    b.add_instruction('NOP', '0000', [])
    b.add_instruction('IAC', '0001', [
        (3, ['ALU_ADD', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU']),
    ])
    b.add_instruction('JCN', '0010', [
        (3, []),  # PC_LOAD or PC_INC added by handler
    ])
    b.add_instruction('JUN', '0100', [
        (3, ['PC_LOAD']),
    ])
    b.add_instruction('ADD', '1000', [
        (3, ['SCRATCH_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ALU_ADD', 'ACC_LOAD_FROM_ALU']),
    ])
    b.add_instruction('SUB', '1001', [
        (3, ['SCRATCH_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ALU_SUB', 'ACC_LOAD_FROM_ALU']),
    ])
    b.add_instruction('LD',  '1010', [
        (3, ['SCRATCH_TO_BUS', 'ACC_LOAD_FROM_BUS']),
    ])
    b.add_instruction('XCH', '1011', [
        (3, ['SCRATCH_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ACC_TO_BUS', 'SCRATCH_LOAD']),
        (5, ['BREG_TO_BUS', 'ACC_LOAD_FROM_BUS']),
    ])
    b.add_instruction('LDM', '1101', [
        (3, ['IMM_TO_BUS', 'ACC_LOAD_FROM_BUS']),
    ])

    return b.build()


if __name__ == '__main__':
    cpu = build_test_cpu()
    cpu.summary()

    # Counting program: IAC, JCN overflow, JUN loop; overflow: XCH R0, IAC, XCH R0
    cpu.load_program({
        0: 0x10,  # IAC
        1: 0x24,  # JCN carry, jump to 4
        2: 0x40,  # JUN 0
        3: 0x00,  # NOP
        4: 0xB0,  # XCH R0
        5: 0x10,  # IAC
        6: 0xB0,  # XCH R0
        7: 0x40,  # JUN 0
    })

    print("\nRunning counting program...")
    trace = cpu.run(n_instructions=55)
    print(f"\nFinal: ACC={trace[-1][1]} C={trace[-1][2]} PC={trace[-1][3]}")
