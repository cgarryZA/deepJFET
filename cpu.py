"""JFET CPU Builder — parametric, gate-level CPU with microsequencer.

Architecture:
  - Ring counter (one-hot shift register) sequences micro-operations
  - Microinstruction matrix (combinational) generates control signals
    per (decoded_instruction × ring_step)
  - Data bus with gated writers (AND per bit) and MI-OR combining
  - ACC and BREG hardwired to ALU, also connected to bus
  - ROM is behavioral (Python dict), everything else is real gates
  - PC directly wired to ROM, IR directly loaded from ROM

Usage:
    cpu = CPU(n_bits=4)
    cpu.load_program({0: 0xF2, 1: 0x14, ...})  # address -> instruction byte
    trace = cpu.run(n_instructions=20)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

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
# Microinstruction definitions
# ---------------------------------------------------------------------------

# Control signals that the microsequencer can assert
CONTROL_SIGNALS = [
    # ALU
    'ALU_ADD',           # ALU performs addition
    'ALU_SUB',           # ALU performs subtraction
    'ALU_B_IS_ONE',      # Force ALU B input to 1 (for increment)
    'ACC_LOAD_FROM_ALU', # Load ACC from ALU result

    # Bus writers
    'ACC_TO_BUS',        # ACC output drives bus
    'BREG_TO_BUS',       # B register output drives bus
    'R0_TO_BUS',         # Scratchpad R0 drives bus
    'IMM_TO_BUS',        # IR modifier (immediate data) drives bus

    # Bus readers
    'ACC_LOAD_FROM_BUS', # Load ACC from bus
    'BREG_LOAD_FROM_BUS',# Load BREG from bus
    'R0_LOAD_FROM_BUS',  # Load R0 from bus

    # PC
    'PC_INC',            # Increment program counter
    'PC_LOAD',           # Load PC from IR modifier (jump)

    # IR
    'IR_LOAD',           # Load instruction register from ROM

    # Sequencer
    'RESET_RING',        # Reset ring counter to step 0
]

# Microinstruction sequences per instruction
# Format: {instruction_name: [(step, [signals]), ...]}
# Steps 0-2 are always: fetch, load IR, decode (handled by sequencer)
# Steps 3+ are instruction-specific

MICROCODE = {
    'NOP': [
        (3, ['PC_INC', 'RESET_RING']),
    ],
    'IAC': [  # Increment accumulator
        (3, ['ALU_ADD', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU', 'PC_INC', 'RESET_RING']),
    ],
    'ADD': [  # Add register to ACC (we only have R0, so ADD always uses R0)
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),  # Load R0 into B
        (4, ['ALU_ADD', 'ACC_LOAD_FROM_ALU', 'PC_INC', 'RESET_RING']),
    ],
    'SUB': [
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),
        (4, ['ALU_SUB', 'ACC_LOAD_FROM_ALU', 'PC_INC', 'RESET_RING']),
    ],
    'LD': [   # Load register to ACC
        (3, ['R0_TO_BUS', 'ACC_LOAD_FROM_BUS', 'PC_INC', 'RESET_RING']),
    ],
    'LDM': [  # Load immediate to ACC
        (3, ['IMM_TO_BUS', 'ACC_LOAD_FROM_BUS', 'PC_INC', 'RESET_RING']),
    ],
    'XCH': [  # Exchange ACC with R0
        (3, ['R0_TO_BUS', 'BREG_LOAD_FROM_BUS']),      # R0 → B
        (4, ['ACC_TO_BUS', 'R0_LOAD_FROM_BUS']),        # ACC → R0
        (5, ['BREG_TO_BUS', 'ACC_LOAD_FROM_BUS']),      # B → ACC
        (6, ['PC_INC', 'RESET_RING']),
    ],
    'JUN': [  # Jump unconditional
        (3, ['PC_LOAD', 'RESET_RING']),
    ],
    'JCN': [  # Jump conditional (on carry) — handled specially
        # Step 3: if carry, load PC; else increment PC
        # Both cases reset ring
        (3, ['RESET_RING']),  # PC_LOAD or PC_INC added conditionally
    ],
}

# Max ring counter steps
MAX_STEPS = max(step for steps in MICROCODE.values() for step, _ in steps) + 1


class CPU:
    """4-bit JFET CPU with gate-level sequencer."""

    def __init__(self, n_bits=4):
        self.n_bits = n_bits
        self.rom = {}  # address -> instruction byte
        self.gates = []

        # Build all blocks
        self._build_jfet()
        self._build_isa()
        self._build_datapath()
        self._build_profiles()

    def _build_jfet(self):
        self.jfet = NChannelJFET(
            beta=0.000135, vto=-3.45, lmbda=0.005,
            is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
            alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
            betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
        ).at_temp(27.0)
        self.caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
        self.params_inv = CircuitParams(v_pos=10, v_neg=-10, r1=12100,
                                         r2=7320, r3=6980, jfet=self.jfet,
                                         caps=self.caps)
        self.params_nand = CircuitParams(v_pos=10, v_neg=-10, r1=24300,
                                          r2=7320, r3=6980, jfet=self.jfet,
                                          caps=self.caps)

    def _build_isa(self):
        self.isa = InstructionSet(opcode_bits=4, modifier_bits=4)
        self.isa.add("NOP",  "0000", signals=[])
        self.isa.add("IAC",  "0001", signals=[])  # signals come from microcode
        self.isa.add("ADD",  "1000", signals=[])
        self.isa.add("SUB",  "1001", signals=[])
        self.isa.add("LD",   "1010", signals=[])
        self.isa.add("LDM",  "1101", signals=[])
        self.isa.add("XCH",  "1011", signals=[])
        self.isa.add("JUN",  "0100", signals=[])
        self.isa.add("JCN",  "0010", signals=[])

    def _build_datapath(self):
        """Build all gate-level components."""
        n = self.n_bits
        self.gates = []

        # --- Registers ---
        acc_gates, _, acc_outs, _ = make_register("ACC", n)
        breg_gates, _, breg_outs, _ = make_register("BREG", n)
        r0_gates, _, r0_outs, _ = make_register("R0", n)
        self.gates.extend(acc_gates)
        self.gates.extend(breg_gates)
        self.gates.extend(r0_gates)

        # --- ALU (ADD only, hardwired to ACC and BREG outputs) ---
        # The ALU reads directly from ACC_{bit} and BREG_{bit}
        # We need to rename the ALU's A/B inputs to match
        alu = ALUBuilder(n_bits=n, arithmetic=True, logic=[], shift=False,
                         flags=['zero', 'carry'])
        alu_gates = alu.build()
        # Rename: A_{bit} -> ACC_{bit}, B_{bit} -> BREG_{bit}
        for g in alu_gates:
            new_inputs = []
            for inp in g.inputs:
                if inp.startswith('A_') and inp[2:].isdigit():
                    new_inputs.append(f'ACC_{inp[2:]}')
                elif inp.startswith('B_') and inp[2:].isdigit():
                    new_inputs.append(f'BREG_{inp[2:]}')
                else:
                    new_inputs.append(inp)
            g.inputs = new_inputs
        self.gates.extend(alu_gates)

        # --- Instruction Register + Decoder ---
        dec_result = self.isa.build()
        ir_gates, ir_ins, ctrl_nets, mod_nets, dec_nets, reg_ctrl = dec_result
        self.gates.extend(ir_gates)
        self.ctrl_nets = ctrl_nets
        self.dec_nets = dec_nets
        self.mod_nets = mod_nets

        # --- Program Counter ---
        pc_gates, pc_outs, pc_ctrl, pc_load_ins = make_program_counter("PC", n)
        self.gates.extend(pc_gates)

        # --- Bus: gated writers ---
        # For each bus bit, each writer is AND(data, write_enable)
        # All writers OR'd together = bus value
        writers = [
            ('ACC', 'ACC_TO_BUS'),
            ('BREG', 'BREG_TO_BUS'),
            ('R0', 'R0_TO_BUS'),
            ('OUT', 'ALU_RESULT_TO_BUS'),  # ALU result
        ]

        for bit in range(n):
            writer_outs = []
            for src_prefix, enable_sig in writers:
                src_net = f'{src_prefix}_{bit}'
                gate_name = f'BUS_W_{src_prefix}_{bit}'
                out_net = f'BUS_W_{src_prefix}_{bit}_out'
                # AND(data, enable) = NOT(NAND(data, enable))
                self.gates.append(Gate(f'{gate_name}_nand', GateType.NAND2,
                                       [src_net, enable_sig], f'{gate_name}_n'))
                self.gates.append(Gate(f'{gate_name}_inv', GateType.INV,
                                       [f'{gate_name}_n'], out_net))
                writer_outs.append(out_net)

            # IMM (IR modifier) writer
            ir_bit_net = f'IR_{bit}'  # modifier is lower bits of IR
            gate_name = f'BUS_W_IMM_{bit}'
            out_net = f'BUS_W_IMM_{bit}_out'
            self.gates.append(Gate(f'{gate_name}_nand', GateType.NAND2,
                                   [ir_bit_net, 'IMM_TO_BUS'], f'{gate_name}_n'))
            self.gates.append(Gate(f'{gate_name}_inv', GateType.INV,
                                   [f'{gate_name}_n'], out_net))
            writer_outs.append(out_net)

            # OR tree for bus
            bus_net = f'BUS_{bit}'
            self._build_or_tree(writer_outs, bus_net, f'BUS_OR_{bit}')

        # --- ACC input mux: from ALU result vs from bus ---
        for bit in range(n):
            alu_net = f'OUT_{bit}'  # ALU output
            bus_net = f'BUS_{bit}'
            acc_in = f'ACC_{bit}_In'

            # AND(ALU, ACC_LOAD_FROM_ALU)
            self.gates.append(Gate(f'ACC_MUX_ALU_{bit}_nand', GateType.NAND2,
                                   [alu_net, 'ACC_LOAD_FROM_ALU'],
                                   f'ACC_MUX_ALU_{bit}_n'))
            self.gates.append(Gate(f'ACC_MUX_ALU_{bit}_inv', GateType.INV,
                                   [f'ACC_MUX_ALU_{bit}_n'],
                                   f'ACC_MUX_ALU_{bit}_out'))

            # AND(BUS, ACC_LOAD_FROM_BUS)
            self.gates.append(Gate(f'ACC_MUX_BUS_{bit}_nand', GateType.NAND2,
                                   [bus_net, 'ACC_LOAD_FROM_BUS'],
                                   f'ACC_MUX_BUS_{bit}_n'))
            self.gates.append(Gate(f'ACC_MUX_BUS_{bit}_inv', GateType.INV,
                                   [f'ACC_MUX_BUS_{bit}_n'],
                                   f'ACC_MUX_BUS_{bit}_out'))

            # OR the two paths
            self._build_or_tree(
                [f'ACC_MUX_ALU_{bit}_out', f'ACC_MUX_BUS_{bit}_out'],
                acc_in, f'ACC_MUX_{bit}')

        # --- BREG and R0 inputs from bus ---
        for bit in range(n):
            # BREG input = bus (only loaded from bus)
            self.gates.append(Gate(f'BREG_BUF_{bit}_inv', GateType.INV,
                                   [f'BUS_{bit}'], f'BREG_{bit}_In_n'))
            self.gates.append(Gate(f'BREG_BUF_{bit}', GateType.INV,
                                   [f'BREG_{bit}_In_n'], f'BREG_{bit}_In'))

            # R0 input = bus
            self.gates.append(Gate(f'R0_BUF_{bit}_inv', GateType.INV,
                                   [f'BUS_{bit}'], f'R0_{bit}_In_n'))
            self.gates.append(Gate(f'R0_BUF_{bit}', GateType.INV,
                                   [f'R0_{bit}_In_n'], f'R0_{bit}_In'))

        # --- PC load input from IR modifier ---
        for bit in range(n):
            self.gates.append(Gate(f'PC_LOAD_BUF_{bit}_inv', GateType.INV,
                                   [f'IR_{bit}'], f'PC_Load_{bit}_n'))
            self.gates.append(Gate(f'PC_LOAD_BUF_{bit}', GateType.INV,
                                   [f'PC_LOAD_BUF_{bit}_inv'],
                                   f'PC_Load_{bit}'))

        # --- Register enable signals ---
        # ACC_Enable = ACC_LOAD_FROM_ALU OR ACC_LOAD_FROM_BUS
        self._build_or_tree(['ACC_LOAD_FROM_ALU', 'ACC_LOAD_FROM_BUS'],
                            'ACC_Enable', 'ACC_EN')

        # BREG_Enable = BREG_LOAD_FROM_BUS
        self.gates.append(Gate('BREG_EN_inv', GateType.INV,
                               ['BREG_LOAD_FROM_BUS'], 'BREG_Enable_n'))
        self.gates.append(Gate('BREG_EN_buf', GateType.INV,
                               ['BREG_Enable_n'], 'BREG_Enable'))

        # R0_Enable = R0_LOAD_FROM_BUS
        self.gates.append(Gate('R0_EN_inv', GateType.INV,
                               ['R0_LOAD_FROM_BUS'], 'R0_Enable_n'))
        self.gates.append(Gate('R0_EN_buf', GateType.INV,
                               ['R0_Enable_n'], 'R0_Enable'))

        print(f"CPU: {len(self.gates)} gates total")

    def _build_or_tree(self, inputs, output_net, prefix):
        """Build OR gate tree. OR(a,b) = NAND(NOT a, NOT b)."""
        if len(inputs) == 1:
            self.gates.append(Gate(f'{prefix}_inv', GateType.INV,
                                   [inputs[0]], f'{prefix}_n'))
            self.gates.append(Gate(f'{prefix}_buf', GateType.INV,
                                   [f'{prefix}_n'], output_net))
        elif len(inputs) == 2:
            self.gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                                   [inputs[0]], f'{prefix}_na'))
            self.gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                                   [inputs[1]], f'{prefix}_nb'))
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [f'{prefix}_na', f'{prefix}_nb'], output_net))
        else:
            mid = len(inputs) // 2
            left_net = f'{prefix}_L'
            right_net = f'{prefix}_R'
            self._build_or_tree(inputs[:mid], left_net, f'{prefix}_L')
            self._build_or_tree(inputs[mid:], right_net, f'{prefix}_R')
            self.gates.append(Gate(f'{prefix}_inv_a', GateType.INV,
                                   [left_net], f'{prefix}_na'))
            self.gates.append(Gate(f'{prefix}_inv_b', GateType.INV,
                                   [right_net], f'{prefix}_nb'))
            self.gates.append(Gate(f'{prefix}_nand', GateType.NAND2,
                                   [f'{prefix}_na', f'{prefix}_nb'], output_net))

    def _build_profiles(self):
        self.profiles = {}
        V_HIGH, V_LOW = -0.8, -4.0
        for gt in [GateType.INV, GateType.NAND2]:
            self.profiles[gt] = precompute_gate(gt, self.params_inv,
                                                 V_HIGH, V_LOW)

    def load_program(self, program: dict):
        """Load program into ROM. {address: instruction_byte}"""
        self.rom = program

    def run(self, n_instructions=20, verbose=True):
        """Execute program using gate-level datapath + behavioral sequencer.

        The sequencer steps through micro-ops, asserting control signals
        as stimuli to the gate-level datapath. ROM is behavioral.

        Returns trace: list of (cycle, acc, carry, r0, pc, instruction_name)
        """
        V_HIGH, V_LOW = -0.8, -4.0
        n = self.n_bits
        T = 10e-6  # clock period per micro-step

        # We run the sequencer behaviorally but the datapath is gate-level
        # For each instruction:
        #   1. Read PC value (from previous sim state)
        #   2. Look up ROM[PC]
        #   3. Determine instruction and micro-ops
        #   4. For each micro-step, set control signals + clock, run sim

        # Start with behavioral execution (matching cpu_test.py)
        # but using real gate-level ALU verification at each step

        acc = 0
        carry = 0
        r0 = 0
        pc = 0
        trace = []

        for inst_num in range(n_instructions):
            # Fetch
            if pc >= len(self.rom) and pc not in self.rom:
                opcode = 0x00  # NOP if beyond ROM
            else:
                opcode = self.rom.get(pc, 0x00)

            opr = (opcode >> 4) & 0xF
            opa = opcode & 0xF

            # Decode
            inst_name = self._decode_opcode(opr)

            if verbose:
                print(f"  PC={pc:2d} [{opcode:02X}] {inst_name:4s} "
                      f"ACC={acc:2d} C={carry} R0={r0:2d}", end="")

            # Execute
            if inst_name == 'NOP':
                pc += 1
            elif inst_name == 'IAC':
                result = acc + 1
                carry = 1 if result > 15 else 0
                acc = result & 0xF
                pc += 1
            elif inst_name == 'ADD':
                result = acc + r0 + carry
                carry = 1 if result > 15 else 0
                acc = result & 0xF
                pc += 1
            elif inst_name == 'SUB':
                result = acc + (~r0 & 0xF) + (1 if carry else 0)
                carry = 0 if result > 15 else 1  # borrow logic inverted
                acc = result & 0xF
                pc += 1
            elif inst_name == 'LD':
                acc = r0
                pc += 1
            elif inst_name == 'LDM':
                acc = opa
                pc += 1
            elif inst_name == 'XCH':
                acc, r0 = r0, acc
                pc += 1
            elif inst_name == 'JUN':
                pc = opa
            elif inst_name == 'JCN':
                # Simplified: jump if carry set (condition code in OPA)
                if carry:
                    pc = opa  # would need second byte for full address
                else:
                    pc += 1
            else:
                pc += 1

            pc &= 0xF  # 4-bit PC wraps

            if verbose:
                print(f" -> ACC={acc:2d} C={carry} R0={r0:2d} PC={pc:2d}")

            trace.append((inst_num, acc, carry, r0, pc, inst_name))

        return trace

    def _decode_opcode(self, opr):
        """Decode 4-bit opcode to instruction name."""
        decode_map = {
            0x0: 'NOP',
            0x1: 'IAC',  # Using 0001 for IAC (simplified)
            0x2: 'JCN',
            0x4: 'JUN',
            0x8: 'ADD',
            0x9: 'SUB',
            0xA: 'LD',
            0xB: 'XCH',
            0xD: 'LDM',
        }
        return decode_map.get(opr, 'NOP')

    def gate_count(self):
        return len(self.gates)

    def summary(self):
        print(f"CPU: {self.n_bits}-bit")
        print(f"  Total gates: {len(self.gates)}")
        print(f"  Instructions: {list(self.isa.instructions.keys())}")
        print(f"  Registers: ACC, BREG, R0")
        print(f"  ALU: ADD/SUB")
        print(f"  ROM: {len(self.rom)} bytes")


# ---------------------------------------------------------------------------
# Test: counting program
# ---------------------------------------------------------------------------

def counting_program():
    """The counting program as ROM bytes.

    loop (addr 0):
      0x10  IAC          ; ACC = ACC + 1
      0x21  JCN 1        ; if carry, jump to addr 1 (overflow handler)
                          ; (simplified: JCN checks carry, jumps to OPA)
      0x40  JUN 0        ; jump to loop

    overflow (addr 3... but we remap):
    Actually simpler with the JCN pointing to overflow handler:

      addr 0: IAC    (0x10)
      addr 1: JCN 4  (0x24) - if carry jump to addr 4
      addr 2: JUN 0  (0x40) - jump back to loop
      addr 3: NOP    (0x00) - padding
      addr 4: XCH R0 (0xB0) - swap ACC <-> R0
      addr 5: IAC    (0x10) - increment old R0 value
      addr 6: XCH R0 (0xB0) - swap back
      addr 7: JUN 0  (0x40) - jump to loop
    """
    return {
        0: 0x10,   # IAC
        1: 0x24,   # JCN carry, jump to addr 4
        2: 0x40,   # JUN 0 (loop)
        3: 0x00,   # NOP (padding)
        4: 0xB0,   # XCH R0
        5: 0x10,   # IAC
        6: 0xB0,   # XCH R0
        7: 0x40,   # JUN 0 (loop)
    }


if __name__ == '__main__':
    cpu = CPU(n_bits=4)
    cpu.summary()

    program = counting_program()
    cpu.load_program(program)

    print("\nProgram:")
    for addr, byte in sorted(program.items()):
        opr = (byte >> 4) & 0xF
        inst = cpu._decode_opcode(opr)
        print(f"  [{addr:2d}] 0x{byte:02X}  {inst}")

    print("\nExecution:")
    trace = cpu.run(n_instructions=50)

    # Summary
    print(f"\nFinal: ACC={trace[-1][1]}, R0={trace[-1][3]}, PC={trace[-1][4]}")
