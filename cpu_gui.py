"""RTL-level CPU debugger GUI.

Shows program listing, register values, and current instruction.
Step through instructions one at a time or run continuously.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import tkinter as tk
from tkinter import font as tkfont
from cpu_builder import CPUBuilder, CPU, build_test_cpu


OPCODE_MAP = {
    0x0: 'NOP', 0x1: 'IAC', 0x2: 'JCN', 0x4: 'JUN',
    0x8: 'ADD', 0x9: 'SUB', 0xA: 'LD', 0xB: 'XCH', 0xD: 'LDM',
}


class CPUGUI(tk.Tk):
    def __init__(self, cpu: CPU, program: dict):
        super().__init__()
        self.title("deepJFET CPU — RTL Debugger")
        self.geometry("900x700")
        self.configure(bg='#1a1a2e')

        self.cpu = cpu
        self.program = program
        self.cpu.load_program(program)

        # CPU state (tracked from gate sim)
        self.cycle = 0
        self.acc = 0
        self.breg = 0
        self.carry = 0
        self.zero = 0
        self.pc = 0
        self.ir = 0
        self.inst_name = "---"
        self.sp_vals = {}
        self.history = []  # list of state snapshots
        self.running = False

        # We need the engine running — do a light init
        self._init_engine()

        self._build_ui()
        self._update_display()

    def _init_engine(self):
        """Initialize the CPU engine (mirrors cpu.run setup)."""
        from simulator.netlist import Netlist
        from simulator.engine import SimulationEngine, Stimulus

        V_HIGH, V_LOW = -0.8, -4.0
        n = self.cpu.n

        outputs = set()
        for prefix in ['ACC', 'BREG', 'PC']:
            for bit in range(n):
                outputs.add(f'{prefix}_{bit}')
        outputs.update(['CY_0', 'CARRY_4', 'FLAG_ZERO'])
        for bit in range(n):
            outputs.update([f'OUT_{bit}', f'BUS_{bit}'])
        if self.cpu.spec.scratchpad_size > 0:
            for reg in range(self.cpu.spec.scratchpad_size):
                for bit in range(n):
                    outputs.add(f'SP_R{reg}_{bit}')

        netlist = Netlist.from_gates(self.cpu.gates, primary_outputs=outputs)
        self.engine = SimulationEngine(netlist, self.cpu.profiles,
                                        v_high=V_HIGH, v_low=V_LOW,
                                        auto_precompute_params=self.cpu.params)
        self.pi = netlist.primary_inputs

        # Power-on reset
        for sig in self.pi:
            self.engine._nets[sig].value = False
            self.engine._nets[sig].voltage = V_LOW

        for net_name, ns in self.engine._nets.items():
            if '_bar' in net_name or '_Qbar' in net_name or '_M_Qbar' in net_name:
                ns.value = True
                ns.voltage = V_HIGH

        self.engine.force_evaluate_all()

        all_ctrl = ['ALU_ADD', 'ALU_SUB', 'ALU_B_IS_ONE', 'ACC_LOAD_FROM_ALU',
                    'ACC_TO_BUS', 'BREG_TO_BUS', 'IMM_TO_BUS', 'ALU_RESULT_TO_BUS',
                    'ACC_LOAD_FROM_BUS', 'BREG_LOAD_FROM_BUS',
                    'SP_READ', 'SP_WRITE', 'PC_INC', 'PC_LOAD', 'IR_LOAD']
        for sig in all_ctrl + ['SUB', 'INC', 'LOAD', 'CLK',
                                'ACC_Enable', 'BREG_Enable', 'IR_Enable',
                                'PC_Enable', 'CY_Enable']:
            if sig in self.pi:
                self.engine.add_stimulus(Stimulus(sig, [0], [False]))
        if 'ALU_B_IS_ONE' in self.pi:
            self.engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [0], [False]))

        self.engine.run(1e-6)
        self.engine.force_evaluate_all()
        self.t_current = 5e-6
        self.T_STEP = 50e-6
        self.all_ctrl = all_ctrl

    def _build_ui(self):
        mono = tkfont.Font(family='Consolas', size=11)
        mono_big = tkfont.Font(family='Consolas', size=14, weight='bold')
        mono_small = tkfont.Font(family='Consolas', size=10)

        # Top: controls
        ctrl_frame = tk.Frame(self, bg='#16213e', height=50)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        self.step_btn = tk.Button(ctrl_frame, text="Step (Space)", command=self._step,
                                   bg='#0f3460', fg='white', font=mono,
                                   activebackground='#1a5276')
        self.step_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.run_btn = tk.Button(ctrl_frame, text="Run (R)", command=self._toggle_run,
                                  bg='#0f3460', fg='white', font=mono,
                                  activebackground='#1a5276')
        self.run_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.reset_btn = tk.Button(ctrl_frame, text="Reset", command=self._reset,
                                    bg='#533483', fg='white', font=mono,
                                    activebackground='#6c4ea0')
        self.reset_btn.pack(side=tk.LEFT, padx=5, pady=5)

        self.cycle_label = tk.Label(ctrl_frame, text="Cycle: 0", bg='#16213e',
                                     fg='#e0e0e0', font=mono_big)
        self.cycle_label.pack(side=tk.RIGHT, padx=20)

        # Main area: left = program, right = registers
        main = tk.Frame(self, bg='#1a1a2e')
        main.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Left: program listing
        prog_frame = tk.LabelFrame(main, text=" Program ", bg='#1a1a2e',
                                    fg='#e94560', font=mono, labelanchor='n')
        prog_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.prog_text = tk.Text(prog_frame, bg='#0f0f23', fg='#cccccc',
                                  font=mono, width=30, insertbackground='white',
                                  selectbackground='#e94560', state='disabled',
                                  relief='flat', padx=10, pady=5)
        self.prog_text.pack(fill=tk.BOTH, expand=True)

        # Configure tags for highlighting
        self.prog_text.tag_configure('current', background='#e94560',
                                      foreground='white')
        self.prog_text.tag_configure('normal', foreground='#cccccc')
        self.prog_text.tag_configure('addr', foreground='#666666')

        # Right: registers
        reg_frame = tk.Frame(main, bg='#1a1a2e')
        reg_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5)

        # Current instruction
        inst_frame = tk.LabelFrame(reg_frame, text=" Instruction ",
                                    bg='#1a1a2e', fg='#e94560', font=mono)
        inst_frame.pack(fill=tk.X, pady=5)

        self.inst_label = tk.Label(inst_frame, text="NOP", bg='#0f0f23',
                                    fg='#FFD700', font=mono_big, anchor='w',
                                    padx=10, pady=8)
        self.inst_label.pack(fill=tk.X)

        # Core registers
        core_frame = tk.LabelFrame(reg_frame, text=" Registers ",
                                    bg='#1a1a2e', fg='#4CAF50', font=mono)
        core_frame.pack(fill=tk.X, pady=5)

        self.reg_labels = {}
        for name, color in [('PC', '#9E9E9E'), ('IR', '#FF9800'),
                             ('ACC', '#4CAF50'), ('BREG', '#2196F3'),
                             ('Carry', '#E91E63'), ('Zero', '#9C27B0')]:
            row = tk.Frame(core_frame, bg='#0f0f23')
            row.pack(fill=tk.X, padx=2, pady=1)
            tk.Label(row, text=f'{name:>6}:', bg='#0f0f23', fg=color,
                     font=mono, width=7, anchor='e').pack(side=tk.LEFT)
            lbl = tk.Label(row, text='0', bg='#0f0f23', fg='white',
                           font=mono_big, anchor='w', padx=10)
            lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.reg_labels[name] = lbl

        # Scratchpad
        if self.cpu.spec.scratchpad_size > 0:
            sp_frame = tk.LabelFrame(reg_frame, text=" Scratchpad ",
                                      bg='#1a1a2e', fg='#FF9800', font=mono)
            sp_frame.pack(fill=tk.X, pady=5)

            self.sp_labels = {}
            n_sp = self.cpu.spec.scratchpad_size
            for reg in range(n_sp):
                row = tk.Frame(sp_frame, bg='#0f0f23')
                row.pack(fill=tk.X, padx=2, pady=1)
                tk.Label(row, text=f'R{reg}:', bg='#0f0f23', fg='#FF9800',
                         font=mono_small, width=5, anchor='e').pack(side=tk.LEFT)
                lbl = tk.Label(row, text='0', bg='#0f0f23', fg='white',
                               font=mono, anchor='w', padx=10)
                lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)
                self.sp_labels[reg] = lbl

        # History
        hist_frame = tk.LabelFrame(reg_frame, text=" History ",
                                    bg='#1a1a2e', fg='#666666', font=mono)
        hist_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.hist_text = tk.Text(hist_frame, bg='#0f0f23', fg='#888888',
                                  font=mono_small, width=40, state='disabled',
                                  relief='flat', padx=5, pady=3)
        self.hist_text.pack(fill=tk.BOTH, expand=True)

        # Key bindings
        self.bind('<space>', lambda e: self._step())
        self.bind('<r>', lambda e: self._toggle_run())
        self.bind('<Escape>', lambda e: self._stop_run())

        # Populate program listing
        self._populate_program()

    def _populate_program(self):
        self.prog_text.config(state='normal')
        self.prog_text.delete('1.0', tk.END)
        max_addr = max(self.program.keys()) if self.program else 0
        for addr in range(max_addr + 1):
            byte = self.program.get(addr, 0x00)
            opr = (byte >> 4) & 0xF
            opa = byte & 0xF
            name = OPCODE_MAP.get(opr, '???')
            line = f" {addr:2d}  {byte:02X}  {name:4s} {opa:X}\n"
            self.prog_text.insert(tk.END, line)
        self.prog_text.config(state='disabled')

    def _update_display(self):
        n = self.cpu.n

        # Update register display
        self.reg_labels['PC'].config(text=f'{self.pc}  (0b{self.pc:04b})')
        self.reg_labels['IR'].config(text=f'0x{self.ir:02X}  {self.inst_name}')
        self.reg_labels['ACC'].config(text=f'{self.acc}  (0b{self.acc:04b})')
        self.reg_labels['BREG'].config(text=f'{self.breg}  (0b{self.breg:04b})')
        self.reg_labels['Carry'].config(text=str(self.carry))
        self.reg_labels['Zero'].config(text=str(self.zero))

        # Scratchpad
        if hasattr(self, 'sp_labels'):
            for reg, lbl in self.sp_labels.items():
                val = self.sp_vals.get(reg, 0)
                lbl.config(text=f'{val}  (0b{val:04b})')

        # Cycle counter
        self.cycle_label.config(text=f"Cycle: {self.cycle}")

        # Instruction display
        opa = self.ir & 0xF
        self.inst_label.config(text=f'{self.inst_name}  (OPA={opa:X})')

        # Highlight current PC in program listing
        self.prog_text.tag_remove('current', '1.0', tk.END)
        line_num = self.pc + 1
        self.prog_text.tag_add('current', f'{line_num}.0', f'{line_num}.end')
        self.prog_text.see(f'{line_num}.0')

    def _read_state(self):
        """Read CPU state from the live engine."""
        n = self.cpu.n
        self.pc = self.cpu._read_val(self.engine, 'PC', self.cpu.spec.addr_width)
        self.acc = self.cpu._read_val(self.engine, 'ACC', n)
        self.breg = self.cpu._read_val(self.engine, 'BREG', n)
        cy = self.engine._nets.get('CY_0')
        self.carry = 1 if cy and cy.value else 0
        zf = self.engine._nets.get('FLAG_ZERO')
        self.zero = 1 if zf and zf.value else 0

        if self.cpu.spec.scratchpad_size > 0:
            for reg in range(self.cpu.spec.scratchpad_size):
                self.sp_vals[reg] = self.cpu._read_val(self.engine, f'SP_R{reg}', n)

    def _execute_one_instruction(self):
        """Execute one instruction through the gate-level datapath."""
        from simulator.engine import Stimulus

        n = self.cpu.n
        T = self.T_STEP
        t = self.t_current

        self._read_state()
        opcode = self.program.get(self.pc, 0x00)
        opr = (opcode >> 4) & 0xF
        self.ir = opcode
        self.inst_name = OPCODE_MAP.get(opr, 'NOP')

        # Find micro-steps
        micro_steps = [(3, ['PC_INC'])]
        for iname, iopc, isteps in self.cpu.spec.instructions:
            if iname == self.inst_name and isteps:
                micro_steps = isteps
                break

        # IR load phase
        for bit in range(8):
            self.engine.add_stimulus(Stimulus(f'IR_{bit}_In', [t],
                                              [bool((opcode >> bit) & 1)]))
        self.engine.add_stimulus(Stimulus('IR_Enable', [t], [True]))
        for sig in self.all_ctrl:
            if sig in self.pi:
                self.engine.add_stimulus(Stimulus(sig, [t], [False]))
        self.engine.add_stimulus(Stimulus('ACC_Enable', [t], [False]))
        self.engine.add_stimulus(Stimulus('BREG_Enable', [t], [False]))
        self.engine.add_stimulus(Stimulus('INC', [t], [False]))
        self.engine.add_stimulus(Stimulus('LOAD', [t], [False]))
        if 'ALU_B_IS_ONE' in self.pi:
            self.engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t], [False]))
        self.engine.add_stimulus(Stimulus('CLK', [t, t+T/4, t+T/2],
                                          [False, True, False]))
        t += T
        self.engine.run(t)

        # Execute micro-steps
        for step, signals in micro_steps:
            active = set(signals)

            if self.inst_name == 'JCN' and step == 3:
                cy = self.engine._nets.get('CY_0')
                if cy and cy.value:
                    active.add('PC_LOAD')
                else:
                    active.add('PC_INC')

            if 'SCRATCH_TO_BUS' in active:
                active.discard('SCRATCH_TO_BUS')
                active.add('SP_READ')
            if 'SCRATCH_LOAD' in active:
                active.discard('SCRATCH_LOAD')
                active.add('SP_WRITE')

            if step == micro_steps[-1][0]:
                if 'PC_LOAD' not in active and 'PC_INC' not in signals:
                    active.add('PC_INC')

            for sig in self.all_ctrl:
                if sig in self.pi:
                    self.engine.add_stimulus(Stimulus(sig, [t], [sig in active]))
            if 'ALU_B_IS_ONE' in self.pi:
                self.engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t],
                                                   ['ALU_B_IS_ONE' in active]))
            self.engine.add_stimulus(Stimulus('SUB', [t], ['ALU_SUB' in active]))

            acc_en = 'ACC_LOAD_FROM_ALU' in active or 'ACC_LOAD_FROM_BUS' in active
            self.engine.add_stimulus(Stimulus('ACC_Enable', [t], [acc_en]))
            self.engine.add_stimulus(Stimulus('BREG_Enable', [t],
                                              ['BREG_LOAD_FROM_BUS' in active]))
            self.engine.add_stimulus(Stimulus('INC', [t], ['PC_INC' in active]))
            self.engine.add_stimulus(Stimulus('LOAD', [t], ['PC_LOAD' in active]))
            self.engine.add_stimulus(Stimulus('PC_Enable', [t], [True]))

            self.engine.add_stimulus(Stimulus('CLK', [t, t+T/4, t+T/2],
                                              [False, True, False]))
            t += T
            self.engine.run(t)

        # Clear
        for sig in self.all_ctrl:
            if sig in self.pi:
                self.engine.add_stimulus(Stimulus(sig, [t], [False]))
        self.engine.add_stimulus(Stimulus('ACC_Enable', [t], [False]))
        self.engine.add_stimulus(Stimulus('BREG_Enable', [t], [False]))
        self.engine.add_stimulus(Stimulus('IR_Enable', [t], [False]))
        if 'ALU_B_IS_ONE' in self.pi:
            self.engine.add_stimulus(Stimulus('ALU_B_IS_ONE', [t], [False]))

        self.t_current = t
        self._read_state()
        self.cycle += 1

        # Add to history
        sp_str = ' '.join(f'R{k}={v}' for k, v in self.sp_vals.items())
        self.history.append(
            f"[{self.cycle:3d}] {self.inst_name:4s}  ACC={self.acc:2d} "
            f"C={self.carry} {sp_str} PC={self.pc}")

    def _step(self):
        self._execute_one_instruction()
        self._update_display()
        self._update_history()

    def _toggle_run(self):
        if self.running:
            self._stop_run()
        else:
            self.running = True
            self.run_btn.config(text="Stop (Esc)", bg='#e94560')
            self._run_loop()

    def _stop_run(self):
        self.running = False
        self.run_btn.config(text="Run (R)", bg='#0f3460')

    def _run_loop(self):
        if not self.running:
            return
        self._step()
        self.after(50, self._run_loop)  # 50ms between steps

    def _reset(self):
        self._stop_run()
        self.cycle = 0
        self.history = []
        self._init_engine()
        self._read_state()
        self._update_display()
        self._update_history()

    def _update_history(self):
        self.hist_text.config(state='normal')
        self.hist_text.delete('1.0', tk.END)
        # Show last 20 entries
        for entry in self.history[-20:]:
            self.hist_text.insert(tk.END, entry + '\n')
        self.hist_text.see(tk.END)
        self.hist_text.config(state='disabled')


if __name__ == '__main__':
    cpu = build_test_cpu()

    program = {
        0: 0x10,  # IAC
        1: 0x24,  # JCN carry, jump to 4
        2: 0x40,  # JUN 0
        3: 0x00,  # NOP
        4: 0xB0,  # XCH R0
        5: 0x10,  # IAC
        6: 0xB0,  # XCH R0
        7: 0x40,  # JUN 0
    }

    app = CPUGUI(cpu, program)
    app.mainloop()
