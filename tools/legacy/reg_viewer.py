#!/usr/bin/env python3
"""
Register viewer for 4004 LTSpice simulation data.

Shows hardware state side-by-side with cycle-accurate expected (simulated)
state. Mismatches highlighted in red.

Loads a .raw file (or extracted/smoothed .raw) and shows:
- All 16 scratchpad registers (HW vs Expected)
- ACC, CY, IR, PC, Bus
- Current decoded instruction
- Mismatch counter
- Step forward/back by clock cycle or instruction

Usage:
    python tools/reg_viewer.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted_smoothed.raw
"""

import sys
import os
import struct
import bisect
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QFrame,
    QSpinBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QPalette, QColor

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from tools.rom_emulator import (
    assemble, CPU4004Sim, get_phase_count,
    CLK_PERIOD, INST_PHASES,
)


# ── .raw reader ──────────────────────────────────────────────────────────

class RawReader:
    """Reads LTSpice .raw files with mixed precision (time=double, rest=float)."""

    def __init__(self, path):
        self.path = path
        self._parse_header()

    def _parse_header(self):
        with open(self.path, "rb") as f:
            header = f.read(2_000_000)

        text = header.decode("utf-16-le", errors="replace")

        self.n_vars = 0
        self.n_points = 0
        self.var_names = {}
        self.var_list = []

        for line in text.split("\n"):
            s = line.strip()
            if s.startswith("No. Variables"):
                self.n_vars = int(s.split(":")[1].strip())
            elif s.startswith("No. Points"):
                self.n_points = int(s.split(":")[1].strip())
            elif "\t" in s:
                parts = s.split("\t")
                if len(parts) >= 3:
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip()
                        self.var_names[name.lower()] = idx
                        while len(self.var_list) <= idx:
                            self.var_list.append("")
                        self.var_list[idx] = name
                    except ValueError:
                        pass
            elif s == "Binary:":
                break

        idx = text.find("Binary:")
        self.data_start = (idx + len("Binary:") + 1) * 2
        self.row_size = 8 + (self.n_vars - 1) * 4

        self._load_time_array()

    def _load_time_array(self):
        self.times = np.zeros(self.n_points)
        with open(self.path, "rb") as f:
            for i in range(self.n_points):
                f.seek(self.data_start + i * self.row_size)
                self.times[i] = struct.unpack_from("d", f.read(8), 0)[0]

    def read_point(self, pt_idx):
        with open(self.path, "rb") as f:
            f.seek(self.data_start + pt_idx * self.row_size)
            row = f.read(self.row_size)

        t = struct.unpack_from("d", row, 0)[0]
        values = {}
        for name, idx in self.var_names.items():
            if idx == 0:
                values[name] = t
            else:
                values[name] = struct.unpack_from("f", row, 8 + (idx - 1) * 4)[0]
        return t, values

    def find_signal(self, name):
        return name.lower() in self.var_names

    def get_nibble(self, values, prefix, n_bits=4, thresh=-2.5):
        val = 0
        for i in range(n_bits):
            key = f"v({prefix}{i})"
            if key in values and values[key] > thresh:
                val |= (1 << i)
        return val


# ── Cycle-accurate simulator trace ───────────────────────────────────────

MNEMONICS = {
    0x0: 'NOP', 0x1: 'JCN', 0x2: 'FIM/SRC', 0x3: 'FIN/JIN',
    0x4: 'JUN', 0x5: 'JMS', 0x6: 'INC', 0x7: 'ISZ',
    0x8: 'ADD', 0x9: 'SUB', 0xA: 'LD', 0xB: 'XCH',
    0xC: 'BBL', 0xD: 'LDM',
}
ACC_MNEMONICS = {
    0xF0: 'CLB', 0xF1: 'CLC', 0xF2: 'IAC', 0xF3: 'CMC', 0xF4: 'CMA',
    0xF5: 'RAL', 0xF6: 'RAR', 0xF7: 'TCC', 0xF8: 'DAC', 0xF9: 'TCS',
    0xFA: 'STC', 0xFB: 'DAA', 0xFC: 'KBP', 0xFD: 'DCL',
}


def decode_mnemonic(opcode):
    if opcode in ACC_MNEMONICS:
        return ACC_MNEMONICS[opcode]
    opr = (opcode >> 4) & 0xF
    opa = opcode & 0xF
    base = MNEMONICS.get(opr, '???')
    if opr in (0x6, 0x8, 0x9, 0xA, 0xB):
        return f"{base} R{opa}"
    elif opr == 0xD:
        return f"LDM {opa}"
    elif opr == 0xC:
        return f"BBL {opa}"
    return f"{base} {opa:X}"


class SimTrace:
    """Clock-cycle-accurate CPU simulation trace.

    Records one snapshot per CLK period (10us). Within each instruction:
    - Micro 1-4: previous state still visible (fetch phases)
    - Micro 5: IR1 (OPR) loads from ROM
    - Micro 6: IR2 (OPA) loads from ROM
    - Last CLK cycle: ACC, CY, regs update (execute completes)

    This matches the real 4004 pipeline where register writes happen
    at the end of the instruction and IR loads during mid-fetch.
    """

    # Which micro-phase the IR nibbles load (0-indexed within instruction)
    IR1_LOAD_PHASE = 4   # Micro5 (0-indexed = 4)
    IR2_LOAD_PHASE = 5   # Micro6 (0-indexed = 5)

    def __init__(self, asm_path, startup_delay=2e-6):
        with open(asm_path) as f:
            asm_text = f.read()
        rom = assemble(asm_text)

        cpu = CPU4004Sim(rom)
        cpu.regs = [0xF] * 16  # JFET power-up for scratchpad
        cpu.acc = 0            # ACC starts at 0
        cpu.cy = 0

        self.snapshots = []   # one per CLK cycle
        self.t_starts = []    # time of each CLK cycle start

        t = startup_delay

        # State that persists across instructions
        prev_acc = cpu.acc
        prev_cy = cpu.cy
        prev_regs = list(cpu.regs)
        prev_pc = cpu.pc
        prev_ir1 = 0x0
        prev_ir2 = 0x0
        prev_ir = 0x00
        prev_mnem = 'INIT'

        for _ in range(600):
            # Snapshot BEFORE executing: this is the "previous" state
            pre_acc = cpu.acc
            pre_cy = cpu.cy
            pre_regs = list(cpu.regs)
            pre_pc = cpu.pc

            # Execute the instruction
            pc_start, b1, b2, two_word = cpu.execute_one()
            n_phases = get_phase_count(b1)
            new_ir1 = (b1 >> 4) & 0xF
            new_ir2 = b1 & 0xF
            new_mnem = decode_mnemonic(b1)

            # Post-execution state
            post_acc = cpu.acc
            post_cy = cpu.cy
            post_regs = list(cpu.regs)
            post_pc = cpu.pc

            # Emit one snapshot per CLK cycle within this instruction
            for phase in range(n_phases):
                # IR updates at specific phases
                if phase < self.IR1_LOAD_PHASE:
                    ir1 = prev_ir1
                    ir2 = prev_ir2
                    ir = prev_ir
                    mnem = prev_mnem
                elif phase < self.IR2_LOAD_PHASE:
                    ir1 = new_ir1
                    ir2 = prev_ir2  # IR2 not loaded yet
                    ir = (new_ir1 << 4) | prev_ir2
                    mnem = new_mnem
                else:
                    ir1 = new_ir1
                    ir2 = new_ir2
                    ir = b1
                    mnem = new_mnem

                # ACC/CY/regs update on the LAST cycle of the instruction
                # PC stays at the instruction's fetch address throughout
                if phase < n_phases - 1:
                    acc = pre_acc
                    cy = pre_cy
                    regs = pre_regs
                else:
                    acc = post_acc
                    cy = post_cy
                    regs = post_regs
                pc = pc_start  # PC shows this instruction's address for all phases

                self.snapshots.append({
                    'acc': acc, 'cy': cy,
                    'regs': list(regs), 'pc': pc,
                    'ir1': ir1, 'ir2': ir2,
                    'ir': ir, 'mnem': mnem,
                })
                self.t_starts.append(t)
                t += CLK_PERIOD

            # Remember for next instruction's early phases
            prev_ir1 = new_ir1
            prev_ir2 = new_ir2
            prev_ir = b1
            prev_mnem = new_mnem

            # Stop after NOP following main code
            if b1 == 0x00 and pc_start > 0x080:
                break

        self.t_starts = np.array(self.t_starts)

    def get_state_at_time(self, t):
        """Return the expected sim state for a given time."""
        if t < self.t_starts[0]:
            return None
        if t > self.t_starts[-1]:
            return self.snapshots[-1]
        idx = bisect.bisect_right(self.t_starts, t) - 1
        idx = max(0, min(idx, len(self.snapshots) - 1))
        return self.snapshots[idx]


# ── GUI ──────────────────────────────────────────────────────────────────

COL_OK_CPU = "#0af"
COL_OK_SP = "#fa0"
COL_FAIL = "#f44"
COL_MATCH_BG = "#222"
COL_FAIL_BG = "#400"
COL_DIM = "#888"


class RegViewer(QMainWindow):
    def __init__(self, raw_path, sim_trace=None):
        super().__init__()
        self.setWindowTitle(f"4004 Register Viewer - {os.path.basename(raw_path)}")
        self.setMinimumSize(1050, 750)

        self.raw = RawReader(raw_path)
        self.sim = sim_trace
        self.current_pt = 0
        self.thresh = -2.5

        self._find_instruction_edges()
        self._build_ui()
        self._update_display()

    def _find_instruction_edges(self):
        self.inst_edges = [0]
        m1_key = "v(micro1)"
        m1_idx = self.raw.var_names.get(m1_key, -1)
        if m1_idx < 0:
            return

        prev_low = True
        with open(self.raw.path, "rb") as f:
            for pt in range(self.raw.n_points):
                f.seek(self.raw.data_start + pt * self.raw.row_size + 8 + (m1_idx - 1) * 4)
                val = struct.unpack_from("f", f.read(4), 0)[0]
                is_high = val > self.thresh
                if is_high and prev_low:
                    self.inst_edges.append(pt)
                prev_low = not is_high

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # ── Mismatch counter ──
        self.lbl_mismatch = QLabel("-- no sim --")
        self.lbl_mismatch.setFont(QFont("Consolas", 13, QFont.Bold))
        self.lbl_mismatch.setAlignment(Qt.AlignCenter)
        self.lbl_mismatch.setStyleSheet(
            "background: #333; color: #888; padding: 6px; border-radius: 4px;"
        )
        layout.addWidget(self.lbl_mismatch)

        # ── Navigation bar ──
        nav = QHBoxLayout()
        self.btn_prev_inst = QPushButton("Prev Inst")
        self.btn_prev = QPushButton("Prev")
        self.btn_next = QPushButton("Next")
        self.btn_next_inst = QPushButton("Next Inst")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, self.raw.n_points - 1)
        self.lbl_time = QLabel("t = 0.000 ms")
        self.lbl_time.setFont(QFont("Consolas", 12, QFont.Bold))
        self.lbl_point = QLabel(f"pt 0 / {self.raw.n_points}")

        self.btn_prev_inst.clicked.connect(self._prev_inst)
        self.btn_prev.clicked.connect(self._prev_point)
        self.btn_next.clicked.connect(self._next_point)
        self.btn_next_inst.clicked.connect(self._next_inst)
        self.slider.valueChanged.connect(self._slider_changed)

        nav.addWidget(self.btn_prev_inst)
        nav.addWidget(self.btn_prev)
        nav.addWidget(self.slider)
        nav.addWidget(self.btn_next)
        nav.addWidget(self.btn_next_inst)
        layout.addLayout(nav)

        time_bar = QHBoxLayout()
        time_bar.addWidget(self.lbl_time)
        time_bar.addStretch()
        time_bar.addWidget(self.lbl_point)
        layout.addLayout(time_bar)

        # ── Instruction decode bar ──
        self.lbl_instr = QLabel("IR: --")
        self.lbl_instr.setFont(QFont("Consolas", 13, QFont.Bold))
        self.lbl_instr.setStyleSheet(f"background: {COL_MATCH_BG}; color: #0f0; padding: 8px;")
        layout.addWidget(self.lbl_instr)

        # ── Main register display ──
        main_row = QHBoxLayout()

        # Left: CPU State — all groups get HW + Exp columns
        left = QVBoxLayout()
        self.cpu_labels = {}  # name -> (hw_label, exp_label)

        # All groups have Exp column (except Bus which sim doesn't model)
        # Extra columns: IR1 gets "Opcode" col, IR2 gets "Dec" col
        all_groups = [
            ("CPU State", [("ACC", True), ("CY", True)]),
            ("Instruction", [("IR1", True), ("IR2", True)]),
            ("Program Counter", [("PC3", True), ("PC2", True), ("PC1", True)]),
            ("Bus", [("Bus", False)]),
        ]

        self.ir_opcode_lbl = None   # extra label for IR1 opcode name
        self.ir2_dec_lbl = None     # extra label for IR2 decimal

        for group_name, regs in all_groups:
            grp = QGroupBox(group_name)
            grid = QGridLayout(grp)
            hdr_hw = QLabel("HW")
            hdr_hw.setFont(QFont("Consolas", 9, QFont.Bold))
            grid.addWidget(QLabel(""), 0, 0)
            grid.addWidget(hdr_hw, 0, 1)

            has_any_exp = any(has_exp for _, has_exp in regs)
            if has_any_exp:
                hdr_exp = QLabel("Exp")
                hdr_exp.setFont(QFont("Consolas", 9, QFont.Bold))
                grid.addWidget(hdr_exp, 0, 2)

            if group_name == "Instruction":
                hdr_extra = QLabel("")
                hdr_extra.setFont(QFont("Consolas", 9, QFont.Bold))
                grid.addWidget(hdr_extra, 0, 3)

            for i, (name, has_exp) in enumerate(regs):
                row = i + 1
                lbl_name = QLabel(f"{name}:")
                lbl_name.setFont(QFont("Consolas", 11))
                lbl_hw = QLabel("--")
                lbl_hw.setFont(QFont("Consolas", 14, QFont.Bold))
                lbl_hw.setStyleSheet(f"color: {COL_OK_CPU};")
                grid.addWidget(lbl_name, row, 0)
                grid.addWidget(lbl_hw, row, 1)

                lbl_exp = None
                if has_exp:
                    lbl_exp = QLabel("--")
                    lbl_exp.setFont(QFont("Consolas", 14, QFont.Bold))
                    lbl_exp.setStyleSheet(f"color: {COL_DIM};")
                    grid.addWidget(lbl_exp, row, 2)

                # Extra column for IR1 (opcode) and IR2 (decimal)
                if name == "IR1":
                    self.ir_opcode_lbl = QLabel("")
                    self.ir_opcode_lbl.setFont(QFont("Consolas", 11))
                    self.ir_opcode_lbl.setStyleSheet(f"color: #8f8;")
                    grid.addWidget(self.ir_opcode_lbl, row, 3)
                elif name == "IR2":
                    self.ir2_dec_lbl = QLabel("")
                    self.ir2_dec_lbl.setFont(QFont("Consolas", 11))
                    self.ir2_dec_lbl.setStyleSheet(f"color: {COL_OK_CPU};")
                    grid.addWidget(self.ir2_dec_lbl, row, 3)

                self.cpu_labels[name] = (lbl_hw, lbl_exp)
            left.addWidget(grp)
        left.addStretch()
        main_row.addLayout(left)

        # Right: Scratchpad 0-15
        grp_sp = QGroupBox("Scratchpad Registers")
        sp_grid = QGridLayout(grp_sp)
        for col, hdr in enumerate(["Reg", "HW", "Exp", "Dec"]):
            lbl = QLabel(hdr)
            lbl.setFont(QFont("Consolas", 10, QFont.Bold))
            sp_grid.addWidget(lbl, 0, col)

        self.sp_labels = {}  # r -> (lbl_hw, lbl_exp, lbl_dec)
        for r in range(16):
            row = r + 1
            lbl_name = QLabel(f"R{r:d}")
            lbl_name.setFont(QFont("Consolas", 10))
            lbl_hw = QLabel("--")
            lbl_hw.setFont(QFont("Consolas", 12, QFont.Bold))
            lbl_hw.setStyleSheet(f"color: {COL_OK_SP};")
            lbl_exp = QLabel("--")
            lbl_exp.setFont(QFont("Consolas", 12, QFont.Bold))
            lbl_exp.setStyleSheet(f"color: {COL_DIM};")
            lbl_dec = QLabel("--")
            lbl_dec.setFont(QFont("Consolas", 10))

            sp_grid.addWidget(lbl_name, row, 0)
            sp_grid.addWidget(lbl_hw, row, 1)
            sp_grid.addWidget(lbl_exp, row, 2)
            sp_grid.addWidget(lbl_dec, row, 3)
            self.sp_labels[r] = (lbl_hw, lbl_exp, lbl_dec)

        main_row.addWidget(grp_sp)
        layout.addLayout(main_row)

    # ── Display update ───────────────────────────────────────────────────

    def _update_display(self):
        t, vals = self.raw.read_point(self.current_pt)
        nib = self.raw.get_nibble

        self.lbl_time.setText(f"t = {t * 1e3:.4f} ms")
        self.lbl_point.setText(f"pt {self.current_pt} / {self.raw.n_points}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_pt)
        self.slider.blockSignals(False)

        # ── Read HW state ──
        acc = nib(vals, "acc")
        ir1 = nib(vals, "ir1_")
        ir2 = nib(vals, "ir2_")
        pc1 = nib(vals, "pc1")
        pc2 = nib(vals, "pc2")
        pc3 = nib(vals, "pc3")
        bus = nib(vals, "bus")
        cy_val = vals.get("v(cf0)", -5)
        cy = 1 if cy_val > self.thresh else 0

        hw_regs = [nib(vals, f"scratch{r}_") for r in range(16)]

        # ── Get expected state ──
        exp = self.sim.get_state_at_time(t) if self.sim else None

        # ── Count mismatches ──
        n_mismatch = 0
        if exp:
            if acc != exp['acc']:
                n_mismatch += 1
            if cy != exp['cy']:
                n_mismatch += 1
            if ir1 != exp['ir1']:
                n_mismatch += 1
            if ir2 != exp['ir2']:
                n_mismatch += 1
            for r in range(16):
                if hw_regs[r] != exp['regs'][r]:
                    n_mismatch += 1

        # ── Mismatch banner ──
        if not exp:
            self.lbl_mismatch.setText("No sim data at this time")
            self.lbl_mismatch.setStyleSheet(
                "background: #333; color: #888; padding: 6px; border-radius: 4px;"
            )
        elif n_mismatch == 0:
            self.lbl_mismatch.setText("All match")
            self.lbl_mismatch.setStyleSheet(
                "background: #141; color: #4f4; padding: 6px; border-radius: 4px;"
            )
        else:
            self.lbl_mismatch.setText(f"{n_mismatch} mismatch{'es' if n_mismatch > 1 else ''}")
            self.lbl_mismatch.setStyleSheet(
                "background: #411; color: #f44; padding: 6px; border-radius: 4px;"
            )

        # ── CPU state labels ──
        def set_cpu(name, hw_str, exp_val=None, hw_val=None):
            lbl_hw, lbl_exp = self.cpu_labels[name]
            lbl_hw.setText(hw_str)
            if lbl_exp is not None and exp is not None and exp_val is not None:
                match = (hw_val == exp_val)
                col = COL_OK_CPU if match else COL_FAIL
                lbl_hw.setStyleSheet(f"color: {col};")
                lbl_exp.setText(f"0x{exp_val:X}" if isinstance(exp_val, int) and exp_val >= 0 else str(exp_val))
                lbl_exp.setStyleSheet(f"color: {col};")
            elif lbl_exp is not None:
                lbl_hw.setStyleSheet(f"color: {COL_OK_CPU};")
                lbl_exp.setText("--")
                lbl_exp.setStyleSheet(f"color: {COL_DIM};")
            else:
                lbl_hw.setStyleSheet(f"color: {COL_OK_CPU};")

        exp_pc = exp['pc'] if exp else None
        exp_pc1 = (exp_pc >> 8) & 0xF if exp_pc is not None else None
        exp_pc2 = (exp_pc >> 4) & 0xF if exp_pc is not None else None
        exp_pc3 = exp_pc & 0xF if exp_pc is not None else None

        set_cpu("ACC", f"0x{acc:X}  ({acc:d})",
                exp['acc'] if exp else None, acc)
        set_cpu("CY", f"{cy}",
                exp['cy'] if exp else None, cy)
        set_cpu("IR1", f"0x{ir1:X}",
                exp['ir1'] if exp else None, ir1)
        set_cpu("IR2", f"0x{ir2:X}",
                exp['ir2'] if exp else None, ir2)

        # IR1 extra: opcode name
        opcode = (ir1 << 4) | ir2
        mnem = decode_mnemonic(opcode)
        if self.ir_opcode_lbl:
            self.ir_opcode_lbl.setText(mnem)
        # IR2 extra: decimal value
        if self.ir2_dec_lbl:
            self.ir2_dec_lbl.setText(f"dec {ir2:d}")
        set_cpu("PC3", f"0x{pc3:X}", exp_pc3, pc3)
        set_cpu("PC2", f"0x{pc2:X}", exp_pc2, pc2)
        set_cpu("PC1", f"0x{pc1:X}", exp_pc1, pc1)
        set_cpu("Bus", f"0x{bus:X}  ({bus:d})")

        # ── Instruction decode ──
        pc = (pc1 << 8) | (pc2 << 4) | pc3

        instr_text = f"  HW: PC=0x{pc:03X}  IR=0x{ir1:X}{ir2:X}  {mnem}  ACC=0x{acc:X}  CY={cy}"
        if exp:
            exp_opcode = exp['ir']
            exp_mnem = exp['mnem']
            instr_text += f"     Exp: PC=0x{exp['pc']:03X}  IR=0x{exp_opcode:02X}  {exp_mnem}  ACC=0x{exp['acc']:X}  CY={exp['cy']}"
        self.lbl_instr.setText(instr_text)

        bar_bg = COL_FAIL_BG if n_mismatch > 0 else COL_MATCH_BG
        self.lbl_instr.setStyleSheet(
            f"background: {bar_bg}; color: #0f0; padding: 8px;"
        )

        # ── Scratchpad registers ──
        for r in range(16):
            hw_val = hw_regs[r]
            lbl_hw, lbl_exp, lbl_dec = self.sp_labels[r]
            lbl_hw.setText(f"0x{hw_val:X}")
            lbl_dec.setText(f"{hw_val:d}")

            if exp:
                exp_val = exp['regs'][r]
                match = (hw_val == exp_val)
                col = COL_OK_SP if match else COL_FAIL
                lbl_hw.setStyleSheet(f"color: {col};")
                lbl_exp.setText(f"0x{exp_val:X}")
                lbl_exp.setStyleSheet(f"color: {col};")
            else:
                lbl_hw.setStyleSheet(f"color: {COL_OK_SP};")
                lbl_exp.setText("--")
                lbl_exp.setStyleSheet(f"color: {COL_DIM};")

    # ── Navigation ───────────────────────────────────────────────────────

    def _slider_changed(self, val):
        self.current_pt = val
        self._update_display()

    def _prev_point(self):
        if self.current_pt > 0:
            self.current_pt -= 1
            self._update_display()

    def _next_point(self):
        if self.current_pt < self.raw.n_points - 1:
            self.current_pt += 1
            self._update_display()

    def _prev_inst(self):
        idx = bisect.bisect_left(self.inst_edges, self.current_pt) - 1
        if idx >= 0:
            self.current_pt = self.inst_edges[idx]
            self._update_display()

    def _next_inst(self):
        idx = bisect.bisect_right(self.inst_edges, self.current_pt)
        if idx < len(self.inst_edges):
            self.current_pt = self.inst_edges[idx]
            self._update_display()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="4004 Register Viewer")
    parser.add_argument("raw_file", nargs="?", help="Path to .raw file")
    parser.add_argument("--asm", default=None, help="Path to .asm (auto-detected if omitted)")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(30, 30, 30))
    palette.setColor(QPalette.WindowText, QColor(200, 200, 200))
    palette.setColor(QPalette.Base, QColor(25, 25, 25))
    palette.setColor(QPalette.AlternateBase, QColor(40, 40, 40))
    palette.setColor(QPalette.Text, QColor(200, 200, 200))
    palette.setColor(QPalette.Button, QColor(50, 50, 50))
    palette.setColor(QPalette.ButtonText, QColor(200, 200, 200))
    app.setPalette(palette)

    raw_path = args.raw_file
    if not raw_path:
        raw_path, _ = QFileDialog.getOpenFileName(
            None, "Open .raw file", "", "Raw Files (*.raw)"
        )
        if not raw_path:
            return

    # Auto-detect .asm
    sim_trace = None
    asm_path = args.asm
    if not asm_path:
        raw_dir = os.path.dirname(os.path.abspath(raw_path))
        for f in os.listdir(raw_dir):
            if f.endswith('.asm'):
                asm_path = os.path.join(raw_dir, f)
                break

    if asm_path and os.path.isfile(asm_path):
        print(f"Loading sim trace from: {asm_path}")
        try:
            sim_trace = SimTrace(asm_path)
            print(f"  {len(sim_trace.snapshots)} instruction snapshots")
        except Exception as e:
            print(f"  WARNING: Failed to build sim trace: {e}")

    win = RegViewer(raw_path, sim_trace)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
