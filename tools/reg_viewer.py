#!/usr/bin/env python3
"""
Register viewer for 4004 LTSpice simulation data.

Loads a .raw file (or extracted .raw) and shows:
- All 16 scratchpad registers
- ACC, IR1-IR4, PC, Carry Flag
- Bus values
- Current decoded instruction
- Step forward/back by clock cycle or instruction

Usage:
    python tools/reg_viewer.py cpus/4004/programs/FloatingPoint/FloatingPoint_extracted.raw
"""

import sys
import os
import struct
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QGroupBox, QGridLayout, QFrame,
    QSpinBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)


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
        self.var_names = {}  # lowercase name -> index
        self.var_list = []   # index -> name

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

        # Read all time values for fast seeking
        self._load_time_array()

    def _load_time_array(self):
        """Load just the time column for fast seeking."""
        self.times = np.zeros(self.n_points)
        with open(self.path, "rb") as f:
            for i in range(self.n_points):
                f.seek(self.data_start + i * self.row_size)
                self.times[i] = struct.unpack_from("d", f.read(8), 0)[0]

    def read_point(self, pt_idx):
        """Read all signals at a specific time point index."""
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


class RegViewer(QMainWindow):
    def __init__(self, raw_path):
        super().__init__()
        self.setWindowTitle(f"4004 Register Viewer — {os.path.basename(raw_path)}")
        self.setMinimumSize(900, 700)

        self.raw = RawReader(raw_path)
        self.current_pt = 0
        self.thresh = -2.5

        # Find micro1 edges for instruction stepping
        self._find_instruction_edges()

        self._build_ui()
        self._update_display()

    def _find_instruction_edges(self):
        """Find all Micro1 rising edges for instruction stepping."""
        self.inst_edges = [0]
        m1_key = "v(micro1)"
        if not self.raw.find_signal("micro1)"):
            return

        prev_low = True
        with open(self.raw.path, "rb") as f:
            m1_idx = self.raw.var_names.get(m1_key, -1)
            if m1_idx < 0:
                return
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

        # Navigation bar
        nav = QHBoxLayout()
        self.btn_prev_inst = QPushButton("◀◀ Prev Inst")
        self.btn_prev = QPushButton("◀ Prev")
        self.btn_next = QPushButton("Next ▶")
        self.btn_next_inst = QPushButton("Next Inst ▶▶")
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

        # Instruction decode
        self.lbl_instr = QLabel("IR: --")
        self.lbl_instr.setFont(QFont("Consolas", 14, QFont.Bold))
        self.lbl_instr.setStyleSheet("background: #222; color: #0f0; padding: 8px;")
        layout.addWidget(self.lbl_instr)

        # Main register display
        main_row = QHBoxLayout()

        # Left: ACC, CY, PC, Bus
        left = QVBoxLayout()
        self.reg_labels = {}

        for group_name, regs in [
            ("CPU State", [("ACC", 4), ("CY", 1), ("IR1", 4), ("IR2", 4)]),
            ("Program Counter", [("PC3", 4), ("PC2", 4), ("PC1", 4)]),
            ("Bus", [("Bus", 4)]),
        ]:
            grp = QGroupBox(group_name)
            grid = QGridLayout(grp)
            for i, (name, bits) in enumerate(regs):
                lbl_name = QLabel(f"{name}:")
                lbl_name.setFont(QFont("Consolas", 11))
                lbl_val = QLabel("--")
                lbl_val.setFont(QFont("Consolas", 14, QFont.Bold))
                lbl_val.setStyleSheet("color: #0af;")
                grid.addWidget(lbl_name, i, 0)
                grid.addWidget(lbl_val, i, 1)
                self.reg_labels[name] = lbl_val
            left.addWidget(grp)
        left.addStretch()
        main_row.addLayout(left)

        # Right: Scratchpad 0-15
        grp_sp = QGroupBox("Scratchpad Registers")
        sp_grid = QGridLayout(grp_sp)
        sp_grid.addWidget(QLabel("Reg"), 0, 0)
        sp_grid.addWidget(QLabel("Hex"), 0, 1)
        sp_grid.addWidget(QLabel("Dec"), 0, 2)

        self.sp_labels = {}
        for r in range(16):
            row = r + 1
            lbl_name = QLabel(f"R{r:d}")
            lbl_name.setFont(QFont("Consolas", 10))
            lbl_hex = QLabel("--")
            lbl_hex.setFont(QFont("Consolas", 12, QFont.Bold))
            lbl_hex.setStyleSheet("color: #fa0;")
            lbl_dec = QLabel("--")
            lbl_dec.setFont(QFont("Consolas", 10))
            sp_grid.addWidget(lbl_name, row, 0)
            sp_grid.addWidget(lbl_hex, row, 1)
            sp_grid.addWidget(lbl_dec, row, 2)
            self.sp_labels[r] = (lbl_hex, lbl_dec)

        main_row.addWidget(grp_sp)
        layout.addLayout(main_row)

    def _update_display(self):
        t, vals = self.raw.read_point(self.current_pt)

        self.lbl_time.setText(f"t = {t * 1e3:.4f} ms")
        self.lbl_point.setText(f"pt {self.current_pt} / {self.raw.n_points}")
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_pt)
        self.slider.blockSignals(False)

        nib = self.raw.get_nibble

        acc = nib(vals, "acc")
        ir1 = nib(vals, "ir1_")
        ir2 = nib(vals, "ir2_")
        pc1 = nib(vals, "pc1")
        pc2 = nib(vals, "pc2")
        pc3 = nib(vals, "pc3")
        bus = nib(vals, "bus")

        # Carry
        cy_val = vals.get("v(cf0)", -5)
        cy = 1 if cy_val > self.thresh else 0 if cy_val > -4 else -1

        self.reg_labels["ACC"].setText(f"0x{acc:X}  ({acc:d})")
        self.reg_labels["CY"].setText(f"{cy}" if cy >= 0 else "?")
        self.reg_labels["IR1"].setText(f"0x{ir1:X}")
        self.reg_labels["IR2"].setText(f"0x{ir2:X}")
        self.reg_labels["PC3"].setText(f"0x{pc3:X}")
        self.reg_labels["PC2"].setText(f"0x{pc2:X}")
        self.reg_labels["PC1"].setText(f"0x{pc1:X}")
        self.reg_labels["Bus"].setText(f"0x{bus:X}  ({bus:d})")

        # Instruction decode
        opcode = (ir1 << 4) | ir2
        pc = (pc3 << 8) | (pc2 << 4) | pc1
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
        if opcode in ACC_MNEMONICS:
            mnem = ACC_MNEMONICS[opcode]
        elif (opcode >> 4) == 0xE:
            io_names = {0: 'WRM', 1: 'WMP', 2: 'WRR', 8: 'SBM', 9: 'RDM', 0xB: 'ADM'}
            mnem = io_names.get(opcode & 0xF, f'IO 0x{opcode:02X}')
        else:
            mnem = MNEMONICS.get(opcode >> 4, '???')
            mnem = f"{mnem} R{ir2:d}" if (opcode >> 4) in (0x6, 0x8, 0x9, 0xA, 0xB) else f"{mnem} {ir2:X}"

        self.lbl_instr.setText(
            f"  PC=0x{pc:03X}  |  IR=0x{ir1:X}{ir2:X}  |  {mnem}  |  ACC=0x{acc:X}  CY={cy}"
        )

        # Scratchpad
        for r in range(16):
            val = nib(vals, f"scratch{r}_")
            lbl_hex, lbl_dec = self.sp_labels[r]
            lbl_hex.setText(f"0x{val:X}")
            lbl_dec.setText(f"{val:d}")

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
        import bisect
        idx = bisect.bisect_left(self.inst_edges, self.current_pt) - 1
        if idx >= 0:
            self.current_pt = self.inst_edges[idx]
            self._update_display()

    def _next_inst(self):
        import bisect
        idx = bisect.bisect_right(self.inst_edges, self.current_pt)
        if idx < len(self.inst_edges):
            self.current_pt = self.inst_edges[idx]
            self._update_display()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="4004 Register Viewer")
    parser.add_argument("raw_file", nargs="?", help="Path to .raw file")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Dark theme
    from PyQt5.QtGui import QPalette, QColor
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
        raw_path, _ = QFileDialog.getOpenFileName(None, "Open .raw file", "", "Raw Files (*.raw)")
        if not raw_path:
            return

    win = RegViewer(raw_path)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
