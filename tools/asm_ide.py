#!/usr/bin/env python3
"""4004 ASM IDE — syntax-highlighted editor with build + run.

Features:
- Open/create Intel 4004 assembly files
- Syntax highlighting: opcodes (green), data (blue), labels (black bold),
  comments (gray italic)
- Line numbers + per-line byte address shown
- Visual page boundary markers (4004 ROM is 256 bytes/page)
- Build & Run (F5): saves to programs/<name>/<name>.asm, assembles via
  rom_emulator, generates PWLs, rebuilds 4004.asc with correct .tran,
  opens LTSpice ready to simulate

Usage:
    python tools/asm_ide.py
    python tools/asm_ide.py cpus/4004/programs/AdderTest/AdderTest.asm
"""

import os
import re
import subprocess
import sys

from PyQt5.QtCore import QRect, QSize, Qt
from PyQt5.QtGui import (
    QColor, QFont, QPainter, QSyntaxHighlighter, QTextCharFormat,
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QFileDialog, QInputDialog, QMainWindow,
    QMessageBox, QPlainTextEdit, QWidget,
)


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LTSPICE = r"C:\Users\z00503ku\AppData\Local\Programs\ADI\LTspice\LTspice.exe"

# ── 4004 ISA ─────────────────────────────────────────────────────────────

OPCODES = {
    'NOP', 'JCN', 'FIM', 'SRC', 'FIN', 'JIN', 'JUN', 'JMS',
    'INC', 'ISZ', 'ADD', 'SUB', 'LD',  'XCH', 'BBL', 'LDM',
    'WRM', 'WMP', 'WRR', 'WPM', 'WR0', 'WR1', 'WR2', 'WR3',
    'SBM', 'RDM', 'RDR', 'ADM', 'RD0', 'RD1', 'RD2', 'RD3',
    'CLB', 'CLC', 'IAC', 'CMC', 'CMA', 'RAL', 'RAR', 'TCC',
    'DAC', 'TCS', 'STC', 'DAA', 'KBP', 'DCL',
}

# 2-byte (2-word) instructions
TWO_WORD = {'JCN', 'FIM', 'JUN', 'JMS', 'ISZ'}

PAGE_SIZE = 256  # 4004 ROM page = 256 bytes


def split_label_and_op(line):
    """Return (label_or_None, remaining_parts_after_label).

    A label is the first whitespace-separated token if it is NOT an opcode.
    """
    parts = re.split(r'\s+', line.strip())
    if not parts or not parts[0]:
        return None, []
    if parts[0].upper() in OPCODES:
        return None, parts
    return parts[0], parts[1:]


def instruction_size(line):
    """Return size in bytes for this line: 0 (blank/comment/label-only), 1, or 2."""
    # Strip comment
    if ';' in line:
        line = line[:line.index(';')]
    line = line.strip()
    if not line:
        return 0
    _, rest = split_label_and_op(line)
    if not rest:
        return 0
    op = rest[0].upper()
    if op in TWO_WORD:
        return 2
    if op in OPCODES:
        return 1
    return 0


def compute_line_addresses(text):
    """For each line, return the starting byte address it occupies."""
    addrs = []
    addr = 0
    for line in text.split('\n'):
        addrs.append(addr)
        addr += instruction_size(line)
    return addrs


# ── Colors ────────────────────────────────────────────────────────────────

COL_OPCODE = QColor('#16a34a')    # green
COL_DATA = QColor('#2563eb')      # blue (numbers / data / register addresses)
COL_LABEL_REF = QColor('#16a34a') # green (label ref = address operand)
COL_LABEL_DEF = QColor('#111827') # near-black, bold (label definitions)
COL_COMMENT = QColor('#6b7280')   # gray italic
COL_BG = QColor('#ffffff')
COL_LINE_AREA_BG = QColor('#f3f4f6')
COL_PAGE_BAR = QColor('#dc2626')  # bright red horizontal divider


# ── Syntax highlighter ────────────────────────────────────────────────────

NUM_RE = re.compile(r'^(0x[0-9A-Fa-f]+|\d+)$')


class AsmHighlighter(QSyntaxHighlighter):
    def __init__(self, parent):
        super().__init__(parent)
        self.fmt_opcode = QTextCharFormat()
        self.fmt_opcode.setForeground(COL_OPCODE)
        self.fmt_opcode.setFontWeight(QFont.Bold)

        self.fmt_data = QTextCharFormat()
        self.fmt_data.setForeground(COL_DATA)

        self.fmt_label_def = QTextCharFormat()
        self.fmt_label_def.setForeground(COL_LABEL_DEF)
        self.fmt_label_def.setFontWeight(QFont.Bold)

        self.fmt_label_ref = QTextCharFormat()
        self.fmt_label_ref.setForeground(COL_LABEL_REF)

        self.fmt_comment = QTextCharFormat()
        self.fmt_comment.setForeground(COL_COMMENT)
        self.fmt_comment.setFontItalic(True)

    def highlightBlock(self, text):
        # Strip comment
        if ';' in text:
            idx = text.index(';')
            self.setFormat(idx, len(text) - idx, self.fmt_comment)
            code = text[:idx]
        else:
            code = text

        # Tokenize (preserve column positions)
        tokens = list(re.finditer(r'\S+', code))
        if not tokens:
            return

        # Find first opcode
        opcode_idx = None
        for i, m in enumerate(tokens):
            if m.group().upper() in OPCODES:
                opcode_idx = i
                break

        if opcode_idx is None:
            # No opcode — line is a label-only line
            m = tokens[0]
            self.setFormat(m.start(), m.end() - m.start(), self.fmt_label_def)
            return

        # Tokens before opcode = labels
        for m in tokens[:opcode_idx]:
            self.setFormat(m.start(), m.end() - m.start(), self.fmt_label_def)

        # Opcode
        m = tokens[opcode_idx]
        self.setFormat(m.start(), m.end() - m.start(), self.fmt_opcode)

        # Operands: numbers = blue, symbols = green (label refs / addresses)
        for m in tokens[opcode_idx + 1:]:
            tok = m.group()
            if NUM_RE.match(tok):
                self.setFormat(m.start(), m.end() - m.start(), self.fmt_data)
            else:
                self.setFormat(m.start(), m.end() - m.start(), self.fmt_label_ref)


# ── Code editor with line numbers and page dividers ──────────────────────

class LineNumberArea(QWidget):
    def __init__(self, editor):
        super().__init__(editor)
        self.editor = editor

    def sizeHint(self):
        return QSize(self.editor.line_number_area_width(), 0)

    def paintEvent(self, event):
        self.editor.line_number_area_paint_event(event)


class CodeEditor(QPlainTextEdit):
    def __init__(self):
        super().__init__()
        font = QFont('Consolas', 11)
        font.setStyleHint(QFont.Monospace)
        self.setFont(font)
        self.setTabStopDistance(self.fontMetrics().horizontalAdvance(' ') * 4)
        self.setStyleSheet(
            f'QPlainTextEdit {{ background: {COL_BG.name()}; color: black; '
            f'selection-background-color: #93c5fd; }}'
        )

        self.line_number_area = LineNumberArea(self)
        self.blockCountChanged.connect(self.update_line_number_area_width)
        self.updateRequest.connect(self.update_line_number_area)
        self.update_line_number_area_width(0)
        self.highlighter = AsmHighlighter(self.document())

    def line_number_area_width(self):
        digits = max(3, len(str(max(1, self.blockCount()))))
        # Width: " 123  0x1FF " style
        adv = self.fontMetrics().horizontalAdvance('9')
        return 12 + adv * (digits + 6)

    def update_line_number_area_width(self, _):
        self.setViewportMargins(self.line_number_area_width(), 0, 0, 0)

    def update_line_number_area(self, rect, dy):
        if dy:
            self.line_number_area.scroll(0, dy)
        else:
            self.line_number_area.update(
                0, rect.y(), self.line_number_area.width(), rect.height()
            )
        if rect.contains(self.viewport().rect()):
            self.update_line_number_area_width(0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        cr = self.contentsRect()
        self.line_number_area.setGeometry(QRect(
            cr.left(), cr.top(),
            self.line_number_area_width(), cr.height()
        ))

    def line_number_area_paint_event(self, event):
        painter = QPainter(self.line_number_area)
        painter.fillRect(event.rect(), COL_LINE_AREA_BG)

        # Compute byte addresses for each line
        addresses = compute_line_addresses(self.toPlainText())

        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        offset = self.contentOffset()
        top = self.blockBoundingGeometry(block).translated(offset).top()
        bottom = top + self.blockBoundingRect(block).height()

        # Page of the line just before the first visible block (for divider logic)
        prev_page = (addresses[block_number - 1] // PAGE_SIZE) if block_number > 0 else -1

        while block.isValid() and top <= event.rect().bottom():
            if block.isVisible() and bottom >= event.rect().top():
                line_num = block_number + 1
                addr = addresses[block_number] if block_number < len(addresses) else 0
                page = addr // PAGE_SIZE

                # Draw page divider above this line if entering a new page
                if page != prev_page and block_number > 0:
                    painter.setPen(COL_PAGE_BAR)
                    painter.drawLine(0, int(top),
                                     self.line_number_area.width(), int(top))
                prev_page = page

                # Draw text: "  N   0xAAA"
                painter.setPen(QColor('#6b7280'))
                txt = f'{line_num:>4d}  0x{addr:03X}'
                painter.drawText(
                    0, int(top),
                    self.line_number_area.width() - 8,
                    self.fontMetrics().height(),
                    Qt.AlignRight, txt
                )

            block = block.next()
            top = bottom
            bottom = top + self.blockBoundingRect(block).height()
            block_number += 1

    def paintEvent(self, event):
        super().paintEvent(event)

        # Draw page-divider line across the editor viewport too
        addresses = compute_line_addresses(self.toPlainText())

        painter = QPainter(self.viewport())
        block = self.firstVisibleBlock()
        block_number = block.blockNumber()
        offset = self.contentOffset()
        top = self.blockBoundingGeometry(block).translated(offset).top()

        prev_page = (addresses[block_number - 1] // PAGE_SIZE) if block_number > 0 else -1

        while block.isValid() and top < self.viewport().rect().bottom():
            if block.isVisible() and block_number < len(addresses):
                page = addresses[block_number] // PAGE_SIZE
                if page != prev_page and block_number > 0:
                    pen = painter.pen()
                    painter.setPen(COL_PAGE_BAR)
                    y = int(top)
                    painter.drawLine(0, y, self.viewport().width(), y)
                    # Annotation "Page N" at right edge
                    label = f'  Page {page}'
                    painter.drawText(
                        self.viewport().width() - 80, y + self.fontMetrics().height() - 2,
                        label
                    )
                    painter.setPen(pen)
                prev_page = page

            block = block.next()
            block_number += 1
            top = self.blockBoundingGeometry(block).translated(offset).top()


# ── Main window ──────────────────────────────────────────────────────────

class AsmIDE(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('4004 ASM IDE')
        self.setMinimumSize(1000, 760)

        self.editor = CodeEditor()
        self.setCentralWidget(self.editor)

        self.current_file = None
        self._set_title()

        # Status bar shows instruction count + ROM size
        self.editor.textChanged.connect(self._update_status)

        # Menu
        menubar = self.menuBar()
        file_menu = menubar.addMenu('&File')
        build_menu = menubar.addMenu('&Build')

        actions = [
            ('&New',          'Ctrl+N', self.new_file, file_menu),
            ('&Open',         'Ctrl+O', self.open_file, file_menu),
            ('&Save',         'Ctrl+S', self.save_file, file_menu),
            ('Save &As',      'Ctrl+Shift+S', self.save_as_file, file_menu),
            ('&Build && Run', 'F5',     self.build_run, build_menu),
        ]
        self._actions = {}
        for name, shortcut, slot, menu in actions:
            act = QAction(name, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            menu.addAction(act)
            self._actions[name] = act

        # Toolbar
        tb = self.addToolBar('Main')
        tb.addAction(self._actions['&New'])
        tb.addAction(self._actions['&Open'])
        tb.addAction(self._actions['&Save'])
        tb.addSeparator()
        tb.addAction(self._actions['&Build && Run'])

        self._update_status()

    def _set_title(self):
        name = self.current_file if self.current_file else '(untitled)'
        self.setWindowTitle(f'4004 ASM IDE — {name}')

    def _update_status(self):
        text = self.editor.toPlainText()
        n_inst = sum(1 for line in text.split('\n') if instruction_size(line) > 0)
        rom_size = sum(instruction_size(line) for line in text.split('\n'))
        n_pages = (rom_size + PAGE_SIZE - 1) // PAGE_SIZE if rom_size else 0
        self.statusBar().showMessage(
            f'{n_inst} instructions  |  {rom_size} bytes  |  {n_pages} page(s)'
        )

    # ── File ops ──

    def new_file(self):
        self.editor.clear()
        self.current_file = None
        self._set_title()

    def open_file(self):
        default_dir = os.path.join(PROJECT_ROOT, 'cpus', '4004', 'programs')
        path, _ = QFileDialog.getOpenFileName(
            self, 'Open ASM file', default_dir,
            'ASM Files (*.asm);;All Files (*)'
        )
        if path:
            with open(path) as f:
                self.editor.setPlainText(f.read())
            self.current_file = path
            self._set_title()

    def save_file(self):
        if not self.current_file:
            return self.save_as_file()
        with open(self.current_file, 'w') as f:
            f.write(self.editor.toPlainText())
        self.statusBar().showMessage(f'Saved {self.current_file}', 3000)

    def save_as_file(self):
        default_dir = os.path.join(PROJECT_ROOT, 'cpus', '4004', 'programs')
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save ASM file', default_dir, 'ASM Files (*.asm)'
        )
        if path:
            if not path.endswith('.asm'):
                path += '.asm'
            self.current_file = path
            self._set_title()
            self.save_file()

    # ── Build ──

    def build_run(self):
        asm_text = self.editor.toPlainText().strip()
        if not asm_text:
            QMessageBox.warning(self, 'Empty', 'Nothing to build.')
            return

        # Default program name from current file
        if self.current_file:
            default_name = os.path.splitext(os.path.basename(self.current_file))[0]
        else:
            default_name = 'Untitled'

        name, ok = QInputDialog.getText(
            self, 'Build & Run', 'Program name:', text=default_name
        )
        if not ok or not name.strip():
            return
        name = name.strip()

        # Sanity check: assemble in-process first to catch errors before
        # writing files and launching the long pipeline
        try:
            sys.path.insert(0, os.path.dirname(__file__))
            from rom_emulator import assemble
            rom = assemble(asm_text)
        except Exception as exc:
            QMessageBox.critical(
                self, 'Assembly Error',
                f'Assembly failed:\n\n{exc}'
            )
            return

        # Estimate cycles needed: count instructions, then * safety factor for loops
        n_inst = sum(1 for line in asm_text.split('\n')
                     if instruction_size(line) > 0)
        # Default 500, but give more for larger programs
        cycles = max(500, n_inst * 4)

        # Ask for cycles
        cycles, ok = QInputDialog.getInt(
            self, 'Build & Run',
            f'Cycles to simulate ({n_inst} instructions in program):',
            cycles, 50, 10000
        )
        if not ok:
            return

        # Write the .asm to the program folder
        prog_dir = os.path.join(PROJECT_ROOT, 'cpus', '4004', 'programs', name)
        os.makedirs(prog_dir, exist_ok=True)
        asm_path = os.path.join(prog_dir, f'{name}.asm')
        with open(asm_path, 'w') as f:
            f.write(asm_text + ('\n' if not asm_text.endswith('\n') else ''))

        # If editing a different file, update current_file to the new location
        if self.current_file != asm_path:
            self.current_file = asm_path
            self._set_title()

        # Run build_and_run_4004 (handles PWLs, 4004.asc rebuild, .tran, .save, .options, LTSpice open)
        self.statusBar().showMessage(f'Building {name} ({cycles} cycles)...')
        QApplication.processEvents()

        try:
            result = subprocess.run(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(__file__), 'build_and_run_4004.py'),
                    name,
                    '--cycles', str(cycles),
                ],
                capture_output=True, text=True, cwd=PROJECT_ROOT, timeout=120
            )
        except subprocess.TimeoutExpired:
            QMessageBox.critical(self, 'Build Error', 'Build timed out (>2 min).')
            return

        if result.returncode != 0:
            QMessageBox.critical(
                self, 'Build Failed',
                f'STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}'
            )
            return

        # Show summary from build output
        summary_lines = [l for l in result.stdout.split('\n')
                         if 'Total:' in l or 'Patched .tran' in l
                         or 'Opening' in l or 'Appended' in l]
        msg = (
            f'Built {name}\n'
            f'  ROM: {len(rom)} bytes ({n_inst} instructions)\n'
            f'  Cycles: {cycles}\n\n' +
            '\n'.join(summary_lines)
        )
        self.statusBar().showMessage(f'Built {name}', 5000)
        QMessageBox.information(self, 'Build Successful', msg)


def main():
    app = QApplication(sys.argv)
    win = AsmIDE()

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        with open(sys.argv[1]) as f:
            win.editor.setPlainText(f.read())
        win.current_file = sys.argv[1]
        win._set_title()

    win.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
