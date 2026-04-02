"""Simulate a simple program on the JFET CPU.

Program:
  loop:
    IAC              ; ACC = ACC + 1, set carry on overflow
    JCN C, overflow  ; if carry, jump to overflow handler
    JUN loop         ; else keep counting
  overflow:
    XCH R0           ; swap ACC <-> scratchpad R0
    IAC              ; increment (old R0 value now in ACC)
    XCH R0           ; swap back (incremented R0, ACC=0)
    JUN loop         ; continue

Expected behavior:
  ACC: 0,1,2,3,...,14,15,0  (then R0 goes 0->1)
  ACC: 0,1,2,3,...,14,15,0  (then R0 goes 1->2)
  etc.

Since we don't have a full CPU sequencer, we simulate at the
behavioral level using the gate-level ALU and registers, stepping
through instructions manually.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk


def run_cpu_program():
    """Execute the counting program behaviorally, recording state per cycle."""

    # CPU state
    acc = 0
    carry = 0
    scratch = 0  # R0

    # Program (simplified — each entry is one instruction)
    # We manually execute the control flow
    history = []  # list of (cycle, acc, carry, scratch, instruction)

    cycle = 0
    max_cycles = 80

    while cycle < max_cycles:
        # IAC: ACC = ACC + 1
        result = acc + 1
        carry = 1 if result > 15 else 0
        acc = result & 0xF
        history.append((cycle, acc, carry, scratch, "IAC"))
        cycle += 1

        if carry:
            # Overflow: handle scratchpad
            # XCH R0: swap ACC <-> R0
            acc, scratch = scratch, acc
            history.append((cycle, acc, carry, scratch, "XCH R0"))
            cycle += 1

            # IAC: increment (old R0 value, now in ACC)
            result = acc + 1
            carry = 1 if result > 15 else 0
            acc = result & 0xF
            history.append((cycle, acc, carry, scratch, "IAC"))
            cycle += 1

            # XCH R0: swap back
            acc, scratch = scratch, acc
            history.append((cycle, acc, carry, scratch, "XCH R0"))
            cycle += 1

    return history


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

V_HIGH, V_LOW = -0.8, -4.0
V_THRESHOLD = (V_HIGH + V_LOW) / 2


class CPUViewer(tk.Tk):
    def __init__(self, history):
        super().__init__()
        self.title("deepJFET CPU — Counting Program")
        self.geometry("1400x700")
        self.configure(bg='#1e1e1e')

        self.history = history
        self.n_cycles = len(history)
        self.cursor_idx = 0

        self._build_ui()
        self._draw()

    def _build_ui(self):
        # Top info bar
        top = tk.Frame(self, bg='#2d2d2d', height=40)
        top.pack(fill=tk.X)
        self.info_label = tk.Label(top, text="", bg='#2d2d2d', fg='#cccccc',
                                   font=('Consolas', 11))
        self.info_label.pack(side=tk.LEFT, padx=10)
        self.state_label = tk.Label(top, text="", bg='#2d2d2d', fg='#FFD700',
                                    font=('Consolas', 13, 'bold'))
        self.state_label.pack(side=tk.RIGHT, padx=20)

        # Canvas area
        self.canvas_frame = tk.Frame(self, bg='#1e1e1e')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        channels = [
            ('ACC', '#4CAF50', 160),
            ('Carry', '#E91E63', 50),
            ('R0 (Scratch)', '#2196F3', 100),
            ('Instruction', '#FF9800', 60),
        ]

        self.canvases = []
        for name, color, height in channels:
            row = tk.Frame(self.canvas_frame, bg='#1e1e1e')
            row.pack(fill=tk.BOTH, expand=True, pady=1)

            lbl = tk.Label(row, text=name, bg='#1e1e1e', fg=color,
                           font=('Consolas', 10, 'bold'), width=12, anchor='e')
            lbl.pack(side=tk.LEFT)

            c = tk.Canvas(row, bg='#0d0d0d', highlightthickness=0, height=height)
            c.pack(fill=tk.BOTH, expand=True, padx=2)
            c.bind('<Motion>', self._on_mouse_move)
            c.bind('<MouseWheel>', self._on_scroll)
            c.bind('<Configure>', lambda e: self._draw())
            self.canvases.append((name, color, c))

        self.view_start = 0
        self.view_end = self.n_cycles

        self.bind('<Left>', lambda e: self._pan(-0.1))
        self.bind('<Right>', lambda e: self._pan(0.1))
        self.bind('<r>', lambda e: self._reset_view())

    def _idx_to_x(self, idx, c):
        w = c.winfo_width()
        span = self.view_end - self.view_start
        if span == 0: return 0
        return (idx - self.view_start) / span * w

    def _x_to_idx(self, x, c):
        w = c.winfo_width()
        if w == 0: return self.view_start
        return self.view_start + x / w * (self.view_end - self.view_start)

    def _draw(self):
        for name, color, canvas in self.canvases:
            canvas.delete('all')
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                continue

            margin = 6

            # Cycle boundaries
            for i in range(int(self.view_start), int(self.view_end) + 1):
                x = self._idx_to_x(i, canvas)
                if 0 <= x <= w:
                    canvas.create_line(x, 0, x, h, fill='#222222')

            if name == 'ACC':
                self._draw_value_channel(canvas, w, h, margin, color,
                                          lambda e: e[1], 0, 15)
            elif name == 'Carry':
                self._draw_binary_channel(canvas, w, h, margin, color,
                                           lambda e: e[2])
            elif name == 'R0 (Scratch)':
                self._draw_value_channel(canvas, w, h, margin, color,
                                          lambda e: e[3], 0, 15)
            elif name == 'Instruction':
                self._draw_instruction_channel(canvas, w, h, color)

            # Cursor
            if self.cursor_idx is not None:
                cx = self._idx_to_x(self.cursor_idx, canvas)
                canvas.create_line(cx, 0, cx, h, fill='#fff', width=1, dash=(3, 3))

    def _draw_value_channel(self, canvas, w, h, margin, color, get_val,
                             v_min, v_max):
        """Draw a numeric value channel with stepped display."""
        vis_start = max(0, int(self.view_start))
        vis_end = min(self.n_cycles, int(self.view_end) + 1)

        for i in range(vis_start, vis_end):
            x1 = self._idx_to_x(i, canvas)
            x2 = self._idx_to_x(i + 1, canvas)
            if x2 < 0 or x1 > w:
                continue

            val = get_val(self.history[i])
            frac = val / max(v_max, 1)
            bar_h = frac * (h - 2 * margin)
            y_top = h - margin - bar_h

            # Bar
            canvas.create_rectangle(x1 + 1, y_top, x2 - 1, h - margin,
                                    fill=color, outline='', stipple='')
            # Dimmer fill
            canvas.create_rectangle(x1 + 1, y_top, x2 - 1, h - margin,
                                    fill='', outline=color)

            # Value text if wide enough
            if x2 - x1 > 18:
                canvas.create_text((x1 + x2) / 2, h / 2, text=str(val),
                                   fill='white', font=('Consolas', 10, 'bold'))

    def _draw_binary_channel(self, canvas, w, h, margin, color, get_val):
        vis_start = max(0, int(self.view_start))
        vis_end = min(self.n_cycles, int(self.view_end) + 1)

        for i in range(vis_start, vis_end):
            x1 = self._idx_to_x(i, canvas)
            x2 = self._idx_to_x(i + 1, canvas)
            val = get_val(self.history[i])
            y = margin if val else h - margin
            if i > vis_start:
                prev_val = get_val(self.history[i - 1])
                prev_y = margin if prev_val else h - margin
                canvas.create_line(x1, prev_y, x1, y, fill=color, width=2)
            canvas.create_line(x1, y, x2, y, fill=color, width=2)

    def _draw_instruction_channel(self, canvas, w, h, color):
        vis_start = max(0, int(self.view_start))
        vis_end = min(self.n_cycles, int(self.view_end) + 1)

        for i in range(vis_start, vis_end):
            x1 = self._idx_to_x(i, canvas)
            x2 = self._idx_to_x(i + 1, canvas)
            inst = self.history[i][4]
            if x2 - x1 > 20:
                canvas.create_text((x1 + x2) / 2, h / 2, text=inst,
                                   fill=color, font=('Consolas', 9, 'bold'))

    def _on_mouse_move(self, event):
        idx = self._x_to_idx(event.x, event.widget)
        self.cursor_idx = idx
        int_idx = max(0, min(int(idx), self.n_cycles - 1))
        entry = self.history[int_idx]
        cycle, acc, carry, scratch, inst = entry

        self.info_label.config(
            text=f"Cycle {cycle}  |  {inst}  |  "
                 f"ACC={acc} (0b{acc:04b})  C={carry}  R0={scratch}")
        self.state_label.config(
            text=f"ACC={acc}  R0={scratch}  C={carry}")
        self._draw()

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.18
        center = self._x_to_idx(event.x, event.widget)
        span = (self.view_end - self.view_start) * factor
        span = max(span, 4)
        span = min(span, self.n_cycles)
        self.view_start = max(center - span / 2, 0)
        self.view_end = min(center + span / 2, self.n_cycles)
        self._draw()

    def _pan(self, frac):
        span = self.view_end - self.view_start
        shift = span * frac
        self.view_start = max(self.view_start + shift, 0)
        self.view_end = min(self.view_end + shift, self.n_cycles)
        self._draw()

    def _reset_view(self):
        self.view_start = 0
        self.view_end = self.n_cycles
        self._draw()


if __name__ == '__main__':
    history = run_cpu_program()

    # Print first 30 cycles
    print(f"{'Cyc':>4} {'ACC':>4} {'C':>2} {'R0':>3} {'Inst':>8}")
    for cycle, acc, carry, scratch, inst in history[:40]:
        print(f"{cycle:>4} {acc:>4} {carry:>2} {scratch:>3} {inst:>8}")

    print(f"\n... {len(history)} total cycles")

    app = CPUViewer(history)
    app.mainloop()
