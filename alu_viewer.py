"""Waveform viewer for ALU: gate-level event simulation (fast, no ODE needed)."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk

from alu import ALUBuilder
from simulator.netlist import Netlist
from simulator.precompute import CircuitParams, precompute_gate
from simulator.engine import SimulationEngine, Stimulus
from model import NChannelJFET, JFETCapacitance, GateType


V_HIGH, V_LOW = -0.8, -4.0
V_THRESHOLD = (V_HIGH + V_LOW) / 2
V_PAD = 0.4


def run_alu_tests():
    """Run a sequence of ALU operations and capture waveforms."""
    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    params = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)

    alu = ALUBuilder(n_bits=4, arithmetic=True, logic=[], shift=False,
                     flags=['zero', 'carry'])
    gates = alu.build()

    profiles = {}
    for gt in [GateType.INV, GateType.NAND2]:
        profiles[gt] = precompute_gate(gt, params, V_HIGH, V_LOW)

    # Test sequence: each operation held for one clock period (10us)
    T = 10e-6
    tests = [
        # (A, B, SUB, label)
        (3,  5,  False, "3+5=8"),
        (7,  1,  False, "7+1=8"),
        (15, 1,  False, "15+1=0 C"),
        (5,  3,  True,  "5-3=2"),
        (3,  5,  True,  "3-5=14"),
        (0,  0,  False, "0+0=0 Z"),
        (9,  6,  False, "9+6=15"),
        (1,  1,  False, "1+1=2"),
    ]

    n_tests = len(tests)
    t_settle = 2e-6  # initial settle
    t_end = t_settle + n_tests * T + 2e-6

    # Build stimulus: toggle HIGH at t=0 (force eval), then sequence values
    a_stims = [[] for _ in range(4)]
    b_stims = [[] for _ in range(4)]
    sub_times = []
    sub_vals = []

    # Initial toggle at t=0
    for bit in range(4):
        a_stims[bit].append((0, True))
        b_stims[bit].append((0, True))
    sub_times.append(0)
    sub_vals.append(True)

    # Test sequence
    for i, (a, b, sub, label) in enumerate(tests):
        t = t_settle + i * T
        for bit in range(4):
            a_stims[bit].append((t, bool((a >> bit) & 1)))
            b_stims[bit].append((t, bool((b >> bit) & 1)))
        sub_times.append(t)
        sub_vals.append(sub)

    # Build engine
    netlist = Netlist.from_gates(gates, primary_outputs=alu.outputs)
    engine = SimulationEngine(netlist, profiles, v_high=V_HIGH, v_low=V_LOW,
                              auto_precompute_params=params)

    for bit in range(4):
        times = [t for t, v in a_stims[bit]]
        vals = [v for t, v in a_stims[bit]]
        engine.add_stimulus(Stimulus(f'A_{bit}', times, vals))

        times = [t for t, v in b_stims[bit]]
        vals = [v for t, v in b_stims[bit]]
        engine.add_stimulus(Stimulus(f'B_{bit}', times, vals))

    engine.add_stimulus(Stimulus('SUB', sub_times, sub_vals))

    print("Running ALU gate-level simulation...")
    result = engine.run(t_end)
    print(f"  {result.events_processed} events processed")

    # Sample waveforms at regular intervals
    dt = 0.1e-6  # 100ns resolution
    t_samples = np.arange(0, t_end, dt)
    n_samples = len(t_samples)

    # Build traces from event history
    def build_trace(net_name):
        ns = result.net_states.get(net_name)
        if ns is None:
            return np.full(n_samples, V_LOW)
        trace = np.full(n_samples, V_LOW)
        # Replay history
        history = ns.history  # list of (time, value, voltage)
        if not history:
            trace[:] = ns.voltage
            return trace
        hi = 0
        for si in range(n_samples):
            t = t_samples[si]
            while hi < len(history) - 1 and history[hi + 1][0] <= t:
                hi += 1
            if hi < len(history):
                trace[si] = history[hi][2]  # voltage
        return trace

    # Build all traces
    traces = {}
    show_nets = []

    # Inputs: A as 4-bit value, B as 4-bit value
    for bit in range(4):
        traces[f'A_{bit}'] = build_trace(f'A_{bit}')
        traces[f'B_{bit}'] = build_trace(f'B_{bit}')
    traces['SUB'] = build_trace('SUB')

    # Outputs
    for bit in range(4):
        traces[f'OUT_{bit}'] = build_trace(f'OUT_{bit}')
    traces['CARRY'] = build_trace('CARRY_4')
    traces['ZERO'] = build_trace('FLAG_ZERO')

    # Composite: A value, B value, OUT value
    def decode_value(prefix, n_bits=4):
        val_trace = np.zeros(n_samples)
        for si in range(n_samples):
            v = 0
            for bit in range(n_bits):
                if traces[f'{prefix}_{bit}'][si] > V_THRESHOLD:
                    v |= (1 << bit)
            val_trace[si] = v
        return val_trace

    traces['A_val'] = decode_value('A')
    traces['B_val'] = decode_value('B')
    traces['OUT_val'] = decode_value('OUT')

    show_nets = ['A_val', 'B_val', 'SUB', 'OUT_val', 'CARRY', 'ZERO']

    return t_samples, traces, show_nets, tests, t_settle, T


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

COLORS = {
    'A_val': '#2196F3',
    'B_val': '#42A5F5',
    'SUB': '#FF9800',
    'OUT_val': '#4CAF50',
    'CARRY': '#E91E63',
    'ZERO': '#9C27B0',
}

LABELS = {
    'A_val': 'A',
    'B_val': 'B',
    'SUB': 'SUB',
    'OUT_val': 'Result',
    'CARRY': 'Carry',
    'ZERO': 'Zero',
}


class ALUViewer(tk.Tk):
    def __init__(self, t, traces, net_names, tests, t_settle, T):
        super().__init__()
        self.title("deepJFET — 4-bit ALU")
        self.geometry("1400x700")
        self.configure(bg='#1e1e1e')

        self.t = t
        self.traces = traces
        self.net_names = net_names
        self.tests = tests
        self.t_settle = t_settle
        self.T = T
        self.t_us = t * 1e6

        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])
        self.cursor_x = None

        self._build_ui()
        self._draw()

    def _build_ui(self):
        top = tk.Frame(self, bg='#2d2d2d', height=36)
        top.pack(fill=tk.X)
        self.info_label = tk.Label(top, text="", bg='#2d2d2d', fg='#cccccc',
                                   font=('Consolas', 11))
        self.info_label.pack(side=tk.LEFT, padx=10)
        self.op_label = tk.Label(top, text="", bg='#2d2d2d', fg='#FFD700',
                                  font=('Consolas', 12, 'bold'))
        self.op_label.pack(side=tk.RIGHT, padx=20)

        self.canvases = []
        self.canvas_frame = tk.Frame(self, bg='#1e1e1e')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for name in self.net_names:
            row = tk.Frame(self.canvas_frame, bg='#1e1e1e')
            row.pack(fill=tk.BOTH, expand=True, pady=1)

            col = COLORS.get(name, '#888')
            lab = LABELS.get(name, name)
            lbl = tk.Label(row, text=lab, bg='#1e1e1e', fg=col,
                           font=('Consolas', 10, 'bold'), width=8, anchor='e')
            lbl.pack(side=tk.LEFT)

            is_value = name.endswith('_val')
            h = 100 if is_value else 60
            c = tk.Canvas(row, bg='#0d0d0d', highlightthickness=0, height=h)
            c.pack(fill=tk.BOTH, expand=True, padx=2)
            c.bind('<Motion>', self._on_mouse_move)
            c.bind('<MouseWheel>', self._on_scroll)
            c.bind('<Configure>', lambda e: self._draw())
            self.canvases.append((name, c))

        self.bind('<Left>', lambda e: self._pan(-0.1))
        self.bind('<Right>', lambda e: self._pan(0.1))
        self.bind('<r>', lambda e: self._reset_view())

    def _t_to_x(self, t_us, c):
        w = c.winfo_width()
        if self.view_end == self.view_start: return 0
        return (t_us - self.view_start) / (self.view_end - self.view_start) * w

    def _x_to_t(self, x, c):
        w = c.winfo_width()
        if w == 0: return self.view_start
        return self.view_start + x / w * (self.view_end - self.view_start)

    def _draw(self):
        for name, canvas in self.canvases:
            canvas.delete('all')
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                continue

            trace = self.traces.get(name)
            if trace is None:
                continue

            is_value = name.endswith('_val')
            col = COLORS.get(name, '#888')

            # Test phase boundaries
            for i in range(len(self.tests) + 1):
                t_phase = (self.t_settle + i * self.T) * 1e6
                if self.view_start <= t_phase <= self.view_end:
                    cx = self._t_to_x(t_phase, canvas)
                    canvas.create_line(cx, 0, cx, h, fill='#333333', dash=(4, 4))

            if is_value:
                # Draw as stepped numeric display
                margin = 4
                mask = (self.t_us >= self.view_start) & (self.t_us <= self.view_end)
                t_vis = self.t_us[mask]
                v_vis = trace[mask]

                if len(t_vis) < 2:
                    continue

                # Draw value labels in each phase
                for i, (a, b, sub, label) in enumerate(self.tests):
                    t_start = (self.t_settle + i * self.T) * 1e6
                    t_end_phase = (self.t_settle + (i + 1) * self.T) * 1e6

                    if t_end_phase < self.view_start or t_start > self.view_end:
                        continue

                    x1 = max(self._t_to_x(t_start, canvas), 0)
                    x2 = min(self._t_to_x(t_end_phase, canvas), w)

                    if x2 - x1 < 15:
                        continue

                    # Get settled value (sample at 80% into the phase)
                    t_sample = self.t_settle + (i + 0.8) * self.T
                    idx = min(np.searchsorted(self.t, t_sample), len(trace) - 1)
                    val = int(trace[idx])

                    mid_x = (x1 + x2) / 2
                    canvas.create_text(mid_x, h / 2, text=str(val),
                                       fill=col, font=('Consolas', 14, 'bold'))

                    # Operation label on result row
                    if name == 'OUT_val':
                        canvas.create_text(mid_x, h - 12,
                                           text=label, fill='#666666',
                                           font=('Consolas', 8))
            else:
                # Binary signal: draw as waveform
                v_min = V_LOW - V_PAD
                v_max = V_HIGH + V_PAD

                mask = (self.t_us >= self.view_start) & (self.t_us <= self.view_end)
                t_vis = self.t_us[mask]
                v_vis = trace[mask]

                if len(t_vis) < 2:
                    continue

                step = max(1, len(t_vis) // (w * 2))
                t_ds = t_vis[::step]
                v_ds = v_vis[::step]

                points = []
                for ti, vi in zip(t_ds, v_ds):
                    x = self._t_to_x(ti, canvas)
                    frac = (vi - v_min) / (v_max - v_min)
                    y = h - 4 - frac * (h - 8)
                    points.extend([x, y])

                if len(points) >= 4:
                    canvas.create_line(*points, fill=col, width=2)

            # Cursor
            if self.cursor_x is not None:
                cx = self._t_to_x(self.cursor_x, canvas)
                canvas.create_line(cx, 0, cx, h, fill='#fff', width=1, dash=(3, 3))

    def _on_mouse_move(self, event):
        self.cursor_x = self._x_to_t(event.x, event.widget)

        idx = min(np.searchsorted(self.t_us, self.cursor_x), len(self.t) - 1)

        a_val = int(self.traces['A_val'][idx])
        b_val = int(self.traces['B_val'][idx])
        out_val = int(self.traces['OUT_val'][idx])
        sub = self.traces['SUB'][idx] > V_THRESHOLD
        carry = self.traces['CARRY'][idx] > V_THRESHOLD
        zero = self.traces['ZERO'][idx] > V_THRESHOLD

        op = '-' if sub else '+'
        self.info_label.config(
            text=f"t={self.cursor_x:.1f}us  |  A={a_val}  {op}  B={b_val}  =  {out_val}"
                 f"  |  C={int(carry)}  Z={int(zero)}")

        # Find which test phase we're in
        t_sec = self.cursor_x * 1e-6
        phase_idx = int((t_sec - self.t_settle) / self.T)
        if 0 <= phase_idx < len(self.tests):
            self.op_label.config(text=self.tests[phase_idx][3])
        else:
            self.op_label.config(text="")

        self._draw()

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.18
        center = self._x_to_t(event.x, event.widget)
        span = (self.view_end - self.view_start) * factor
        span = max(span, 0.5)
        span = min(span, float(self.t_us[-1] - self.t_us[0]))
        self.view_start = max(center - span / 2, float(self.t_us[0]))
        self.view_end = min(center + span / 2, float(self.t_us[-1]))
        self._draw()

    def _pan(self, frac):
        span = self.view_end - self.view_start
        shift = span * frac
        self.view_start = max(self.view_start + shift, float(self.t_us[0]))
        self.view_end = min(self.view_end + shift, float(self.t_us[-1]))
        self._draw()

    def _reset_view(self):
        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])
        self._draw()


if __name__ == '__main__':
    t, traces, nets, tests, t_settle, T = run_alu_tests()
    app = ALUViewer(t, traces, nets, tests, t_settle, T)
    app.mainloop()
