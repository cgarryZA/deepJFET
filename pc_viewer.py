"""Waveform viewer for Program Counter."""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk

from program_counter import make_program_counter
from simulator.netlist import Netlist
from simulator.precompute import CircuitParams, precompute_gate
from simulator.engine import SimulationEngine, Stimulus
from model import NChannelJFET, JFETCapacitance, GateType


V_HIGH, V_LOW = -0.8, -4.0
V_THRESHOLD = (V_HIGH + V_LOW) / 2
V_PAD = 0.4


def run_pc():
    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12)
    params = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)

    N = 4
    T = 10e-6
    t_settle = 2e-6

    gates, outs, ctrl, load_ins = make_program_counter("PC", N)

    profiles = {}
    for gt in [GateType.INV, GateType.NAND2]:
        profiles[gt] = precompute_gate(gt, params, V_HIGH, V_LOW)

    # Test sequence:
    # Cycles 0-7:  INC (count 0 to 7)
    # Cycle 8:     LOAD value 12
    # Cycles 9-13: INC (count 13 to 15, then wrap to 0, 1)
    # Cycle 14:    LOAD value 0
    # Cycles 15-18: INC (count 1 to 4)

    n_cycles = 20
    t_end = t_settle + (n_cycles + 1) * T

    netlist = Netlist.from_gates(gates, primary_outputs=set(outs))
    engine = SimulationEngine(netlist, profiles, v_high=V_HIGH, v_low=V_LOW,
                              auto_precompute_params=params)

    # CLK
    clk_times = [0]
    clk_vals = [True]
    for i in range(n_cycles + 2):
        t = t_settle + i * T
        clk_times.extend([t, t + T / 2])
        clk_vals.extend([True, False])
    engine.add_stimulus(Stimulus('CLK', clk_times, clk_vals))

    # INC: high during CLK-low phase, except during LOAD cycles
    load_cycles = {8, 14}
    inc_times = [0]
    inc_vals = [True]
    for i in range(n_cycles + 2):
        t = t_settle + i * T
        if i in load_cycles:
            inc_times.extend([t, t + T - 0.1e-6])
            inc_vals.extend([False, False])
        else:
            inc_times.extend([t, t + T / 2 - 0.1e-6, t + T / 2, t + T - 0.1e-6])
            inc_vals.extend([False, False, True, True])
    engine.add_stimulus(Stimulus('INC', inc_times, inc_vals))

    # LOAD: pulse during load cycles (during CLK-high phase)
    load_times = [0]
    load_vals = [True]
    for i in range(n_cycles + 2):
        t = t_settle + i * T
        if i in load_cycles:
            load_times.extend([t, t + T / 2 - 0.1e-6])
            load_vals.extend([True, True])
        else:
            load_times.extend([t, t + T / 2 - 0.1e-6])
            load_vals.extend([False, False])
    engine.add_stimulus(Stimulus('LOAD', load_times, load_vals))

    # PC_Enable: always high
    engine.add_stimulus(Stimulus('PC_Enable', [0, t_settle], [True, True]))

    # Load inputs: set to 12 (1100) for cycle 8, 0 (0000) for cycle 14
    load_values = {8: 12, 14: 0}
    for bit in range(N):
        lt = [0]
        lv = [True]
        for i in range(n_cycles + 2):
            t = t_settle + i * T
            if i in load_values:
                val = bool((load_values[i] >> bit) & 1)
            else:
                val = False
            lt.extend([t, t + T - 0.1e-6])
            lv.extend([val, val])
        engine.add_stimulus(Stimulus(f'PC_Load_{bit}', lt, lv))

    print("Running 4-bit PC simulation...")
    result = engine.run(t_end)
    print(f"  {result.events_processed} events")

    # Sample waveforms
    dt = 0.1e-6
    t_samples = np.arange(0, t_end, dt)
    n_samples = len(t_samples)

    def build_trace(net_name):
        ns = result.net_states.get(net_name)
        if ns is None:
            return np.full(n_samples, V_LOW)
        trace = np.full(n_samples, V_LOW)
        if not ns.history:
            trace[:] = ns.voltage
            return trace
        hi = 0
        for si in range(n_samples):
            t = t_samples[si]
            while hi < len(ns.history) - 1 and ns.history[hi + 1][0] <= t:
                hi += 1
            if hi < len(ns.history):
                trace[si] = ns.history[hi][2]
        return trace

    traces = {}
    traces['CLK'] = build_trace('CLK')
    traces['INC'] = build_trace('INC')
    traces['LOAD'] = build_trace('LOAD')
    for bit in range(N):
        traces[f'PC_{bit}'] = build_trace(f'PC_{bit}')

    # Compute decimal value
    pc_val = np.zeros(n_samples)
    for si in range(n_samples):
        v = 0
        for bit in range(N):
            if traces[f'PC_{bit}'][si] > V_THRESHOLD:
                v |= (1 << bit)
        pc_val[si] = v
    traces['PC_val'] = pc_val

    show_nets = ['CLK', 'INC', 'LOAD', 'PC_val']
    return t_samples, traces, show_nets, N, T, t_settle, n_cycles


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

COLORS = {
    'CLK': '#9E9E9E',
    'INC': '#FF9800',
    'LOAD': '#E91E63',
    'PC_val': '#4CAF50',
}


class PCViewer(tk.Tk):
    def __init__(self, t, traces, net_names, n_bits, T, t_settle, n_cycles):
        super().__init__()
        self.title("deepJFET — 4-bit Program Counter")
        self.geometry("1400x600")
        self.configure(bg='#1e1e1e')

        self.t = t
        self.traces = traces
        self.net_names = net_names
        self.n_bits = n_bits
        self.T = T
        self.t_settle = t_settle
        self.n_cycles = n_cycles
        self.t_us = t * 1e6

        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])
        self.cursor_x = None

        self._build_ui()
        self._draw()

    def _build_ui(self):
        top = tk.Frame(self, bg='#2d2d2d', height=40)
        top.pack(fill=tk.X)
        self.info_label = tk.Label(top, text="", bg='#2d2d2d', fg='#cccccc',
                                   font=('Consolas', 11))
        self.info_label.pack(side=tk.LEFT, padx=10)
        self.pc_label = tk.Label(top, text="", bg='#2d2d2d', fg='#FFD700',
                                 font=('Consolas', 14, 'bold'))
        self.pc_label.pack(side=tk.RIGHT, padx=20)

        self.canvases = []
        self.canvas_frame = tk.Frame(self, bg='#1e1e1e')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        heights = {'CLK': 50, 'INC': 50, 'LOAD': 50, 'PC_val': 150}
        labels = {'CLK': 'CLK', 'INC': 'INC', 'LOAD': 'LOAD', 'PC_val': 'PC'}

        for name in self.net_names:
            row = tk.Frame(self.canvas_frame, bg='#1e1e1e')
            row.pack(fill=tk.BOTH, expand=True, pady=1)

            col = COLORS.get(name, '#888')
            lbl = tk.Label(row, text=labels.get(name, name), bg='#1e1e1e', fg=col,
                           font=('Consolas', 10, 'bold'), width=6, anchor='e')
            lbl.pack(side=tk.LEFT)

            c = tk.Canvas(row, bg='#0d0d0d', highlightthickness=0,
                          height=heights.get(name, 80))
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

            col = COLORS.get(name, '#888')

            # Clock cycle boundaries
            for i in range(self.n_cycles + 2):
                t_phase = (self.t_settle + i * self.T) * 1e6
                if self.view_start <= t_phase <= self.view_end:
                    cx = self._t_to_x(t_phase, canvas)
                    canvas.create_line(cx, 0, cx, h, fill='#333333', dash=(4, 4))

            if name == 'PC_val':
                # Draw decimal value in each cycle
                for i in range(self.n_cycles + 1):
                    t_start = (self.t_settle + i * self.T) * 1e6
                    t_end_c = (self.t_settle + (i + 1) * self.T) * 1e6
                    if t_end_c < self.view_start or t_start > self.view_end:
                        continue

                    x1 = max(self._t_to_x(t_start, canvas), 0)
                    x2 = min(self._t_to_x(t_end_c, canvas), w)
                    if x2 - x1 < 15:
                        continue

                    t_sample = self.t_settle + (i + 0.8) * self.T
                    idx = min(np.searchsorted(self.t, t_sample), len(trace) - 1)
                    val = int(trace[idx])

                    mid_x = (x1 + x2) / 2
                    canvas.create_text(mid_x, h / 2, text=str(val),
                                       fill=col, font=('Consolas', 16, 'bold'))
                    bin_str = format(val, f'0{self.n_bits}b')
                    canvas.create_text(mid_x, h / 2 + 20, text=f'0b{bin_str}',
                                       fill='#666666', font=('Consolas', 9))
            else:
                # Binary signal waveform
                v_min = V_LOW - V_PAD
                v_max = V_HIGH + V_PAD
                mask = (self.t_us >= self.view_start) & (self.t_us <= self.view_end)
                t_vis = self.t_us[mask]
                v_vis = trace[mask]
                if len(t_vis) < 2:
                    continue
                step = max(1, len(t_vis) // (w * 2))
                points = []
                for ti, vi in zip(t_vis[::step], v_vis[::step]):
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

        pc_val = int(self.traces['PC_val'][idx])
        inc = self.traces['INC'][idx] > V_THRESHOLD
        load = self.traces['LOAD'][idx] > V_THRESHOLD
        clk = self.traces['CLK'][idx] > V_THRESHOLD

        bin_str = format(pc_val, f'0{self.n_bits}b')
        self.info_label.config(
            text=f"t={self.cursor_x:.1f}us  |  CLK={'H' if clk else 'L'}  "
                 f"INC={'H' if inc else 'L'}  LOAD={'H' if load else 'L'}")
        self.pc_label.config(text=f"PC = {pc_val} (0b{bin_str})")
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
    t, traces, nets, n_bits, T, t_settle, n_cycles = run_pc()
    app = PCViewer(t, traces, nets, n_bits, T, t_settle, n_cycles)
    app.mainloop()
