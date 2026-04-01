"""Interactive waveform viewer for multi-gate transient simulations.

Shows inputs and outputs of a circuit with zoom, pan, and cursor.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk
from tkinter import ttk

from model import NChannelJFET, JFETCapacitance, GateType
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams
from transient.engine.multi_gate import MultiGateCircuit, simulate_multi, compute_multi_ic, multi_gate_ode
from model.network import gate_type_to_network
from scipy.integrate import solve_ivp


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_d_latch():
    """Run D latch simulation and return results."""
    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12, pb=1.0, m=0.407, fc=0.5)

    gates = [
        Gate('inv_d',   GateType.INV,   ['D'],              'D_bar'),
        Gate('nand_s',  GateType.NAND2, ['D', 'EN'],        'S_bar'),
        Gate('nand_r',  GateType.NAND2, ['D_bar', 'EN'],    'R_bar'),
        Gate('nand_q',  GateType.NAND2, ['S_bar', 'Q_bar'], 'Q'),
        Gate('nand_qb', GateType.NAND2, ['R_bar', 'Q'],     'Q_bar'),
    ]
    netlist = Netlist.from_gates(gates, primary_outputs={'Q', 'Q_bar'})

    params_inv = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                               jfet=jfet, caps=caps)
    params_nand = CircuitParams(v_pos=10, v_neg=-10, r1=24300, r2=7320, r3=6980,
                                jfet=jfet, caps=caps)
    gate_params = {
        'inv_d': params_inv,
        'nand_s': params_nand, 'nand_r': params_nand,
        'nand_q': params_nand, 'nand_qb': params_nand,
    }
    gate_networks = {g.name: gate_type_to_network(g.gate_type) for g in gates}

    V_HIGH, V_LOW = -0.8, -4.0
    t_settle = 5e-6
    T = 10e-6  # clock period = 100kHz, high for 5us then low for 5us

    # All signals are clock-aligned: each is HIGH for exactly 5us (half period)
    # then LOW for 5us, transitions only on clock edges.
    #
    # Clock cycle:  0    1    2    3    4    5    6    7    8    9   10   11   12   13
    # Half-period:  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |  8 |  9 | 10 | 11 | 12 | 13 |
    #
    # Test sequence (value during each half-period):
    # Cycle  EN  D   Expected Q behavior
    #  0     H   L   Q -> L (transparent, D=L)
    #  1     H   H   Q -> H (transparent, D=H)
    #  2     H   L   Q -> L (transparent, D=L)
    #  3     H   H   Q -> H (transparent, D=H)
    #  4     L   H   Q holds H (latched)
    #  5     L   L   Q holds H (D changes ignored)
    #  6     L   H   Q holds H (D changes ignored)
    #  7     L   L   Q holds H (D changes ignored)
    #  8     H   L   Q -> L (re-enabled, D=L)
    #  9     H   H   Q -> H (transparent)
    # 10     L   L   Q holds H (latched)
    # 11     L   H   Q holds H (ignored)
    # 12     H   L   Q -> L (re-enabled, D=L)
    # 13     L   L   Q holds L (latched with L)

    en_pattern = [1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0]
    d_pattern  = [0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0]
    n_cycles = len(en_pattern)
    half_T = T / 2  # 5us

    def make_clock_stim(pattern):
        def fn(t):
            t_adj = t - t_settle
            if t_adj < 0:
                return V_LOW
            idx = int(t_adj / half_T)
            if idx >= len(pattern):
                return V_LOW
            return V_HIGH if pattern[idx] else V_LOW
        return fn

    v_d = make_clock_stim(d_pattern)
    v_en = make_clock_stim(en_pattern)

    circuit = MultiGateCircuit(
        netlist=netlist, gate_params=gate_params, gate_networks=gate_networks,
        stimuli={'D': v_d, 'EN': v_en},
        jfet=jfet, caps=caps, temp_c=27.0,
    )

    y0 = compute_multi_ic(circuit)
    for gname in circuit.gate_order:
        start, n = circuit.state_map[gname]
        v_a = y0[start]
        for k in range(3, n):
            if abs(y0[start + k] - 5.0) < 0.01:
                y0[start + k] = v_a * 0.3
    y0 = np.clip(y0, -15, 15)

    t_end = t_settle + n_cycles * half_T + 5e-6  # extra settle at end
    t_eval = np.linspace(0, t_end, 8000)

    def rhs(t, y):
        y_clamped = np.clip(y, -20, 20)
        return multi_gate_ode(t, y_clamped, circuit)

    print("Running D latch simulation...")
    sol = solve_ivp(rhs, (0, t_end), y0, method='LSODA',
                    t_eval=t_eval, max_step=0.5e-6, rtol=1e-3, atol=1e-6)
    print(f"  Done: {sol.nfev} evals")

    # Build traces
    traces = {}
    for net_name in netlist.nets:
        if net_name in circuit.net_to_state:
            traces[net_name] = sol.y[circuit.net_to_state[net_name]]
        elif net_name in circuit.stimuli:
            traces[net_name] = np.array([circuit.stimuli[net_name](ti) for ti in sol.t])

    # Only return inputs and outputs
    # Generate CLK trace (100kHz square wave)
    clk_trace = np.array([V_HIGH if ((ti - t_settle) % T) < half_T and ti >= t_settle
                           else V_LOW for ti in sol.t])
    traces['CLK'] = clk_trace

    show_nets = ['CLK', 'D', 'EN', 'Q', 'Q_bar']
    return sol.t, {n: traces[n] for n in show_nets if n in traces}, show_nets, 'CLK'


# ---------------------------------------------------------------------------
# Waveform Viewer GUI
# ---------------------------------------------------------------------------

COLORS = {
    'CLK': '#9E9E9E',
    'D': '#2196F3',
    'EN': '#FF9800',
    'Q': '#4CAF50',
    'Q_bar': '#E91E63',
}
DEFAULT_COLOR = '#888888'

V_HIGH = -0.8
V_LOW = -4.0
V_PAD = 0.4


class WaveformViewer(tk.Tk):
    def __init__(self, t, traces, net_names, clock_net=None):
        super().__init__()
        self.title("deepJFET Waveform Viewer — D Latch")
        self.geometry("1200x700")
        self.configure(bg='#1e1e1e')

        self.t = t
        self.traces = traces
        self.net_names = net_names
        self.t_us = t * 1e6
        self.clock_net = clock_net

        # Precompute clock edge times
        self.clock_edges = []
        if clock_net and clock_net in traces:
            clk = traces[clock_net]
            threshold = (V_HIGH + V_LOW) / 2
            for j in range(1, len(clk)):
                crossed = (clk[j-1] < threshold and clk[j] >= threshold) or \
                          (clk[j-1] >= threshold and clk[j] < threshold)
                if crossed:
                    # Interpolate exact crossing
                    frac = (threshold - clk[j-1]) / (clk[j] - clk[j-1]) \
                           if clk[j] != clk[j-1] else 0.5
                    t_cross = self.t_us[j-1] + frac * (self.t_us[j] - self.t_us[j-1])
                    rising = clk[j] > clk[j-1]
                    self.clock_edges.append((t_cross, rising))

        self.v_min = V_LOW - V_PAD
        self.v_max = V_HIGH + V_PAD

        # View window (in us)
        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])

        self.cursor_x = None

        self._build_ui()
        self._draw()

    def _build_ui(self):
        # Top bar with info
        top = tk.Frame(self, bg='#2d2d2d', height=30)
        top.pack(fill=tk.X)
        self.info_label = tk.Label(top, text="", bg='#2d2d2d', fg='#cccccc',
                                   font=('Consolas', 10))
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Legend
        legend_frame = tk.Frame(top, bg='#2d2d2d')
        legend_frame.pack(side=tk.RIGHT, padx=10)
        for name in self.net_names:
            col = COLORS.get(name, DEFAULT_COLOR)
            tk.Label(legend_frame, text=f"■ {name}", bg='#2d2d2d', fg=col,
                     font=('Consolas', 10, 'bold')).pack(side=tk.LEFT, padx=8)

        # Canvas for each waveform
        self.canvases = []
        self.canvas_frame = tk.Frame(self, bg='#1e1e1e')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i, name in enumerate(self.net_names):
            row = tk.Frame(self.canvas_frame, bg='#1e1e1e')
            row.pack(fill=tk.BOTH, expand=True, pady=1)

            # Label
            col = COLORS.get(name, DEFAULT_COLOR)
            lbl = tk.Label(row, text=name, bg='#1e1e1e', fg=col,
                           font=('Consolas', 11, 'bold'), width=6, anchor='e')
            lbl.pack(side=tk.LEFT)

            # Canvas
            c = tk.Canvas(row, bg='#0d0d0d', highlightthickness=0, height=120)
            c.pack(fill=tk.BOTH, expand=True, padx=2)
            c.bind('<Motion>', self._on_mouse_move)
            c.bind('<ButtonPress-1>', self._on_click)
            c.bind('<MouseWheel>', self._on_scroll)
            c.bind('<Button-4>', lambda e: self._on_scroll_linux(e, 1))
            c.bind('<Button-5>', lambda e: self._on_scroll_linux(e, -1))
            c.bind('<Configure>', lambda e: self._draw())
            self.canvases.append((name, c))

        # Time axis at bottom
        self.time_canvas = tk.Canvas(self.canvas_frame, bg='#0d0d0d',
                                      highlightthickness=0, height=25)
        self.time_canvas.pack(fill=tk.X, padx=2)

        # Key bindings
        self.bind('<Left>', lambda e: self._pan(-0.1))
        self.bind('<Right>', lambda e: self._pan(0.1))
        self.bind('<plus>', lambda e: self._zoom(0.8))
        self.bind('<minus>', lambda e: self._zoom(1.25))
        self.bind('<Home>', lambda e: self._reset_view())
        self.bind('<r>', lambda e: self._reset_view())

    def _t_to_x(self, t_us, canvas):
        w = canvas.winfo_width()
        if self.view_end == self.view_start:
            return 0
        return (t_us - self.view_start) / (self.view_end - self.view_start) * w

    def _x_to_t(self, x, canvas):
        w = canvas.winfo_width()
        if w == 0:
            return self.view_start
        return self.view_start + x / w * (self.view_end - self.view_start)

    def _v_to_y(self, v, canvas):
        h = canvas.winfo_height()
        margin = 8
        usable = h - 2 * margin
        frac = (v - self.v_min) / (self.v_max - self.v_min)
        return h - margin - frac * usable

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

            # Grid lines at V_HIGH and V_LOW
            y_high = self._v_to_y(V_HIGH, canvas)
            y_low = self._v_to_y(V_LOW, canvas)
            canvas.create_line(0, y_high, w, y_high, fill='#333333', dash=(2, 4))
            canvas.create_line(0, y_low, w, y_low, fill='#333333', dash=(2, 4))

            # Clock edge lines
            for t_edge, rising in self.clock_edges:
                if self.view_start <= t_edge <= self.view_end:
                    cx = self._t_to_x(t_edge, canvas)
                    if rising:
                        canvas.create_line(cx, 0, cx, h, fill='#556600',
                                           width=1, dash=(6, 3))
                    else:
                        canvas.create_line(cx, 0, cx, h, fill='#443300',
                                           width=1, dash=(3, 6))

            # Waveform
            col = COLORS.get(name, DEFAULT_COLOR)
            mask = (self.t_us >= self.view_start) & (self.t_us <= self.view_end)
            t_vis = self.t_us[mask]
            v_vis = trace[mask]

            if len(t_vis) < 2:
                continue

            # Downsample if too many points
            step = max(1, len(t_vis) // (w * 2))
            t_ds = t_vis[::step]
            v_ds = v_vis[::step]

            points = []
            for ti, vi in zip(t_ds, v_ds):
                x = self._t_to_x(ti, canvas)
                y = self._v_to_y(vi, canvas)
                points.extend([x, y])

            if len(points) >= 4:
                canvas.create_line(*points, fill=col, width=2, smooth=False)

            # Cursor line
            if self.cursor_x is not None:
                cx = self._t_to_x(self.cursor_x, canvas)
                canvas.create_line(cx, 0, cx, h, fill='#ffffff', width=1, dash=(3, 3))

                # Value at cursor
                idx = np.searchsorted(self.t_us, self.cursor_x)
                idx = min(idx, len(trace) - 1)
                val = trace[idx]
                logic = "H" if val > (V_HIGH + V_LOW) / 2 else "L"
                canvas.create_text(cx + 5, 12, text=f"{val:.2f}V ({logic})",
                                   fill=col, anchor='w', font=('Consolas', 9))

        # Time axis
        self.time_canvas.delete('all')
        tc = self.time_canvas
        w = tc.winfo_width()
        if w < 10:
            return

        # Time ticks
        span = self.view_end - self.view_start
        # Choose tick spacing: 1, 2, 5, 10, 20, 50 us etc.
        raw_step = span / 8
        mag = 10 ** np.floor(np.log10(max(raw_step, 1e-10)))
        for nice in [1, 2, 5, 10]:
            if mag * nice >= raw_step:
                tick_step = mag * nice
                break
        else:
            tick_step = mag * 10

        t_tick = np.ceil(self.view_start / tick_step) * tick_step
        while t_tick <= self.view_end:
            x = self._t_to_x(t_tick, tc)
            tc.create_line(x, 0, x, 8, fill='#666666')
            tc.create_text(x, 15, text=f"{t_tick:.1f}us", fill='#999999',
                           font=('Consolas', 8))
            t_tick += tick_step

    def _on_mouse_move(self, event):
        canvas = event.widget
        t_val = self._x_to_t(event.x, canvas)
        self.cursor_x = t_val

        # Update info
        parts = [f"t={t_val:.2f}us"]
        idx = np.searchsorted(self.t_us, t_val)
        idx = min(idx, len(self.t) - 1)
        for name in self.net_names:
            trace = self.traces.get(name)
            if trace is not None:
                val = trace[idx]
                logic = "H" if val > (V_HIGH + V_LOW) / 2 else "L"
                parts.append(f"{name}={val:.2f}V({logic})")
        self.info_label.config(text="  |  ".join(parts))

        self._draw()

    def _on_click(self, event):
        canvas = event.widget
        self.cursor_x = self._x_to_t(event.x, canvas)
        self._draw()

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.18
        self._zoom(factor, event.x, event.widget)

    def _on_scroll_linux(self, event, direction):
        factor = 0.85 if direction > 0 else 1.18
        self._zoom(factor, event.x, event.widget)

    def _zoom(self, factor, mouse_x=None, canvas=None):
        if mouse_x is not None and canvas is not None:
            center = self._x_to_t(mouse_x, canvas)
        else:
            center = (self.view_start + self.view_end) / 2

        span = (self.view_end - self.view_start) * factor
        span = max(span, 0.1)  # min 100ns view
        span = min(span, float(self.t_us[-1] - self.t_us[0]))

        self.view_start = center - span / 2
        self.view_end = center + span / 2

        # Clamp
        if self.view_start < self.t_us[0]:
            self.view_start = float(self.t_us[0])
            self.view_end = self.view_start + span
        if self.view_end > self.t_us[-1]:
            self.view_end = float(self.t_us[-1])
            self.view_start = self.view_end - span

        self._draw()

    def _pan(self, fraction):
        span = self.view_end - self.view_start
        shift = span * fraction
        self.view_start += shift
        self.view_end += shift

        if self.view_start < self.t_us[0]:
            self.view_start = float(self.t_us[0])
            self.view_end = self.view_start + span
        if self.view_end > self.t_us[-1]:
            self.view_end = float(self.t_us[-1])
            self.view_start = self.view_end - span

        self._draw()

    def _reset_view(self):
        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])
        self._draw()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    t, traces, net_names, clock_net = run_d_latch()
    app = WaveformViewer(t, traces, net_names, clock_net=clock_net)
    app.mainloop()
