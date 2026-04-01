"""Interactive waveform viewer for multi-gate transient simulations."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import tkinter as tk
from tkinter import ttk

from model import NChannelJFET, JFETCapacitance, GateType
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams
from transient.engine.multi_gate import MultiGateCircuit, compute_multi_ic, multi_gate_ode
from model.network import gate_type_to_network
from scipy.integrate import solve_ivp
from register import make_register


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def run_2bit_register():
    """Run 2-bit register simulation: latch 3, then 1, then 0."""
    jfet = NChannelJFET(
        beta=0.000135, vto=-3.45, lmbda=0.005,
        is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
        alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
        betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
    ).at_temp(27.0)
    caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12, pb=1.0, m=0.407, fc=0.5)

    gates, input_nets, output_nets, control_nets = make_register("R0", 2)
    netlist = Netlist.from_gates(gates, primary_outputs=set(output_nets))

    params_inv = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                               jfet=jfet, caps=caps)
    params_nand = CircuitParams(v_pos=10, v_neg=-10, r1=24300, r2=7320, r3=6980,
                                jfet=jfet, caps=caps)
    gate_params = {}
    for g in gates:
        if g.gate_type == GateType.INV:
            gate_params[g.name] = params_inv
        else:
            gate_params[g.name] = params_nand
    gate_networks = {g.name: gate_type_to_network(g.gate_type) for g in gates}

    V_HIGH, V_LOW = -0.8, -4.0
    t_settle = 5e-6
    T = 10e-6      # clock period
    half_T = T / 2  # 5us

    # Test: latch 3 (11), then 1 (01), then 0 (00)
    # All transitions on rising CLK edge
    #
    # Cycle  EN  D1  D0  Action
    #  0      L   L   L  idle (register holds initial = 0)
    #  1      H   H   H  latch 3 (both bits high)
    #  2      L   H   H  hold (EN off, D doesn't matter)
    #  3      L   L   L  hold (still 3)
    #  4      H   L   H  latch 1 (bit0=H, bit1=L)
    #  5      L   L   H  hold
    #  6      L   L   L  hold (still 1)
    #  7      H   L   L  latch 0
    #  8      L   L   L  hold
    #  9      L   H   H  hold (D changes ignored, EN off)
    # 10      H   H   L  latch 2
    # 11      L   H   L  hold
    # 12      L   L   L  hold (still 2)

    en_cycle = [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
    d1_cycle = [0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0]  # MSB
    d0_cycle = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0]  # LSB
    n_cycles = len(en_cycle)

    def v_clk(t):
        t_adj = t - t_settle
        if t_adj < 0:
            return V_LOW
        return V_HIGH if (t_adj % T) < half_T else V_LOW

    def make_cycle_stim(pattern):
        def fn(t):
            t_adj = t - t_settle
            if t_adj < 0:
                return V_LOW
            idx = int(t_adj / T)
            if idx >= len(pattern):
                return V_LOW
            return V_HIGH if pattern[idx] else V_LOW
        return fn

    stimuli = {
        'CLK': v_clk,
        'R0_Enable': make_cycle_stim(en_cycle),
        'R0_0_In': make_cycle_stim(d0_cycle),  # bit 0 = LSB
        'R0_1_In': make_cycle_stim(d1_cycle),  # bit 1 = MSB
    }

    circuit = MultiGateCircuit(
        netlist=netlist, gate_params=gate_params, gate_networks=gate_networks,
        stimuli=stimuli, jfet=jfet, caps=caps, temp_c=27.0,
    )

    y0 = compute_multi_ic(circuit)
    for gname in circuit.gate_order:
        start, n = circuit.state_map[gname]
        v_a = y0[start]
        for k in range(3, n):
            if abs(y0[start + k] - 5.0) < 0.01:
                y0[start + k] = v_a * 0.3
    y0 = np.clip(y0, -15, 15)

    t_end = t_settle + n_cycles * T + 5e-6
    t_eval = np.linspace(0, t_end, 10000)

    def rhs(t, y):
        y_clamped = np.clip(y, -20, 20)
        return multi_gate_ode(t, y_clamped, circuit)

    print(f"Running 2-bit register: {len(gates)} gates, {circuit.total_states} state vars...")
    sol = solve_ivp(rhs, (0, t_end), y0, method='LSODA',
                    t_eval=t_eval, max_step=0.5e-6, rtol=1e-3, atol=1e-6)
    print(f"  Done: {sol.nfev} evals, {sol.message}")

    # Build traces
    traces = {}
    for net_name in netlist.nets:
        if net_name in circuit.net_to_state:
            traces[net_name] = sol.y[circuit.net_to_state[net_name]]
        elif net_name in circuit.stimuli:
            traces[net_name] = np.array([circuit.stimuli[net_name](ti) for ti in sol.t])

    show_nets = ['CLK', 'R0_Enable', 'R0_0_In', 'R0_1_In', 'R0_0', 'R0_1']
    register_q_nets = ['R0_0', 'R0_1']  # LSB first
    return sol.t, {n: traces[n] for n in show_nets if n in traces}, \
           show_nets, 'CLK', register_q_nets


# ---------------------------------------------------------------------------
# Waveform Viewer GUI
# ---------------------------------------------------------------------------

COLORS = {
    'CLK': '#9E9E9E',
    'R0_Enable': '#FF9800',
    'R0_0_In': '#42A5F5',
    'R0_1_In': '#2196F3',
    'R0_0': '#66BB6A',
    'R0_1': '#4CAF50',
}
DEFAULT_COLOR = '#888888'

V_HIGH = -0.8
V_LOW = -4.0
V_PAD = 0.4
V_THRESHOLD = (V_HIGH + V_LOW) / 2


class WaveformViewer(tk.Tk):
    def __init__(self, t, traces, net_names, clock_net=None, register_q_nets=None):
        super().__init__()
        self.title("deepJFET — 2-bit Register R0")
        self.geometry("1400x800")
        self.configure(bg='#1e1e1e')

        self.t = t
        self.traces = traces
        self.net_names = net_names
        self.t_us = t * 1e6
        self.clock_net = clock_net
        self.register_q_nets = register_q_nets or []

        self.v_min = V_LOW - V_PAD
        self.v_max = V_HIGH + V_PAD

        self.view_start = float(self.t_us[0])
        self.view_end = float(self.t_us[-1])
        self.cursor_x = None

        # Precompute clock edges
        self.clock_edges = []
        if clock_net and clock_net in traces:
            clk = traces[clock_net]
            for j in range(1, len(clk)):
                crossed = (clk[j-1] < V_THRESHOLD and clk[j] >= V_THRESHOLD) or \
                          (clk[j-1] >= V_THRESHOLD and clk[j] < V_THRESHOLD)
                if crossed:
                    frac = (V_THRESHOLD - clk[j-1]) / (clk[j] - clk[j-1]) \
                           if clk[j] != clk[j-1] else 0.5
                    t_cross = self.t_us[j-1] + frac * (self.t_us[j] - self.t_us[j-1])
                    rising = clk[j] > clk[j-1]
                    self.clock_edges.append((t_cross, rising))

        self._build_ui()
        self._draw()

    def _build_ui(self):
        top = tk.Frame(self, bg='#2d2d2d', height=36)
        top.pack(fill=tk.X)
        self.info_label = tk.Label(top, text="", bg='#2d2d2d', fg='#cccccc',
                                   font=('Consolas', 10))
        self.info_label.pack(side=tk.LEFT, padx=10)

        # Register value display
        self.reg_label = tk.Label(top, text="", bg='#2d2d2d', fg='#FFD700',
                                  font=('Consolas', 12, 'bold'))
        self.reg_label.pack(side=tk.RIGHT, padx=20)

        legend_frame = tk.Frame(top, bg='#2d2d2d')
        legend_frame.pack(side=tk.RIGHT, padx=10)
        for name in self.net_names:
            col = COLORS.get(name, DEFAULT_COLOR)
            short = name.split('_')[-1] if '_' in name else name
            tk.Label(legend_frame, text=f"■ {short}", bg='#2d2d2d', fg=col,
                     font=('Consolas', 9, 'bold')).pack(side=tk.LEFT, padx=4)

        self.canvases = []
        self.canvas_frame = tk.Frame(self, bg='#1e1e1e')
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        for i, name in enumerate(self.net_names):
            row = tk.Frame(self.canvas_frame, bg='#1e1e1e')
            row.pack(fill=tk.BOTH, expand=True, pady=1)

            col = COLORS.get(name, DEFAULT_COLOR)
            short = name.replace('R0_', '').replace('_In', ' In')
            lbl = tk.Label(row, text=short, bg='#1e1e1e', fg=col,
                           font=('Consolas', 10, 'bold'), width=8, anchor='e')
            lbl.pack(side=tk.LEFT)

            c = tk.Canvas(row, bg='#0d0d0d', highlightthickness=0, height=90)
            c.pack(fill=tk.BOTH, expand=True, padx=2)
            c.bind('<Motion>', self._on_mouse_move)
            c.bind('<ButtonPress-1>', self._on_click)
            c.bind('<MouseWheel>', self._on_scroll)
            c.bind('<Configure>', lambda e: self._draw())
            self.canvases.append((name, c))

        self.time_canvas = tk.Canvas(self.canvas_frame, bg='#0d0d0d',
                                      highlightthickness=0, height=25)
        self.time_canvas.pack(fill=tk.X, padx=2)

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

    def _get_register_value(self, idx):
        """Get decimal register value at sample index."""
        val = 0
        for bit, net in enumerate(self.register_q_nets):
            trace = self.traces.get(net)
            if trace is not None and idx < len(trace):
                if trace[idx] > V_THRESHOLD:
                    val |= (1 << bit)
        return val

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

            # Grid lines
            y_high = self._v_to_y(V_HIGH, canvas)
            y_low = self._v_to_y(V_LOW, canvas)
            canvas.create_line(0, y_high, w, y_high, fill='#333333', dash=(2, 4))
            canvas.create_line(0, y_low, w, y_low, fill='#333333', dash=(2, 4))

            # Clock edges
            for t_edge, rising in self.clock_edges:
                if self.view_start <= t_edge <= self.view_end:
                    cx = self._t_to_x(t_edge, canvas)
                    if rising:
                        canvas.create_line(cx, 0, cx, h, fill='#556600',
                                           width=1, dash=(6, 3))
                    else:
                        canvas.create_line(cx, 0, cx, h, fill='#443300',
                                           width=1, dash=(3, 6))

            # Register decimal value overlay (only on Q output panes)
            if name in self.register_q_nets and name == self.register_q_nets[-1]:
                # Draw on the MSB pane — show decimal value between clock edges
                prev_t = self.view_start
                prev_val = self._get_register_value(
                    np.searchsorted(self.t_us, self.view_start))
                for t_edge, rising in self.clock_edges:
                    if t_edge > self.view_end:
                        break
                    if t_edge < self.view_start:
                        prev_t = t_edge
                        prev_val = self._get_register_value(
                            np.searchsorted(self.t_us, t_edge + 0.1))
                        continue
                    # Draw value in the region from prev_t to t_edge
                    x1 = self._t_to_x(max(prev_t, self.view_start), canvas)
                    x2 = self._t_to_x(t_edge, canvas)
                    if x2 - x1 > 20:
                        mid_x = (x1 + x2) / 2
                        canvas.create_text(mid_x, h/2, text=str(prev_val),
                                           fill='#FFD700', font=('Consolas', 11, 'bold'))
                    prev_t = t_edge
                    prev_val = self._get_register_value(
                        min(np.searchsorted(self.t_us, t_edge + 0.5),
                            len(self.t_us) - 1))
                # Last segment
                x1 = self._t_to_x(max(prev_t, self.view_start), canvas)
                x2 = self._t_to_x(self.view_end, canvas)
                if x2 - x1 > 20:
                    mid_x = (x1 + x2) / 2
                    canvas.create_text(mid_x, h/2, text=str(prev_val),
                                       fill='#FFD700', font=('Consolas', 11, 'bold'))

            # Waveform
            col = COLORS.get(name, DEFAULT_COLOR)
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
                y = self._v_to_y(vi, canvas)
                points.extend([x, y])

            if len(points) >= 4:
                canvas.create_line(*points, fill=col, width=2, smooth=False)

            # Cursor
            if self.cursor_x is not None:
                cx = self._t_to_x(self.cursor_x, canvas)
                canvas.create_line(cx, 0, cx, h, fill='#ffffff', width=1, dash=(3, 3))

                idx = min(np.searchsorted(self.t_us, self.cursor_x), len(trace) - 1)
                val = trace[idx]
                logic = "H" if val > V_THRESHOLD else "L"
                canvas.create_text(cx + 5, 12, text=f"{val:.2f}V ({logic})",
                                   fill=col, anchor='w', font=('Consolas', 9))

        # Time axis
        self.time_canvas.delete('all')
        tc = self.time_canvas
        w = tc.winfo_width()
        if w < 10:
            return

        span = self.view_end - self.view_start
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

        idx = min(np.searchsorted(self.t_us, t_val), len(self.t) - 1)

        parts = [f"t={t_val:.2f}us"]
        for name in self.net_names:
            trace = self.traces.get(name)
            if trace is not None:
                val = trace[idx]
                logic = "H" if val > V_THRESHOLD else "L"
                short = name.replace('R0_', '').replace('_In', 'i')
                parts.append(f"{short}={logic}")
        self.info_label.config(text="  |  ".join(parts))

        # Register value
        reg_val = self._get_register_value(idx)
        n_bits = len(self.register_q_nets)
        bin_str = format(reg_val, f'0{n_bits}b')
        self.reg_label.config(text=f"R0 = {reg_val} (0b{bin_str})")

        self._draw()

    def _on_click(self, event):
        canvas = event.widget
        self.cursor_x = self._x_to_t(event.x, canvas)
        self._draw()

    def _on_scroll(self, event):
        factor = 0.85 if event.delta > 0 else 1.18
        self._zoom(factor, event.x, event.widget)

    def _zoom(self, factor, mouse_x=None, canvas=None):
        if mouse_x is not None and canvas is not None:
            center = self._x_to_t(mouse_x, canvas)
        else:
            center = (self.view_start + self.view_end) / 2

        span = (self.view_end - self.view_start) * factor
        span = max(span, 0.1)
        span = min(span, float(self.t_us[-1] - self.t_us[0]))

        self.view_start = center - span / 2
        self.view_end = center + span / 2

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
    t, traces, net_names, clock_net, reg_q_nets = run_2bit_register()
    app = WaveformViewer(t, traces, net_names, clock_net=clock_net,
                         register_q_nets=reg_q_nets)
    app.mainloop()
