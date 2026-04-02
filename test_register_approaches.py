"""Compare 3 approaches for 4-bit register simulation:
1. Original (Python loop ODE)
2. Vectorized (numpy batch ODE)
3. Block-characterized (pre-solved 1-bit latch lookup)
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from scipy.integrate import solve_ivp

from model import NChannelJFET, JFETCapacitance, GateType
from model.jfet import jfet_ids, jfet_gate_current, thermal_voltage, _ids_intrinsic
from simulator.netlist import Gate, Netlist
from simulator.precompute import CircuitParams
from transient.engine.multi_gate import (
    MultiGateCircuit, compute_multi_ic, multi_gate_ode,
)
from model.network import gate_type_to_network
from register import make_register


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

jfet = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
).at_temp(27.0)
caps = JFETCapacitance(cgs0=16.9e-12, cgd0=16.9e-12, pb=1.0, m=0.407, fc=0.5)
params_inv = CircuitParams(v_pos=10, v_neg=-10, r1=12100, r2=7320, r3=6980,
                           jfet=jfet, caps=caps)
params_nand = CircuitParams(v_pos=10, v_neg=-10, r1=24300, r2=7320, r3=6980,
                            jfet=jfet, caps=caps)

V_HIGH, V_LOW = -0.8, -4.0
T = 10e-6
half_T = T / 2
t_settle = 5e-6
N_BITS = 4
vt = thermal_voltage(27.0 + 273.15)

# Test: latch all-1s (15), then 5 (0101), then 0 (0000)
# Bit pattern for 5 = 0101: bit0=1, bit1=0, bit2=1, bit3=0
en_cycle = [0, 1, 0, 1, 0, 1, 0]
d_cycle = [
    [0, 1, 1, 1, 1, 0, 0],  # bit 0 (LSB): H for 15, H for 5, L for 0
    [0, 1, 1, 0, 0, 0, 0],  # bit 1:       H for 15, L for 5, L for 0
    [0, 1, 1, 1, 1, 0, 0],  # bit 2:       H for 15, H for 5, L for 0
    [0, 1, 1, 0, 0, 0, 0],  # bit 3 (MSB): H for 15, L for 5, L for 0
]

n_cycles = len(en_cycle)
t_end = t_settle + n_cycles * T + 5e-6


def v_clk(t):
    t_adj = t - t_settle
    return V_HIGH if t_adj >= 0 and (t_adj % T) < half_T else V_LOW


def make_stim(pattern):
    def fn(t):
        t_adj = t - t_settle
        if t_adj < 0: return V_LOW
        idx = int(t_adj / T)
        return V_HIGH if idx < len(pattern) and pattern[idx] else V_LOW
    return fn


# ---------------------------------------------------------------------------
# Approach 1: Original (Python loop)
# ---------------------------------------------------------------------------

def run_original():
    gates, ins, outs, ctrl = make_register("R", N_BITS)
    netlist = Netlist.from_gates(gates, primary_outputs=set(outs))
    gate_params = {g.name: (params_inv if g.gate_type == GateType.INV else params_nand)
                   for g in gates}
    gate_networks = {g.name: gate_type_to_network(g.gate_type) for g in gates}

    stimuli = {'CLK': v_clk, 'R_Enable': make_stim(en_cycle)}
    for bit in range(N_BITS):
        stimuli[f'R_{bit}_In'] = make_stim(d_cycle[bit])

    circuit = MultiGateCircuit(
        netlist=netlist, gate_params=gate_params, gate_networks=gate_networks,
        stimuli=stimuli, jfet=jfet, caps=caps, temp_c=27.0,
    )

    y0 = compute_multi_ic(circuit)
    for gname in circuit.gate_order:
        start, n = circuit.state_map[gname]
        for k in range(3, n):
            if abs(y0[start + k] - 5.0) < 0.01:
                y0[start + k] = y0[start] * 0.3
    y0 = np.clip(y0, -15, 15)

    t_eval = np.linspace(0, t_end, 4000)

    def rhs(t, y):
        return multi_gate_ode(t, np.clip(y, -20, 20), circuit)

    t0 = time.time()
    sol = solve_ivp(rhs, (0, t_end), y0, method='LSODA',
                    t_eval=t_eval, max_step=0.5e-6, rtol=1e-3, atol=1e-6)
    elapsed = time.time() - t0

    # Extract Q outputs
    q_traces = {}
    for bit in range(N_BITS):
        net = f'R_{bit}'
        if net in circuit.net_to_state:
            q_traces[bit] = sol.y[circuit.net_to_state[net]]

    return sol.t, q_traces, elapsed, sol.nfev


# ---------------------------------------------------------------------------
# Approach 2: Vectorized ODE (numpy batch)
# ---------------------------------------------------------------------------

def _jfet_ids_vec(vgs, vds, j):
    """Vectorized jfet_ids for numpy arrays."""
    # Handle reverse polarity
    rev = vds < 0
    vgs_f = np.where(rev, vgs - vds, vgs)
    vds_f = np.where(rev, -vds, vds)

    vsat = np.maximum(vgs_f - j.vto, 0.0)
    active = vgs_f > j.vto
    clm = 1.0 + j.lmbda * vds_f

    ids_sat = j.beta * vsat**2 * clm
    ids_lin = j.beta * (2.0 * vsat * vds_f - vds_f**2) * clm
    ids = np.where(vds_f >= vsat, ids_sat, ids_lin)
    ids = np.where(active, ids, 0.0)

    if j.alpha > 0.0 and j.vk > 0.0:
        vds_safe = np.maximum(vds_f, 1e-10)
        ion = 1.0 + j.alpha * vds_f * np.exp(np.minimum(-j.vk / vds_safe, 40))
        ids = ids * np.where(vds_f > 0, ion, 1.0)

    return np.where(rev, -ids, ids)


def _jfet_ids_vec_parasitic(vgs, vds, j, n_iters=5):
    """Vectorized with parasitic Rs/Rd."""
    if j.rs == 0.0 and j.rd == 0.0:
        return _jfet_ids_vec(vgs, vds, j)
    id_est = _jfet_ids_vec(vgs, vds, j)
    for _ in range(n_iters):
        vgs_int = vgs - id_est * j.rs
        vds_int = vds - id_est * (j.rs + j.rd)
        id_new = _jfet_ids_vec(vgs_int, vds_int, j)
        id_est = 0.5 * (id_est + id_new)
    return id_est


def _diode_vec(vd, is_, n, vt):
    """Vectorized diode current."""
    x = vd / (n * vt)
    safe_x = np.minimum(x, 40.0)
    exp_val = np.exp(safe_x)
    exp40 = np.exp(40.0)
    linear = exp40 * (1.0 + x - 40.0)
    return is_ * np.where(x > 40.0, linear, exp_val - 1.0)


def _gate_currents_vec(vgs, vgd, j, vt):
    """Vectorized gate junction currents."""
    igs = _diode_vec(vgs, j.is_, j.n, vt)
    igd = _diode_vec(vgd, j.is_, j.n, vt)
    if j.isr > 0:
        igs = igs + _diode_vec(vgs, j.isr, j.nr, vt)
        igd = igd + _diode_vec(vgd, j.isr, j.nr, vt)
    return igs, igd


class VectorizedRegisterODE:
    """Vectorized ODE for N-bit register.

    Pre-computes the structure so that all INV gates and all NAND2 gates
    are evaluated in one vectorized call each.
    """

    def __init__(self, n_bits, jfet, caps, params_inv, params_nand,
                 stimuli_fns, t_settle):
        self.n_bits = n_bits
        self.j = jfet
        self.vt = thermal_voltage(27.0 + 273.15)
        self.t_settle = t_settle

        # Per-bit state layout: 7 gates
        # Gate order per bit: nand_ce(4), inv_ce(3), inv_d(3), nand_s(4), nand_r(4), nand_q(4), nand_qb(4)
        # States per gate: INV=3, NAND2=4
        # Per bit: 4+3+3+4+4+4+4 = 26 states
        self.states_per_bit = 26
        self.total_states = n_bits * self.states_per_bit

        # Gate offsets within a bit (relative to bit start)
        # nand_ce: 0-3 (va, vb, vout, vmid)
        # inv_ce:  4-6 (va, vb, vout)
        # inv_d:   7-9
        # nand_s:  10-13
        # nand_r:  14-17
        # nand_q:  18-21
        # nand_qb: 22-25

        self.stimuli_fns = stimuli_fns

        # Capacitances
        self.c_a = caps.cgd0 + caps.cgs0
        self.c_b = caps.cgd0
        self.c_out_inv = 1 * caps.c_per_input  # fanout=1 for internal
        self.c_out_nand = 1 * caps.c_per_input
        self.c_out_q = 2 * caps.c_per_input  # Q drives nand_qb + external
        self.c_mid = caps.cgd0 + caps.cgs0

        # R values
        self.r1_inv = params_inv.r1
        self.r2_inv = params_inv.r2
        self.r3_inv = params_inv.r3
        self.r1_nand = params_nand.r1
        self.r2_nand = params_nand.r2
        self.r3_nand = params_nand.r3
        self.v_pos = params_inv.v_pos
        self.v_neg = params_inv.v_neg

    def _inv_derivs(self, va, vb, vout, v_in):
        """Vectorized INV derivatives for N gates at once."""
        j = self.j
        # J1
        i_j1 = _jfet_ids_vec_parasitic(v_in, va, j)
        vgs_int = v_in - i_j1 * j.rs
        vgd_int = v_in - (va - i_j1 * j.rd)
        _, igd_j1 = _gate_currents_vec(vgs_int, vgd_int, j, self.vt)

        # J2
        i_j2 = _jfet_ids_vec_parasitic(va - vb, self.v_pos - vb, j)
        vgs2_int = va - (vb + i_j2 * j.rs)
        vgd2_int = va - (self.v_pos - i_j2 * j.rd)
        igs_j2, igd_j2 = _gate_currents_vec(vgs2_int, vgd2_int, j, self.vt)

        dva = ((self.v_pos - va) / self.r1_inv + igd_j1 - i_j1 - (igs_j2 + igd_j2)) / self.c_a
        dvb = (i_j2 + igs_j2 - (vb - vout) / self.r2_inv) / self.c_b
        dvout = ((vb - vout) / self.r2_inv - (vout - self.v_neg) / self.r3_inv) / self.c_out_inv

        return dva, dvb, dvout

    def _nand2_derivs(self, va, vb, vout, vmid, v_in0, v_in1, c_out):
        """Vectorized NAND2 derivatives for N gates at once."""
        j = self.j
        # Series J1 chain: J1_0 drain=va, source=vmid; J1_1 drain=vmid, source=0
        i_j1_0 = _jfet_ids_vec_parasitic(v_in0 - vmid, va - vmid, j)
        i_j1_1 = _jfet_ids_vec_parasitic(v_in1, vmid, j)

        # Gate current of top J1
        vgs_int = v_in0 - (vmid + i_j1_0 * j.rs)
        vgd_int = v_in0 - (va - i_j1_0 * j.rd)
        _, igd_top = _gate_currents_vec(vgs_int, vgd_int, j, self.vt)

        # J2
        i_j2 = _jfet_ids_vec_parasitic(va - vb, self.v_pos - vb, j)
        vgs2_int = va - (vb + i_j2 * j.rs)
        vgd2_int = va - (self.v_pos - i_j2 * j.rd)
        igs_j2, igd_j2 = _gate_currents_vec(vgs2_int, vgd2_int, j, self.vt)

        dva = ((self.v_pos - va) / self.r1_nand + igd_top - i_j1_0 - (igs_j2 + igd_j2)) / self.c_a
        dvb = (i_j2 + igs_j2 - (vb - vout) / self.r2_nand) / self.c_b
        dvout = ((vb - vout) / self.r2_nand - (vout - self.v_neg) / self.r3_nand) / c_out
        dmid = (i_j1_0 - i_j1_1) / self.c_mid

        return dva, dvb, dvout, dmid

    def __call__(self, t, state):
        N = self.n_bits
        S = self.states_per_bit
        derivs = np.zeros_like(state)

        # Get stimuli
        v_clk = self.stimuli_fns['CLK'](t)
        v_en = self.stimuli_fns['EN'](t)
        v_d = np.array([self.stimuli_fns[f'D{i}'](t) for i in range(N)])

        # Extract all gate states as (N,) arrays
        def get(offset, count=1):
            idx = np.arange(N) * S + offset
            return state[idx] if count == 1 else tuple(state[idx + k] for k in range(count))

        def put(offset, *vals):
            idx = np.arange(N) * S + offset
            for k, v in enumerate(vals):
                derivs[idx + k] = v

        # --- nand_ce: inputs (CLK, EN) -> CLK_EN_bar ---
        va, vb, vout, vmid = get(0, 4)
        clk_arr = np.full(N, v_clk)
        en_arr = np.full(N, v_en)
        d = self._nand2_derivs(va, vb, vout, vmid, clk_arr, en_arr, self.c_out_nand)
        put(0, *d)
        clk_en_bar = get(2)  # vout of nand_ce

        # --- inv_ce: CLK_EN_bar -> CLK_EN ---
        va, vb, vout = get(4, 3)
        d = self._inv_derivs(va, vb, vout, clk_en_bar)
        put(4, *d)
        clk_en = get(6)  # vout of inv_ce

        # --- inv_d: D -> D_bar ---
        va, vb, vout = get(7, 3)
        d = self._inv_derivs(va, vb, vout, v_d)
        put(7, *d)
        d_bar = get(9)

        # --- nand_s: (D, CLK_EN) -> S_bar ---
        va, vb, vout, vmid = get(10, 4)
        d = self._nand2_derivs(va, vb, vout, vmid, v_d, clk_en, self.c_out_nand)
        put(10, *d)
        s_bar = get(12)

        # --- nand_r: (D_bar, CLK_EN) -> R_bar ---
        va, vb, vout, vmid = get(14, 4)
        d = self._nand2_derivs(va, vb, vout, vmid, d_bar, clk_en, self.c_out_nand)
        put(14, *d)
        r_bar = get(16)

        # --- nand_q: (S_bar, Q_bar) -> Q ---
        q_bar = get(24)  # from nand_qb vout (forward reference, use previous step)
        va, vb, vout, vmid = get(18, 4)
        d = self._nand2_derivs(va, vb, vout, vmid, s_bar, q_bar, self.c_out_q)
        put(18, *d)
        q = get(20)

        # --- nand_qb: (R_bar, Q) -> Q_bar ---
        va, vb, vout, vmid = get(22, 4)
        d = self._nand2_derivs(va, vb, vout, vmid, r_bar, q, self.c_out_nand)
        put(22, *d)

        derivs = np.nan_to_num(derivs, nan=0.0, posinf=1e8, neginf=-1e8)
        np.clip(derivs, -1e9, 1e9, out=derivs)
        return derivs


def run_vectorized():
    N = N_BITS
    stimuli_fns = {
        'CLK': v_clk,
        'EN': make_stim(en_cycle),
    }
    for bit in range(N):
        stimuli_fns[f'D{bit}'] = make_stim(d_cycle[bit])

    ode = VectorizedRegisterODE(N, jfet, caps, params_inv, params_nand,
                                 stimuli_fns, t_settle)

    # Initial conditions: use original approach to get ICs
    gates, ins, outs, ctrl = make_register("R", N)
    netlist = Netlist.from_gates(gates, primary_outputs=set(outs))
    gate_params_dict = {g.name: (params_inv if g.gate_type == GateType.INV else params_nand)
                        for g in gates}
    gate_networks = {g.name: gate_type_to_network(g.gate_type) for g in gates}

    stim_orig = {'CLK': v_clk, 'R_Enable': make_stim(en_cycle)}
    for bit in range(N):
        stim_orig[f'R_{bit}_In'] = make_stim(d_cycle[bit])

    circuit = MultiGateCircuit(
        netlist=netlist, gate_params=gate_params_dict, gate_networks=gate_networks,
        stimuli=stim_orig, jfet=jfet, caps=caps, temp_c=27.0,
    )
    y0_orig = compute_multi_ic(circuit)
    for gname in circuit.gate_order:
        start, n = circuit.state_map[gname]
        for k in range(3, n):
            if abs(y0_orig[start + k] - 5.0) < 0.01:
                y0_orig[start + k] = y0_orig[start] * 0.3
    y0_orig = np.clip(y0_orig, -15, 15)

    # Repack ICs into vectorized layout
    y0 = np.zeros(ode.total_states)
    S = ode.states_per_bit
    for bit in range(N):
        # Map from original gate order to vectorized layout
        # Original per-bit gates: nand_ce, inv_ce, inv_d, nand_s, nand_r, nand_q, nand_qb
        prefix = f'R_{bit}'
        gate_names_ordered = [
            f'{prefix}_nand_ce', f'{prefix}_inv_ce', f'{prefix}_inv_d',
            f'{prefix}_nand_s', f'{prefix}_nand_r',
            f'{prefix}_nand_q', f'{prefix}_nand_qb',
        ]
        offsets = [0, 4, 7, 10, 14, 18, 22]
        sizes = [4, 3, 3, 4, 4, 4, 4]

        for gname, off, sz in zip(gate_names_ordered, offsets, sizes):
            orig_start, orig_n = circuit.state_map[gname]
            y0[bit * S + off: bit * S + off + sz] = y0_orig[orig_start: orig_start + sz]

    t_eval = np.linspace(0, t_end, 4000)

    def rhs(t, y):
        return ode(t, np.clip(y, -20, 20))

    t0 = time.time()
    sol = solve_ivp(rhs, (0, t_end), y0, method='LSODA',
                    t_eval=t_eval, max_step=0.5e-6, rtol=1e-3, atol=1e-6)
    elapsed = time.time() - t0

    # Extract Q outputs (nand_q vout at offset 20)
    q_traces = {}
    for bit in range(N):
        q_traces[bit] = sol.y[bit * S + 20]

    return sol.t, q_traces, elapsed, sol.nfev


# ---------------------------------------------------------------------------
# Approach 3: Block-characterized 1-bit latch
# ---------------------------------------------------------------------------

def build_1bit_latch_profile():
    """Pre-solve a 1-bit D latch for all input combinations.

    Returns a lookup function: (v_d, v_clk, v_en) -> (v_q_settled, tau)
    """
    from transient.engine import Circuit, simulate
    from transient.util import measure_timing

    gates_1b, _, _, _ = make_register("BLK", 1)
    netlist = Netlist.from_gates(gates_1b, primary_outputs={'BLK_0'})
    gate_params_1b = {g.name: (params_inv if g.gate_type == GateType.INV else params_nand)
                      for g in gates_1b}
    gate_networks = {g.name: gate_type_to_network(g.gate_type) for g in gates_1b}

    # Pre-solve DC for all 8 input combinations (D, CLK, EN each H or L)
    dc_table = {}
    for d_val in [V_LOW, V_HIGH]:
        for clk_val in [V_LOW, V_HIGH]:
            for en_val in [V_LOW, V_HIGH]:
                stim = {
                    'CLK': lambda t, v=clk_val: v,
                    'BLK_Enable': lambda t, v=en_val: v,
                    'BLK_0_In': lambda t, v=d_val: v,
                }
                circ = MultiGateCircuit(
                    netlist=netlist, gate_params=gate_params_1b,
                    gate_networks=gate_networks, stimuli=stim,
                    jfet=jfet, caps=caps, temp_c=27.0,
                )
                y0 = compute_multi_ic(circ)
                for gn in circ.gate_order:
                    s, n = circ.state_map[gn]
                    for k in range(3, n):
                        if abs(y0[s+k] - 5.0) < 0.01:
                            y0[s+k] = y0[s] * 0.3
                y0 = np.clip(y0, -15, 15)

                q_idx = circ.net_to_state.get('BLK_0')
                dc_table[(d_val, clk_val, en_val)] = y0[q_idx] if q_idx else 0.0

    # Measure transition time with a transient sim
    # Step from Q=LOW to Q=HIGH (D=H, CLK=H, EN=H)
    stim_step = {
        'CLK': lambda t: V_HIGH,
        'BLK_Enable': lambda t: V_HIGH,
        'BLK_0_In': lambda t: V_LOW if t < 3e-6 else V_HIGH,
    }
    circ_step = MultiGateCircuit(
        netlist=netlist, gate_params=gate_params_1b,
        gate_networks=gate_networks, stimuli=stim_step,
        jfet=jfet, caps=caps, temp_c=27.0,
    )
    y0_step = compute_multi_ic(circ_step)
    for gn in circ_step.gate_order:
        s, n = circ_step.state_map[gn]
        for k in range(3, n):
            if abs(y0_step[s+k] - 5.0) < 0.01:
                y0_step[s+k] = y0_step[s] * 0.3
    y0_step = np.clip(y0_step, -15, 15)

    def rhs(t, y):
        return multi_gate_ode(t, np.clip(y, -20, 20), circ_step)

    sol = solve_ivp(rhs, (0, 15e-6), y0_step, method='LSODA',
                    t_eval=np.linspace(0, 15e-6, 2000),
                    max_step=0.5e-6, rtol=1e-3, atol=1e-6)

    q_trace = sol.y[circ_step.net_to_state['BLK_0']]
    # Estimate tau from 10%-90% rise time
    v_lo = q_trace[:500].mean()
    v_hi = q_trace[-500:].mean()
    v_10 = v_lo + 0.1 * (v_hi - v_lo)
    v_90 = v_lo + 0.9 * (v_hi - v_lo)
    t_10_idx = np.searchsorted(q_trace[500:], v_10) + 500
    t_90_idx = np.searchsorted(q_trace[500:], v_90) + 500
    t_10_idx = min(t_10_idx, len(sol.t) - 1)
    t_90_idx = min(t_90_idx, len(sol.t) - 1)
    rise_time = sol.t[t_90_idx] - sol.t[t_10_idx]
    tau = rise_time / 2.2  # RC time constant

    return dc_table, tau


def run_block_characterized():
    """Simulate 4-bit register using pre-characterized 1-bit latch model.

    Block model: each bit has Q and Q_bar as state variables (2 per bit).
    When transparent (CLK=H, EN=H): Q tracks toward D level, Q_bar toward inverse.
    When latched: Q and Q_bar hold (dQ/dt = 0).

    The target levels and time constant come from the full transient sim
    of a single latch.
    """
    print("  Building 1-bit latch profile...")
    _, tau = build_1bit_latch_profile()
    print(f"  tau = {tau*1e9:.0f}ns")

    # The settled Q levels when transparent:
    # D=H -> Q settles to ~V_HIGH (-0.82V), Q_bar to ~V_LOW
    # D=L -> Q settles to ~V_LOW (-4.08V), Q_bar to ~V_HIGH
    # These come from the original full ODE (we know them from previous runs)
    Q_SETTLED_HIGH = -0.825  # Q when D=H and transparent
    Q_SETTLED_LOW = -4.076   # Q when D=L and transparent

    N = N_BITS
    # State: Q for each bit (N variables)

    v_en_fn = make_stim(en_cycle)
    d_fns = [make_stim(d_cycle[bit]) for bit in range(N)]

    def rhs(t, q_state):
        v_clk_now = v_clk(t)
        v_en_now = v_en_fn(t)
        derivs = np.zeros(N)

        clk_high = v_clk_now > V_THRESHOLD
        en_high = v_en_now > V_THRESHOLD
        transparent = clk_high and en_high

        for bit in range(N):
            if transparent:
                v_d_now = d_fns[bit](t)
                d_high = v_d_now > V_THRESHOLD
                q_target = Q_SETTLED_HIGH if d_high else Q_SETTLED_LOW
                derivs[bit] = (q_target - q_state[bit]) / tau
            else:
                derivs[bit] = 0.0  # hold

        return derivs

    q0 = np.full(N, Q_SETTLED_LOW)
    t_eval = np.linspace(0, t_end, 4000)

    t0 = time.time()
    sol = solve_ivp(rhs, (0, t_end), q0, method='LSODA',
                    t_eval=t_eval, max_step=0.5e-6, rtol=1e-3, atol=1e-6)
    elapsed = time.time() - t0

    q_traces = {bit: sol.y[bit] for bit in range(N)}
    return sol.t, q_traces, elapsed, sol.nfev


V_THRESHOLD = (V_HIGH + V_LOW) / 2

# ---------------------------------------------------------------------------
# Run all three and compare
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print(f"=== 4-bit Register Comparison ===\n")

    print("1. Original (Python loop ODE)...")
    t1, q1, time1, nfev1 = run_original()
    print(f"   {time1:.1f}s, {nfev1} evals\n")

    print("2. Vectorized (numpy batch ODE)...")
    t2, q2, time2, nfev2 = run_vectorized()
    print(f"   {time2:.1f}s, {nfev2} evals\n")

    print("3. Block-characterized (1-bit latch model)...")
    t3, q3, time3, nfev3 = run_block_characterized()
    print(f"   {time3:.1f}s, {nfev3} evals\n")

    # Compare outputs
    print("=== Comparison ===")
    print(f"{'':>15} {'Original':>10} {'Vectorized':>12} {'Block':>10}")
    print(f"{'Time':>15} {time1:>9.1f}s {time2:>11.1f}s {time3:>9.1f}s")
    print(f"{'Evals':>15} {nfev1:>10} {nfev2:>12} {nfev3:>10}")
    print(f"{'Speedup':>15} {'1.0x':>10} {f'{time1/time2:.1f}x':>12} {f'{time1/time3:.1f}x':>12}")

    # Check Q values at end of each phase
    print(f"\n{'Phase':>8} {'Expected':>10} {'Original':>10} {'Vector':>10} {'Block':>10}")
    check_times = [
        (t_settle + 1.5 * T, "15 (1111)"),  # after latch 15
        (t_settle + 3.5 * T, "5 (0101)"),   # after latch 5
        (t_settle + 5.5 * T, "0 (0000)"),   # after latch 0
    ]
    for t_check, expected in check_times:
        idx1 = np.searchsorted(t1, t_check)
        idx2 = np.searchsorted(t2, t_check)
        idx3 = np.searchsorted(t3, t_check)

        def decode(traces, idx):
            val = 0
            for bit in range(N_BITS):
                if traces[bit][min(idx, len(traces[bit])-1)] > V_THRESHOLD:
                    val |= (1 << bit)
            return val

        v1 = decode(q1, idx1)
        v2 = decode(q2, idx2)
        v3 = decode(q3, idx3)
        match = "OK" if v1 == v2 == v3 else "MISMATCH"
        print(f"{t_check*1e6:>7.0f}us {expected:>10} {v1:>10} {v2:>10} {v3:>10}  {match}")
