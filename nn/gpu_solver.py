"""GPU-accelerated circuit solver using PyTorch.

All samples x all truth table entries x all initial guesses are stacked into
ONE mega-batch and solved in a single Newton pass. No Python loops in the
hot path except Newton steps and Jacobian columns (unavoidable with FD).
"""

import torch
import numpy as np

K_BOLTZ = 1.380649e-23
Q_ELEC = 1.602176634e-19
TNOM_K = 300.15
_EXP40 = np.exp(40.0)  # precomputed constant


# ---------------------------------------------------------------------------
# JFET physics — branchless tensor ops
# ---------------------------------------------------------------------------

def _ids_intrinsic(vgs, vds, beta, vto, lmbda, alpha, vk):
    """Shichman-Hodges channel current, branchless."""
    rev = vds < 0
    vgs_f = torch.where(rev, vgs - vds, vgs)
    vds_f = torch.where(rev, -vds, vds)

    vsat = (vgs_f - vto).clamp(min=0)
    active = vgs_f > vto
    clm = 1.0 + lmbda * vds_f

    ids_sat = beta * vsat.pow(2) * clm
    ids_lin = beta * (2.0 * vsat * vds_f - vds_f.pow(2)) * clm
    ids = torch.where(vds_f >= vsat, ids_sat, ids_lin)
    ids = torch.where(active, ids, torch.zeros_like(ids))

    if alpha > 0.0 and vk > 0.0:
        vds_safe = vds_f.clamp(min=1e-10)
        ion = 1.0 + alpha * vds_f * torch.exp((-vk / vds_safe).clamp(max=40))
        ids = ids * torch.where(vds_f > 0, ion, torch.ones_like(ion))

    return torch.where(rev, -ids, ids)


def _ids_with_rs_rd(vgs, vds, beta, vto, lmbda, alpha, vk, rs, rd):
    """Channel current with parasitic Rs/Rd. Early-exit iteration."""
    if rs == 0.0 and rd == 0.0:
        return _ids_intrinsic(vgs, vds, beta, vto, lmbda, alpha, vk)
    id_est = _ids_intrinsic(vgs, vds, beta, vto, lmbda, alpha, vk)
    for _ in range(5):  # 5 iters sufficient for rs=rd=1 ohm
        vgs_int = vgs - id_est * rs
        vds_int = vds - id_est * (rs + rd)
        id_new = _ids_intrinsic(vgs_int, vds_int, beta, vto, lmbda, alpha, vk)
        id_est = 0.5 * (id_est + id_new)
    return id_est


def _diode_current(vd, is_, n, vt):
    """PN junction diode current with overflow protection."""
    x = vd / (n * vt)
    safe_x = x.clamp(max=40.0)
    exp_val = torch.exp(safe_x)
    linear = _EXP40 * (1.0 + x - 40.0)
    return is_ * torch.where(x > 40.0, linear, exp_val - 1.0)


def _gate_currents(vgs, vgd, is_, n, isr, nr, vt):
    """Gate junction currents (igs, igd)."""
    igs = _diode_current(vgs, is_, n, vt)
    igd = _diode_current(vgd, is_, n, vt)
    if isr > 0.0:
        igs = igs + _diode_current(vgs, isr, nr, vt)
        igd = igd + _diode_current(vgd, isr, nr, vt)
    return igs, igd


def _ids_j(vgs, vds, jp):
    """Shorthand for full JFET current."""
    return _ids_with_rs_rd(vgs, vds, jp['beta'], jp['vto'], jp['lmbda'],
                           jp['alpha'], jp['vk'], jp['rs'], jp['rd'])


# ---------------------------------------------------------------------------
# Recursive network current (GPU tensors) — arbitrary topology
# ---------------------------------------------------------------------------

def _network_current_gpu(net, v_top, v_bot, v_inputs_dict, jp,
                         midpoint_cols, mid_offset):
    """Recursive J1 current computation on GPU tensors.

    Args:
        net: PulldownNetwork (Leaf/Series/Parallel)
        v_top: (B,) top node voltage
        v_bot: (B,) bottom node voltage
        v_inputs_dict: dict of input_name -> (B,) tensor
        jp: JFET params dict
        midpoint_cols: (B, total_midpoints) tensor of midpoint voltages
        mid_offset: [int] mutable counter, indexes into midpoint_cols

    Returns:
        (i_drain, igd_top, mid_residuals)
        i_drain: (B,) current from top to bottom
        igd_top: (B,) gate-drain current of topmost JFET
        mid_residuals: list of (B,) KCL residuals at midpoints
    """
    from model.network import Leaf, Series, Parallel

    if isinstance(net, Leaf):
        vgs = v_inputs_dict[net.input_name] - v_bot
        vds = v_top - v_bot
        i_d = _ids_j(vgs, vds, jp)
        # Gate junction current of this JFET
        vgs_int = v_inputs_dict[net.input_name] - (v_bot + i_d * jp['rs'])
        vgd_int = v_inputs_dict[net.input_name] - (v_top - i_d * jp['rd'])
        _, igd = _gate_currents(vgs_int, vgd_int, jp['is_'], jp['n'],
                                jp['isr'], jp['nr'],
                                torch.zeros_like(v_top) + K_BOLTZ * 300.15 / Q_ELEC)
        return i_d, igd, []

    if isinstance(net, Parallel):
        i_total = torch.zeros_like(v_top)
        igd_total = torch.zeros_like(v_top)
        all_residuals = []
        for child in net.children:
            i_k, igd_k, res_k = _network_current_gpu(
                child, v_top, v_bot, v_inputs_dict, jp,
                midpoint_cols, mid_offset)
            i_total = i_total + i_k
            igd_total = igd_total + igd_k
            all_residuals.extend(res_k)
        return i_total, igd_total, all_residuals

    if isinstance(net, Series):
        n = len(net.children)
        # Build node voltages
        nodes = [v_top]
        for _ in range(n - 1):
            idx = mid_offset[0]
            mid_offset[0] += 1
            nodes.append(midpoint_cols[:, idx])
        nodes.append(v_bot)

        # Current through each child (just drain current, no gate effects)
        currents = []
        all_residuals = []
        for k, child in enumerate(net.children):
            i_k, _, res_k = _network_current_gpu(
                child, nodes[k], nodes[k + 1], v_inputs_dict, jp,
                midpoint_cols, mid_offset)
            currents.append(i_k)
            all_residuals.extend(res_k)

        # KCL at midpoints
        for k in range(n - 1):
            all_residuals.append(currents[k] - currents[k + 1])

        # Gate current of topmost — use topmost leaf
        i_top = currents[0]
        top_leaf = _find_topmost_leaf(net)
        vgs_int = v_inputs_dict[top_leaf.input_name] - (nodes[1] + i_top * jp['rs'])
        vgd_int = v_inputs_dict[top_leaf.input_name] - (v_top - i_top * jp['rd'])
        _, igd_top = _gate_currents(vgs_int, vgd_int, jp['is_'], jp['n'],
                                     jp['isr'], jp['nr'],
                                     torch.zeros_like(v_top) + K_BOLTZ * 300.15 / Q_ELEC)
        return i_top, igd_top, all_residuals


def _find_topmost_leaf(net):
    from model.network import Leaf
    if isinstance(net, Leaf):
        return net
    return _find_topmost_leaf(net.children[0])


def _network_residuals_gpu(x, v_ins_dict, r1, r_load, v_pos, v_neg, vt, jp, network):
    """KCL residuals for arbitrary topology. x:(B, n_vars). Returns (B, n_vars).

    x columns: [v_a, v_b, mid_0, mid_1, ...]
    """
    from model.network import count_midpoints
    B = x.shape[0]
    v_a = x[:, 0]
    v_b = x[:, 1]
    n_mids = count_midpoints(network)

    if n_mids > 0:
        midpoint_cols = x[:, 2:2 + n_mids]
    else:
        midpoint_cols = torch.zeros(B, 0, device=x.device, dtype=x.dtype)

    mid_offset = [0]
    i_j1, igd_j1, mid_residuals = _network_current_gpu(
        network, v_a, torch.zeros_like(v_a), v_ins_dict, jp,
        midpoint_cols, mid_offset)

    # J2 (same for all topologies)
    i_j2 = _ids_j(v_a - v_b, v_pos - v_b, jp)
    vgs2_int = v_a - (v_b + i_j2 * jp['rs'])
    vgd2_int = v_a - (v_pos - i_j2 * jp['rd'])
    igs_j2, igd_j2 = _gate_currents(vgs2_int, vgd2_int, jp['is_'], jp['n'],
                                      jp['isr'], jp['nr'], vt)

    eq_a = (v_pos - v_a) / r1 + igd_j1 - i_j1 - (igs_j2 + igd_j2)
    eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load

    all_eqs = [eq_a, eq_b] + mid_residuals
    return torch.stack(all_eqs, dim=1)


# ---------------------------------------------------------------------------
# Circuit residuals (legacy gate-type-specific, kept for backward compat)
# ---------------------------------------------------------------------------

def _inv_nor_residuals(x, v_ins, r1, r_load, v_pos, v_neg, vt, jp):
    """KCL for INV/NOR. x:(B,2), v_ins:(B,N_inputs). Returns (B,2)."""
    v_a, v_b = x[:, 0], x[:, 1]

    i_j1_total = torch.zeros_like(v_a)
    igd_j1_total = torch.zeros_like(v_a)
    for k in range(v_ins.shape[1]):
        v_in_k = v_ins[:, k]
        i_j1 = _ids_j(v_in_k, v_a, jp)
        i_j1_total = i_j1_total + i_j1
        vgs_int = v_in_k - i_j1 * jp['rs']
        vgd_int = v_in_k - (v_a - i_j1 * jp['rd'])
        _, igd = _gate_currents(vgs_int, vgd_int, jp['is_'], jp['n'],
                                jp['isr'], jp['nr'], vt)
        igd_j1_total = igd_j1_total + igd

    i_j2 = _ids_j(v_a - v_b, v_pos - v_b, jp)
    vgs2_int = v_a - (v_b + i_j2 * jp['rs'])
    vgd2_int = v_a - (v_pos - i_j2 * jp['rd'])
    igs_j2, igd_j2 = _gate_currents(vgs2_int, vgd2_int, jp['is_'], jp['n'],
                                      jp['isr'], jp['nr'], vt)

    eq_a = (v_pos - v_a) / r1 + igd_j1_total - i_j1_total - (igs_j2 + igd_j2)
    eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load
    return torch.stack([eq_a, eq_b], dim=1)


def _nand_residuals(x, v_ins, r1, r_load, v_pos, v_neg, vt, jp, n_inputs):
    """KCL for NAND-N. x:(B,N+1), v_ins:(B,N). Returns (B,N+1)."""
    B = x.shape[0]
    v_a, v_b = x[:, 0], x[:, -1]

    # Node stack: [v_a, mid_0..mid_{N-2}, 0]
    nodes = torch.zeros(B, n_inputs + 1, device=x.device, dtype=x.dtype)
    nodes[:, 0] = v_a
    if n_inputs > 1:
        nodes[:, 1:n_inputs] = x[:, 1:n_inputs]

    # Chain currents
    i_chain = []
    for k in range(n_inputs):
        i_k = _ids_j(v_ins[:, k] - nodes[:, k + 1],
                      nodes[:, k] - nodes[:, k + 1], jp)
        i_chain.append(i_k)

    # J1[0] gate current
    i_top = i_chain[0]
    vgs_int = v_ins[:, 0] - (nodes[:, 1] + i_top * jp['rs'])
    vgd_int = v_ins[:, 0] - (v_a - i_top * jp['rd'])
    _, igd_top = _gate_currents(vgs_int, vgd_int, jp['is_'], jp['n'],
                                 jp['isr'], jp['nr'], vt)

    # J2
    i_j2 = _ids_j(v_a - v_b, v_pos - v_b, jp)
    vgs2_int = v_a - (v_b + i_j2 * jp['rs'])
    vgd2_int = v_a - (v_pos - i_j2 * jp['rd'])
    igs_j2, igd_j2 = _gate_currents(vgs2_int, vgd2_int, jp['is_'], jp['n'],
                                      jp['isr'], jp['nr'], vt)

    res = torch.zeros(B, n_inputs + 1, device=x.device, dtype=x.dtype)
    res[:, 0] = (v_pos - v_a) / r1 + igd_top - i_top - (igs_j2 + igd_j2)
    for k in range(n_inputs - 1):
        res[:, 1 + k] = i_chain[k] - i_chain[k + 1]
    res[:, -1] = i_j2 + igs_j2 - (v_b - v_neg) / r_load
    return res


# ---------------------------------------------------------------------------
# Newton solver
# ---------------------------------------------------------------------------

def _newton_batch(residual_fn, x0, n_vars, max_steps=25, tol=1e-8):
    """Batched Newton. Masks converged samples to prevent divergence."""
    x = x0.clone()
    B = x.shape[0]
    dev, dt = x.device, x.dtype
    converged = torch.zeros(B, dtype=torch.bool, device=dev)

    for step in range(max_steps):
        F = residual_fn(x)
        max_res = F.abs().amax(dim=1)
        converged = converged | (max_res < tol)
        if converged.all():
            break

        active = ~converged

        # Finite-difference Jacobian
        eps = (x.abs() * 1e-6).clamp(min=1e-6)
        J = torch.zeros(B, n_vars, n_vars, device=dev, dtype=dt)
        for i in range(n_vars):
            x_p = x.clone()
            x_p[:, i] += eps[:, i]
            J[:, :, i] = (residual_fn(x_p) - F) / eps[:, i].unsqueeze(1)

        # Regularize, solve, clamp, mask
        reg = 1e-10 * torch.eye(n_vars, device=dev, dtype=dt).unsqueeze(0)
        try:
            dx = torch.linalg.solve(J + reg, -F.unsqueeze(-1)).squeeze(-1)
        except Exception:
            break
        dx = dx.clamp(-20, 20) * active.unsqueeze(1)
        x = x + dx

    return x, converged


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _temp_scale_jfet(jfet_params, temp_c):
    d = jfet_params
    temp_k = temp_c + 273.15
    dt_c = temp_c - 27.0
    ratio = temp_k / TNOM_K
    eg_term = d["eg"] * Q_ELEC * dt_c / (K_BOLTZ * temp_k * TNOM_K)
    return {
        'beta': d["beta"] * np.exp(d["betatce"] / 100.0 * dt_c),
        'vto': d["vto"] + d["vtotc"] * dt_c,
        'lmbda': d["lmbda"],
        'is_': d["is_"] * (ratio ** (d["xti"] / d["n"])) * np.exp(eg_term / d["n"]),
        'n': d["n"],
        'isr': (d["isr"] * (ratio ** (d["xti"] / d["nr"])) * np.exp(eg_term / d["nr"])
                if d["isr"] > 0 else 0.0),
        'nr': d["nr"],
        'alpha': d["alpha"], 'vk': d["vk"],
        'rs': d["rs"], 'rd': d["rd"],
    }


def _truth_table(gate_type_str, n_inputs):
    if gate_type_str.startswith("NOR"):
        expected = lambda c: not any(c)
    elif gate_type_str.startswith("NAND"):
        expected = lambda c: not all(c)
    elif gate_type_str == "INV":
        expected = lambda c: not c[0]
    else:
        raise ValueError(f"Unsupported: {gate_type_str}")

    if n_inputs <= 8:
        entries = []
        for i in range(2 ** n_inputs):
            combo = tuple(bool((i >> bit) & 1) for bit in range(n_inputs))
            entries.append((combo, expected(combo)))
        return entries
    else:
        entries = []
        all_lo = tuple(False for _ in range(n_inputs))
        entries.append((all_lo, expected(all_lo)))
        all_hi = tuple(True for _ in range(n_inputs))
        entries.append((all_hi, expected(all_hi)))
        for k in range(n_inputs):
            one_hot = tuple(True if i == k else False for i in range(n_inputs))
            entries.append((one_hot, expected(one_hot)))
            one_cold = tuple(False if i == k else True for i in range(n_inputs))
            entries.append((one_cold, expected(one_cold)))
        return entries


def _cpu_seed_guesses(gate_type_str, n_inputs, table, board_dict, mean_temp):
    """Run CPU solver on ONE representative R-combo per truth table entry.
    Returns list of (n_vars,) lists — one per truth table entry.
    Cost: ~10ms per entry, negligible.
    """
    from model import NChannelJFET, GateType, solve_any_gate
    jfet = NChannelJFET(**board_dict["jfet_params"]).at_temp(mean_temp)
    gt = GateType(gate_type_str)
    v_map = {False: board_dict["v_low"], True: board_dict["v_high"]}
    v_pos = board_dict["v_pos"]
    v_neg = board_dict["v_neg"]

    results = []
    for combo, _ in table:
        v_ins = [v_map[b] for b in combo]
        try:
            res = solve_any_gate(gt, v_ins, v_pos, v_neg,
                                 10000.0, 3000.0, 3000.0,
                                 jfet, jfet, mean_temp)
            if gate_type_str.startswith("NAND") and n_inputs > 1:
                vals = [res["v_a"]] + list(res.get("v_mids", [])) + [res["v_b"]]
            else:
                vals = [res["v_a"], res["v_b"]]
        except Exception:
            if gate_type_str.startswith("NAND") and n_inputs > 1:
                vals = [5.0] + [2.5] * (n_inputs - 1) + [3.0]
            else:
                vals = [5.0, 3.0]
        results.append(vals)
    return results


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def gpu_solve_batch(gate_type_str: str, X: np.ndarray,
                    board_dict: dict, device: str = None) -> np.ndarray:
    """Solve circuits on GPU. All TT entries x guesses in one mega-batch.

    Args:
        gate_type_str: "INV", "NAND2", "NOR3", etc.
        X: (N, 6) — [R1, R2, R3, V+, V-, temp]
        board_dict: from _board_to_dict
        device: 'cuda', 'cpu', or None

    Returns:
        (N, 4) — [v_out_high, v_out_low, avg_power, converged]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dt = torch.float64

    # Parse gate type
    if gate_type_str == "INV":
        n_inputs, is_nand = 1, False
    elif gate_type_str.startswith("NAND"):
        n_inputs, is_nand = int(gate_type_str[4:]), True
    elif gate_type_str.startswith("NOR"):
        n_inputs, is_nand = int(gate_type_str[3:]), False
    else:
        raise ValueError(f"Unsupported: {gate_type_str}")

    N = X.shape[0]
    n_vars = (n_inputs + 1) if is_nand else 2

    # Circuit params
    r1 = torch.tensor(X[:, 0], dtype=dt, device=dev)
    r2 = torch.tensor(X[:, 1], dtype=dt, device=dev)
    r3 = torch.tensor(X[:, 2], dtype=dt, device=dev)
    v_pos = torch.tensor(X[:, 3], dtype=dt, device=dev)
    v_neg = torch.tensor(X[:, 4], dtype=dt, device=dev)
    r_load = r2 + r3
    mean_temp = float(X[:, 5].mean())
    vt_val = K_BOLTZ * (mean_temp + 273.15) / Q_ELEC
    vt = torch.full((N,), vt_val, dtype=dt, device=dev)

    jp = _temp_scale_jfet(board_dict["jfet_params"], mean_temp)
    v_high, v_low = board_dict["v_high"], board_dict["v_low"]
    table = _truth_table(gate_type_str, n_inputs)
    n_tt = len(table)

    # --- Build initial guesses ---
    # Hardcoded guesses matching CPU solver (model/gate.py)
    if is_nand:
        hardcoded_ab = [(5.0, 3.0), (5.0, 0.0), (1.0, -1.0), (10.0, 5.0)]
    else:
        hardcoded_ab = [(5.0, 3.0), (5.0, 0.0), (1.0, -1.0)]

    # CPU seed: one solve per truth table entry
    cpu_seeds = _cpu_seed_guesses(gate_type_str, n_inputs, table,
                                   board_dict, mean_temp)
    n_hardcoded = len(hardcoded_ab)
    n_guesses = n_hardcoded + 1  # hardcoded + 1 CPU seed

    # --- Build mega-batch: N samples x T truth table entries x G guesses ---
    # Total rows: N * T * G
    mega_size = N * n_tt * n_guesses

    # Pre-build input voltage tensor for all TT entries: (T, N, n_inputs)
    v_ins_per_tt = torch.zeros(n_tt, N, n_inputs, dtype=dt, device=dev)
    for t_idx, (combo, _) in enumerate(table):
        for k, b in enumerate(combo):
            v_ins_per_tt[t_idx, :, k] = v_high if b else v_low

    # Build x0 mega-batch: (mega_size, n_vars)
    # Layout: [TT0_guess0(N), TT0_guess1(N), ..., TT0_guessG(N),
    #          TT1_guess0(N), ..., TTT_guessG(N)]
    x0 = torch.zeros(mega_size, n_vars, dtype=dt, device=dev)
    # Also build corresponding circuit params
    r1_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    r_load_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    v_pos_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    v_neg_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    vt_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    v_ins_mega = torch.zeros(mega_size, n_inputs, dtype=dt, device=dev)

    idx = 0
    for t_idx in range(n_tt):
        for g_idx in range(n_guesses):
            sl = slice(idx, idx + N)
            # Circuit params — same for all guesses/TT entries
            r1_mega[sl] = r1
            r_load_mega[sl] = r_load
            v_pos_mega[sl] = v_pos
            v_neg_mega[sl] = v_neg
            vt_mega[sl] = vt
            v_ins_mega[sl] = v_ins_per_tt[t_idx]

            # Initial guess
            if g_idx < n_hardcoded:
                # Hardcoded guess
                va_g, vb_g = hardcoded_ab[g_idx]
                if is_nand:
                    x0[sl, 0] = va_g
                    for mk in range(n_inputs - 1):
                        x0[sl, 1 + mk] = va_g * (n_inputs - 1 - mk) / n_inputs
                    x0[sl, -1] = vb_g
                else:
                    x0[sl, 0] = va_g
                    x0[sl, 1] = vb_g
            else:
                # CPU seed guess
                seed = cpu_seeds[t_idx]
                for v_idx, val in enumerate(seed):
                    x0[sl, v_idx] = val

            idx += N

    # --- ONE Newton solve on the entire mega-batch ---
    if is_nand:
        def res_fn(x):
            return _nand_residuals(x, v_ins_mega, r1_mega, r_load_mega,
                                   v_pos_mega, v_neg_mega, vt_mega, jp, n_inputs)
    else:
        def res_fn(x):
            return _inv_nor_residuals(x, v_ins_mega, r1_mega, r_load_mega,
                                      v_pos_mega, v_neg_mega, vt_mega, jp)

    x_sol, conv_mega = _newton_batch(res_fn, x0, n_vars,
                                      max_steps=25, tol=1e-8)

    # Compute final residuals for picking best guess
    F_final = res_fn(x_sol)
    res_mag = F_final.abs().amax(dim=1)  # (mega_size,)

    # --- Unstack and pick best guess per (sample, TT entry) ---
    # Reshape to (n_tt, n_guesses, N)
    res_mag = res_mag.view(n_tt, n_guesses, N)
    x_sol = x_sol.view(n_tt, n_guesses, N, n_vars)
    conv_mega = conv_mega.view(n_tt, n_guesses, N)

    # Best guess per (TT entry, sample): lowest residual
    best_g = res_mag.argmin(dim=1)  # (n_tt, N)
    sample_idx = torch.arange(N, device=dev)

    # Results per truth table entry
    v_out_high = torch.zeros(N, dtype=dt, device=dev)
    v_out_low = torch.zeros(N, dtype=dt, device=dev)
    total_power = torch.zeros(N, dtype=dt, device=dev)
    all_converged = torch.ones(N, dtype=torch.bool, device=dev)

    for t_idx, (combo, _) in enumerate(table):
        bg = best_g[t_idx]  # (N,)
        x_best = x_sol[t_idx, bg, sample_idx]  # (N, n_vars)
        res_best = res_mag[t_idx, bg, sample_idx]  # (N,)
        entry_conv = res_best < 1e-4
        all_converged = all_converged & entry_conv

        v_a = x_best[:, 0]
        v_b = x_best[:, -1] if is_nand else x_best[:, 1]

        # Output voltage
        i_load = (v_b - v_neg) / r_load
        v_out = v_neg + i_load * r3

        # Power
        i_r1 = (v_pos - v_a) / r1
        i_j2 = _ids_j(v_a - v_b, v_pos - v_b, jp)
        total_power = total_power + v_pos * (i_r1 + i_j2) + (-v_neg) * i_load

        if all(not b for b in combo):
            v_out_high = v_out.clone()
        if all(b for b in combo):
            v_out_low = v_out.clone()

    avg_power = total_power / n_tt

    result = torch.stack([
        v_out_high, v_out_low, avg_power,
        all_converged.to(dt),
    ], dim=1)

    return result.cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Network-based GPU solver (arbitrary topology)
# ---------------------------------------------------------------------------

def gpu_solve_network(network, X: np.ndarray, board_dict: dict,
                      device: str = None) -> np.ndarray:
    """Solve circuits with arbitrary pull-down topology on GPU.

    Args:
        network: PulldownNetwork (Leaf/Series/Parallel)
        X: (N, 6) — [R1, R2, R3, V+, V-, temp]
        board_dict: from _board_to_dict
        device: 'cuda', 'cpu', or None

    Returns:
        (N, 4) — [v_out_high, v_out_low, avg_power, converged]
    """
    from model.network import (
        input_names, n_solver_vars, network_truth_table,
        count_midpoints, Leaf, Series, Parallel,
    )
    from model import solve_network as cpu_solve_network, NChannelJFET

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    dt = torch.float64

    N = X.shape[0]
    names = input_names(network)
    n_inputs = len(names)
    n_vars = n_solver_vars(network)
    n_mids = count_midpoints(network)
    table = network_truth_table(network)
    n_tt = len(table)

    r1 = torch.tensor(X[:, 0], dtype=dt, device=dev)
    r2 = torch.tensor(X[:, 1], dtype=dt, device=dev)
    r3 = torch.tensor(X[:, 2], dtype=dt, device=dev)
    v_pos = torch.tensor(X[:, 3], dtype=dt, device=dev)
    v_neg = torch.tensor(X[:, 4], dtype=dt, device=dev)
    r_load = r2 + r3

    mean_temp = float(X[:, 5].mean())
    vt_val = K_BOLTZ * (mean_temp + 273.15) / Q_ELEC
    vt = torch.full((N,), vt_val, dtype=dt, device=dev)
    jp = _temp_scale_jfet(board_dict["jfet_params"], mean_temp)
    v_high, v_low = board_dict["v_high"], board_dict["v_low"]

    # --- Initial guesses ---
    hardcoded_ab = [(5.0, 3.0), (5.0, 0.0), (1.0, -1.0), (10.0, 5.0)]

    # CPU seed per truth table entry
    jfet_cpu = NChannelJFET(**board_dict["jfet_params"]).at_temp(mean_temp)
    cpu_seeds = []
    v_map = {False: v_low, True: v_high}
    for combo, _ in table:
        v_ins_dict = {names[i]: v_map[combo[i]] for i in range(n_inputs)}
        try:
            res = cpu_solve_network(network, v_ins_dict,
                                    board_dict["v_pos"], board_dict["v_neg"],
                                    10000.0, 3000.0, 3000.0,
                                    jfet_cpu, jfet_cpu, mean_temp)
            vals = [res["v_a"], res["v_b"]] + list(res.get("v_mids", []))
        except Exception:
            vals = [5.0, 3.0] + [2.5] * n_mids
        cpu_seeds.append(vals)

    n_hardcoded = len(hardcoded_ab)
    n_guesses = n_hardcoded + 1  # hardcoded + CPU seed

    # --- Build mega-batch ---
    mega_size = N * n_tt * n_guesses

    x0 = torch.zeros(mega_size, n_vars, dtype=dt, device=dev)
    r1_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    r_load_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    v_pos_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    v_neg_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    vt_mega = torch.zeros(mega_size, dtype=dt, device=dev)
    # Input voltages per JFET input name
    v_ins_mega = {name: torch.zeros(mega_size, dtype=dt, device=dev) for name in names}

    idx = 0
    for t_idx, (combo, _) in enumerate(table):
        for g_idx in range(n_guesses):
            sl = slice(idx, idx + N)
            r1_mega[sl] = r1
            r_load_mega[sl] = r_load
            v_pos_mega[sl] = v_pos
            v_neg_mega[sl] = v_neg
            vt_mega[sl] = vt

            for k, name in enumerate(names):
                v_ins_mega[name][sl] = v_high if combo[k] else v_low

            # Initial guess
            if g_idx < n_hardcoded:
                va_g, vb_g = hardcoded_ab[g_idx]
                x0[sl, 0] = va_g
                x0[sl, 1] = vb_g
                for mk in range(n_mids):
                    x0[sl, 2 + mk] = va_g * (n_mids - mk) / max(n_mids + 1, 1)
            else:
                seed = cpu_seeds[t_idx]
                for v_idx, val in enumerate(seed):
                    if v_idx < n_vars:
                        x0[sl, v_idx] = val

            idx += N

    # --- ONE Newton solve ---
    def res_fn(x):
        return _network_residuals_gpu(x, v_ins_mega, r1_mega, r_load_mega,
                                       v_pos_mega, v_neg_mega, vt_mega, jp, network)

    x_sol, conv_mega = _newton_batch(res_fn, x0, n_vars, max_steps=25, tol=1e-8)

    F_final = res_fn(x_sol)
    res_mag = F_final.abs().amax(dim=1)

    # --- Unstack and pick best ---
    res_mag = res_mag.view(n_tt, n_guesses, N)
    x_sol = x_sol.view(n_tt, n_guesses, N, n_vars)

    best_g = res_mag.argmin(dim=1)
    sample_idx = torch.arange(N, device=dev)

    v_out_high = torch.zeros(N, dtype=dt, device=dev)
    v_out_low = torch.zeros(N, dtype=dt, device=dev)
    total_power = torch.zeros(N, dtype=dt, device=dev)
    all_converged = torch.ones(N, dtype=torch.bool, device=dev)

    for t_idx, (combo, _) in enumerate(table):
        bg = best_g[t_idx]
        x_best = x_sol[t_idx, bg, sample_idx]
        res_best = res_mag[t_idx, bg, sample_idx]
        entry_conv = res_best < 1e-4
        all_converged = all_converged & entry_conv

        v_a = x_best[:, 0]
        v_b = x_best[:, 1]

        i_load = (v_b - v_neg) / r_load
        v_out = v_neg + i_load * r3

        i_r1 = (v_pos - v_a) / r1
        i_j2 = _ids_j(v_a - v_b, v_pos - v_b, jp)
        total_power = total_power + v_pos * (i_r1 + i_j2) + (-v_neg) * i_load

        if all(not b for b in combo):
            v_out_high = v_out.clone()
        if all(b for b in combo):
            v_out_low = v_out.clone()

    avg_power = total_power / n_tt

    result = torch.stack([
        v_out_high, v_out_low, avg_power,
        all_converged.to(dt),
    ], dim=1)

    return result.cpu().numpy().astype(np.float32)
