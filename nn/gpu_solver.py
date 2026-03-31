"""GPU-accelerated circuit solver using PyTorch.

Drop-in replacement for the CPU fsolve-based data generation.
All JFET physics and Newton root-finding run as batched tensor ops,
enabling millions of circuit solves simultaneously on GPU.

Works on CPU too (just slower) — no CUDA required for correctness.
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_BOLTZ = 1.380649e-23
Q_ELEC = 1.602176634e-19
TNOM_K = 300.15


# ---------------------------------------------------------------------------
# JFET parameters as a tensor-friendly struct
# ---------------------------------------------------------------------------

@dataclass
class JFETParams:
    """JFET parameters stored as scalars (shared across entire batch)."""
    beta: float
    vto: float
    lmbda: float
    is_: float
    n: float
    isr: float
    nr: float
    alpha: float
    vk: float
    rd: float
    rs: float

    @classmethod
    def from_dict(cls, d: dict, temp_c: float = 27.0) -> "JFETParams":
        """Create from board_dict jfet_params, with temperature scaling."""
        temp_k = temp_c + 273.15
        dt_c = temp_c - 27.0
        ratio = temp_k / TNOM_K

        beta = d["beta"] * np.exp(d["betatce"] / 100.0 * dt_c)
        vto = d["vto"] + d["vtotc"] * dt_c

        eg_term = d["eg"] * Q_ELEC * dt_c / (K_BOLTZ * temp_k * TNOM_K)
        is_ = d["is_"] * (ratio ** (d["xti"] / d["n"])) * np.exp(eg_term / d["n"])

        isr = 0.0
        if d["isr"] > 0:
            isr = d["isr"] * (ratio ** (d["xti"] / d["nr"])) * np.exp(eg_term / d["nr"])

        return cls(
            beta=beta, vto=vto, lmbda=d["lmbda"],
            is_=is_, n=d["n"], isr=isr, nr=d["nr"],
            alpha=d["alpha"], vk=d["vk"],
            rd=d["rd"], rs=d["rs"],
        )


# ---------------------------------------------------------------------------
# Layer 1: JFET Device Physics (batched tensor ops)
# ---------------------------------------------------------------------------

def _ids_intrinsic(vgs: torch.Tensor, vds: torch.Tensor,
                   p: JFETParams) -> torch.Tensor:
    """Shichman-Hodges channel current, branchless via torch.where.

    Handles reverse polarity, cutoff, linear, and saturation regions.
    """
    # Handle reverse polarity: if vds < 0, ids = -ids(vgs-vds, -vds)
    rev_mask = vds < 0
    vgs_fwd = torch.where(rev_mask, vgs - vds, vgs)
    vds_fwd = torch.where(rev_mask, -vds, vds)

    # Cutoff: vgs <= vto
    vsat = vgs_fwd - p.vto  # > 0 when not in cutoff
    active = vgs_fwd > p.vto

    # Saturation vs linear
    sat_mask = vds_fwd >= vsat
    clm = 1.0 + p.lmbda * vds_fwd

    ids_sat = p.beta * vsat.clamp(min=0).pow(2) * clm
    ids_lin = p.beta * (2.0 * vsat.clamp(min=0) * vds_fwd - vds_fwd.pow(2)) * clm

    ids = torch.where(sat_mask, ids_sat, ids_lin)
    ids = torch.where(active, ids, torch.zeros_like(ids))

    # Ionization current
    if p.alpha > 0.0 and p.vk > 0.0:
        # Clamp vds_fwd to avoid div-by-zero in exp(-vk/vds)
        vds_safe = vds_fwd.clamp(min=1e-10)
        ion = 1.0 + p.alpha * vds_fwd * torch.exp(-p.vk / vds_safe)
        ion = torch.where(vds_fwd > 0, ion, torch.ones_like(ion))
        ids = ids * ion

    # Apply reverse polarity sign
    ids = torch.where(rev_mask, -ids, ids)
    return ids


def ids_batched(vgs: torch.Tensor, vds: torch.Tensor,
                p: JFETParams, n_iters: int = 10) -> torch.Tensor:
    """Channel current with parasitic Rs/Rd, unrolled iteration."""
    if p.rs == 0.0 and p.rd == 0.0:
        return _ids_intrinsic(vgs, vds, p)

    id_est = _ids_intrinsic(vgs, vds, p)
    for _ in range(n_iters):
        vgs_int = vgs - id_est * p.rs
        vds_int = vds - id_est * (p.rs + p.rd)
        id_new = _ids_intrinsic(vgs_int, vds_int, p)
        id_est = 0.5 * (id_est + id_new)
    return id_est


def _diode_current(vd: torch.Tensor, is_: float, n: float,
                   vt: torch.Tensor) -> torch.Tensor:
    """PN junction diode current with overflow clamping."""
    x = vd / (n * vt)
    # Linear approximation for x > 40 (prevents exp overflow)
    exp_clamped = torch.exp(x.clamp(max=40.0))
    linear_tail = torch.exp(torch.tensor(40.0, device=vd.device)) * (1.0 + x - 40.0)
    result = is_ * torch.where(x > 40.0, linear_tail, exp_clamped - 1.0)
    return result


def gate_currents_batched(vgs: torch.Tensor, vgd: torch.Tensor,
                          p: JFETParams, vt: torch.Tensor):
    """Gate junction currents (gate-source and gate-drain diodes).

    Returns (igs, igd).
    """
    igs = _diode_current(vgs, p.is_, p.n, vt)
    igd = _diode_current(vgd, p.is_, p.n, vt)
    if p.isr > 0.0:
        igs = igs + _diode_current(vgs, p.isr, p.nr, vt)
        igd = igd + _diode_current(vgd, p.isr, p.nr, vt)
    return igs, igd


def thermal_voltage(temp_c: torch.Tensor) -> torch.Tensor:
    """Thermal voltage kT/q from temperature in Celsius."""
    return K_BOLTZ * (temp_c + 273.15) / Q_ELEC


# ---------------------------------------------------------------------------
# Layer 2: Circuit Equations (batched KCL residuals)
# ---------------------------------------------------------------------------

def _inv_nor_residuals(x: torch.Tensor, v_ins_list: list,
                       r1: torch.Tensor, r_load: torch.Tensor,
                       v_pos: torch.Tensor, v_neg: torch.Tensor,
                       p_j1: JFETParams, p_j2: JFETParams,
                       vt: torch.Tensor) -> torch.Tensor:
    """KCL residuals for INV or NOR-N gate.

    x: (batch, 2) — [v_a, v_b]
    v_ins_list: list of (batch,) tensors — input voltages per J1
    Returns: (batch, 2) residuals
    """
    v_a = x[:, 0]
    v_b = x[:, 1]

    # Sum of all parallel J1 currents and gate currents
    i_j1_total = torch.zeros_like(v_a)
    igd_j1_total = torch.zeros_like(v_a)
    for v_in in v_ins_list:
        i_j1 = ids_batched(v_in, v_a, p_j1)
        i_j1_total = i_j1_total + i_j1
        # Gate current for topmost (for KCL at node A)
        vgs_int = v_in - i_j1 * p_j1.rs
        vgd_int = v_in - (v_a - i_j1 * p_j1.rd)
        _, igd = gate_currents_batched(vgs_int, vgd_int, p_j1, vt)
        igd_j1_total = igd_j1_total + igd

    # J2 (load transistor)
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = ids_batched(vgs2, vds2, p_j2)
    vgs2_int = v_a - (v_b + i_j2 * p_j2.rs)
    vgd2_int = v_a - (v_pos - i_j2 * p_j2.rd)
    igs_j2, igd_j2 = gate_currents_batched(vgs2_int, vgd2_int, p_j2, vt)

    # KCL at Node A
    eq_a = (v_pos - v_a) / r1 + igd_j1_total - i_j1_total - (igs_j2 + igd_j2)
    # KCL at Node B
    eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load

    return torch.stack([eq_a, eq_b], dim=1)


def _solve_nand_midpoints(v_a: torch.Tensor, v_ins_list: list,
                          p_j1: JFETParams,
                          n_iters: int = 30) -> torch.Tensor:
    """Solve for series chain midpoint voltages given v_a.

    For NAND-N, the series chain has N JFETs. Given v_a (top node),
    find midpoint voltages such that current is continuous through the chain.

    Uses bisection-style iteration: current through J1[0] determines the
    chain current, then each midpoint adjusts to pass that current.
    """
    N = len(v_ins_list)
    batch = v_a.shape[0]

    if N <= 1:
        return torch.zeros(batch, 0, device=v_a.device, dtype=torch.float64)

    # Initialize midpoints: linear from v_a to 0
    mids = torch.zeros(batch, N - 1, device=v_a.device, dtype=torch.float64)
    for k in range(N - 1):
        frac = (N - 1 - k) / N
        mids[:, k] = v_a * frac

    # Iteratively adjust midpoints for current continuity
    for _ in range(n_iters):
        # Build nodes: [v_a, mid_0, ..., mid_{N-2}, 0]
        nodes = torch.zeros(batch, N + 1, device=v_a.device, dtype=torch.float64)
        nodes[:, 0] = v_a
        nodes[:, 1:N] = mids
        # nodes[:, N] = 0 (GND)

        # Compute all chain currents
        i_chain = torch.zeros(batch, N, device=v_a.device, dtype=torch.float64)
        for k in range(N):
            vgs_k = v_ins_list[k] - nodes[:, k + 1]
            vds_k = nodes[:, k] - nodes[:, k + 1]
            i_chain[:, k] = ids_batched(vgs_k, vds_k, p_j1)

        # Target: all currents equal to the average
        i_target = i_chain.mean(dim=1, keepdim=True)

        # Adjust midpoints: if i_chain[k] > i_target, increase mid[k] (reduce VDS)
        for k in range(N - 1):
            err = i_chain[:, k] - i_chain[:, k + 1]
            # Move midpoint to equalize: positive err means too much current above,
            # so raise the midpoint
            mids[:, k] = mids[:, k] + 0.3 * err / (i_chain[:, k].abs().clamp(min=1e-12) + 1e-12)
            mids[:, k] = torch.min(torch.max(mids[:, k], torch.zeros_like(v_a)), v_a)

    return mids


def _nand_residuals_reduced(x_ab: torch.Tensor, v_ins_list: list,
                            r1: torch.Tensor, r_load: torch.Tensor,
                            v_pos: torch.Tensor, v_neg: torch.Tensor,
                            p_j1: JFETParams, p_j2: JFETParams,
                            vt: torch.Tensor,
                            n_inputs: int) -> torch.Tensor:
    """Reduced 2x2 NAND residuals: solve midpoints internally.

    x_ab: (batch, 2) — [v_a, v_b]
    Returns: (batch, 2) residuals [eq_a, eq_b]
    """
    v_a = x_ab[:, 0]
    v_b = x_ab[:, 1]
    batch = v_a.shape[0]
    N = n_inputs

    # Solve midpoints for this v_a
    mids = _solve_nand_midpoints(v_a, v_ins_list, p_j1)

    # Build full nodes: [v_a, mid_0, ..., mid_{N-2}, 0]
    nodes = torch.zeros(batch, N + 1, device=v_a.device, dtype=torch.float64)
    nodes[:, 0] = v_a
    if N > 1:
        nodes[:, 1:N] = mids
    # nodes[:, N] = 0

    # J1 chain current (use topmost)
    vgs_top = v_ins_list[0] - nodes[:, 1]
    vds_top = v_a - nodes[:, 1]
    i_j1_top = ids_batched(vgs_top, vds_top, p_j1)

    # Gate current of J1[0]
    vgs_int = v_ins_list[0] - (nodes[:, 1] + i_j1_top * p_j1.rs)
    vgd_int = v_ins_list[0] - (v_a - i_j1_top * p_j1.rd)
    _, igd_j1_top = gate_currents_batched(vgs_int, vgd_int, p_j1, vt)

    # J2
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = ids_batched(vgs2, vds2, p_j2)
    vgs2_int = v_a - (v_b + i_j2 * p_j2.rs)
    vgd2_int = v_a - (v_pos - i_j2 * p_j2.rd)
    igs_j2, igd_j2 = gate_currents_batched(vgs2_int, vgd2_int, p_j2, vt)

    eq_a = (v_pos - v_a) / r1 + igd_j1_top - i_j1_top - (igs_j2 + igd_j2)
    eq_b = i_j2 + igs_j2 - (v_b - v_neg) / r_load

    return torch.stack([eq_a, eq_b], dim=1)


def _nand_residuals(x: torch.Tensor, v_ins_list: list,
                    r1: torch.Tensor, r_load: torch.Tensor,
                    v_pos: torch.Tensor, v_neg: torch.Tensor,
                    p_j1: JFETParams, p_j2: JFETParams,
                    vt: torch.Tensor) -> torch.Tensor:
    """KCL residuals for NAND-N gate.

    x: (batch, N+1) — [v_a, v_mid_0, ..., v_mid_{N-2}, v_b]
    v_ins_list: list of N (batch,) tensors — input voltages
    Returns: (batch, N+1) residuals
    """
    N = len(v_ins_list)
    batch = x.shape[0]

    v_a = x[:, 0]
    v_b = x[:, -1]

    # Build node voltage stack: [v_a, v_mid_0, ..., v_mid_{N-2}, GND=0]
    # For the series chain: J1[k] connects nodes[k] (drain) to nodes[k+1] (source)
    nodes = torch.zeros(batch, N + 1, device=x.device)
    nodes[:, 0] = v_a
    if N > 1:
        nodes[:, 1:N] = x[:, 1:N]  # midpoints
    # nodes[:, N] = 0.0 (GND, already zeros)

    # Compute J1 chain currents
    i_j1 = torch.zeros(batch, N, device=x.device)
    for k in range(N):
        vgs_k = v_ins_list[k] - nodes[:, k + 1]
        vds_k = nodes[:, k] - nodes[:, k + 1]
        i_j1[:, k] = ids_batched(vgs_k, vds_k, p_j1)

    # Gate current of topmost J1 (for KCL at node A)
    i_j1_top = i_j1[:, 0]
    vgs_top_int = v_ins_list[0] - (nodes[:, 1] + i_j1_top * p_j1.rs)
    vgd_top_int = v_ins_list[0] - (v_a - i_j1_top * p_j1.rd)
    _, igd_j1_top = gate_currents_batched(vgs_top_int, vgd_top_int, p_j1, vt)

    # J2 (load transistor)
    vgs2 = v_a - v_b
    vds2 = v_pos - v_b
    i_j2 = ids_batched(vgs2, vds2, p_j2)
    vgs2_int = v_a - (v_b + i_j2 * p_j2.rs)
    vgd2_int = v_a - (v_pos - i_j2 * p_j2.rd)
    igs_j2, igd_j2 = gate_currents_batched(vgs2_int, vgd2_int, p_j2, vt)

    # Residuals
    residuals = torch.zeros(batch, N + 1, device=x.device)

    # KCL at Node A
    residuals[:, 0] = (v_pos - v_a) / r1 + igd_j1_top - i_j1_top - (igs_j2 + igd_j2)

    # Current continuity at midpoints: i_j1[k] - i_j1[k+1] = 0
    for k in range(N - 1):
        residuals[:, 1 + k] = i_j1[:, k] - i_j1[:, k + 1]

    # KCL at Node B
    residuals[:, -1] = i_j2 + igs_j2 - (v_b - v_neg) / r_load

    return residuals


# ---------------------------------------------------------------------------
# Layer 3: Batched Newton Solver
# ---------------------------------------------------------------------------

def _newton_solve(residual_fn, x0: torch.Tensor,
                  max_steps: int = 20, tol: float = 1e-8,
                  damping: float = 1.0) -> tuple:
    """Batched Newton solver using autograd Jacobians.

    Args:
        residual_fn: callable(x) -> (batch, n_vars) residuals
        x0: (batch, n_vars) initial guess
        max_steps: maximum Newton iterations
        tol: convergence tolerance on max residual
        damping: step damping factor (1.0 = full Newton step)

    Returns:
        (x_solution, converged_mask)
    """
    x = x0.clone().to(torch.float64)
    batch, n_vars = x.shape
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)

    for step in range(max_steps):
        F_val = residual_fn(x).to(torch.float64)

        # Compute Jacobian via finite differences
        # Use adaptive eps based on magnitude of x
        J = torch.zeros(batch, n_vars, n_vars, dtype=torch.float64, device=x.device)
        for i in range(n_vars):
            eps_i = (x[:, i].abs() * 1e-6).clamp(min=1e-6)
            x_p = x.clone()
            x_p[:, i] += eps_i
            F_p = residual_fn(x_p).to(torch.float64)
            J[:, :, i] = (F_p - F_val) / eps_i.unsqueeze(1)

        # Check convergence
        max_residual = F_val.abs().max(dim=1).values
        newly_converged = max_residual < tol
        converged = converged | newly_converged

        if converged.all():
            break

        # Newton step: dx = -J^{-1} @ F
        # Only update non-converged circuits
        active = ~converged
        if active.any():
            J_active = J[active]
            F_active = F_val[active]

            # Regularize Jacobian to avoid singularity
            reg = 1e-10 * torch.eye(n_vars, dtype=torch.float64,
                                     device=x.device).unsqueeze(0)
            J_reg = J_active + reg

            try:
                dx = torch.linalg.solve(J_reg, -F_active.unsqueeze(-1))
                dx = dx.squeeze(-1)
                dx = dx.clamp(-20, 20)
                x_new = x.clone()
                x_new[active] = x_new[active] + damping * dx
                x = x_new
            except Exception:
                # Singular Jacobian — try with more regularization
                reg2 = 1e-6 * torch.eye(n_vars, dtype=torch.float64,
                                          device=x.device).unsqueeze(0)
                try:
                    dx = torch.linalg.solve(J_active + reg2,
                                            -F_active.unsqueeze(-1))
                    dx = dx.squeeze(-1).clamp(-20, 20)
                    x_new = x.clone()
                    x_new[active] = x_new[active] + damping * 0.5 * dx
                    x = x_new
                except Exception:
                    pass

    return x, converged


def _newton_solve_2x2(residual_fn, x0: torch.Tensor,
                      max_steps: int = 20, tol: float = 1e-8,
                      damping: float = 1.0) -> tuple:
    """Optimized Newton solver for 2x2 systems (INV/NOR).

    Uses analytical Jacobian inversion instead of torch.linalg.solve.
    Computes Jacobian via finite differences (faster than autograd for 2x2).
    """
    x = x0.clone()
    batch = x.shape[0]
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)
    eps = 1e-7

    for step in range(max_steps):
        F_val = residual_fn(x)

        # Check convergence
        max_residual = F_val.abs().max(dim=1).values
        newly_converged = max_residual < tol
        converged = converged | newly_converged

        if converged.all():
            break

        # Jacobian via finite differences (2x2 = 4 evaluations)
        x_pa = x.clone(); x_pa[:, 0] += eps
        x_pb = x.clone(); x_pb[:, 1] += eps
        F_pa = residual_fn(x_pa)
        F_pb = residual_fn(x_pb)

        dFda = (F_pa - F_val) / eps  # (batch, 2)
        dFdb = (F_pb - F_val) / eps  # (batch, 2)

        # Analytical 2x2 inverse: J = [[a,b],[c,d]], J^-1 = 1/det * [[d,-b],[-c,a]]
        a = dFda[:, 0]; b = dFdb[:, 0]
        c = dFda[:, 1]; d = dFdb[:, 1]
        det = a * d - b * c
        det = det.clamp(min=1e-20)  # prevent div by zero

        # dx = -J^{-1} @ F
        dx0 = -(d * F_val[:, 0] - b * F_val[:, 1]) / det
        dx1 = -(-c * F_val[:, 0] + a * F_val[:, 1]) / det

        # Clamp step size to prevent divergence
        max_step = 20.0
        dx0 = dx0.clamp(-max_step, max_step)
        dx1 = dx1.clamp(-max_step, max_step)

        # Adaptive damping: reduce step if residual would increase
        x_trial = x.clone()
        x_trial[:, 0] = x[:, 0] + damping * dx0
        x_trial[:, 1] = x[:, 1] + damping * dx1
        F_trial = residual_fn(x_trial)
        trial_res = F_trial.abs().max(dim=1).values
        # If trial is worse, use half step
        worse = trial_res > max_residual * 1.5
        dx0 = torch.where(worse, dx0 * 0.25, dx0)
        dx1 = torch.where(worse, dx1 * 0.25, dx1)

        # Update only non-converged
        active = ~converged
        x[active, 0] = x[active, 0] + damping * dx0[active]
        x[active, 1] = x[active, 1] + damping * dx1[active]

    return x, converged


# ---------------------------------------------------------------------------
# Layer 4: High-level solve interface
# ---------------------------------------------------------------------------

def _make_guess_2x2(batch: int, v_pos: torch.Tensor,
                    device: torch.device,
                    a_frac: float, b_frac: float) -> torch.Tensor:
    """Create a 2x2 initial guess: v_a = v_pos*a_frac, v_b = v_pos*b_frac."""
    x0 = torch.zeros(batch, 2, dtype=torch.float64, device=device)
    x0[:, 0] = v_pos * a_frac
    x0[:, 1] = v_pos * b_frac
    return x0


def _nand_cpu_seed_guess(combo: tuple, n_inputs: int,
                         board_dict: dict,
                         device: torch.device) -> torch.Tensor:
    """Run CPU solver on ONE representative R-combo to seed NAND initial guess.

    Costs ~10ms for NAND2, negligible vs batch solve time.
    Returns (1, N+1) tensor: [v_a, mid_0, ..., mid_{N-2}, v_b]
    """
    from model import NChannelJFET, GateType, solve_any_gate

    jfet = NChannelJFET(**board_dict["jfet_params"]).at_temp(board_dict["temp_c"])
    v_pos = board_dict["v_pos"]
    v_neg = board_dict["v_neg"]
    v_map = {False: board_dict["v_low"], True: board_dict["v_high"]}
    v_ins = [v_map[b] for b in combo]

    # Use median R values
    r1, r2, r3 = 10000.0, 3000.0, 3000.0
    gt_str = f"NAND{n_inputs}"
    gt = GateType(gt_str)

    try:
        res = solve_any_gate(gt, v_ins, v_pos, v_neg, r1, r2, r3,
                             jfet, jfet, board_dict["temp_c"])
        v_a = res["v_a"]
        v_b = res["v_b"]
        mids = res.get("v_mids", [])
        vals = [v_a] + list(mids) + [v_b]
    except Exception:
        # Fallback
        vals = [v_pos * 0.5] + [2.5] * (n_inputs - 1) + [v_pos * 0.4]

    return torch.tensor([vals], dtype=torch.float64, device=device)


def _make_nand_guess_physics(batch: int, n_inputs: int,
                             v_pos: torch.Tensor,
                             device: torch.device,
                             a_frac: float = 0.8,
                             mid_val: float = 2.5,
                             b_frac: float = 0.7) -> torch.Tensor:
    """Physics-informed NAND guess.

    Based on observed CPU solutions:
    - JFETs off: v_a near v_pos, midpoints at ~2.5V, v_b near v_a
    - JFETs on: v_a small, midpoints small, v_b small but > midpoints
    """
    n_vars = n_inputs + 1  # v_a, mid_0..mid_{N-2}, v_b
    x0 = torch.zeros(batch, n_vars, dtype=torch.float64, device=device)
    x0[:, 0] = v_pos * a_frac  # v_a
    # Midpoints: linearly interpolate from mid_val toward 0
    for k in range(n_inputs - 1):
        frac = (n_inputs - 1 - k) / n_inputs
        x0[:, 1 + k] = mid_val * frac
    x0[:, -1] = v_pos * b_frac  # v_b
    return x0


def _multi_guess_solve(residual_fn, batch, device, guess_fns, solver_fn,
                       max_steps=30, tol=1e-8):
    """Try multiple initial guesses, keep the best solution per sample."""
    best_x = None
    best_residual = torch.full((batch,), float('inf'), dtype=torch.float64,
                               device=device)
    best_converged = torch.zeros(batch, dtype=torch.bool, device=device)

    for guess_fn in guess_fns:
        x0 = guess_fn()
        x_sol, conv = solver_fn(residual_fn, x0, max_steps=max_steps, tol=tol)

        # Compute residual magnitude for each sample
        with torch.no_grad():
            F_val = residual_fn(x_sol)
            residual_mag = F_val.abs().max(dim=1).values.to(torch.float64)

        # Update best where this guess is better
        improved = residual_mag < best_residual
        if best_x is None:
            best_x = x_sol.clone()
        else:
            best_x[improved] = x_sol[improved]
        best_residual[improved] = residual_mag[improved]
        best_converged = best_converged | conv

        # Early exit if all converged
        if best_converged.all():
            break

    return best_x, best_converged


def _make_initial_guess_inv_nor(batch: int, v_pos: torch.Tensor,
                                device: torch.device) -> torch.Tensor:
    """Initial guess for INV/NOR: (v_a, v_b) = (v_pos/2, 0)."""
    x0 = torch.zeros(batch, 2, dtype=torch.float64, device=device)
    x0[:, 0] = v_pos * 0.5
    x0[:, 1] = 0.0
    return x0


def _make_initial_guess_nand(batch: int, n_inputs: int,
                             v_pos: torch.Tensor,
                             device: torch.device) -> torch.Tensor:
    """Default NAND guess."""
    return _make_nand_guess(batch, n_inputs, v_pos, device, 0.1, 0.15)


def _extract_output(v_b: torch.Tensor, v_neg: torch.Tensor,
                    r2: torch.Tensor, r3: torch.Tensor) -> torch.Tensor:
    """Compute v_out from v_b and the R2/R3 divider."""
    r_load = r2 + r3
    i_load = (v_b - v_neg) / r_load
    return v_neg + i_load * r3


def _compute_power(v_a: torch.Tensor, v_b: torch.Tensor,
                   v_pos: torch.Tensor, v_neg: torch.Tensor,
                   r1: torch.Tensor, r_load: torch.Tensor,
                   p_j2: JFETParams) -> torch.Tensor:
    """Compute power consumption from solved node voltages."""
    i_r1 = (v_pos - v_a) / r1
    i_j2 = ids_batched(v_a - v_b, v_pos - v_b, p_j2)
    i_load = (v_b - v_neg) / r_load
    return v_pos * (i_r1 + i_j2) + (-v_neg) * i_load


def _truth_table_inputs(gate_type: str, n_inputs: int):
    """Generate truth table input patterns.

    For N <= 8: full 2^N truth table.
    For N > 8: key patterns (all-low, all-high, one-hot, one-cold).

    Returns list of tuples: (input_bools, expected_output_high)
    """
    if gate_type.startswith("NOR"):
        # NOR: output high only when ALL inputs low
        def expected(combo):
            return not any(combo)
    elif gate_type.startswith("NAND"):
        # NAND: output low only when ALL inputs high
        def expected(combo):
            return not all(combo)
    elif gate_type == "INV":
        def expected(combo):
            return not combo[0]
    else:
        raise ValueError(f"Unsupported gate type: {gate_type}")

    if n_inputs <= 8:
        # Full truth table
        entries = []
        for i in range(2 ** n_inputs):
            combo = tuple(bool((i >> bit) & 1) for bit in range(n_inputs))
            entries.append((combo, expected(combo)))
        return entries
    else:
        # Key patterns only
        entries = []
        # All low
        combo = tuple(False for _ in range(n_inputs))
        entries.append((combo, expected(combo)))
        # All high
        combo = tuple(True for _ in range(n_inputs))
        entries.append((combo, expected(combo)))
        # One-hot (one input high, rest low)
        for k in range(n_inputs):
            combo = tuple(True if i == k else False for i in range(n_inputs))
            entries.append((combo, expected(combo)))
        # One-cold (one input low, rest high) — important for NAND
        for k in range(n_inputs):
            combo = tuple(False if i == k else True for i in range(n_inputs))
            entries.append((combo, expected(combo)))
        return entries


def gpu_solve_batch(gate_type_str: str, X: np.ndarray,
                    board_dict: dict, device: str = None) -> np.ndarray:
    """Solve a batch of circuits on GPU (or CPU).

    Args:
        gate_type_str: e.g. "INV", "NAND2", "NOR3"
        X: (N, 6) float array — [R1, R2, R3, V+, V-, temp]
        board_dict: dict with jfet_params, v_high, v_low, etc.
        device: 'cuda', 'cpu', or None (auto-detect)

    Returns:
        (N, 4) float array — [v_out_high, v_out_low, avg_power, converged]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    batch = X.shape[0]

    # Parse gate type
    if gate_type_str == "INV":
        n_inputs = 1
        is_nand = False
    elif gate_type_str.startswith("NAND"):
        n_inputs = int(gate_type_str[4:])
        is_nand = True
    elif gate_type_str.startswith("NOR"):
        n_inputs = int(gate_type_str[3:])
        is_nand = False
    else:
        raise ValueError(f"Unsupported: {gate_type_str}")

    # Extract circuit parameters
    r1 = torch.tensor(X[:, 0], dtype=torch.float64, device=dev)
    r2 = torch.tensor(X[:, 1], dtype=torch.float64, device=dev)
    r3 = torch.tensor(X[:, 2], dtype=torch.float64, device=dev)
    v_pos = torch.tensor(X[:, 3], dtype=torch.float64, device=dev)
    v_neg = torch.tensor(X[:, 4], dtype=torch.float64, device=dev)
    temp_c = torch.tensor(X[:, 5], dtype=torch.float64, device=dev)

    r_load = r2 + r3
    vt = thermal_voltage(temp_c)

    # JFET params (same for all samples, but temp-scaled per sample if varying)
    # For now: use mean temp for parameter scaling (close enough for small ranges)
    mean_temp = float(temp_c.mean().item())
    p_j1 = JFETParams.from_dict(board_dict["jfet_params"], mean_temp)
    p_j2 = JFETParams.from_dict(board_dict["jfet_params"], mean_temp)

    # Logic level voltages for truth table
    v_high = board_dict["v_high"]
    v_low = board_dict["v_low"]

    # Truth table
    table = _truth_table_inputs(gate_type_str, n_inputs)

    # Results accumulators
    v_out_high = torch.zeros(batch, dtype=torch.float64, device=dev)
    v_out_low = torch.zeros(batch, dtype=torch.float64, device=dev)
    total_power = torch.zeros(batch, dtype=torch.float64, device=dev)
    all_converged = torch.ones(batch, dtype=torch.bool, device=dev)

    for combo, out_high in table:
        # Map boolean inputs to voltages
        v_ins_list = []
        for b in combo:
            v_in_val = v_high if b else v_low
            v_ins_list.append(torch.full((batch,), v_in_val,
                                         dtype=torch.float64, device=dev))

        if is_nand and n_inputs > 1:
            # NAND: full (N+1) system
            # Use CPU solver on ONE midpoint-R combo to get a good initial guess
            def residual_fn(x):
                return _nand_residuals(x, v_ins_list, r1, r_load,
                                       v_pos, v_neg, p_j1, p_j2, vt)

            cpu_guess = _nand_cpu_seed_guess(
                combo, n_inputs, board_dict, dev)

            x_sol, conv = _multi_guess_solve(
                residual_fn, batch, dev,
                guess_fns=[
                    lambda: cpu_guess.expand(batch, -1).clone(),
                    lambda: _make_nand_guess_physics(batch, n_inputs, v_pos, dev,
                                                     a_frac=0.8, mid_val=2.5, b_frac=0.7),
                    lambda: _make_nand_guess_physics(batch, n_inputs, v_pos, dev,
                                                     a_frac=0.1, mid_val=0.5, b_frac=0.15),
                ],
                solver_fn=_newton_solve,
                max_steps=30, tol=1e-8,
            )
            v_a = x_sol[:, 0]
            v_b = x_sol[:, -1]
        else:
            # INV or NOR: 2 unknowns — try multiple guesses
            def residual_fn(x):
                return _inv_nor_residuals(x, v_ins_list, r1, r_load,
                                          v_pos, v_neg, p_j1, p_j2, vt)

            x_sol, conv = _multi_guess_solve(
                residual_fn, batch, dev,
                guess_fns=[
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.5, 0.0),
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.3, -0.1),
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.8, 0.3),
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.1, -0.3),
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.7, 0.1),
                    lambda: _make_guess_2x2(batch, v_pos, dev, 0.15, -0.5),
                ],
                solver_fn=_newton_solve_2x2,
                max_steps=30, tol=1e-8,
            )
            v_a = x_sol[:, 0]
            v_b = x_sol[:, 1]

        all_converged = all_converged & conv

        # Extract output
        v_out = _extract_output(v_b, v_neg, r2, r3)

        # Power
        power = _compute_power(v_a, v_b, v_pos, v_neg, r1, r_load, p_j2)
        total_power = total_power + power

        # Record logic levels
        all_low = all(not b for b in combo)
        all_high = all(b for b in combo)
        if all_low:
            v_out_high = v_out
        if all_high:
            v_out_low = v_out

    # Average power over truth table entries
    avg_power = total_power / len(table)

    # Pack results
    result = torch.stack([
        v_out_high, v_out_low, avg_power,
        all_converged.to(torch.float64),
    ], dim=1)

    return result.cpu().numpy().astype(np.float32)
