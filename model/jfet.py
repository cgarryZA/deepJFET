"""
JFET Device Model — Full SPICE NJF DC model with temperature scaling.

Implements the Shichman-Hodges channel current model with:
  - Gate junction diodes (Is/N diffusion + Isr/Nr recombination)
  - Parasitic Rs/Rd resistance
  - Alpha/Vk ionization current
  - SPICE-standard temperature scaling (Betatce, Vtotc, Xti, Eg)
"""

import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

K_BOLTZ = 1.380649e-23     # Boltzmann constant (J/K)
Q_ELEC = 1.602176634e-19   # Elementary charge (C)
TNOM_K = 300.15             # SPICE nominal temperature 27C in Kelvin


def thermal_voltage(temp_k: float) -> float:
    """Thermal voltage kT/q at a given temperature in Kelvin."""
    return K_BOLTZ * temp_k / Q_ELEC


# ---------------------------------------------------------------------------
# JFET dataclass
# ---------------------------------------------------------------------------

@dataclass
class NChannelJFET:
    """N-channel JFET parameters — full SPICE NJF model for DC analysis.

    All values are specified at Tnom (27C / 300.15K) and come directly
    from the SPICE .model card.
    """
    beta: float      # Transconductance coefficient (A/V^2)
    vto: float       # Pinch-off / threshold voltage (V), negative for N-ch
    lmbda: float     # Channel-length modulation (1/V)
    is_: float       # Gate junction saturation current (A)
    n: float         # Gate junction emission coefficient
    isr: float = 0.0 # Gate junction recombination current (A)
    nr: float = 2.0  # Recombination emission coefficient
    alpha: float = 0.0  # Ionization coefficient (1/V)
    vk: float = 0.0     # Ionization knee voltage (V)
    rd: float = 0.0  # Parasitic drain resistance (ohm)
    rs: float = 0.0  # Parasitic source resistance (ohm)
    # Temperature coefficients
    betatce: float = 0.0  # Beta exponential temp coeff (%/C)
    vtotc: float = 0.0    # Vto linear temp coeff (V/C)
    xti: float = 3.0      # Is saturation current temp exponent
    eg: float = 1.11      # Energy gap (eV) — 1.11 for Si, 3.26 for SiC

    def __post_init__(self):
        if self.vto >= 0:
            raise ValueError(
                f"vto must be negative for N-channel JFET, got {self.vto}"
            )

    @property
    def idss(self) -> float:
        """Saturation current at Vgs=0 (A) at Tnom."""
        return self.beta * self.vto ** 2

    def at_temp(self, temp_c: float) -> "NChannelJFET":
        """Return a new JFET with parameters scaled to the given temperature.

        Uses standard SPICE temperature scaling equations:
          Beta(T)  = Beta(Tnom) * exp(Betatce/100 * dT)
          Vto(T)   = Vto(Tnom) + Vtotc * dT
          Is(T)    = Is(Tnom) * (T/Tnom)^(Xti/N)
                     * exp(Eg*q*dT / (N*k*T*Tnom))
          Isr(T)   = same form with Nr instead of N
        """
        temp_k = temp_c + 273.15
        dt_c = temp_c - (TNOM_K - 273.15)

        beta_t = self.beta * np.exp(self.betatce / 100.0 * dt_c)
        vto_t = self.vto + self.vtotc * dt_c

        ratio = temp_k / TNOM_K
        eg_term = self.eg * Q_ELEC * dt_c / (K_BOLTZ * temp_k * TNOM_K)
        is_t = self.is_ * (ratio ** (self.xti / self.n)) * np.exp(eg_term / self.n)

        isr_t = 0.0
        if self.isr > 0:
            isr_t = (self.isr * (ratio ** (self.xti / self.nr))
                     * np.exp(eg_term / self.nr))

        return NChannelJFET(
            beta=beta_t, vto=vto_t, lmbda=self.lmbda,
            is_=is_t, n=self.n, isr=isr_t, nr=self.nr,
            alpha=self.alpha, vk=self.vk, rd=self.rd, rs=self.rs,
            betatce=self.betatce, vtotc=self.vtotc,
            xti=self.xti, eg=self.eg,
        )


# ---------------------------------------------------------------------------
# Channel current (Shichman-Hodges)
# ---------------------------------------------------------------------------

def _ids_intrinsic(vgs: float, vds: float, j: NChannelJFET) -> float:
    """Intrinsic drain-source channel current (no parasitic R)."""
    if vds < 0:
        return -_ids_intrinsic(vgs - vds, -vds, j)
    if vgs <= j.vto:
        return 0.0

    vsat = vgs - j.vto
    if vds >= vsat:
        ids = j.beta * vsat ** 2 * (1.0 + j.lmbda * vds)
    else:
        ids = j.beta * (2.0 * vsat * vds - vds ** 2) * (1.0 + j.lmbda * vds)

    if j.alpha > 0.0 and vds > 0.0 and j.vk > 0.0:
        ids *= (1.0 + j.alpha * vds * np.exp(-j.vk / vds))
    return ids


def jfet_ids(vgs: float, vds: float, j: NChannelJFET) -> float:
    """Drain-to-source current including parasitic Rs/Rd.

    Iteratively solves for internal voltages across the intrinsic device
    accounting for voltage drops across Rs and Rd.
    """
    if j.rs == 0.0 and j.rd == 0.0:
        return _ids_intrinsic(vgs, vds, j)

    id_est = _ids_intrinsic(vgs, vds, j)
    for _ in range(20):
        vgs_int = vgs - id_est * j.rs
        vds_int = vds - id_est * (j.rs + j.rd)
        id_new = _ids_intrinsic(vgs_int, vds_int, j)
        if abs(id_new - id_est) < 1e-12:
            break
        id_est = 0.5 * (id_est + id_new)
    return id_est


# ---------------------------------------------------------------------------
# Gate junction diode
# ---------------------------------------------------------------------------

def _diode_current(vd: float, is_: float, n: float, vt: float) -> float:
    """PN junction diode current. Clamps exponent to avoid overflow."""
    x = vd / (n * vt)
    if x > 40.0:
        exp40 = np.exp(40.0)
        return is_ * (exp40 + exp40 * (x - 40.0))
    return is_ * (np.exp(x) - 1.0)


def jfet_gate_current(
    vgs: float, vgd: float, j: NChannelJFET, vt: float,
) -> tuple:
    """Gate junction currents (gate-source and gate-drain diodes).

    Includes both diffusion (Is/N) and recombination (Isr/Nr) components.
    Returns (Igs, Igd) — positive when forward-biased.
    """
    igs = _diode_current(vgs, j.is_, j.n, vt)
    igd = _diode_current(vgd, j.is_, j.n, vt)
    if j.isr > 0.0:
        igs += _diode_current(vgs, j.isr, j.nr, vt)
        igd += _diode_current(vgd, j.isr, j.nr, vt)
    return igs, igd


# ---------------------------------------------------------------------------
# Operating region helper
# ---------------------------------------------------------------------------

def region_name(vgs: float, vds: float, j: NChannelJFET) -> str:
    """Return the JFET operating region as a string."""
    if vgs <= j.vto:
        return "cutoff"
    if vds >= (vgs - j.vto):
        return "saturation"
    return "linear"
