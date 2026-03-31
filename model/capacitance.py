"""Junction capacitance model for transient JFET simulation.

Phase 1: Constant capacitance (Cgs0, Cgd0 from SPICE model card).
Phase 2 (future): Voltage-dependent depletion capacitance:
    Cj(V) = Cj0 / (1 - V/Pb)^M     for V < Fc*Pb
    Cj(V) = Cj0/(1-Fc)^(1+M) * (1 - Fc*(1+M) + M*V/Pb)  for V >= Fc*Pb
"""

from dataclasses import dataclass


@dataclass
class JFETCapacitance:
    """Junction capacitances for transient simulation.

    Values from SPICE .model card: Cgd=16.9p Cgs=16.9p Pb=1 M=407m Fc=0.5
    """
    cgs0: float = 16.9e-12   # Gate-source capacitance (F)
    cgd0: float = 16.9e-12   # Gate-drain capacitance (F)
    pb: float = 1.0          # Built-in potential (V)
    m: float = 0.407         # Junction grading exponent
    fc: float = 0.5          # Forward bias depletion cap coefficient

    @property
    def c_per_input(self) -> float:
        """Total capacitance presented by one gate input (Cgs + Cgd)."""
        return self.cgs0 + self.cgd0

    def cgs(self, vgs: float = 0.0) -> float:
        """Gate-source capacitance. Phase 1: constant."""
        return self.cgs0

    def cgd(self, vgd: float = 0.0) -> float:
        """Gate-drain capacitance. Phase 1: constant."""
        return self.cgd0
