"""Helpers for emitting SPICE netlists from cpus/<cpu>/config.py.

Used by tools/gen_freq_testbench.py and tools/gen_sensitivity_testbench.py
to keep the JFET model card, supplies, and gate resistor values in sync
with the live CPU configuration. If you tune config.py, regenerate the
test-bench .cir files and the corner sweeps automatically pick up the
new parameters.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_cpu_config(cpu_name: str = "4004"):
    cfg_path = PROJECT_ROOT / "cpus" / cpu_name / "config.py"
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    spec = importlib.util.spec_from_file_location(f"_cpu_{cpu_name}", cfg_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def model_card(cfg, model_name: str = "DR") -> str:
    """Emit a SPICE .model line for the JFET defined in config.py."""
    j = cfg.JFET_MODEL
    c = cfg.CAPS
    # SPICE NJF parameter names (LTspice / Berkeley SPICE3).
    return (
        f".model {model_name} NJF ("
        f"Beta={j.beta:.4g} Betatce={j.betatce:.4g} "
        f"Vto={j.vto:.4g} Vtotc={j.vtotc:.4g} "
        f"Lambda={j.lmbda:.4g} "
        f"Is={j.is_:.4g} Xti={j.xti:.4g} "
        f"Isr={j.isr:.4g} Nr={j.nr:.4g} "
        f"Alpha={j.alpha:.4g} Vk={j.vk:.4g} "
        f"N={j.n:.4g} "
        f"Rd={j.rd:.4g} Rs={j.rs:.4g} "
        f"Cgs={c.cgs0:.4g} Cgd={c.cgd0:.4g}"
        f")"
    )


def inverter_subckt(cfg, r1: str = "{r1}", r2: str = "{r2}", r3: str = "{r3}",
                    model_name: str = "DR", subckt_name: str = "INV2J",
                    pos_node: str = "VPOS", neg_node: str = "VNEG") -> str:
    """Emit the 2-JFET RTL inverter as a subcircuit.

    Nodes (in this order): vin, vout, vpos, vneg

    Internal topology mirrors the schematic in cpus/4004/alu.asc and
    the description in optimization/README.md:

        VPOS --R1-- A    J1: drain=A, source=GND, gate=VIN
        VPOS --J2(D)     J2: source=B, gate=A
                 |
                 B --R2-- VOUT --R3-- VNEG

    R values can be passed as numeric strings ("50k") or as braced
    parameter references ("{r1}") so the caller can .step them.
    """
    return "\n".join([
        f".subckt {subckt_name} vin vout {pos_node} {neg_node}",
        f"R1 {pos_node} A {r1}",
        f"J1 A 0 vin {model_name}",
        f"J2 {pos_node} B A {model_name}",
        f"R2 B vout {r2}",
        f"R3 vout {neg_node} {r3}",
        f".ends {subckt_name}",
    ])


def supplies_block(cfg) -> str:
    """Emit V_POS and V_NEG sources at the nominal voltages from config.py.

    Caller can override these with .param vpos=... vneg=... and switch
    the values to {vpos} / {vneg} via supplies_block_param().
    """
    return "\n".join([
        f"VPOS VPOS 0 {cfg.V_POS:.4g}",
        f"VNEG VNEG 0 {cfg.V_NEG:.4g}",
    ])


def supplies_block_param() -> str:
    return "\n".join([
        "VPOS VPOS 0 {vpos}",
        "VNEG VNEG 0 {vneg}",
    ])


def gate_resistors(cfg, gate_name: str = "INV") -> dict:
    """Get R1/R2/R3 for a gate type out of config.GATES."""
    g = cfg.GATES[gate_name]
    return {"r1": g["r1"], "r2": g["r2"], "r3": g["r3"]}


def fmt_r(value: float) -> str:
    """Format a resistor value as 'NNNk' / 'NNN' / 'N.Nmeg' for SPICE."""
    if value >= 1e6:
        return f"{value/1e6:g}meg"
    if value >= 1e3:
        return f"{value/1e3:g}k"
    return f"{value:g}"
