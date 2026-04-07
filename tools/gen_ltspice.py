#!/usr/bin/env python3
"""Generate LTSpice SPICE netlists + PWL stimulus files from a flattened circuit.

This tool takes a Module (from blocks/ or a CPU component) and a CPU config,
then produces:
  1. A .net SPICE netlist with JFET subcircuits for each gate type
  2. .pwl files for each driven input net

Usage:
    # From Python — generate netlist for a 4-bit register
    from tools.gen_ltspice import generate
    from blocks import register
    generate(register(4), config_path="cpus/4004/config.py", output_dir="ltspice/output")

    # CLI — drive specific nets with waveforms
    python tools/gen_ltspice.py --module "register:4" --config cpus/4004/config.py \\
        --drive "clk:square:10u" --drive "d0:pulse:0:5u:20u" --output ltspice/output

Waveform types:
    square:period           — 50% duty cycle square wave
    pulse:delay:width:period — single-width pulse repeating
    high                    — constant logic HIGH
    low                     — constant logic LOW
    custom:t0,v0,t1,v1,... — arbitrary PWL points

All times in seconds (use suffix: 1u=1e-6, 1m=1e-3, 1n=1e-9).
"""

import argparse
import os
import sys
import importlib.util

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _root)

from simulator.module import flatten_top
from simulator.netlist import Netlist


def _parse_time(s: str) -> float:
    """Parse a time string with optional suffix (u, m, n, p)."""
    s = s.strip()
    suffixes = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3}
    if s[-1] in suffixes:
        return float(s[:-1]) * suffixes[s[-1]]
    return float(s)


def _load_config(config_path: str) -> dict:
    """Load a CPU config.py as a module and return its attributes as a dict."""
    spec = importlib.util.spec_from_file_location("cpu_config", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return {
        "v_pos": mod.V_POS,
        "v_neg": mod.V_NEG,
        "gates": mod.GATES,
        "jfet": mod.JFET_MODEL,
        "temp_c": mod.TEMP_C,
    }


def _jfet_model_card(jfet) -> str:
    """Generate SPICE .model card from NChannelJFET parameters."""
    return (
        f".model JFET_SIC NJF("
        f"Beta={jfet.beta} Vto={jfet.vto} Lambda={jfet.lmbda} "
        f"Is={jfet.is_} N={jfet.n} Isr={jfet.isr} Nr={jfet.nr} "
        f"Alpha={jfet.alpha} Vk={jfet.vk} Rd={jfet.rd} Rs={jfet.rs} "
        f"Betatce={jfet.betatce} Vtotc={jfet.vtotc} Xti={jfet.xti} Eg={jfet.eg})"
    )


def _inv_subckt(r1, r2, r3) -> str:
    """SPICE subcircuit for INV gate."""
    return f""".subckt INV in out vpos vneg
R1 vpos node_a {r1}
J1 node_a in 0 JFET_SIC
J2 vpos node_a node_b JFET_SIC
R2 node_b out {r2}
R3 out vneg {r3}
.ends INV"""


def _nand2_subckt(r1, r2, r3) -> str:
    """SPICE subcircuit for NAND2 gate."""
    return f""".subckt NAND2 in1 in2 out vpos vneg
R1 vpos node_a {r1}
J1a node_a in1 mid JFET_SIC
J1b mid in2 0 JFET_SIC
J2 vpos node_a node_b JFET_SIC
R2 node_b out {r2}
R3 out vneg {r3}
.ends NAND2"""


def _nor2_subckt(r1, r2, r3) -> str:
    """SPICE subcircuit for NOR2 gate."""
    return f""".subckt NOR2 in1 in2 out vpos vneg
R1 vpos node_a {r1}
J1a node_a in1 0 JFET_SIC
J1b node_a in2 0 JFET_SIC
J2 vpos node_a node_b JFET_SIC
R2 node_b out {r2}
R3 out vneg {r3}
.ends NOR2"""


_SUBCKT_GENERATORS = {
    "INV": _inv_subckt,
    "NAND2": _nand2_subckt,
    "NOR2": _nor2_subckt,
}


def generate_pwl(waveform_spec: str, v_high: float, v_low: float,
                 end_time: float = 100e-6) -> list:
    """Generate PWL time-voltage pairs from a waveform specification.

    Returns list of (time, voltage) tuples.
    """
    parts = waveform_spec.split(":")

    if parts[0] == "high":
        return [(0, v_high), (end_time, v_high)]

    elif parts[0] == "low":
        return [(0, v_low), (end_time, v_low)]

    elif parts[0] == "square":
        period = _parse_time(parts[1])
        half = period / 2
        points = []
        t = 0
        rise_time = period * 0.01  # 1% of period
        while t < end_time:
            points.append((t, v_low))
            points.append((t + rise_time, v_high))
            points.append((t + half, v_high))
            points.append((t + half + rise_time, v_low))
            t += period
        points.append((end_time, v_low))
        return points

    elif parts[0] == "pulse":
        delay = _parse_time(parts[1])
        width = _parse_time(parts[2])
        period = _parse_time(parts[3]) if len(parts) > 3 else end_time
        rise_time = width * 0.01
        points = [(0, v_low)]
        t = delay
        while t < end_time:
            points.append((t, v_low))
            points.append((t + rise_time, v_high))
            points.append((t + width, v_high))
            points.append((t + width + rise_time, v_low))
            t += period
        points.append((end_time, v_low))
        return points

    elif parts[0] == "custom":
        # custom:t0,v0,t1,v1,...
        vals = parts[1].split(",")
        points = []
        for i in range(0, len(vals), 2):
            t = _parse_time(vals[i])
            v = float(vals[i + 1])
            points.append((t, v))
        return points

    else:
        raise ValueError(f"Unknown waveform type: {parts[0]}")


def write_pwl_file(filepath: str, points: list):
    """Write a .pwl file from time-voltage pairs."""
    with open(filepath, "w") as f:
        for t, v in points:
            f.write(f"{t:.9e} {v:.4f}\n")


def generate_netlist(module, config: dict, drives: dict = None,
                     end_time: float = 100e-6, output_dir: str = "ltspice/output"):
    """Generate a complete SPICE netlist + PWL files for a module.

    Args:
        module: a simulator.Module to export
        config: dict from _load_config (v_pos, v_neg, gates, jfet)
        drives: dict of net_name -> waveform_spec (e.g. {"clk": "square:10u"})
        end_time: simulation end time in seconds
        output_dir: where to write output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Flatten module to gates
    flat_gates = flatten_top(module)
    nl = Netlist.from_gates(flat_gates)

    # Determine logic levels from config (solve a quick INV)
    jfet = config["jfet"].at_temp(config["temp_c"])
    from model import solve_gate
    inv_r = config["gates"]["INV"]
    res_hi = solve_gate(0.0, config["v_pos"], config["v_neg"],
                        inv_r["r1"], inv_r["r2"], inv_r["r3"], jfet, jfet)
    res_lo = solve_gate(jfet.vto * 1.2, config["v_pos"], config["v_neg"],
                        inv_r["r1"], inv_r["r2"], inv_r["r3"], jfet, jfet)
    v_high = res_lo["v_out"]
    v_low = res_hi["v_out"]

    # Build SPICE netlist
    lines = []
    lines.append(f"* Auto-generated SPICE netlist for {module.name}")
    lines.append(f"* Generated by deepJFET gen_ltspice.py")
    lines.append("")

    # JFET model
    lines.append(_jfet_model_card(config["jfet"]))
    lines.append("")

    # Gate subcircuits
    for gate_name, gen_func in _SUBCKT_GENERATORS.items():
        if gate_name in config["gates"]:
            r = config["gates"][gate_name]
            lines.append(gen_func(r["r1"], r["r2"], r["r3"]))
            lines.append("")

    # Supply sources
    lines.append(f"Vpos vpos 0 {config['v_pos']}")
    lines.append(f"Vneg vneg 0 {config['v_neg']}")
    lines.append("")

    # PWL voltage sources for driven inputs
    if drives:
        for net_name, waveform_spec in drives.items():
            pwl_points = generate_pwl(waveform_spec, v_high, v_low, end_time)
            pwl_file = os.path.join(output_dir, f"{net_name}.pwl")
            write_pwl_file(pwl_file, pwl_points)
            lines.append(f"V_{net_name} {net_name} 0 PWL file={net_name}.pwl")

    # Undriven primary inputs get a default (logic LOW)
    for pi in sorted(nl.primary_inputs):
        if drives and pi in drives:
            continue
        lines.append(f"V_{pi} {pi} 0 {v_low}")

    lines.append("")

    # Gate instances — replace dots with underscores in net names for SPICE compatibility
    def spice_net(name):
        return name.replace(".", "_")

    for gate in flat_gates:
        gt = gate.gate_type.value  # "INV", "NAND2", etc.
        inst_name = f"X_{gate.name}".replace(".", "_")
        inputs = [spice_net(i) for i in gate.inputs]
        output = spice_net(gate.output)

        if gt == "INV":
            lines.append(f"{inst_name} {inputs[0]} {output} vpos vneg INV")
        elif gt == "NAND2":
            lines.append(f"{inst_name} {inputs[0]} {inputs[1]} {output} vpos vneg NAND2")
        elif gt == "NOR2":
            lines.append(f"{inst_name} {inputs[0]} {inputs[1]} {output} vpos vneg NOR2")
        else:
            lines.append(f"* WARNING: unsupported gate type {gt} for gate {gate.name}")

    lines.append("")

    # Simulation command
    lines.append(f".tran {end_time:.6e}")
    lines.append(".end")

    # Write netlist
    netlist_path = os.path.join(output_dir, f"{module.name}.net")
    with open(netlist_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Generated: {netlist_path}")
    print(f"  Gates: {len(flat_gates)}")
    print(f"  Primary inputs: {sorted(nl.primary_inputs)}")
    print(f"  Logic levels: HIGH={v_high:.3f}V  LOW={v_low:.3f}V")
    if drives:
        for net, spec in drives.items():
            print(f"  Drive: {net} <- {spec}")

    return netlist_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate LTSpice netlist + PWL from a block/module"
    )
    parser.add_argument(
        "--module", "-m", required=True,
        help="Block to export, format: name[:param] (e.g. 'register:4', 'dff', 'mux2to1:8')"
    )
    parser.add_argument(
        "--config", "-c", required=True,
        help="Path to CPU config.py"
    )
    parser.add_argument(
        "--drive", "-d", action="append", default=[],
        help="Drive a net: 'net_name:waveform_spec' (repeatable). "
             "Waveforms: square:period, pulse:delay:width[:period], high, low, "
             "custom:t0,v0,t1,v1,..."
    )
    parser.add_argument(
        "--end-time", "-t", type=float, default=100e-6,
        help="Simulation end time in seconds (default: 100us)"
    )
    parser.add_argument(
        "--output", "-o", default="ltspice/output",
        help="Output directory (default: ltspice/output)"
    )

    args = parser.parse_args()

    # Parse module spec
    mod_parts = args.module.split(":")
    block_name = mod_parts[0]
    block_param = int(mod_parts[1]) if len(mod_parts) > 1 else None

    from blocks.registry import get_block
    if block_param is not None:
        module = get_block(block_name, n_bits=block_param)
    else:
        module = get_block(block_name)

    # Load config
    config = _load_config(args.config)

    # Parse drives
    drives = {}
    for d in args.drive:
        net_name, *waveform_parts = d.split(":", 1)
        drives[net_name] = waveform_parts[0] if waveform_parts else "low"

    generate_netlist(module, config, drives, args.end_time, args.output)


if __name__ == "__main__":
    main()
