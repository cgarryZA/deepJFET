"""Export a Netlist to LTSpice .asc schematic format.

Each gate becomes: R1 + J1 pull-down network + J2 (load) + R2 + R3
arranged vertically. Gates are placed side by side horizontally.
Wires connect gate outputs to downstream gate inputs.
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, input_names, count_midpoints,
)
from model import GateType


# LTSpice coordinate grid: symbols snap to 16-unit grid
# One gate takes about 400 units wide, 800 units tall

GATE_WIDTH = 800
GATE_HEIGHT = 1200
GATE_SPACING = 200


def _sym(sym_type, x, y, rot="R0", inst_name=None, value=None, windows=None):
    """Generate SYMBOL + SYMATTR lines."""
    lines = [f"SYMBOL {sym_type} {x} {y} {rot}"]
    if windows:
        for w in windows:
            lines.append(w)
    if inst_name:
        lines.append(f"SYMATTR InstName {inst_name}")
    if value:
        lines.append(f"SYMATTR Value {value}")
    return lines


def _wire(x1, y1, x2, y2):
    return f"WIRE {x1} {y1} {x2} {y2}"


def _flag(x, y, label):
    return f"FLAG {x} {y} {label}"


def _text(x, y, text, size=2):
    return f"TEXT {x} {y} Left {size} !{text}"


def _comment(x, y, text, size=2):
    return f"TEXT {x} {y} Left {size} ;{text}"


def _export_inv_gate(gate_name, params, x_base, y_base,
                     input_net_labels, output_net_label,
                     jfet_model, counter):
    """Generate LTSpice elements for a single INV/NOR gate.

    For NOR: multiple J1s in parallel at Node A.
    Returns (lines, updated_counter).
    """
    lines = []
    r1, r2, r3 = params.r1, params.r2, params.r3
    n_inputs = len(input_net_labels)

    # Coordinate layout (vertical, top to bottom):
    # VDD ---- R1 ---- Node_A ---- J1(drain)  J1(source) ---- GND
    #                   |
    #          VDD ---- J2(drain)  J2(source) ---- Node_B
    #                                               |
    #                                         R2 -- V_OUT -- R3 -- VSS

    x = x_base
    y = y_base

    # Node A position
    node_a_x = x
    node_a_y = y + 400

    # R1: from VDD down to Node A
    r1_y = y + 100
    lines.append(_wire(x, y, x, r1_y))  # VDD to R1 top
    lines.append(_flag(x, y, "VDD"))
    lines.extend(_sym("res", x - 16, r1_y, "R0",
                       f"R1_{counter[0]}", f"{r1:.0f}"))
    counter[0] += 1
    lines.append(_wire(x, r1_y + 96, x, node_a_y))  # R1 bottom to Node A

    # J1s: parallel, each from Node A to GND
    for k in range(n_inputs):
        j1_x = x - 200 - k * 300
        j1_y = node_a_y

        # Wire from Node A to J1 drain
        lines.append(_wire(node_a_x, node_a_y, j1_x + 64, node_a_y))
        # J1 symbol (njf): gate on left, drain on top, source on bottom
        lines.extend(_sym("njf", j1_x, j1_y, "R0",
                           f"J1_{gate_name}_{k}", jfet_model))
        # J1 source to GND
        j1_source_y = j1_y + 128
        lines.append(_wire(j1_x + 64, j1_source_y, j1_x + 64, j1_source_y + 50))
        lines.append(_flag(j1_x + 64, j1_source_y + 50, "0"))
        # J1 gate label (input net)
        lines.append(_wire(j1_x, j1_y + 64, j1_x - 80, j1_y + 64))
        lines.append(_flag(j1_x - 80, j1_y + 64, input_net_labels[k]))

    # J2: load transistor. Gate = Node A, Drain = VDD, Source = Node B
    j2_x = x + 200
    j2_y = node_a_y - 100

    # J2 gate wire from Node A
    lines.append(_wire(node_a_x, node_a_y, j2_x, node_a_y))
    lines.append(_wire(j2_x, node_a_y, j2_x, j2_y + 64))  # down to gate

    # J2 symbol
    lines.extend(_sym("njf", j2_x, j2_y, "M0",  # mirrored for load
                       f"J2_{gate_name}", jfet_model))
    # J2 drain to VDD
    lines.append(_wire(j2_x + 64, j2_y, j2_x + 64, j2_y - 50))
    lines.append(_flag(j2_x + 64, j2_y - 50, "VDD"))
    # J2 source = Node B
    node_b_x = j2_x + 64
    node_b_y = j2_y + 128

    # R2: Node B to V_OUT
    r2_y = node_b_y + 50
    lines.append(_wire(node_b_x, node_b_y, node_b_x, r2_y))
    lines.extend(_sym("res", node_b_x - 16, r2_y, "R0",
                       f"R2_{counter[0]}", f"{r2:.0f}"))
    counter[0] += 1

    # V_OUT node
    vout_y = r2_y + 96 + 30
    lines.append(_wire(node_b_x, r2_y + 96, node_b_x, vout_y))
    lines.append(_flag(node_b_x, vout_y, output_net_label))

    # R3: V_OUT to VSS
    r3_y = vout_y + 30
    lines.append(_wire(node_b_x, vout_y, node_b_x, r3_y))
    lines.extend(_sym("res", node_b_x - 16, r3_y, "R0",
                       f"R3_{counter[0]}", f"{r3:.0f}"))
    counter[0] += 1
    lines.append(_wire(node_b_x, r3_y + 96, node_b_x, r3_y + 150))
    lines.append(_flag(node_b_x, r3_y + 150, "VSS"))

    return lines


def export_netlist_asc(
    netlist: Netlist,
    gate_params: dict,
    gate_networks: dict = None,
    stimuli: dict = None,
    jfet_model: str = "DR",
    jfet_model_card: str = None,
    v_pos: float = 10.0,
    v_neg: float = -10.0,
    sim_time: float = 40e-6,
    output_path: str = "circuit.asc",
):
    """Export a Netlist to LTSpice .asc format.

    Args:
        netlist: Gate netlist
        gate_params: gate_name -> CircuitParams
        gate_networks: gate_name -> PulldownNetwork (optional, inferred from GateType)
        stimuli: net_name -> dict with 'type', 'v1', 'v2', 'td', 'tr', 'tf', 'pw', 'per'
        jfet_model: SPICE model name
        jfet_model_card: full .model line (auto-generated if None)
        v_pos, v_neg: supply voltages
        sim_time: simulation time
        output_path: where to save the .asc file
    """
    lines = [
        "Version 4",
        "SHEET 1 10000 10000",
    ]

    if gate_networks is None:
        gate_networks = {}
        for g in netlist.gates.values():
            gate_networks[g.name] = gate_type_to_network(g.gate_type)

    # Layout gates horizontally
    ordered, feedback = netlist.topological_sort()
    all_gates = ordered + feedback

    counter = [0]  # mutable counter for unique component names
    x_pos = 0

    for gname in all_gates:
        gate = netlist.gates[gname]
        params = gate_params[gname]
        network = gate_networks[gname]

        # Get input/output net labels
        in_labels = gate.inputs
        out_label = gate.output

        gate_lines = _export_inv_gate(
            gname, params, x_pos, 0,
            in_labels, out_label,
            jfet_model, counter,
        )
        lines.extend(gate_lines)

        # Add gate name comment
        lines.append(_comment(x_pos - 50, -100, f"{gname} ({gate.gate_type.value})"))

        x_pos += GATE_WIDTH + GATE_SPACING

    # Supply voltage sources
    supply_x = -500
    lines.extend(_sym("voltage", supply_x, 0, "R0", "V_VDD", f"{v_pos}"))
    lines.append(_flag(supply_x, 0, "VDD"))
    lines.append(_flag(supply_x, 96, "0"))

    lines.extend(_sym("voltage", supply_x, 200, "R0", "V_VSS", f"{v_neg}"))
    lines.append(_flag(supply_x, 200, "VSS"))
    lines.append(_flag(supply_x, 296, "0"))

    # Input stimulus sources
    if stimuli:
        stim_x = -500
        stim_y = 500
        for net_name, stim in stimuli.items():
            if isinstance(stim, dict):
                v1 = stim.get("v1", -4.0)
                v2 = stim.get("v2", 0.0)
                td = stim.get("td", 10e-6)
                tr = stim.get("tr", 0.1e-6)
                tf = stim.get("tf", 0.1e-6)
                pw = stim.get("pw", 5e-6)
                per = stim.get("per", 10e-6)
                val = f"PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})"
            else:
                val = str(stim)

            lines.extend(_sym("voltage", stim_x, stim_y, "R0",
                               f"V_{net_name}", val))
            lines.append(_flag(stim_x, stim_y, net_name))
            lines.append(_flag(stim_x, stim_y + 96, "0"))
            stim_y += 200

    # JFET model card
    if jfet_model_card is None:
        jfet_model_card = (
            f".model {jfet_model} NJF(Beta=0.135m Betatce=-0.5 Vto=-3.45 "
            "Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f Nr=4 "
            "Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
            "Vk=123.7 M=407m Pb=1)"
        )

    text_y = -300
    lines.append(_text(-500, text_y, f".tran 0 {sim_time} 0 {sim_time/4000}"))
    lines.append(_text(-500, text_y - 50, jfet_model_card))

    # Write file
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Exported {len(all_gates)} gates to {output_path}")
