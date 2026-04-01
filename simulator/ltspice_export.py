"""Export a Netlist to LTSpice .asc schematic format.

Supports INV, NOR-N, NAND-N, and arbitrary series/parallel topologies.
Layout derived from hand-built LTSpice reference schematics.

NJF pins from (x,y): drain=(x+48,y), source=(x+48,y+96), gate=(x,y+64)
RES pins from (x,y): top=(x+16,y), bottom=(x+16,y+80)
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, input_names, count_jfets,
)
from model import GateType


def _layout_j1_network(net, x, y, lines, jfet_model, gate_name, counter,
                       input_labels, input_idx):
    """Recursively lay out J1 pull-down network. Returns (width, height, node_a_points).

    Parallel: J1s side by side at same y, all drains at y
    Series: J1s stacked vertically, source-to-drain chain
    Leaf: single JFET

    Returns:
        top_x: x of top drain connection
        top_y: y of top drain connection
        bot_y: y of bottom source connection
        node_a_points: list of (x, y) drain connection points for wiring
    """
    if isinstance(net, Leaf):
        # Single JFET at (x, y)
        lines.append(f"SYMBOL njf {x} {y} R0")
        lines.append(f"SYMATTR InstName J1_{gate_name}_{counter[0]}")
        lines.append(f"SYMATTR Value {jfet_model}")
        counter[0] += 1

        # Gate wire to input label
        idx = input_idx[0]
        input_idx[0] += 1
        label = input_labels[idx] if idx < len(input_labels) else f"in{idx}"
        lines.append(f"WIRE {x} {y+64} {x-16} {y+64}")
        lines.append(f"FLAG {x-16} {y+64} {label}")

        drain_x = x + 48
        drain_y = y
        source_y = y + 96
        return drain_x, drain_y, source_y, [(drain_x, drain_y)]

    if isinstance(net, Parallel):
        # Lay out children side by side
        all_points = []
        child_x = x
        max_bot_y = 0
        first_drain_y = None

        for i, child in enumerate(net.children):
            dx, dy, by, pts = _layout_j1_network(
                child, child_x, y, lines, jfet_model, gate_name,
                counter, input_labels, input_idx)
            all_points.extend(pts)
            max_bot_y = max(max_bot_y, by)
            if first_drain_y is None:
                first_drain_y = dy

            # Compute width of this child for spacing
            if isinstance(child, Leaf):
                child_x += 96
            elif isinstance(child, Series):
                child_x += 96
            else:
                n = count_jfets(child)
                child_x += n * 96

        # Connect all drain points with horizontal wire if multiple
        if len(all_points) > 1:
            xs = [p[0] for p in all_points]
            min_x, max_x = min(xs), max(xs)
            lines.append(f"WIRE {max_x} {y} {min_x} {y}")

        # GND for bottom sources
        for child in net.children:
            # Each parallel branch needs its own GND
            pass  # handled by leaf/series source connections

        return all_points[0][0], y, max_bot_y, all_points

    if isinstance(net, Series):
        # Stack children vertically
        curr_y = y
        top_drain = None
        all_points = []

        for i, child in enumerate(net.children):
            dx, dy, by, pts = _layout_j1_network(
                child, x, curr_y, lines, jfet_model, gate_name,
                counter, input_labels, input_idx)
            if top_drain is None:
                top_drain = (dx, dy)
                all_points = pts
            curr_y = by  # next child starts where this one's source is

        # GND at bottom of chain
        last_source_x = x + 48
        lines.append(f"FLAG {last_source_x} {curr_y} 0")

        return top_drain[0], top_drain[1], curr_y, all_points


def _emit_gate(lines, gate_name, params, network, xb, yb,
               input_labels, output_label, jfet_model, counter):
    """Emit a complete gate with arbitrary topology."""
    r1, r2, r3 = params.r1, params.r2, params.r3

    # R1: from VDD down to Node A
    r1_x = xb
    r1_y = yb + 32
    lines.append(f"SYMBOL res {r1_x-16} {r1_y} R0")
    lines.append(f"SYMATTR InstName R1_{counter[0]}")
    lines.append(f"SYMATTR Value {r1:.0f}")
    counter[0] += 1
    lines.append(f"FLAG {r1_x} {yb+48-16} VDD")

    # R1 bottom pin at (r1_x, r1_y+80) = (xb, yb+112)
    # Node A is at yb+128 (where J1 drains connect)
    node_a_y = yb + 128

    # Wire R1 bottom to Node A level
    # (R1 bottom is at yb+112, Node A at yb+128 — but they might need a wire)

    # Layout J1 network
    j1_x = xb - 48  # J1 network starts to the left
    input_idx = [0]
    j1_counter = [counter[0]]

    drain_x, drain_y, source_y, node_a_pts = _layout_j1_network(
        network, j1_x, node_a_y, lines, jfet_model, gate_name,
        j1_counter, input_labels, input_idx)
    counter[0] = j1_counter[0]

    # Wire from R1 bottom to Node A (connect to rightmost drain point)
    rightmost_x = max(p[0] for p in node_a_pts)
    lines.append(f"WIRE {r1_x} {node_a_y} {r1_x} {r1_y+80}")

    # Extend Node A wire to include R1 position
    if r1_x != rightmost_x:
        lines.append(f"WIRE {rightmost_x} {node_a_y} {r1_x} {node_a_y}")

    # J2 column: to the right of R1
    j2_x = rightmost_x + 48
    j2_y = yb + 64

    # Wire from Node A to J2 gate
    j2_gate_x = j2_x
    j2_gate_y = j2_y + 64  # = yb + 128 = node_a_y
    lines.append(f"WIRE {j2_gate_x} {j2_gate_y} {rightmost_x} {j2_gate_y}")

    # J2
    lines.append(f"SYMBOL njf {j2_x} {j2_y} R0")
    lines.append(f"SYMATTR InstName J2_{gate_name}")
    lines.append(f"SYMATTR Value {jfet_model}")

    j2_drain_x = j2_x + 48  # where R2/R3 column is
    lines.append(f"FLAG {j2_drain_x} {j2_y} VDD")

    # R2: below J2 source
    r2_x = j2_drain_x - 16
    r2_y = j2_y + 96 + 48  # J2 source at j2_y+96, small gap
    lines.append(f"SYMBOL res {r2_x} {r2_y} R0")
    lines.append(f"SYMATTR InstName R2_{counter[0]}")
    lines.append(f"SYMATTR Value {r2:.0f}")
    counter[0] += 1

    # R3: below R2
    r3_y = r2_y + 80
    lines.append(f"SYMBOL res {r2_x} {r3_y} R0")
    lines.append(f"SYMATTR InstName R3_{counter[0]}")
    lines.append(f"SYMATTR Value {r3:.0f}")
    counter[0] += 1

    # Output flag between R2 and R3
    out_y = r3_y
    lines.append(f"WIRE {j2_drain_x+48} {out_y} {j2_drain_x} {out_y}")
    lines.append(f"FLAG {j2_drain_x+48} {out_y} {output_label}")

    # VSS at R3 bottom
    lines.append(f"FLAG {j2_drain_x} {r3_y+80} VSS")


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
    """Export a Netlist to LTSpice .asc format."""
    lines = ["Version 4", "SHEET 1 10000 10000"]

    if gate_networks is None:
        gate_networks = {}
        for g in netlist.gates.values():
            gate_networks[g.name] = gate_type_to_network(g.gate_type)

    ordered, feedback = netlist.topological_sort()
    all_gates = ordered + feedback

    counter = [0]
    x_pos = 0

    for gname in all_gates:
        gate = netlist.gates[gname]
        params = gate_params[gname]
        network = gate_networks[gname]

        _emit_gate(lines, gname, params, network, x_pos, 0,
                   gate.inputs, gate.output, jfet_model, counter)

        lines.append(f"TEXT {x_pos-48} -104 Left 2 ;{gname} ({gate.gate_type.value})")

        # Estimate width for next gate position
        n_j = count_jfets(network)
        gate_width = max(n_j * 96 + 200, 320)
        x_pos += gate_width

    # Supplies
    sx = -500
    lines.append(f"FLAG {sx} 16 VDD")
    lines.append(f"FLAG {sx} 96 0")
    lines.append(f"SYMBOL voltage {sx} 0 R0")
    lines.append(f"SYMATTR InstName V_VDD")
    lines.append(f"SYMATTR Value {v_pos}")
    lines.append(f"FLAG {sx} 216 VSS")
    lines.append(f"FLAG {sx} 296 0")
    lines.append(f"SYMBOL voltage {sx} 200 R0")
    lines.append(f"SYMATTR InstName V_VSS")
    lines.append(f"SYMATTR Value {v_neg}")

    if stimuli:
        stim_y = 500
        for net_name, stim in stimuli.items():
            if isinstance(stim, dict):
                v1 = stim.get("v1", -4.0)
                v2 = stim.get("v2", -0.8)
                td = stim.get("td", 10e-6)
                tr = stim.get("tr", 0.1e-6)
                tf = stim.get("tf", 0.1e-6)
                pw = stim.get("pw", 5e-6)
                per = stim.get("per", 10e-6)
                val = f"PULSE({v1} {v2} {td} {tr} {tf} {pw} {per})"
            else:
                val = str(stim)
            lines.append(f"FLAG {sx} {stim_y+16} {net_name}")
            lines.append(f"FLAG {sx} {stim_y+96} 0")
            lines.append(f"SYMBOL voltage {sx} {stim_y} R0")
            lines.append(f"SYMATTR InstName V_{net_name}")
            lines.append(f"SYMATTR Value {val}")
            stim_y += 200

    if jfet_model_card is None:
        jfet_model_card = (
            f".model {jfet_model} NJF(Beta=0.135m Betatce=-0.5 Vto=-3.45 "
            "Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f Nr=4 "
            "Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
            "Vk=123.7 M=407m Pb=1)"
        )
    lines.append(f"TEXT {sx-4} -304 Left 2 !.tran 0 {sim_time} 0 {sim_time/4000}")
    lines.append(f"TEXT {sx-4} -352 Left 2 !{jfet_model_card}")

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Exported {len(all_gates)} gates to {output_path}")
