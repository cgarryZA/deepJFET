"""Export a Netlist to LTSpice .asc schematic format.

Supports arbitrary series/parallel JFET topologies.
All coordinates computed from component pin positions to guarantee connectivity.

Component pin positions (from symbol origin x, y with rotation R0):
  NJF:  drain=(x+48, y), source=(x+48, y+96), gate=(x, y+64)
  RES:  top=(x+16, y), bottom=(x+16, y+80)
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, input_names, count_jfets,
)
from model import GateType

# Pin offsets from symbol origin
NJF_DRAIN = (48, 0)
NJF_SOURCE = (48, 96)
NJF_GATE = (0, 64)
RES_TOP = (16, 0)
RES_BOT = (16, 80)

# Supply source position (must be on 16-grid)
SUPPLY_X = -496


def _grid(v):
    """Snap to 16-unit grid."""
    return round(v / 16) * 16


class _LayoutState:
    def __init__(self):
        self.lines = []
        self.counter = 0

    def njf(self, x, y, name, model):
        self.lines.append(f"SYMBOL njf {x} {y} R0")
        self.lines.append(f"SYMATTR InstName {name}")
        self.lines.append(f"SYMATTR Value {model}")
        self.counter += 1
        return (x + NJF_DRAIN[0], y + NJF_DRAIN[1],    # drain
                x + NJF_SOURCE[0], y + NJF_SOURCE[1],   # source
                x + NJF_GATE[0], y + NJF_GATE[1])       # gate

    def res(self, x, y, name, value):
        self.lines.append(f"SYMBOL res {x} {y} R0")
        self.lines.append(f"SYMATTR InstName {name}")
        self.lines.append(f"SYMATTR Value {value:.0f}")
        self.counter += 1
        return (x + RES_TOP[0], y + RES_TOP[1],         # top
                x + RES_BOT[0], y + RES_BOT[1])         # bottom

    def wire(self, x1, y1, x2, y2):
        if (x1, y1) != (x2, y2):
            self.lines.append(f"WIRE {x1} {y1} {x2} {y2}")

    def flag(self, x, y, label):
        self.lines.append(f"FLAG {x} {y} {label}")


def _layout_j1(net, x, y, ls, gate_name, jfet_model, input_labels, input_idx):
    """Recursively lay out J1 pull-down network.

    Args:
        x, y: position for topmost/leftmost JFET symbol origin
        ls: _LayoutState

    Returns:
        drain_x, drain_y: top connection point (Node A side)
        source_x, source_y: bottom connection point (GND side)
        width: horizontal extent of this subtree
    """
    if isinstance(net, Leaf):
        idx = input_idx[0]
        input_idx[0] += 1
        label = input_labels[idx] if idx < len(input_labels) else f"in{idx}"

        d_x, d_y, s_x, s_y, g_x, g_y = ls.njf(
            x, y, f"J1_{gate_name}_{idx}", jfet_model)

        ls.wire(g_x, g_y, g_x - 16, g_y)
        ls.flag(g_x - 16, g_y, label)

        return d_x, d_y, s_x, s_y, 96

    if isinstance(net, Parallel):
        # Children side by side at same y, 96-unit spacing between symbol origins
        child_x = x
        drain_points = []
        source_points = []
        total_width = 0

        for i, child in enumerate(net.children):
            d_x, d_y, s_x, s_y, w = _layout_j1(
                child, child_x, y, ls, gate_name, jfet_model,
                input_labels, input_idx)
            drain_points.append((d_x, d_y))
            source_points.append((s_x, s_y))
            child_x += w
            total_width += w

        # Wire connecting all drain points horizontally
        if len(drain_points) > 1:
            all_dx = [p[0] for p in drain_points]
            ls.wire(min(all_dx), drain_points[0][1], max(all_dx), drain_points[0][1])

        # GND on each parallel leg's source
        for s_x, s_y in source_points:
            ls.flag(s_x, s_y, "0")

        # Return leftmost drain as connection point
        return drain_points[0][0], drain_points[0][1], \
               source_points[0][0], source_points[0][1], total_width

    if isinstance(net, Series):
        # Children stacked vertically at same x, 96 apart (source=drain)
        curr_y = y
        first_drain = None
        total_width = 96  # single column

        for i, child in enumerate(net.children):
            d_x, d_y, s_x, s_y, w = _layout_j1(
                child, x, curr_y, ls, gate_name, jfet_model,
                input_labels, input_idx)
            if first_drain is None:
                first_drain = (d_x, d_y)
            total_width = max(total_width, w)
            curr_y = s_y  # next child starts at this source

        # GND at bottom of chain
        ls.flag(s_x, s_y, "0")

        return first_drain[0], first_drain[1], s_x, s_y, total_width


def _emit_gate(ls, gate_name, params, network, xb, yb,
               input_labels, output_label, jfet_model):
    """Emit a complete gate at (xb, yb)."""
    r1, r2, r3 = params.r1, params.r2, params.r3
    ctr = ls.counter

    # R1: symbol at (xb-16, yb+16)
    r1_top_x, r1_top_y, r1_bot_x, r1_bot_y = ls.res(
        xb - 16, yb + 16, f"R1_{ctr}", r1)
    ls.flag(r1_top_x, r1_top_y, "VDD")

    # Node A y: R1 bottom + 16 gap = where J1 drains connect
    node_a_y = r1_bot_y + 16

    # Layout J1 network, starting to the left of R1
    input_idx = [0]
    j1_start_x = xb - 48  # parallel leg starts here (or further left)

    # For complex topologies, put parallel leg at xb-48, series at xb+48
    if isinstance(network, Parallel):
        # Multiple children — lay them out
        d_x, d_y, s_x, s_y, j1_width = _layout_j1(
            network, j1_start_x - (count_jfets(network) - 2) * 48,
            node_a_y, ls, gate_name, jfet_model, input_labels, input_idx)
    else:
        d_x, d_y, s_x, s_y, j1_width = _layout_j1(
            network, j1_start_x, node_a_y, ls, gate_name, jfet_model,
            input_labels, input_idx)

    # Wire R1 bottom to Node A (drain level)
    ls.wire(r1_bot_x, r1_bot_y, r1_bot_x, node_a_y)

    # J2: to the right, gate connects to Node A
    # J2 symbol x = rightmost drain x + 48
    rightmost_drain_x = d_x
    # Find rightmost drain by checking all parallel children
    if isinstance(network, Parallel):
        # The wire already connects them, rightmost is at d_x + j1_width - 96 + 48
        rightmost_drain_x = max(d_x, r1_bot_x)  # at least at R1

    j2_x = rightmost_drain_x + 48
    j2_y = node_a_y - 64  # so J2 gate (at j2_y+64) aligns with node_a_y

    # Wire from R1/J1 drain area to J2 gate
    j2_gate_x = j2_x
    j2_gate_y = j2_y + NJF_GATE[1]  # = node_a_y
    ls.wire(r1_bot_x, node_a_y, j2_gate_x, j2_gate_y)

    j2_drain_x, j2_drain_y, j2_src_x, j2_src_y, j2_gx, j2_gy = ls.njf(
        j2_x, j2_y, f"J2_{gate_name}", jfet_model)
    ls.flag(j2_drain_x, j2_drain_y, "VDD")

    # R2: top pin must connect to J2 source
    # J2 source at (j2_src_x, j2_src_y). R2 top at (sym_x+16, sym_y).
    # Place R2 so top pin is 16 below J2 source (matching reference)
    r2_sym_x = j2_src_x - RES_TOP[0]
    r2_sym_y = j2_src_y - 16  # R2 top at j2_src_y - 16, close to J2 source
    r2_top_x, r2_top_y, r2_bot_x, r2_bot_y = ls.res(
        r2_sym_x, r2_sym_y, f"R2_{ls.counter}", r2)

    # R3: top = R2 bottom (zero gap)
    r3_sym_y = r2_bot_y
    r3_top_x, r3_top_y, r3_bot_x, r3_bot_y = ls.res(
        r2_sym_x, r3_sym_y, f"R3_{ls.counter}", r3)

    # Output flag at R2/R3 junction, wire goes right
    out_y = r2_bot_y
    ls.wire(r2_bot_x + 48, out_y, r2_bot_x, out_y)
    ls.flag(r2_bot_x + 48, out_y, output_label)

    # VSS at R3 bottom
    ls.flag(r3_bot_x, r3_bot_y, "VSS")

    return j1_width + 200  # approximate gate width


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
    if gate_networks is None:
        gate_networks = {}
        for g in netlist.gates.values():
            gate_networks[g.name] = gate_type_to_network(g.gate_type)

    ordered, feedback = netlist.topological_sort()
    all_gates = ordered + feedback

    ls = _LayoutState()

    x_pos = 0
    for gname in all_gates:
        gate = netlist.gates[gname]
        params = gate_params[gname]
        network = gate_networks[gname]

        width = _emit_gate(ls, gname, params, network, x_pos, 0,
                           gate.inputs, gate.output, jfet_model)

        ls.lines.append(f"TEXT {x_pos-48} -104 Left 2 ;{gname} ({gate.gate_type.value})")
        # Pitch must be multiple of 96 to keep all children on 16-grid
        pitch = ((width + 144) // 96 + 1) * 96
        x_pos += pitch

    # Supplies
    sx = SUPPLY_X
    ls.flag(sx, 16, "VDD")
    ls.flag(sx, 96, "0")
    ls.lines.append(f"SYMBOL voltage {sx} 0 R0")
    ls.lines.append(f"SYMATTR InstName V_VDD")
    ls.lines.append(f"SYMATTR Value {v_pos}")
    ls.flag(sx, 208, "VSS")
    ls.flag(sx, 304, "0")
    ls.lines.append(f"SYMBOL voltage {sx} 192 R0")
    ls.lines.append(f"SYMATTR InstName V_VSS")
    ls.lines.append(f"SYMATTR Value {v_neg}")

    if stimuli:
        stim_y = 496
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
            ls.flag(sx, stim_y + 16, net_name)
            ls.flag(sx, stim_y + 96, "0")
            ls.lines.append(f"SYMBOL voltage {sx} {stim_y} R0")
            ls.lines.append(f"SYMATTR InstName V_{net_name}")
            ls.lines.append(f"SYMATTR Value {val}")
            stim_y += 192

    if jfet_model_card is None:
        jfet_model_card = (
            f".model {jfet_model} NJF(Beta=0.135m Betatce=-0.5 Vto=-3.45 "
            "Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f Nr=4 "
            "Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
            "Vk=123.7 M=407m Pb=1)"
        )
    ls.lines.append(f"TEXT {sx-4} -304 Left 2 !.tran 0 {sim_time} 0 {sim_time/4000}")
    ls.lines.append(f"TEXT {sx-4} -352 Left 2 !{jfet_model_card}")

    header = ["Version 4", "SHEET 1 10000 10000"]
    with open(output_path, "w") as f:
        f.write("\n".join(header + ls.lines) + "\n")
    print(f"Exported {len(all_gates)} gates to {output_path}")
