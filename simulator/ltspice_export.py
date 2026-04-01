"""Export a Netlist to LTSpice .asc schematic format.

Layout derived from EXAMPLE.asc hand-built reference.
The J2/R1/R2/R3 invariant structure never changes position.
Only the J1 pull-down network varies (parallel=left, series=down).

All coordinates relative to xb (R1 top pin x-position).
Fixed y-coordinates from the reference:
  R1 sym y=-128, J2 sym y=-96, J1 starts at y=-32
  R2 sym y=-16, R3 sym y=64
  VDD at y=-112, Out at y=80, VSS at y=160
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, count_jfets,
)
from model import GateType

SUPPLY_X = -496

# Fixed y-coordinates (from EXAMPLE.asc)
Y_R1 = -128       # R1 symbol y
Y_J2 = -96        # J2 symbol y
Y_NODE_A = -32     # J1 drain level / J2 gate level
Y_R2 = -16        # R2 symbol y
Y_R3 = 64         # R3 symbol y
Y_VDD = -112      # VDD flag
Y_OUT = 80        # output flag
Y_VSS = 160       # VSS flag

# NJF pin offsets: drain=(+48,0), source=(+48,+96), gate=(0,+64)
# RES pin offsets: top=(+16,0), bottom=(+16,+80)


def _layout_j1(net, x, y, lines, gate_name, jfet_model, input_labels, idx):
    """Layout J1 network. Parallel goes LEFT, series goes DOWN.

    x, y: symbol origin of the first/topmost JFET.
    Returns: (leftmost_x, bottommost_y) for sizing.
    """
    if isinstance(net, Leaf):
        i = idx[0]; idx[0] += 1
        label = input_labels[i] if i < len(input_labels) else f"in{i}"
        lines.append(f"SYMBOL njf {x} {y} R0")
        lines.append(f"SYMATTR InstName J1_{gate_name}_{i}")
        lines.append(f"SYMATTR Value {jfet_model}")
        # Gate label at gate pin: (x, y+64)
        lines.append(f"FLAG {x} {y+64} {label}")
        # Source to GND at source pin: (x+48, y+96)
        lines.append(f"FLAG {x+48} {y+96} 0")
        return x, y + 96

    if isinstance(net, Parallel):
        cx = x
        min_x = x
        max_bot = y
        for child in net.children:
            lx, by = _layout_j1(child, cx, y, lines, gate_name,
                                jfet_model, input_labels, idx)
            min_x = min(min_x, lx)
            max_bot = max(max_bot, by)
            # Next parallel child 96 to the LEFT
            if isinstance(child, Leaf):
                cx -= 96
            elif isinstance(child, Series):
                cx -= 96
            else:
                cx -= count_jfets(child) * 96
        # Horizontal wire connecting all parallel drains
        rightmost = x + 48   # first child drain
        leftmost = cx + 48 + 96  # last child placed drain
        if rightmost != leftmost:
            lines.append(f"WIRE {rightmost} {y} {leftmost} {y}")
        return min_x, max_bot

    if isinstance(net, Series):
        cy = y
        top_x = x
        for i, child in enumerate(net.children):
            lx, by = _layout_j1(child, x, cy, lines, gate_name,
                                jfet_model, input_labels, idx)
            if i == 0:
                top_x = lx
            # Remove the GND flag from intermediate nodes (only bottom gets GND)
            if i < len(net.children) - 1:
                # Remove last FLAG line (the GND we just added)
                for j in range(len(lines) - 1, -1, -1):
                    if lines[j].startswith("FLAG") and " 0" in lines[j]:
                        lines.pop(j)
                        break
            cy = by  # next child at source of current
        return top_x, cy


def _emit_gate(lines, gate_name, params, network, xb, input_labels,
               output_label, jfet_model):
    """Emit one gate. xb = R1 top pin x-position.

    The invariant (J2/R2/R3) is at fixed offsets from xb.
    J1 network hangs off Node A to the left.
    """
    r1, r2, r3 = params.r1, params.r2, params.r3

    # Invariant structure
    # R1: sym at (xb-16, Y_R1), top pin (xb, Y_R1), bot pin (xb, Y_R1+80)
    lines.append(f"SYMBOL res {xb-16} {Y_R1} R0")
    lines.append(f"SYMATTR InstName R1_{gate_name}")
    lines.append(f"SYMATTR Value {r1:.0f}")

    # J2: sym at (xb+16, Y_J2), drain=(xb+64, Y_J2), src=(xb+64, Y_J2+96), gate=(xb+16, Y_J2+64=Y_NODE_A)
    lines.append(f"SYMBOL njf {xb+16} {Y_J2} R0")
    lines.append(f"SYMATTR InstName J2_{gate_name}")
    lines.append(f"SYMATTR Value {jfet_model}")

    # R2: sym at (xb+48, Y_R2), top=(xb+64, Y_R2), bot=(xb+64, Y_R2+80)
    lines.append(f"SYMBOL res {xb+48} {Y_R2} R0")
    lines.append(f"SYMATTR InstName R2_{gate_name}")
    lines.append(f"SYMATTR Value {r2:.0f}")

    # R3: sym at (xb+48, Y_R3), top=(xb+64, Y_R3), bot=(xb+64, Y_R3+80)
    lines.append(f"SYMBOL res {xb+48} {Y_R3} R0")
    lines.append(f"SYMATTR InstName R3_{gate_name}")
    lines.append(f"SYMATTR Value {r3:.0f}")

    # J1 pull-down network at Node A (y = Y_NODE_A)
    # First J1 symbol at (xb-48, Y_NODE_A)
    idx = [0]
    min_x, max_bot = _layout_j1(network, xb - 48, Y_NODE_A, lines,
                                 gate_name, jfet_model, input_labels, idx)

    # --- Wires ---
    # VDD: R1 top (xb, -128) across to J2 drain (xb+64, -96)
    # Wire at Y_VDD connecting R1 top pin area to J2 drain area
    lines.append(f"WIRE {xb+64} {Y_VDD} {xb} {Y_VDD}")
    lines.append(f"WIRE {xb+80} {Y_VDD} {xb+64} {Y_VDD}")
    # Bridge J2 drain pin (xb+64, Y_J2) up to VDD wire (xb+64, Y_VDD)
    lines.append(f"WIRE {xb+64} {Y_J2} {xb+64} {Y_VDD}")

    # Node A: R1 bot (xb, -48) to J1 drain (xb, -32) to J2 gate (xb+16, -32)
    lines.append(f"WIRE {xb} {Y_NODE_A} {xb+16} {Y_NODE_A}")
    # If parallel J1s, extend Node A wire left to leftmost drain
    leftmost_drain = min_x + 48  # leftmost J1 symbol + 48 = drain x
    rightmost_drain = xb  # first J1 drain at xb
    if leftmost_drain < rightmost_drain:
        lines.append(f"WIRE {rightmost_drain} {Y_NODE_A} {leftmost_drain} {Y_NODE_A}")

    # Out: wire from R2/R3 junction (xb+64, Y_OUT) right to flag
    lines.append(f"WIRE {xb+96} {Y_OUT} {xb+64} {Y_OUT}")

    # --- Flags ---
    lines.append(f"FLAG {xb+80} {Y_VDD} VDD")
    lines.append(f"FLAG {xb+96} {Y_OUT} {output_label}")
    lines.append(f"FLAG {xb+64} {Y_VSS} VSS")

    # Width for gate spacing
    width = (xb - min_x) + 192
    return width


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

    lines = []
    x_pos = 0

    for gname in all_gates:
        gate = netlist.gates[gname]
        params = gate_params[gname]
        network = gate_networks[gname]

        width = _emit_gate(lines, gname, params, network, x_pos,
                           gate.inputs, gate.output, jfet_model)

        lines.append(f"TEXT {x_pos-48} {Y_R1-64} Left 2 ;{gname} ({gate.gate_type.value})")

        # Next gate: round up to multiple of 96
        pitch = ((width + 96) // 96) * 96
        x_pos += pitch

    # Supplies
    sx = SUPPLY_X
    lines.append(f"SYMBOL voltage {sx} 0 R0")
    lines.append(f"SYMATTR InstName V_VDD")
    lines.append(f"SYMATTR Value {v_pos}")
    lines.append(f"FLAG {sx} 16 VDD")
    lines.append(f"FLAG {sx} 96 0")

    lines.append(f"SYMBOL voltage {sx} 192 R0")
    lines.append(f"SYMATTR InstName V_VSS")
    lines.append(f"SYMATTR Value {v_neg}")
    lines.append(f"FLAG {sx} 208 VSS")
    lines.append(f"FLAG {sx} 304 0")

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
            lines.append(f"SYMBOL voltage {sx} {stim_y} R0")
            lines.append(f"SYMATTR InstName V_{net_name}")
            lines.append(f"SYMATTR Value {val}")
            lines.append(f"FLAG {sx} {stim_y+16} {net_name}")
            lines.append(f"FLAG {sx} {stim_y+96} 0")
            stim_y += 192

    if jfet_model_card is None:
        jfet_model_card = (
            f".model {jfet_model} NJF(Beta=0.135m Betatce=-0.5 Vto=-3.45 "
            "Vtotc=-2.5m Lambda=0.005 Is=205.2f Xti=3 Isr=1988f Nr=4 "
            "Alpha=20.98u N=3 Rd=1 Rs=1 Cgd=16.9p Cgs=16.9p Fc=0.5 "
            "Vk=123.7 M=407m Pb=1)"
        )
    lines.append(f"TEXT {sx-16} -304 Left 2 !.tran 0 {sim_time} 0 {sim_time/4000}")
    lines.append(f"TEXT {sx-16} -352 Left 2 !{jfet_model_card}")

    header = ["Version 4", "SHEET 1 10000 10000"]
    with open(output_path, "w") as f:
        f.write("\n".join(header + lines) + "\n")
    print(f"Exported {len(all_gates)} gates to {output_path}")
