"""Export a Netlist to LTSpice .asc schematic format.

Each gate becomes: R1 + J1 (pull-down) + J2 (load) + R2 + R3.
Layout matches hand-built LTSpice schematics with proper grid alignment.
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, input_names, count_midpoints,
)
from model import GateType


# LTSpice snaps to 16-unit grid. All coordinates must be multiples of 16.
# Gate horizontal spacing ~1000 units.
GATE_PITCH = 1056


def _align(v):
    """Snap to nearest 16-unit grid."""
    return round(v / 16) * 16


def _emit_gate_inv(lines, gate_name, params, x, y, input_labels,
                   output_label, jfet_model, counter):
    """Emit one INV gate (or NOR with parallel J1s) at position (x, y).

    Layout (relative to x, y as top-left origin):
      R1:  (x, y+100) vertical, VDD at top
      J1:  (x-184, y+400) pull-down, gate left to input label
      J2:  (x+136, y+336) load, drain up to (x+184, y+298)=VDD
      R2:  (x+168, y+462)
      R3:  (x+168, y+602)
      Output label at (x+184, y+572)
      VSS at (x+184, y+752)
    """
    r1, r2, r3 = params.r1, params.r2, params.r3
    n_ins = len(input_labels)

    # -- R1: VDD to Node A --
    lines.append(f"WIRE {x} {y+116} {x} {y+16}")
    lines.append(f"FLAG {x} {y+16} VDD")
    lines.append(f"SYMBOL res {x-16} {y+100} R0")
    lines.append(f"SYMATTR InstName R1_{counter[0]}")
    lines.append(f"SYMATTR Value {r1:.0f}")
    counter[0] += 1

    # Node A area: y+400
    na_y = y + 400

    # Wire from R1 bottom to Node A
    lines.append(f"WIRE {x} {na_y} {x} {y+196}")

    # -- J1 pull-down JFETs (parallel for NOR) --
    for k in range(n_ins):
        j1_x = x - 184 - k * 256
        j1_y = na_y

        # Wire from Node A to J1 drain
        lines.append(f"WIRE {x} {na_y} {j1_x+48} {na_y}")
        # J1 symbol
        lines.append(f"SYMBOL njf {j1_x} {j1_y} R0")
        lines.append(f"SYMATTR InstName J1_{gate_name}_{k}")
        lines.append(f"SYMATTR Value {jfet_model}")
        # J1 source to GND
        lines.append(f"WIRE {j1_x+48} {j1_y+96} {j1_x+48} {j1_y+144}")
        lines.append(f"FLAG {j1_x+48} {j1_y+144} 0")
        # J1 gate wire to input label
        lines.append(f"WIRE {j1_x} {j1_y+64} {j1_x-80} {j1_y+64}")
        lines.append(f"FLAG {j1_x-80} {j1_y+64} {input_labels[k]}")

    # -- J2 load transistor --
    j2_x = x + 136
    j2_y = y + 336

    # Wire from Node A right to J2 area
    lines.append(f"WIRE {x} {na_y} {j2_x} {na_y}")

    # J2 symbol (R0, same orientation as J1)
    lines.append(f"SYMBOL njf {j2_x} {j2_y} R0")
    lines.append(f"SYMATTR InstName J2_{gate_name}")
    lines.append(f"SYMATTR Value {jfet_model}")

    # J2 drain goes UP to VDD
    j2_drain_x = j2_x + 48
    lines.append(f"WIRE {j2_drain_x} {j2_y} {j2_drain_x} {y+298}")
    lines.append(f"FLAG {j2_drain_x} {y+298} VDD")

    # J2 gate wire down from Node A
    # J2 gate pin is at (j2_x, j2_y+64) — need wire from Node A level
    lines.append(f"WIRE {j2_x} {j2_y+64} {j2_x} {na_y}")

    # J2 source = Node B, goes down to R2
    node_b_x = j2_drain_x
    node_b_y = j2_y + 96

    # -- R2 --
    r2_y = y + 462
    lines.append(f"WIRE {node_b_x} {node_b_y} {node_b_x} {r2_y}")
    lines.append(f"SYMBOL res {node_b_x-16} {r2_y} R0")
    lines.append(f"SYMATTR InstName R2_{counter[0]}")
    lines.append(f"SYMATTR Value {r2:.0f}")
    counter[0] += 1

    # -- Output node between R2 and R3 --
    out_y = y + 572
    lines.append(f"WIRE {node_b_x} {r2_y+96} {node_b_x} {out_y}")
    lines.append(f"FLAG {node_b_x} {out_y} {output_label}")

    # -- R3 --
    r3_y = y + 602
    lines.append(f"WIRE {node_b_x} {out_y} {node_b_x} {r3_y}")
    lines.append(f"SYMBOL res {node_b_x-16} {r3_y} R0")
    lines.append(f"SYMATTR InstName R3_{counter[0]}")
    lines.append(f"SYMATTR Value {r3:.0f}")
    counter[0] += 1

    # VSS
    vss_y = y + 752
    lines.append(f"WIRE {node_b_x} {r3_y+96} {node_b_x} {vss_y}")
    lines.append(f"FLAG {node_b_x} {vss_y} VSS")


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
        gate_networks: gate_name -> PulldownNetwork (optional)
        stimuli: net_name -> dict with PULSE params or string
        jfet_model: SPICE model name
        jfet_model_card: full .model line
        v_pos, v_neg: supply voltages
        sim_time: transient simulation time
        output_path: output file path
    """
    lines = [
        "Version 4",
        "SHEET 1 10000 10000",
    ]

    if gate_networks is None:
        gate_networks = {}
        for g in netlist.gates.values():
            gate_networks[g.name] = gate_type_to_network(g.gate_type)

    ordered, feedback = netlist.topological_sort()
    all_gates = ordered + feedback

    counter = [0]
    y_origin = 0

    for i, gname in enumerate(all_gates):
        gate = netlist.gates[gname]
        params = gate_params[gname]
        x = i * GATE_PITCH

        _emit_gate_inv(
            lines, gname, params, x, y_origin,
            gate.inputs, gate.output,
            jfet_model, counter,
        )

        # Gate name label
        lines.append(f"TEXT {x-48} {y_origin-104} Left 2 ;{gname} ({gate.gate_type.value})")

    # Supply voltage sources (far left)
    sx = -500
    lines.append(f"FLAG {sx} {16} VDD")
    lines.append(f"FLAG {sx} {96} 0")
    lines.append(f"SYMBOL voltage {sx} 0 R0")
    lines.append(f"SYMATTR InstName V_VDD")
    lines.append(f"SYMATTR Value {v_pos}")

    lines.append(f"FLAG {sx} {216} VSS")
    lines.append(f"FLAG {sx} {296} 0")
    lines.append(f"SYMBOL voltage {sx} 200 R0")
    lines.append(f"SYMATTR InstName V_VSS")
    lines.append(f"SYMATTR Value {v_neg}")

    # Stimulus sources
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

    # Model card and simulation command
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
