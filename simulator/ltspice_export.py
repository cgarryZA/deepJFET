"""Export a Netlist to LTSpice .asc schematic format.

Layout matched to hand-corrected LTSpice schematics.
NJF pin offsets: drain=(+48, 0), source=(+48, +96), gate=(0, +64)
RES pin offsets: top=(+16, 0), bottom=(+16, +80)
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import (
    PulldownNetwork, Leaf, Series, Parallel,
    gate_type_to_network, input_names,
)
from model import GateType

# NJF pin offsets from symbol origin (x, y)
#   drain:  (x+48, y)
#   source: (x+48, y+96)
#   gate:   (x, y+64)

# RES pin offsets from symbol origin (x, y) R0
#   top:    (x+16, y)
#   bottom: (x+16, y+80)

GATE_PITCH = 1056  # horizontal spacing between gates


def _emit_gate(lines, gate_name, params, xb, yb, input_labels,
               output_label, jfet_model, counter):
    """Emit one INV/NOR gate at (xb, yb).

    Coordinates derived from the hand-corrected fixed reference:
      R1 symbol:  (xb-16, yb+100)   → pins at (xb, yb+100) and (xb, yb+180)
      J1 symbol:  (xb-184, yb+400)  → drain(-136), source(-136,+496), gate(-184,+464)
      J2 symbol:  (xb+136, yb+336)  → drain(+184,+336), source(+184,+432), gate(+136,+400)
      R2 symbol:  (xb+168, yb+446)  → pins at (+184,+446) and (+184,+526)
      R3 symbol:  (xb+168, yb+570)  → pins at (+184,+570) and (+184,+650)
    """
    r1, r2, r3 = params.r1, params.r2, params.r3
    n_ins = len(input_labels)

    # -- Wires --
    # VDD down to R1 top
    lines.append(f"WIRE {xb} {yb+116} {xb} {yb+16}")
    # R1 bottom down to Node A
    lines.append(f"WIRE {xb} {yb+400} {xb} {yb+196}")

    # Node A to J1 drain(s)
    for k in range(n_ins):
        j1_x = xb - 184 - k * 256
        j1_drain_x = j1_x + 48
        lines.append(f"WIRE {xb} {yb+400} {j1_drain_x} {yb+400}")

    # Node A to J2 gate area
    j2_sym_x = xb + 136
    lines.append(f"WIRE {j2_sym_x} {yb+400} {xb} {yb+400}")

    # J2 drain up to VDD
    j2_drain_x = j2_sym_x + 48  # = xb+184
    lines.append(f"WIRE {j2_drain_x} {yb+336} {j2_drain_x} {yb+298}")

    # J2 source down to R2 top
    lines.append(f"WIRE {j2_drain_x} {yb+462} {j2_drain_x} {yb+432}")

    # J1 source(s) to GND
    for k in range(n_ins):
        j1_x = xb - 184 - k * 256
        j1_src_x = j1_x + 48
        lines.append(f"WIRE {j1_src_x} {yb+544} {j1_src_x} {yb+496}")

    # R2 bottom to output label
    lines.append(f"WIRE {j2_drain_x} {yb+556} {j2_drain_x} {yb+542}")

    # Output to R3 top
    lines.append(f"WIRE {j2_drain_x} {yb+586} {j2_drain_x} {yb+556}")

    # R3 bottom to VSS
    lines.append(f"WIRE {j2_drain_x} {yb+720} {j2_drain_x} {yb+666}")

    # J1 gate wires to input labels
    for k in range(n_ins):
        j1_x = xb - 184 - k * 256
        j1_gate_y = yb + 464
        lines.append(f"WIRE {j1_x} {j1_gate_y} {j1_x - 80} {j1_gate_y}")

    # -- Flags --
    lines.append(f"FLAG {xb} {yb+16} VDD")
    lines.append(f"FLAG {j2_drain_x} {yb+298} VDD")
    lines.append(f"FLAG {j2_drain_x} {yb+556} {output_label}")
    lines.append(f"FLAG {j2_drain_x} {yb+720} VSS")
    for k in range(n_ins):
        j1_x = xb - 184 - k * 256
        lines.append(f"FLAG {j1_x + 48} {yb+544} 0")
        lines.append(f"FLAG {j1_x - 80} {yb+464} {input_labels[k]}")

    # -- Symbols --
    # R1
    lines.append(f"SYMBOL res {xb-16} {yb+100} R0")
    lines.append(f"SYMATTR InstName R1_{counter[0]}")
    lines.append(f"SYMATTR Value {r1:.0f}")
    counter[0] += 1

    # J1(s)
    for k in range(n_ins):
        j1_x = xb - 184 - k * 256
        lines.append(f"SYMBOL njf {j1_x} {yb+400} R0")
        lines.append(f"SYMATTR InstName J1_{gate_name}_{k}")
        lines.append(f"SYMATTR Value {jfet_model}")

    # J2
    lines.append(f"SYMBOL njf {j2_sym_x} {yb+336} R0")
    lines.append(f"SYMATTR InstName J2_{gate_name}")
    lines.append(f"SYMATTR Value {jfet_model}")

    # R2
    lines.append(f"SYMBOL res {xb+168} {yb+446} R0")
    lines.append(f"SYMATTR InstName R2_{counter[0]}")
    lines.append(f"SYMATTR Value {r2:.0f}")
    counter[0] += 1

    # R3
    lines.append(f"SYMBOL res {xb+168} {yb+570} R0")
    lines.append(f"SYMATTR InstName R3_{counter[0]}")
    lines.append(f"SYMATTR Value {r3:.0f}")
    counter[0] += 1


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

    # Emit all wires and symbols first, then flags, to match LTSpice ordering
    for i, gname in enumerate(all_gates):
        gate = netlist.gates[gname]
        params = gate_params[gname]
        x = i * GATE_PITCH

        _emit_gate(lines, gname, params, x, 0,
                   gate.inputs, gate.output,
                   jfet_model, counter)

        lines.append(f"TEXT {x-48} -104 Left 2 ;{gname} ({gate.gate_type.value})")

    # Supply sources
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

    # Stimuli
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

    # Model card + sim command
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
