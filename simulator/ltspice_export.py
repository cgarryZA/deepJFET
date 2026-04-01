"""Export a Netlist to LTSpice .asc schematic format.

Layout matched to the minimal hand-built LTSpice inverter.
Components packed tight — pins connect directly or with minimal wires.

NJF pins from symbol (x,y): drain=(x+48,y), source=(x+48,y+96), gate=(x,y+64)
RES pins from symbol (x,y): top=(x+16,y), bottom=(x+16,y+80)
"""

from simulator.netlist import Netlist, Gate
from simulator.precompute import CircuitParams
from model.network import gate_type_to_network, input_names
from model import GateType

# Minimal inverter layout (from hand-built reference):
#
#   R1 sym (-112, 32)   top=(-96,32)=VDD   bot=(-96,112)
#   J1 sym (-144, 128)  drain=(-96,128)     src=(-96,224)=GND  gate=(-144,192)=INPUT
#   J2 sym (-48, 64)    drain=(0,64)=VDD    src=(0,160)        gate=(-48,128)=NodeA
#   R2 sym (-16, 144)   top=(0,144)         bot=(0,224)
#   R3 sym (-16, 224)   top=(0,224)=OUT     bot=(0,304)=VSS
#
# Connections:
#   R1 bot (-96,112) → J1 drain (-96,128): wire, 16 units
#   Node A = junction of R1/J1/J2gate at (-96,128) area
#   Wire (-48,128) to (-96,128) connects J2 gate to Node A
#   J2 src (0,160) → R2 top (0,144): 16 unit overlap (LTSpice connects)
#   R2 bot (0,224) = R3 top (0,224): direct connection
#
# Total gate footprint: ~200 wide, ~280 tall

GATE_PITCH = 320  # compact horizontal spacing


def _emit_gate(lines, gate_name, params, xb, yb, input_labels,
               output_label, jfet_model, counter):
    """Emit one INV/NOR gate at (xb, yb) using minimal layout."""
    r1, r2, r3 = params.r1, params.r2, params.r3
    n_ins = len(input_labels)

    # Offset base: R1/J1 column at xb, J2/R2/R3 column at xb+96

    # -- Wires (only where pins don't directly touch) --
    # Node A: connect R1 bot to J1 drain to J2 gate
    # R1 bot = (xb, 112), J1 drain = (xb, 128) → 16 gap, need wire
    # J2 gate = (xb+48, 128), connect to Node A at (xb, 128)
    lines.append(f"WIRE {xb+48} {yb+128} {xb} {yb+128}")

    # J1 gate to input label
    for k in range(n_ins):
        j1_x = xb - 48 - k * 160
        lines.append(f"WIRE {j1_x} {yb+192} {j1_x-16} {yb+192}")

    # J2 source (xb+96, 160) to R2 top (xb+96, 144) — overlap, but add wire for clarity
    # R2 bot (xb+96, 224) to R3 top (xb+96, 224) — same point, auto-connects
    # Output label wire
    lines.append(f"WIRE {xb+96+48} {yb+240} {xb+96} {yb+240}")

    # -- Flags --
    lines.append(f"FLAG {xb} {yb+48} VDD")          # R1 top
    lines.append(f"FLAG {xb+96} {yb+64} VDD")       # J2 drain
    lines.append(f"FLAG {xb+96+48} {yb+240} {output_label}")  # between R2/R3
    lines.append(f"FLAG {xb+96} {yb+320} VSS")      # R3 bottom
    for k in range(n_ins):
        j1_x = xb - 48 - k * 160
        lines.append(f"FLAG {xb} {yb+224} 0")       # J1 source = GND
        lines.append(f"FLAG {j1_x-16} {yb+192} {input_labels[k]}")

    # -- Symbols --
    # R1
    lines.append(f"SYMBOL res {xb-16} {yb+32} R0")
    lines.append(f"SYMATTR InstName R1_{counter[0]}")
    lines.append(f"SYMATTR Value {r1:.0f}")
    counter[0] += 1

    # J1(s)
    for k in range(n_ins):
        j1_x = xb - 48 - k * 160
        lines.append(f"SYMBOL njf {j1_x} {yb+128} R0")
        lines.append(f"SYMATTR InstName J1_{gate_name}_{k}")
        lines.append(f"SYMATTR Value {jfet_model}")

    # J2
    lines.append(f"SYMBOL njf {xb+48} {yb+64} R0")
    lines.append(f"SYMATTR InstName J2_{gate_name}")
    lines.append(f"SYMATTR Value {jfet_model}")

    # R2
    lines.append(f"SYMBOL res {xb+80} {yb+144} R0")
    lines.append(f"SYMATTR InstName R2_{counter[0]}")
    lines.append(f"SYMATTR Value {r2:.0f}")
    counter[0] += 1

    # R3
    lines.append(f"SYMBOL res {xb+80} {yb+224} R0")
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
