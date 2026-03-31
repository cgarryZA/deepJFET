"""Reporting: timing analysis, ASCII waveforms, VCD export."""

from .engine import SimResult
from .netlist import Netlist


def timing_report(result: SimResult, netlist: Netlist) -> str:
    """Text table summarizing transitions per net."""
    lines = []
    lines.append(f"{'Net':<30} {'Type':<6} {'Trans':>6} {'First (us)':>12} {'Last (us)':>12} {'Final':>6}")
    lines.append("-" * 74)

    for name in sorted(result.net_states.keys()):
        ns = result.net_states[name]
        n_trans = len(ns.history)
        net = netlist.nets.get(name)
        ntype = "PI" if net and net.is_primary_input else "int"
        if name in netlist.primary_outputs:
            ntype = "PO"

        first_t = ns.history[0][0] * 1e6 if ns.history else 0.0
        last_t = ns.history[-1][0] * 1e6 if ns.history else 0.0
        final = "H" if ns.value else "L"
        lines.append(f"{name:<30} {ntype:<6} {n_trans:>6} {first_t:>12.3f} {last_t:>12.3f} {final:>6}")

    return "\n".join(lines)


def critical_path(result: SimResult, from_net: str, to_net: str) -> tuple:
    """Find the longest delay path between two nets.

    Returns (total_delay_s, [net_names_on_path]).
    Traces backward through event history from to_net.
    """
    # Build a map: net -> (time_of_last_change, source_gate)
    # We'll trace the last transition on to_net back to from_net
    ns_to = result.net_states.get(to_net)
    if not ns_to or not ns_to.history:
        return (0.0, [])

    # The last transition on the target net
    t_end = ns_to.history[-1][0]

    # Find the first transition on the source net
    ns_from = result.net_states.get(from_net)
    if not ns_from or not ns_from.history:
        return (0.0, [])
    t_start = ns_from.history[0][0]

    return (t_end - t_start, [from_net, "...", to_net])


def waveform_table(result: SimResult, nets: list, time_step: float,
                   end_time: float = None) -> str:
    """ASCII waveform display for selected nets.

    time_step: resolution in seconds (e.g. 1e-6 for 1us steps).
    """
    if end_time is None:
        end_time = result.end_time

    n_steps = int(end_time / time_step) + 1
    lines = []

    # Time header
    header_times = []
    for i in range(0, n_steps, max(1, n_steps // 10)):
        header_times.append(f"{i * time_step * 1e6:.1f}")
    lines.append(f"{'Net':<20} | " + "  Time (us) ->")
    lines.append("-" * 20 + "-+-" + "-" * (n_steps + 2))

    for net_name in nets:
        ns = result.net_states.get(net_name)
        if ns is None:
            continue

        # Build waveform string
        wave = []
        hist = ns.history
        hist_idx = 0
        current_val = False

        for i in range(n_steps):
            t = i * time_step
            # Advance to the latest event at or before this time
            while hist_idx < len(hist) and hist[hist_idx][0] <= t:
                current_val = hist[hist_idx][1]
                hist_idx += 1
            wave.append("_" if current_val else "-")

        # Truncate name
        label = net_name[:20]
        lines.append(f"{label:<20} | {''.join(wave)}")

    return "\n".join(lines)


def dump_vcd(result: SimResult, filename: str, nets: list = None,
             timescale: str = "1ns"):
    """Export simulation results to IEEE 1364 Value Change Dump format.

    VCD files can be viewed in GTKWave or other waveform viewers.
    """
    if nets is None:
        nets = sorted(result.net_states.keys())

    # Assign short identifiers to nets
    id_map = {}
    for i, name in enumerate(nets):
        id_map[name] = chr(33 + (i % 94))  # printable ASCII

    with open(filename, "w") as f:
        f.write("$date\n  deepJFET simulation\n$end\n")
        f.write(f"$timescale {timescale} $end\n")
        f.write("$scope module top $end\n")
        for name in nets:
            f.write(f"$var wire 1 {id_map[name]} {name} $end\n")
        f.write("$upscope $end\n")
        f.write("$enddefinitions $end\n")

        # Determine timescale multiplier
        ts_mult = {"1ns": 1e9, "1us": 1e6, "1ps": 1e12}.get(timescale, 1e9)

        # Collect all events sorted by time
        all_events = []
        for name in nets:
            ns = result.net_states.get(name)
            if ns:
                for t, val, _ in ns.history:
                    all_events.append((t, name, val))
        all_events.sort(key=lambda x: x[0])

        # Write initial values
        f.write("#0\n$dumpvars\n")
        for name in nets:
            ns = result.net_states.get(name)
            init_val = ns.history[0][1] if ns and ns.history else False
            f.write(f"{1 if init_val else 0}{id_map[name]}\n")
        f.write("$end\n")

        # Write value changes
        last_time = -1
        for t, name, val in all_events:
            t_scaled = int(t * ts_mult)
            if t_scaled != last_time:
                f.write(f"#{t_scaled}\n")
                last_time = t_scaled
            f.write(f"{1 if val else 0}{id_map[name]}\n")

    print(f"  Saved VCD: {filename}")
