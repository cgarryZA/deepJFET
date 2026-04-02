"""Event-driven gate-level simulation engine."""

import heapq
from dataclasses import dataclass, field
from typing import Optional

from model import GateType
from .netlist import Netlist, Gate
from .precompute import GateProfile


NET_CHANGE = 0
GATE_EVAL = 1


@dataclass(order=True)
class Event:
    """A scheduled event -- net value change or gate re-evaluation."""
    time: float
    seq: int = field(compare=True)
    event_type: int = field(compare=False)
    net_name: str = field(default="", compare=False)
    new_value: bool = field(default=False, compare=False)
    new_voltage: float = field(default=0.0, compare=False)
    gate_name: str = field(default="", compare=False)
    source: str = field(default="", compare=False)


@dataclass
class NetState:
    """Current state of a net."""
    name: str
    value: bool = False
    voltage: float = 0.0
    last_change_time: float = 0.0
    history: list = field(default_factory=list)


@dataclass
class Stimulus:
    """External input waveform for a primary input net."""
    net_name: str
    times: list
    values: list


@dataclass
class SimResult:
    """Simulation results."""
    net_states: dict
    events_processed: int
    end_time: float
    delta_cycle_warnings: list


class SimulationEngine:
    """Event-driven digital simulation engine.

    All gates share the same global logic levels (v_high, v_low, threshold).
    Gate re-evaluations are scheduled after propagation delay so that
    simultaneous input changes are handled correctly.
    """

    def __init__(
        self,
        netlist: Netlist,
        profiles: dict,
        v_high: float,
        v_low: float,
        custom_profiles: dict = None,
        max_delta_cycles: int = 1000,
        auto_precompute_params=None,
    ):
        """
        Args:
            auto_precompute_params: optional CircuitParams. If set, any gate type
                found in the netlist that's missing from profiles will be
                auto-precomputed using these params before simulation starts.
        """
        self.netlist = netlist
        self.profiles = dict(profiles)  # copy so we can add to it
        self.custom_profiles = custom_profiles or {}
        self.max_delta = max_delta_cycles

        self.v_high = v_high
        self.v_low = v_low
        self.v_threshold = (v_high + v_low) / 2.0

        # Auto-precompute missing gate types
        if auto_precompute_params is not None:
            self._auto_precompute(auto_precompute_params)

        self._fan_out = netlist.fan_out_map()

        self._nets = {}
        for name in netlist.nets:
            self._nets[name] = NetState(name=name, value=False,
                                        voltage=self.v_low)

        self._queue = []
        self._seq = 0
        self._pending_evals = {}

    def _auto_precompute(self, params):
        """Detect gate types in the netlist missing from profiles and precompute them."""
        from .precompute import precompute_gate
        missing = set()
        for gate in self.netlist.gates.values():
            if gate.gate_type == GateType.CUSTOM:
                continue
            if gate.gate_type not in self.profiles:
                missing.add(gate.gate_type)
        if missing:
            print(f"Auto-precomputing missing gate types: "
                  f"{[gt.value for gt in missing]}")
            for gt in missing:
                profile = precompute_gate(gt, params, self.v_high, self.v_low)
                self.profiles[gt] = profile
                n = profile.n_inputs
                all_hi = tuple([True] * n)
                print(f"  {gt.value}: all-HIGH->{profile.dc_table[all_hi]:.4f}V, "
                      f"delay(fo=1)={profile.delay_table[1]*1e9:.1f}ns")

    def _push(self, ev):
        heapq.heappush(self._queue, ev)
        self._seq += 1

    def add_stimulus(self, stim: Stimulus):
        """Schedule input events from a stimulus definition."""
        for t, v in zip(stim.times, stim.values):
            voltage = self.v_high if v else self.v_low
            self._push(Event(
                time=t, seq=self._seq, event_type=NET_CHANGE,
                net_name=stim.net_name, new_value=v, new_voltage=voltage,
                source="stimulus",
            ))

    def force_evaluate_all(self):
        """Evaluate every gate once in topological order.

        This establishes consistent initial state for all gates, including
        those whose inputs are constant (which the event-driven sim would
        otherwise never evaluate). Call after setting initial net values
        and before running the simulation.
        """
        ordered, feedback = self.netlist.topological_sort()
        # Evaluate ordered gates first, then feedback gates
        all_gates = ordered + feedback
        for gate_name in all_gates:
            gate = self.netlist.gates[gate_name]
            new_val, new_voltage = self._evaluate_gate(gate)
            ns = self._nets[gate.output]
            changed = (ns.value != new_val)
            ns.value = new_val
            ns.voltage = new_voltage
            ns.history.append((0.0, new_val, new_voltage))
            # If output changed, schedule downstream gates
            if changed:
                out_net = self.netlist.nets[gate.output]
                for load_name in out_net.loads:
                    load_gate = self.netlist.gates[load_name]
                    delay = self._get_delay(load_gate)
                    self._schedule_gate_eval(load_name, delay)

        # Run a few iterations for feedback gates to settle
        for _ in range(3):
            for gate_name in feedback:
                gate = self.netlist.gates[gate_name]
                new_val, new_voltage = self._evaluate_gate(gate)
                ns = self._nets[gate.output]
                ns.value = new_val
                ns.voltage = new_voltage

    def set_initial_state(self, net_values: dict):
        """Set initial net values before simulation."""
        for name, val in net_values.items():
            if name in self._nets:
                self._nets[name].value = val
                self._nets[name].voltage = self.v_high if val else self.v_low

    def _get_profile(self, gate: Gate) -> GateProfile:
        if gate.gate_type == GateType.CUSTOM:
            return self.custom_profiles[gate.name]
        return self.profiles[gate.gate_type]

    def _evaluate_gate(self, gate: Gate) -> tuple:
        """Evaluate gate output from current input states. Returns (bool, voltage)."""
        profile = self._get_profile(gate)
        input_values = tuple(
            self._nets[inp].voltage > self.v_threshold
            for inp in gate.inputs
        )
        v_out = profile.dc_table.get(input_values)
        if v_out is None:
            raise ValueError(
                f"No DC table entry for gate '{gate.name}' "
                f"({gate.gate_type.value}) inputs={input_values}"
            )
        return v_out > self.v_threshold, v_out

    def _get_delay(self, gate: Gate) -> float:
        profile = self._get_profile(gate)
        fo = self._fan_out.get(gate.name, 0)
        max_fo = max(profile.delay_table.keys()) if profile.delay_table else 1
        fo_key = min(fo, max_fo)
        return profile.delay_table.get(fo_key, profile.delay_table.get(1, 1e-9))

    def _schedule_gate_eval(self, gate_name: str, at_time: float):
        prev = self._pending_evals.get(gate_name)
        if prev is not None and prev >= at_time:
            return
        self._pending_evals[gate_name] = at_time
        self._push(Event(
            time=at_time, seq=self._seq, event_type=GATE_EVAL,
            gate_name=gate_name,
        ))

    def run(self, end_time: float) -> SimResult:
        """Run the event-driven simulation until end_time."""
        events_processed = 0
        delta_warnings = []
        toggle_counts = {}

        while self._queue:
            ev = heapq.heappop(self._queue)
            if ev.time > end_time:
                break

            events_processed += 1

            if ev.event_type == NET_CHANGE:
                ns = self._nets[ev.net_name]
                if ns.value == ev.new_value and abs(ns.voltage - ev.new_voltage) < 1e-6:
                    continue

                key = (ev.net_name, ev.time)
                toggle_counts[key] = toggle_counts.get(key, 0) + 1
                if toggle_counts[key] > self.max_delta:
                    delta_warnings.append(
                        f"Net '{ev.net_name}' exceeded {self.max_delta} toggles "
                        f"at t={ev.time*1e9:.1f}ns"
                    )
                    continue

                ns.value = ev.new_value
                ns.voltage = ev.new_voltage
                ns.last_change_time = ev.time
                ns.history.append((ev.time, ev.new_value, ev.new_voltage))

                net = self.netlist.nets[ev.net_name]
                for gate_name in net.loads:
                    gate = self.netlist.gates[gate_name]
                    delay = self._get_delay(gate)
                    self._schedule_gate_eval(gate_name, ev.time + delay)

            elif ev.event_type == GATE_EVAL:
                self._pending_evals.pop(ev.gate_name, None)
                gate = self.netlist.gates[ev.gate_name]
                new_val, new_v = self._evaluate_gate(gate)
                out_ns = self._nets[gate.output]

                if new_val == out_ns.value and abs(new_v - out_ns.voltage) < 1e-6:
                    continue

                key = (gate.output, ev.time)
                toggle_counts[key] = toggle_counts.get(key, 0) + 1
                if toggle_counts[key] > self.max_delta:
                    delta_warnings.append(
                        f"Net '{gate.output}' exceeded {self.max_delta} toggles "
                        f"at t={ev.time*1e9:.1f}ns"
                    )
                    continue

                out_ns.value = new_val
                out_ns.voltage = new_v
                out_ns.last_change_time = ev.time
                out_ns.history.append((ev.time, new_val, new_v))

                out_net = self.netlist.nets[gate.output]
                for load_name in out_net.loads:
                    load_gate = self.netlist.gates[load_name]
                    delay = self._get_delay(load_gate)
                    self._schedule_gate_eval(load_name, ev.time + delay)

        for ns in self._nets.values():
            if not ns.history:
                ns.history.append((0.0, ns.value, ns.voltage))

        return SimResult(
            net_states=dict(self._nets),
            events_processed=events_processed,
            end_time=end_time,
            delta_cycle_warnings=delta_warnings,
        )
