"""Gate-level netlist: Gate, Net, Netlist with topological sort."""

from dataclasses import dataclass, field
from typing import Optional
from collections import deque

from model import GateType, gate_input_count


@dataclass
class Gate:
    """A single logic gate in the netlist."""
    name: str
    gate_type: GateType
    inputs: list          # net names
    output: str           # net name

    def __post_init__(self):
        expected = gate_input_count(self.gate_type)
        if expected is not None and len(self.inputs) != expected:
            raise ValueError(
                f"Gate '{self.name}' ({self.gate_type.value}) expects "
                f"{expected} inputs, got {len(self.inputs)}"
            )


@dataclass
class Net:
    """A wire connecting gate outputs to gate inputs."""
    name: str
    driver: Optional[str] = None      # gate name that drives this net
    loads: list = field(default_factory=list)  # gate names that read this net

    @property
    def fan_out(self) -> int:
        return len(self.loads)

    @property
    def is_primary_input(self) -> bool:
        return self.driver is None


@dataclass
class Netlist:
    """Complete gate-level netlist with connectivity."""
    gates: dict           # name -> Gate
    nets: dict            # name -> Net
    primary_inputs: set   # net names not driven by any gate
    primary_outputs: set  # net names explicitly marked as outputs

    @classmethod
    def from_gates(cls, gate_list: list, primary_outputs: set = None) -> "Netlist":
        """Build a Netlist from a list of Gate objects.

        Connectivity is inferred from shared net names.
        Nets not driven by any gate are primary inputs.
        """
        gates = {}
        nets = {}

        def get_net(name):
            if name not in nets:
                nets[name] = Net(name=name)
            return nets[name]

        for g in gate_list:
            if g.name in gates:
                raise ValueError(f"Duplicate gate name: '{g.name}'")
            gates[g.name] = g

            # Register output net
            out_net = get_net(g.output)
            if out_net.driver is not None:
                raise ValueError(
                    f"Net '{g.output}' driven by both '{out_net.driver}' and '{g.name}'"
                )
            out_net.driver = g.name

            # Register input nets
            for inp in g.inputs:
                net = get_net(inp)
                net.loads.append(g.name)

        pi = {name for name, net in nets.items() if net.is_primary_input}
        po = primary_outputs or set()

        return cls(gates=gates, nets=nets, primary_inputs=pi, primary_outputs=po)

    def topological_sort(self) -> tuple:
        """Kahn's algorithm. Returns (ordered_gate_names, feedback_gate_names).

        Feedback gates are those in cycles (latches, flip-flops).
        """
        # Build adjacency: gate -> gates it feeds
        in_degree = {name: 0 for name in self.gates}
        successors = {name: [] for name in self.gates}

        for g in self.gates.values():
            out_net = self.nets[g.output]
            for load_name in out_net.loads:
                successors[g.name].append(load_name)
                in_degree[load_name] += 1

        # Gates whose inputs are all primary inputs start with in_degree from
        # internal nets only. Subtract contributions from primary inputs.
        for g in self.gates.values():
            for inp in g.inputs:
                net = self.nets[inp]
                if net.is_primary_input:
                    pass  # no internal driver, doesn't count

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        ordered = []

        while queue:
            name = queue.popleft()
            ordered.append(name)
            for succ in successors[name]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)

        feedback = [name for name in self.gates if name not in set(ordered)]
        return ordered, feedback

    def fan_out_map(self) -> dict:
        """Return gate_name -> output fan-out."""
        return {
            g.name: self.nets[g.output].fan_out
            for g in self.gates.values()
        }

    def validate(self) -> list:
        """Check for common netlist errors. Returns list of warning strings."""
        warnings = []
        for name, net in self.nets.items():
            if not net.is_primary_input and net.driver is None:
                warnings.append(f"Net '{name}' has no driver and is not a primary input")
            if net.fan_out == 0 and name not in self.primary_outputs:
                warnings.append(f"Net '{name}' has no loads (dangling output)")
        return warnings
