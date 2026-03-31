"""Hierarchical module system for RTL-level circuit composition."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .netlist import Gate
from model import GateType


class PortDir(Enum):
    IN = "IN"
    OUT = "OUT"


@dataclass
class Port:
    """A module port."""
    name: str
    direction: PortDir


@dataclass
class ModuleInstance:
    """An instance of a module within a parent module."""
    name: str
    module: "Module"
    connections: dict    # port_name -> net_name in parent scope


@dataclass
class Module:
    """A reusable hierarchical module containing gates and/or sub-module instances."""
    name: str
    ports: list                              # list of Port
    gates: list = field(default_factory=list)           # list of Gate (local names)
    submodules: list = field(default_factory=list)      # list of ModuleInstance

    def input_ports(self):
        return [p for p in self.ports if p.direction == PortDir.IN]

    def output_ports(self):
        return [p for p in self.ports if p.direction == PortDir.OUT]


def flatten(instance: ModuleInstance) -> list:
    """Recursively flatten a module instance into a flat list of Gates.

    All gate names and internal net names are prefixed with the instance path
    to avoid collisions. Port connections map internal port nets to external nets.
    """
    return _flatten_recursive(instance, prefix="")


def _flatten_recursive(instance: ModuleInstance, prefix: str) -> list:
    mod = instance.module
    inst_prefix = f"{prefix}{instance.name}." if prefix else f"{instance.name}."

    # Build the net name mapping: internal name -> external name
    # Port names map to the connected external nets
    net_map = {}
    for port in mod.ports:
        if port.name in instance.connections:
            net_map[port.name] = instance.connections[port.name]

    def resolve_net(internal_name):
        """Map an internal net name to its external (flattened) name."""
        if internal_name in net_map:
            return net_map[internal_name]
        return f"{inst_prefix}{internal_name}"

    flat_gates = []

    # Flatten local gates
    for g in mod.gates:
        flat_name = f"{inst_prefix}{g.name}"
        flat_inputs = [resolve_net(inp) for inp in g.inputs]
        flat_output = resolve_net(g.output)
        flat_gates.append(Gate(flat_name, g.gate_type, flat_inputs, flat_output))

    # Recursively flatten sub-module instances
    for sub_inst in mod.submodules:
        # Remap sub-instance connections through our net_map
        remapped_connections = {}
        for port_name, net_name in sub_inst.connections.items():
            remapped_connections[port_name] = resolve_net(net_name)

        remapped_inst = ModuleInstance(
            name=sub_inst.name,
            module=sub_inst.module,
            connections=remapped_connections,
        )
        flat_gates.extend(_flatten_recursive(remapped_inst, inst_prefix))

    return flat_gates


def flatten_top(module: Module, port_connections: dict = None) -> list:
    """Flatten a top-level module (no instance prefix for ports).

    port_connections maps port names to top-level net names.
    If not provided, port names are used as-is.
    """
    if port_connections is None:
        port_connections = {p.name: p.name for p in module.ports}

    inst = ModuleInstance(name="top", module=module, connections=port_connections)
    return _flatten_recursive(inst, prefix="")
