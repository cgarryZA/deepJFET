"""Arbitrary series/parallel JFET pull-down network topology.

The pull-down network sits between Node_A (drain side) and GND (source side).
It determines the J1 drain current that flows into Node_A. Everything else
(J2 buffer, R1, R2, R3, output divider) is topology-independent.

Topology is a recursive tree:
  Leaf("A")                        — single JFET, gate driven by input A
  Parallel((Leaf("A"), Leaf("B"))) — NOR: A and B in parallel
  Series((Leaf("A"), Leaf("B")))   — NAND: A and B in series
  Parallel((Leaf("A"), Series((Leaf("B"), Leaf("C")))))  — A NOR (B NAND C)

Frozen dataclasses with tuples → hashable, immutable, cacheable.
"""

import itertools
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Leaf:
    """Single JFET. Gate driven by input_name, drain=top node, source=bottom node."""
    input_name: str


@dataclass(frozen=True)
class Series:
    """JFETs/sub-networks in series. Top of first child = top of this block,
    bottom of last child = bottom of this block. Adjacent children share
    midpoint nodes."""
    children: tuple  # of PulldownNetwork


@dataclass(frozen=True)
class Parallel:
    """JFETs/sub-networks in parallel. All children share the same top and
    bottom nodes."""
    children: tuple  # of PulldownNetwork


# Type alias
PulldownNetwork = Leaf | Series | Parallel


# ---------------------------------------------------------------------------
# Tree queries
# ---------------------------------------------------------------------------

def count_midpoints(net: PulldownNetwork) -> int:
    """Number of internal midpoint nodes that need solver variables.

    Series(N children) creates N-1 midpoints + children's internal midpoints.
    Parallel just sums children's. Leaf has none.
    """
    if isinstance(net, Leaf):
        return 0
    if isinstance(net, Parallel):
        return sum(count_midpoints(c) for c in net.children)
    if isinstance(net, Series):
        return (len(net.children) - 1) + sum(count_midpoints(c) for c in net.children)


def count_jfets(net: PulldownNetwork) -> int:
    """Total number of JFETs in the network."""
    if isinstance(net, Leaf):
        return 1
    return sum(count_jfets(c) for c in net.children)


def input_names(net: PulldownNetwork) -> list:
    """Ordered, deduplicated list of input signal names."""
    seen = set()
    result = []
    for name in _walk_inputs(net):
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _walk_inputs(net):
    if isinstance(net, Leaf):
        yield net.input_name
    else:
        for c in net.children:
            yield from _walk_inputs(c)


def n_solver_vars(net: PulldownNetwork) -> int:
    """Total Newton solver variables: 2 (v_a, v_b) + midpoints."""
    return 2 + count_midpoints(net)


# ---------------------------------------------------------------------------
# Canonical string (for hashing, registry keys, serialization)
# ---------------------------------------------------------------------------

def canonical_str(net: PulldownNetwork) -> str:
    """Deterministic string representation.

    Parallel children are sorted (electrically equivalent regardless of order).
    Series children are NOT sorted (position in chain matters).
    """
    if isinstance(net, Leaf):
        return net.input_name
    if isinstance(net, Series):
        inner = ",".join(canonical_str(c) for c in net.children)
        return f"S({inner})"
    if isinstance(net, Parallel):
        parts = sorted(canonical_str(c) for c in net.children)
        return f"P({','.join(parts)})"


# ---------------------------------------------------------------------------
# JSON serialization
# ---------------------------------------------------------------------------

def to_dict(net: PulldownNetwork) -> dict:
    if isinstance(net, Leaf):
        return {"type": "leaf", "input": net.input_name}
    if isinstance(net, Series):
        return {"type": "series", "children": [to_dict(c) for c in net.children]}
    if isinstance(net, Parallel):
        return {"type": "parallel", "children": [to_dict(c) for c in net.children]}


def from_dict(d: dict) -> PulldownNetwork:
    if d["type"] == "leaf":
        return Leaf(d["input"])
    children = tuple(from_dict(c) for c in d["children"])
    if d["type"] == "series":
        return Series(children)
    return Parallel(children)


# ---------------------------------------------------------------------------
# Truth table generation
# ---------------------------------------------------------------------------

def _conducts(net: PulldownNetwork, assignment: dict) -> bool:
    """Does the network have a conducting path from top to bottom?

    Parallel = OR (any child conducts).
    Series = AND (all children conduct).
    Leaf = True if input is high (JFET on).
    """
    if isinstance(net, Leaf):
        return assignment[net.input_name]
    if isinstance(net, Parallel):
        return any(_conducts(c, assignment) for c in net.children)
    if isinstance(net, Series):
        return all(_conducts(c, assignment) for c in net.children)


def network_truth_table(net: PulldownNetwork) -> list:
    """Generate truth table for the network.

    For N <= 8 inputs: exhaustive 2^N evaluation.
    For N > 8: key patterns (all-low, all-high, one-hot, one-cold).

    Returns list of (combo_tuple, output_is_high) where output_is_high
    is True when the pull-down does NOT conduct (output stays high).
    """
    names = input_names(net)
    n = len(names)

    if n <= 8:
        entries = []
        for combo in itertools.product([False, True], repeat=n):
            assignment = dict(zip(names, combo))
            conducts = _conducts(net, assignment)
            entries.append((combo, not conducts))
        return entries
    else:
        entries = []
        # All low
        combo = tuple(False for _ in range(n))
        assignment = dict(zip(names, combo))
        entries.append((combo, not _conducts(net, assignment)))
        # All high
        combo = tuple(True for _ in range(n))
        assignment = dict(zip(names, combo))
        entries.append((combo, not _conducts(net, assignment)))
        # One-hot
        for k in range(n):
            combo = tuple(True if i == k else False for i in range(n))
            assignment = dict(zip(names, combo))
            entries.append((combo, not _conducts(net, assignment)))
        # One-cold
        for k in range(n):
            combo = tuple(False if i == k else True for i in range(n))
            assignment = dict(zip(names, combo))
            entries.append((combo, not _conducts(net, assignment)))
        return entries


# ---------------------------------------------------------------------------
# Recursive current computation (CPU, for use in Newton solver)
# ---------------------------------------------------------------------------

def network_current(net, v_top, v_bot, v_inputs, j1, jfet_ids_fn,
                    midpoints, residuals):
    """Compute current flowing from v_top to v_bot through the network.

    Args:
        net: PulldownNetwork topology
        v_top: voltage at top terminal (drain side)
        v_bot: voltage at bottom terminal (source side)
        v_inputs: dict of input_name -> voltage
        j1: NChannelJFET device
        jfet_ids_fn: callable(vgs, vds, j) -> current
        midpoints: list of midpoint voltages, consumed left-to-right (mutated)
        residuals: list of KCL residuals at midpoints (appended to, mutated)

    Returns:
        Total current flowing from v_top through the network to v_bot.
    """
    if isinstance(net, Leaf):
        vgs = v_inputs[net.input_name] - v_bot
        vds = v_top - v_bot
        return jfet_ids_fn(vgs=vgs, vds=vds, j=j1)

    if isinstance(net, Parallel):
        total = 0.0
        for child in net.children:
            total += network_current(child, v_top, v_bot, v_inputs, j1,
                                     jfet_ids_fn, midpoints, residuals)
        return total

    if isinstance(net, Series):
        n = len(net.children)
        # Build node voltages: [v_top, mid_0, mid_1, ..., mid_{n-2}, v_bot]
        nodes = [v_top]
        for _ in range(n - 1):
            nodes.append(midpoints.pop(0))
        nodes.append(v_bot)

        # Current through each child
        currents = []
        for k, child in enumerate(net.children):
            i_k = network_current(child, nodes[k], nodes[k + 1], v_inputs,
                                  j1, jfet_ids_fn, midpoints, residuals)
            currents.append(i_k)

        # KCL at midpoints: current continuity
        for k in range(n - 1):
            residuals.append(currents[k] - currents[k + 1])

        return currents[0]


# ---------------------------------------------------------------------------
# Gate current with gate junction effects (for full KCL at Node A)
# ---------------------------------------------------------------------------

def network_current_with_gate(net, v_top, v_bot, v_inputs, j1,
                              jfet_ids_fn, gate_current_fn, vt,
                              midpoints, residuals):
    """Like network_current but also returns the gate-drain junction current
    of the topmost JFET (needed for KCL at Node A).

    Returns (i_drain, igd_top).
    """
    if isinstance(net, Leaf):
        vgs = v_inputs[net.input_name] - v_bot
        vds = v_top - v_bot
        i_d = jfet_ids_fn(vgs=vgs, vds=vds, j=j1)
        # Gate junction currents
        vgs_int = v_inputs[net.input_name] - (v_bot + i_d * j1.rs)
        vgd_int = v_inputs[net.input_name] - (v_top - i_d * j1.rd)
        _, igd = gate_current_fn(vgs_int, vgd_int, j1, vt)
        return i_d, igd

    if isinstance(net, Parallel):
        i_total = 0.0
        igd_total = 0.0
        for child in net.children:
            i_k, igd_k = network_current_with_gate(
                child, v_top, v_bot, v_inputs, j1,
                jfet_ids_fn, gate_current_fn, vt, midpoints, residuals)
            i_total += i_k
            igd_total += igd_k
        return i_total, igd_total

    if isinstance(net, Series):
        n = len(net.children)
        nodes = [v_top]
        for _ in range(n - 1):
            nodes.append(midpoints.pop(0))
        nodes.append(v_bot)

        currents = []
        for k, child in enumerate(net.children):
            i_k = network_current(child, nodes[k], nodes[k + 1], v_inputs,
                                  j1, jfet_ids_fn, midpoints, residuals)
            currents.append(i_k)

        for k in range(n - 1):
            residuals.append(currents[k] - currents[k + 1])

        # Gate-drain current of topmost child
        i_top = currents[0]
        # Get topmost leaf for gate current calculation
        top_child = net.children[0]
        top_leaf = _topmost_leaf(top_child)
        vgs_int = v_inputs[top_leaf.input_name] - (nodes[1] + i_top * j1.rs)
        vgd_int = v_inputs[top_leaf.input_name] - (v_top - i_top * j1.rd)
        _, igd = gate_current_fn(vgs_int, vgd_int, j1, vt)

        return i_top, igd


def _topmost_leaf(net: PulldownNetwork) -> Leaf:
    """Find the topmost (drain-side) leaf in a network."""
    if isinstance(net, Leaf):
        return net
    return _topmost_leaf(net.children[0])


# ---------------------------------------------------------------------------
# Backward compatibility: GateType -> PulldownNetwork
# ---------------------------------------------------------------------------

def gate_type_to_network(gate_type) -> PulldownNetwork:
    """Convert a GateType enum to a PulldownNetwork."""
    gt = gate_type.value
    if gt == "INV":
        return Leaf("A")

    if gt.startswith("NOR"):
        n = int(gt[3:])
        names = [chr(ord('A') + i) for i in range(n)]
        return Parallel(tuple(Leaf(name) for name in names))

    if gt.startswith("NAND"):
        n = int(gt[4:])
        names = [chr(ord('A') + i) for i in range(n)]
        return Series(tuple(Leaf(name) for name in names))

    raise ValueError(f"Cannot convert {gt} to network")
