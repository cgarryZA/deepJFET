"""N-to-2^N decoder built from INV and NAND gates.

Tiling pattern:
    1-to-2:  1 input  -> 2 outputs  (1 INV)
    2-to-4:  2 inputs -> 4 outputs  (2 INV + 4 NAND2)
    3-to-8:  3 inputs -> 8 outputs  (3 INV + 8 NAND3)  [uses NAND3 if available]

Implementation (2-to-4 example):
    i0 = INV(in0)    -> in0_bar
    i1 = INV(in1)    -> in1_bar
    n0 = NAND2(in0_bar, in1_bar) -> out0_bar   (select when in=00)
    n1 = NAND2(in0,     in1_bar) -> out1_bar   (select when in=01)
    n2 = NAND2(in0_bar, in1    ) -> out2_bar   (select when in=10)
    n3 = NAND2(in0,     in1    ) -> out3_bar   (select when in=11)

    Outputs are active-LOW (NAND output). Add INV per output if active-HIGH needed.

For n_bits > 2, we use NAND2 trees instead of NAND3/NAND4 to stay universal:
    3-to-8: pairs of 2-input partial decodes ANDed together via NAND-INV.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from simulator.module import Module, Port, PortDir
from simulator.netlist import Gate
from model import GateType

IN, OUT = PortDir.IN, PortDir.OUT


def decoder(n_bits: int, active_high: bool = False) -> Module:
    """N-to-2^N line decoder.

    Ports:
        in0..in{N-1}          (IN)  — address inputs
        out0..out{2^N - 1}    (OUT) — decoded outputs (active-LOW unless active_high=True)

    For n_bits=1: trivial inverter decoder (out0=in_bar, out1=in).
    For n_bits=2: 2 INV + 4 NAND2 (+ 4 INV if active_high).
    """
    if n_bits < 1:
        raise ValueError("n_bits must be >= 1")

    n_outputs = 2 ** n_bits

    ports = []
    for i in range(n_bits):
        ports.append(Port(f"in{i}", IN))
    for i in range(n_outputs):
        ports.append(Port(f"out{i}", OUT))

    gates = []

    # Generate inverted versions of each input
    for i in range(n_bits):
        gates.append(Gate(f"i{i}", GateType.INV, [f"in{i}"], f"in{i}_bar"))

    if n_bits == 1:
        # Special case: out0 = in0_bar, out1 = in0
        # out0 is already in0_bar from the inverter — need to route it
        # Use an inverter of the inverter for out1
        # Actually: out0 = NOT(in0), out1 = in0
        # But we need gates to drive the output nets.
        # Use: out0 driven by i0 (already "in0_bar"), rename output
        gates.clear()
        gates.append(Gate("i0", GateType.INV, ["in0"], "out0"))       # out0 = NOT(in0)
        gates.append(Gate("i0_buf", GateType.INV, ["out0"], "out1"))  # out1 = in0 (double inv)
        if not active_high:
            # Active-low: invert meaning. out0 active when in0=0 (already correct).
            pass

    else:
        # General case: for each output index, determine which inputs are 0/1
        for out_idx in range(n_outputs):
            # Build the minterm: for each bit, pick in{i} or in{i}_bar
            input_nets = []
            for bit in range(n_bits):
                if (out_idx >> bit) & 1:
                    input_nets.append(f"in{bit}")
                else:
                    input_nets.append(f"in{bit}_bar")

            # Chain NAND2 gates for minterms > 2 inputs
            out_net = f"out{out_idx}_nand" if active_high else f"out{out_idx}"

            if len(input_nets) == 2:
                gates.append(Gate(
                    f"d{out_idx}", GateType.NAND2,
                    input_nets, out_net
                ))
            else:
                # Tree of NAND2+INV pairs to AND all inputs
                # First pair
                gates.append(Gate(
                    f"d{out_idx}_0", GateType.NAND2,
                    [input_nets[0], input_nets[1]],
                    f"d{out_idx}_p0"
                ))
                gates.append(Gate(
                    f"d{out_idx}_0i", GateType.INV,
                    [f"d{out_idx}_p0"],
                    f"d{out_idx}_a0"
                ))
                # Chain remaining inputs
                prev = f"d{out_idx}_a0"
                for j in range(2, len(input_nets)):
                    is_last = (j == len(input_nets) - 1)
                    nand_out = out_net if (is_last and not active_high) else f"d{out_idx}_p{j-1}"
                    gates.append(Gate(
                        f"d{out_idx}_{j-1}", GateType.NAND2,
                        [prev, input_nets[j]],
                        nand_out
                    ))
                    if not is_last or active_high:
                        inv_out = f"d{out_idx}_a{j-1}" if not is_last else out_net
                        # For active_high last stage, we want the AND result
                        # NAND+INV = AND, so the INV output is the AND
                        if is_last and active_high:
                            inv_out = f"out{out_idx}"
                        gates.append(Gate(
                            f"d{out_idx}_{j-1}i", GateType.INV,
                            [nand_out],
                            inv_out
                        ))
                        prev = inv_out
                    else:
                        prev = nand_out

            # If active_high, invert the NAND output
            if active_high and len(input_nets) == 2:
                gates.append(Gate(
                    f"d{out_idx}_inv", GateType.INV,
                    [out_net], f"out{out_idx}"
                ))

    return Module(
        name=f"decoder_{n_bits}to{n_outputs}",
        ports=ports,
        gates=gates,
    )
