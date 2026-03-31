# Static JFET Logic Gate Solver

DC operating point solver for the 2-JFET inverter + source-follower logic gate.
Matches LTSpice to within 3mV. Full SPICE NJF model with temperature scaling.

## Circuit

```
+V_POS -- R1 -- Node_A -- J1(drain)   J1(source) -- GND   J1(gate) -- V_IN
+V_POS -- J2(drain)   J2(source) -- Node_B -- R2 -- V_OUT -- R3 -- V_NEG
J2(gate) -- Node_A
```

## Structure

```
static/
  model/             JFET device model and gate solver (library)
    jfet.py            Shichman-Hodges + gate diode + parasitic R + temp scaling
    gate.py            solve_gate(), sweep()

  analysis/          Analysis tools (library)
    inverse.py         Find R1/R2/R3 for target logic levels
    fanout.py          Fan-out and cascade analysis
    power.py           Min-power optimizer with output RC constraint
    design_space.py    R2/R3 design space contour maps

  util/              Utilities (library)
    plot.py            All plotting functions
    waveform.py        PULSE waveform generator, SPICE data loader

  examples/          Runnable scripts
    forward_solve.py   Basic: solve gate, plot transfer curve, time domain
    find_resistors.py  Inverse: find resistors for target logic levels
    power_sweep.py     Optimize power across frequency sweep
    fanout_test.py     Cascade and fan-out degradation
    design_space_explore.py   R2/R3 contour maps
```

## Quick Start

```bash
cd static
pip install numpy scipy matplotlib

# Run any example:
cd examples
python forward_solve.py       # transfer curves, time domain, temp sweep
python find_resistors.py      # inverse solver
python power_sweep.py         # min-power frequency sweep (slow: ~10min)
python fanout_test.py         # cascade and fan-out analysis
python design_space_explore.py  # R2/R3 contour maps (slow: ~5min)
```

Plots are saved to `static/examples/plots/`.

## Using as a Library

```python
import sys; sys.path.insert(0, "/path/to/static")

from model import NChannelJFET, solve_gate, sweep
from analysis import find_resistors, optimize_for_frequency
from util import plot_gate_nodes

# Define your JFET from SPICE .model card
jfet = NChannelJFET(
    beta=0.000135, vto=-3.45, lmbda=0.005,
    is_=205.2e-15, n=3.0, isr=1988e-15, nr=4.0,
    alpha=20.98e-6, vk=123.7, rd=1.0, rs=1.0,
    betatce=-0.5, vtotc=-0.0025, xti=3.0, eg=3.26,
).at_temp(27.0)

# Forward solve
result = solve_gate(v_in=-4.0, v_pos=24, v_neg=-20,
                    r1=50e3, r2=1e3, r3=4.5e3, j1=jfet, j2=jfet)
print(f"V_OUT = {result['v_out']:.4f} V")

# Find resistors for target levels
from analysis import find_resistors
r = find_resistors(v_in_low=-4, v_in_high=0,
                   v_out_target_when_low=-0.8,
                   v_out_target_when_high=-3.6,
                   v_pos=24, v_neg=-20, jfet=jfet)
print(f"R1={r['r1']/1e3:.1f}k, R2={r['r2']/1e3:.1f}k, R3={r['r3']/1e3:.1f}k")
```

## Temperature

Set `eg=3.26` for SiC JFETs (default) or `eg=1.11` for silicon. Call
`jfet.at_temp(400.0)` to get temperature-scaled parameters.

## SPICE Model Reference

```
.model DR NJF(Beta=0.135m Betatce=-0.5 Vto=-3.45 Vtotc=-2.5m
  Lambda=0.005 Is=205.2f Xti=3 Isr=1988f Nr=4 Alpha=20.98u
  Vk=123.7 N=3 Rd=1 Rs=1)
```
