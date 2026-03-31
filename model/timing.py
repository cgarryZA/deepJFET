"""Shared timing utilities for RC delay estimation and speed constraints."""


def max_r_out_for_freq(f_target, cgd, cgs, n_fanout=1):
    """Maximum R_out (R2||R3) for a given frequency and fan-out.

    Requires 5*tau settling within half a clock period:
      5 * R_out * N_fanout * C_gate <= T/2
    """
    return 1.0 / (10.0 * f_target * n_fanout * (cgd + cgs))


def estimate_prop_delay(r1, r2, r3, cgd, cgs, n_fanout=1):
    """Propagation delay from both Node A and output RC paths.

    Returns delay in seconds.
    """
    c = cgd + cgs
    tau_a = r1 * c                      # Node A rise through R1
    r_out = (r2 * r3) / (r2 + r3)
    tau_out = r_out * n_fanout * c       # Output driving fan-out
    return 0.7 * max(tau_a, tau_a * 0.3, tau_out)
