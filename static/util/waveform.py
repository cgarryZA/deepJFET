"""Waveform generation and SPICE data loading."""

import numpy as np


def pulse_waveform(t, v1, v2, td, tr, tf, pw, per):
    """Generate SPICE-style PULSE waveform."""
    v = np.full_like(t, v1)
    for i, ti in enumerate(t):
        tc = (ti - td) % per if ti >= td else -1.0
        if tc < 0:
            v[i] = v1
        elif tc < tr:
            v[i] = v1 + (v2 - v1) * tc / tr
        elif tc < tr + pw:
            v[i] = v2
        elif tc < tr + pw + tf:
            v[i] = v2 + (v1 - v2) * (tc - tr - pw) / tf
        else:
            v[i] = v1
    return v


def load_spice_data(filepath: str) -> dict:
    """Load LTSpice .txt export (tab-separated: time, V(1), V(n002))."""
    raw = np.loadtxt(filepath, skiprows=1)
    return {"time": raw[:, 0], "v_out": raw[:, 1], "v_in": raw[:, 2]}
