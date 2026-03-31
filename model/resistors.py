"""Standard E-series resistor values.

E12 (10% tolerance): 12 values per decade
E24 (5% tolerance):  24 values per decade
E96 (1% tolerance):  96 values per decade
"""

# Base values per decade (multiply by powers of 10 for actual values)
E12_BASE = [1.0, 1.2, 1.5, 1.8, 2.2, 2.7, 3.3, 3.9, 4.7, 5.6, 6.8, 8.2]

E24_BASE = [
    1.0, 1.1, 1.2, 1.3, 1.5, 1.6, 1.8, 2.0, 2.2, 2.4, 2.7, 3.0,
    3.3, 3.6, 3.9, 4.3, 4.7, 5.1, 5.6, 6.2, 6.8, 7.5, 8.2, 9.1,
]

E96_BASE = [
    1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24,
    1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47, 1.50, 1.54, 1.58,
    1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00,
    2.05, 2.10, 2.15, 2.21, 2.26, 2.32, 2.37, 2.43, 2.49, 2.55,
    2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24,
    3.32, 3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12,
    4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99, 5.11, 5.23,
    5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65,
    6.81, 6.98, 7.15, 7.32, 7.50, 7.68, 7.87, 8.06, 8.25, 8.45,
    8.66, 8.87, 9.09, 9.31, 9.53, 9.76,
]


def e_series_values(series, r_min, r_max):
    """Generate all E-series resistor values within a range.

    Args:
        series: "E12", "E24", or "E96"
        r_min: minimum resistance (ohms)
        r_max: maximum resistance (ohms)

    Returns sorted list of resistor values in ohms.
    """
    base = {"E12": E12_BASE, "E24": E24_BASE, "E96": E96_BASE}[series]
    values = []
    for decade_exp in range(-1, 7):  # 0.1 ohm to 1M ohm
        multiplier = 10.0 ** decade_exp
        for b in base:
            r = b * multiplier
            if r_min <= r <= r_max:
                values.append(r)
    return sorted(set(values))


def nearest_e_series(value, series="E24"):
    """Find the nearest E-series value to a given resistance."""
    base = {"E12": E12_BASE, "E24": E24_BASE, "E96": E96_BASE}[series]
    best = None
    best_err = float("inf")
    for decade_exp in range(-1, 7):
        multiplier = 10.0 ** decade_exp
        for b in base:
            r = b * multiplier
            err = abs(r - value) / value
            if err < best_err:
                best = r
                best_err = err
    return best


def e_series_neighbourhood(center, series, n_steps=2):
    """Get E-series values within n_steps of a center value.

    Returns values from (center - n_steps) to (center + n_steps)
    in the E-series sequence.
    """
    base = {"E12": E12_BASE, "E24": E24_BASE, "E96": E96_BASE}[series]

    # Build full sorted list around the center
    all_vals = []
    for decade_exp in range(-1, 7):
        multiplier = 10.0 ** decade_exp
        for b in base:
            all_vals.append(b * multiplier)
    all_vals = sorted(set(all_vals))

    # Find the closest index
    idx = min(range(len(all_vals)), key=lambda i: abs(all_vals[i] - center))

    lo = max(0, idx - n_steps)
    hi = min(len(all_vals) - 1, idx + n_steps)
    return all_vals[lo:hi + 1]
