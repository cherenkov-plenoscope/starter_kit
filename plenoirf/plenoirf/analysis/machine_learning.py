import numpy as np


def range_of_values_within_containment(values, containment_factor):
    """
    Returns (start, stop) of values range while excluding outliers
    outside of the containment-factor.
    """
    assert containment_factor > 0.
    assert containment_factor <= 1.0
    sorted_values = np.sort(values)
    num = len(values)
    ff = 0.5*(1.0-containment_factor)
    start_idx = int(np.ceil(num*ff))
    stop_idx = int(np.floor(num*(containment_factor + ff)))
    if start_idx >= num:
        start_idx = num - 1
    if stop_idx >= num:
        stop_idx = num - 1
    return sorted_values[start_idx], sorted_values[stop_idx]
