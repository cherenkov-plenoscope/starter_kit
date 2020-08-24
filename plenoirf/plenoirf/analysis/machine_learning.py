import numpy as np

import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x
from sympy import lambdify


def function_from_string(function_string="1*x"):
    expression = sympy.parsing.sympy_parser.parse_expr(function_string)
    function = sympy.lambdify(sympy.abc.x, expression)
    return function


def range_of_values_in_quantile(values, quantile_range):
    """
    Returns (start, stop) of values range while excluding outliers
    outside of the quantile-range.
    """
    start = quantile_range[0]
    stop = quantile_range[1]
    assert start >= 0.
    assert start < 1.
    assert stop > 0.
    assert stop <= 1.
    sorted_values = np.sort(values)
    num = len(values)
    start_idx = int(np.ceil(num*start))
    stop_idx = int(np.floor(num*stop))
    if start_idx >= num:
        start_idx = num - 1
    if stop_idx >= num:
        stop_idx = num - 1
    return sorted_values[start_idx], sorted_values[stop_idx]


