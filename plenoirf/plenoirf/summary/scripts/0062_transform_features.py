#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy.abc import x
from sympy import lambdify

import sklearn
from sklearn import neural_network
from sklearn import gaussian_process

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]
FEATURES = irf.table.STRUCTURE["features"]

sfs = {}
transformed_features = {}
for sk in SITES:
    sfs[sk] = {}
    for pk in PARTICLES:
        sfs[sk][pk] = {}

        features = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )["features"]

        for fk in FEATURES:
            sfs[sk][pk][fk] = {}

            f_raw = features[fk]

            # replace

            # apply function
            func = irf.analysis.machine_learning.function_from_string(
                function_string=FEATURES[fk]["transformation"]["function"]
            )
            f_trans = func(f_raw)
            sfs[sk][pk][fk]["function"] = FEATURES[fk]["transformation"]["function"]

            # find quantile

            (
                start,
                stop,
            ) = irf.analysis.machine_learning.range_of_values_in_quantile(
                values=f_trans, quantile_range=[0.01, 0.99]
            )
            mask_quanitle = np.logical_and(
                f_trans >= start,
                f_trans <= stop
            )
            sfs[sk][pk][fk]["start"] = start
            sfs[sk][pk][fk]["stop"] = stop

            # find scaling in quantile

            mean = np.mean(f_trans[mask_quanitle])
            std = np.std(f_trans[mask_quanitle])

            sfs[sk][pk][fk]["mean"] = mean
            sfs[sk][pk][fk]["std"] = std

            f_scaled = (f_trans - mean)/std

            print(
                sk,
                pk,
                fk,
                np.min(features[fk]),
                start,
                stop,
                np.max(features[fk]),
            )
