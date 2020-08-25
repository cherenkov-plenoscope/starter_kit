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
ORIGINAL_FEATURES = irf.table.STRUCTURE["features"]
COMBINED_FEATURES = irf.features.combined_features.COMBINED_FEATURES

fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]


FEATURES = {}
for fk in ORIGINAL_FEATURES:
    FEATURES[fk] = dict(ORIGINAL_FEATURES[fk])
for fk in COMBINED_FEATURES:
    FEATURES[fk] = dict(COMBINED_FEATURES[fk])

sfs = {}
for sk in SITES:
    sfs[sk] = {}
    for pk in ["gamma"]:

        features = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )["features"]

        for fk in FEATURES:
            print(sk, pk, fk)
            sfs[sk][fk] = {}

            if fk in ORIGINAL_FEATURES:
                f_raw = features[fk]
            else:
                f_raw = COMBINED_FEATURES[fk]["generator"](features)

            sfs[sk][fk] = irf.features.find_transformation(
                feature_raw=f_raw,
                transformation_instruction=FEATURES[fk]["transformation"],
            )


transformed_features = {}
for sk in SITES:
    transformed_features[sk] = {}
    for pk in PARTICLES:
        transformed_features[sk][pk] = {}

        features = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )["features"]

        for fk in FEATURES:

            if fk in ORIGINAL_FEATURES:
                f_raw = features[fk]
            else:
                f_raw = COMBINED_FEATURES[fk]["generator"](features)

            transformed_features[sk][pk][fk] = irf.features.transform(
                feature_raw=f_raw, transformation=sfs[sk][fk]
            )


for sk in SITES:
    for fk in transformed_features[sk]["gamma"]:

        fig_path = os.path.join(pa["out_dir"], "{:s}_{:s}.jpg".format(sk, fk))

        if not os.path.exists(fig_path):

            fig = irf.summary.figure.figure(fig_16_by_9)
            ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

            for pk in PARTICLES:
                start = -5
                stop = 5

                bin_edges_fk = np.linspace(start, stop, 101)
                bin_counts_fk = np.histogram(
                    transformed_features[sk][pk][fk], bins=bin_edges_fk
                )[0]

                bin_counts_unc_fk = irf.analysis.effective_quantity._divide_silent(
                    numerator=np.sqrt(bin_counts_fk),
                    denominator=bin_counts_fk,
                    default=np.nan,
                )
                bin_counts_norm_fk = irf.analysis.effective_quantity._divide_silent(
                    numerator=bin_counts_fk,
                    denominator=np.sum(bin_counts_fk),
                    default=0,
                )

                irf.summary.figure.ax_add_hist(
                    ax=ax,
                    bin_edges=bin_edges_fk,
                    bincounts=bin_counts_norm_fk,
                    linestyle="-",
                    linecolor=particle_colors[pk],
                    linealpha=1.0,
                    bincounts_upper=bin_counts_norm_fk
                    * (1 + bin_counts_unc_fk),
                    bincounts_lower=bin_counts_norm_fk
                    * (1 - bin_counts_unc_fk),
                    face_color=particle_colors[pk],
                    face_alpha=0.3,
                )

            ax.semilogy()
            irf.summary.figure.mark_ax_thrown_spectrum(ax=ax)
            ax.spines["right"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_xlabel("transformed {:s} / 1".format(fk))
            ax.set_ylabel("relative intensity / 1")
            ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            ax.set_xlim([start, stop])
            ax.set_ylim([1e-5, 1.0])
            fig.savefig(fig_path)
            plt.close(fig)
