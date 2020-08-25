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

fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]


def generate_diff_image_and_light_front(features):
    f_raw = np.hypot(
        features["image_infinity_cx_mean"] - features["light_front_cx"],
        features["image_infinity_cy_mean"] - features["light_front_cy"],
    )
    return f_raw


def generate_paxel_intensity_offset(features):
    slope = np.hypot(
        features["paxel_intensity_median_x"],
        features["paxel_intensity_median_y"]
    )
    return np.sqrt(slope)


def generate_image_infinity_std_density(features):
    std = np.hypot(
        features["image_infinity_cx_std"],
        features["image_infinity_cx_std"]
    )
    return np.log10(features["num_photons"])/std**2.0


def generate_A1(features):
    shift = np.hypot(
        features["image_half_depth_shift_cx"],
        features["image_half_depth_shift_cy"],
    )
    return (
        features["num_photons"] * shift /
        features["image_smallest_ellipse_half_depth"]
    )


def generate_A2(features):
    return (
        features["num_photons"] /
        features["image_smallest_ellipse_object_distance"]**2.0
    )


def generate_A3(features):
    return (
        features["paxel_intensity_peakness_std_over_mean"] /
        np.log10(features["image_smallest_ellipse_object_distance"])
    )




combined_features = {
    "diff_image_and_light_front": {
        "generator": generate_diff_image_and_light_front,
        "dtype": "<f8",
        "unit": "rad",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
    "paxel_intensity_offset": {
        "generator": generate_paxel_intensity_offset,
        "dtype": "<f8",
        "unit": "$m^{1/2}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
    "image_infinity_std_density": {
        "generator": generate_image_infinity_std_density,
        "dtype": "<f8",
        "unit": "$sr^{-1}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
    "A1": {
        "generator": generate_A1,
        "dtype": "<f8",
        "unit": "$sr m^{-1}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
    "A2": {
        "generator": generate_A2,
        "dtype": "<f8",
        "unit": "$m^{-2}$",
        "transformation": {
            "function": "log(x)",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
    "A3": {
        "generator": generate_A3,
        "dtype": "<f8",
        "unit": "$1$",
        "transformation": {
            "function": "x",
            "shift": "mean(x)",
            "scale": "std(x)",
            "quantile_range": [0.01, 0.99],
        },
    },
}


FEATURES = {}
for fk in ORIGINAL_FEATURES:
    FEATURES[fk] = dict(ORIGINAL_FEATURES[fk])
for fk in combined_features:
    FEATURES[fk] = dict(combined_features[fk])


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
            print(
                sk,
                pk,
                fk
            )
            sfs[sk][fk] = {}

            if fk in ORIGINAL_FEATURES:
                f_raw = features[fk]
            else:
                f_raw = combined_features[fk]["generator"](features)

            # replace

            # apply function
            func = irf.analysis.machine_learning.function_from_string(
                function_string=FEATURES[fk]["transformation"]["function"]
            )
            f_trans = func(f_raw)
            sfs[sk][fk]["function"] = FEATURES[fk]["transformation"]["function"]

            # find quantile

            (
                start,
                stop,
            ) = irf.analysis.machine_learning.range_of_values_in_quantile(
                values=f_trans,
                quantile_range=FEATURES[fk]["transformation"]["quantile_range"]
            )
            mask_quanitle = np.logical_and(
                f_trans >= start,
                f_trans <= stop
            )
            sfs[sk][fk]["quantile_range"] = [start, stop]

            # scale

            shift_func = irf.analysis.machine_learning.function_from_string(
                function_string=FEATURES[fk]["transformation"]["shift"]
            )
            scale_func = irf.analysis.machine_learning.function_from_string(
                function_string=FEATURES[fk]["transformation"]["scale"]
            )

            sfs[sk][fk]["shift"] = shift_func(f_trans[mask_quanitle])
            sfs[sk][fk]["scale"] = scale_func(f_trans[mask_quanitle])

            f_scaled = (f_trans - sfs[sk][fk]["shift"])/sfs[sk][fk]["scale"]

            print(
                sk,
                pk,
                fk,
                np.min(f_raw),
                start,
                stop,
                np.max(f_raw),
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
                f_raw = combined_features[fk]["generator"](features)

            # replace
            # apply function
            func = irf.analysis.machine_learning.function_from_string(
                function_string=FEATURES[fk]["transformation"]["function"]
            )
            f_trans = func(f_raw)
            # scale
            f_scaled = (f_trans - sfs[sk][fk]["shift"])/sfs[sk][fk]["scale"]
            transformed_features[sk][pk][fk] = f_scaled


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
                with np.errstate(divide="ignore"):
                    bin_counts_unc_fk = np.sqrt(bin_counts_fk) / bin_counts_fk
                    bin_counts_norm_fk = bin_counts_fk / np.sum(
                        bin_counts_fk
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
