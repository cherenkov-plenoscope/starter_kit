#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
import numpy as np
import sklearn
import pickle
import json
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn import svm
from sklearn import ensemble
from sklearn import model_selection
from sklearn import feature_selection
from sklearn import utils

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

train_test = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0030_splitting_train_and_test_sample",)
)

transformed_features_dir = os.path.join(
    pa["summary_dir"], "0062_transform_features"
)

trigger_config = sum_config["trigger"]
quality_config = sum_config["quality"]
random_seed = sum_config["random_seed"]

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

fig_16_by_9 = sum_config["plot"]["16_by_9"]
fig_1_by_1 = fig_16_by_9.copy()
fig_1_by_1["rows"] = fig_16_by_9["rows"] * (16 / 9)


targets = {
    "energy": {
        "idx": 0,
        "start": 1e-1,
        "stop": 1e3,
        "num_bins": 20,
        "label": "energy",
        "unit": "GeV",
    },
    "airshower_maximum": {
        "idx": 1,
        "start": 7.5e3,
        "stop": 25e3,
        "num_bins": 20,
        "label": "airshower maximum",
        "unit": "m",
    },
}

level_keys = [
    "primary",
    "cherenkovpool",
    "transformed_features",
]

min_number_samples = 100

for sk in SITES:
    gamma_frame = irf.summary.read_train_test_frame(
        site_key=sk,
        particle_key="gamma",
        run_dir=pa["run_dir"],
        transformed_features_dir=transformed_features_dir,
        trigger_config=trigger_config,
        quality_config=quality_config,
        train_test=train_test,
        level_keys=level_keys,
    )
    gf = gamma_frame

    # prepare sets
    # ------------
    MA = {}
    for mk in ["test", "train"]:
        MA[mk] = {}

        MA[mk]["x"] = np.array(
            [
                gf[mk]["transformed_features.num_photons"].values,
                gf[mk][
                    "transformed_features.image_smallest_ellipse_object_distance"
                ].values,
                gf[mk][
                    "transformed_features.image_smallest_ellipse_solid_angle"
                ].values,
                gf[mk][
                    "transformed_features.image_smallest_ellipse_half_depth"
                ].values,
                gf[mk]["transformed_features.combi_A"].values,
                gf[mk]["transformed_features.combi_B"].values,
                gf[mk]["transformed_features.combi_C"].values,
                gf[mk][
                    "transformed_features.combi_image_infinity_std_density"
                ].values,
                gf[mk][
                    "transformed_features.combi_paxel_intensity_median_hypot"
                ].values,
                gf[mk][
                    "transformed_features.combi_diff_image_and_light_front"
                ].values,
            ]
        ).T
        MA[mk]["y"] = np.array(
            [
                np.log10(gf[mk]["primary.energy_GeV"].values),
                np.log10(gf[mk]["cherenkovpool.maximum_asl_m"].values),
            ]
        ).T

    num_features = MA["train"]["x"].shape[1]
    models = {}

    models["MultiLayerPerceptron"] = sklearn.neural_network.MLPRegressor(
        solver="lbfgs",
        alpha=1e-2,
        hidden_layer_sizes=(num_features, num_features, num_features),
        random_state=random_seed,
        verbose=False,
        max_iter=5000,
        learning_rate_init=0.1,
    )
    models["RandomForest"] = sklearn.ensemble.RandomForestRegressor(
        random_state=random_seed, n_estimators=10,
    )

    _X_shuffle, _y_shuffle = sklearn.utils.shuffle(
        MA["train"]["x"], MA["train"]["y"], random_state=random_seed
    )

    for mk in models:
        models[mk].fit(_X_shuffle, _y_shuffle)

        model_gh_path = os.path.join(pa["out_dir"], mk + "_gamma_hadron_model")
        with open(model_gh_path + ".pkl", "wb") as fout:
            fout.write(pickle.dumps(models[mk]))

        _y_score = models[mk].predict(MA["test"]["x"])

        for tk in targets:
            y_true = 10 ** MA["test"]["y"][:, targets[tk]["idx"]]
            y_score = 10 ** _y_score[:, targets[tk]["idx"]]

            out = {}
            out["comment"] = "Reconstructed from the test-set."
            out["learner"] = mk
            out[tk] = y_score
            out["unit"] = targets[tk]["unit"]
            out["idx"] = np.array(gf["test"]["idx"])

            site_particle_dir = os.path.join(pa["out_dir"], sk, "gamma")
            os.makedirs(site_particle_dir, exist_ok=True)
            irf.json_numpy.write(
                os.path.join(site_particle_dir, tk + ".json"), out
            )

            # plot
            # ====
            bin_edges = np.geomspace(
                targets[tk]["start"],
                targets[tk]["stop"],
                targets[tk]["num_bins"] + 1,
            )

            cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
                x=y_true,
                y=y_score,
                bin_edges=bin_edges,
                min_exposure_x=min_number_samples,
                default_low_exposure=0.0,
            )

            fig = irf.summary.figure.figure(fig_1_by_1)
            ax = irf.summary.figure.add_axes(fig, [0.1, 0.23, 0.7, 0.7])
            ax_h = irf.summary.figure.add_axes(fig, [0.1, 0.07, 0.7, 0.1])
            ax_cb = fig.add_axes([0.85, 0.23, 0.02, 0.7])
            _pcm_confusion = ax.pcolormesh(
                cm["bin_edges"],
                cm["bin_edges"],
                np.transpose(cm["confusion_bins_normalized_columns"]),
                cmap="Greys",
                norm=plt_colors.PowerNorm(gamma=0.5),
            )
            ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
            irf.summary.figure.mark_ax_thrown_spectrum(ax=ax)
            ax.set_aspect("equal")
            ax.set_title("normalized for each column")
            ax.set_ylabel(
                "reco. {:s} / {:s}".format(
                    targets[tk]["label"], targets[tk]["unit"]
                )
            )
            ax.loglog()
            ax_h.loglog()
            ax_h.set_xlim([np.min(cm["bin_edges"]), np.max(cm["bin_edges"])])
            ax_h.set_xlabel(
                "true {:s} / {:s}".format(
                    targets[tk]["label"], targets[tk]["unit"]
                )
            )
            ax_h.set_ylabel("num. events / 1")
            irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
            ax_h.axhline(min_number_samples, linestyle=":", color="k")
            irf.summary.figure.ax_add_hist(
                ax=ax_h,
                bin_edges=cm["bin_edges"],
                bincounts=cm["exposure_bins_x_no_weights"],
                linestyle="-",
                linecolor="k",
            )
            plt.savefig(os.path.join(pa["out_dir"], mk + "_" + tk + ".jpg"))
            plt.close("all")
