#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
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

particle_colors = sum_config["plot"]["particle_colors"]

for sk in SITES:
    gamma_frame = irf.summary.read_train_test_frames(
        site_key=sk,
        particle_keys=["gamma"],
        run_dir=pa["run_dir"],
        transformed_features_dir=transformed_features_dir,
        trigger_config=trigger_config,
        quality_config=quality_config,
        train_test=train_test,
    )

    hadron_frame = irf.summary.read_train_test_frames(
        site_key=sk,
        particle_keys=["proton", "helium"],
        run_dir=pa["run_dir"],
        transformed_features_dir=transformed_features_dir,
        trigger_config=trigger_config,
        quality_config=quality_config,
        train_test=train_test,
    )

    ET = {}
    GM = {}
    for kk in ["train", "test"]:
        num_gammas = gamma_frame[kk].shape[0]
        num_hadrons = hadron_frame[kk].shape[0]
        ET[kk] = pandas.concat([gamma_frame[kk], hadron_frame[kk]])
        GM[kk] = np.concatenate([np.ones(num_gammas), np.zeros(num_hadrons)])

    # prepare sets
    # ------------
    MA = {}
    for mk in ["test", "train"]:
        MA[mk] = {}

        MA[mk]["x"] = np.array(
            [
                ET[mk]["transformed_features.num_photons"].values,
                ET[mk][
                    "transformed_features.image_smallest_ellipse_object_distance"
                ].values,
                ET[mk][
                    "transformed_features.image_smallest_ellipse_solid_angle"
                ].values,
                ET[mk][
                    "transformed_features.image_smallest_ellipse_half_depth"
                ].values,
                ET[mk]["transformed_features.combi_A"].values,
                ET[mk]["transformed_features.combi_B"].values,
                ET[mk]["transformed_features.combi_C"].values,
                ET[mk][
                    "transformed_features.combi_image_infinity_std_density"
                ].values,
                ET[mk][
                    "transformed_features.combi_paxel_intensity_median_hypot"
                ].values,
                ET[mk][
                    "transformed_features.combi_diff_image_and_light_front"
                ].values,
            ]
        ).T
        MA[mk]["y"] = np.array(
            [
                GM[mk],
                np.log10(ET[mk]["primary.energy_GeV"].values),
                np.log10(ET[mk]["cherenkovpool.maximum_asl_m"].values),
            ]
        ).T

    num_features = MA["train"]["x"].shape[1]
    models = {}

    models["MultiLayerPerceptron"] = sklearn.neural_network.MLPRegressor(
        solver="lbfgs",
        alpha=1e-2,
        hidden_layer_sizes=(num_features, num_features, num_features),
        random_state=random_seed,
        verbose=True,
        max_iter=5000,
        learning_rate_init=0.1,
    )
    models["RandomForest"] = sklearn.ensemble.RandomForestRegressor(
        random_state=random_seed
    )

    _X_shuffle, _y_shuffle = sklearn.utils.shuffle(
        MA["train"]["x"], MA["train"]["y"], random_state=random_seed
    )

    for mk in models:
        models[mk].fit(_X_shuffle, _y_shuffle)

        model_gh_path = os.path.join(pa["out_dir"], mk + "_gamma_hadron_model")
        with open(model_gh_path + ".pkl", "wb") as fout:
            fout.write(pickle.dumps(models[mk]))

        y_score = models[mk].predict(MA["test"]["x"])
        y_gammaness_true = MA["test"]["y"][:, 0]
        y_gammaness_score = y_score[:, 0]

        fpr_gh, tpr_gh, thresholds_gh = sklearn.metrics.roc_curve(
            y_true=y_gammaness_true, y_score=y_gammaness_score
        )

        auc_gh = sklearn.metrics.roc_auc_score(
            y_true=y_gammaness_true, y_score=y_gammaness_score
        )

        roc_gh = {
            "false_positive_rate": fpr_gh.tolist(),
            "true_positive_rate": tpr_gh.tolist(),
            "gamma_hadron_threshold": thresholds_gh.tolist(),
            "area_under_curve": float(auc_gh),
            "num_events_for_training": int(MA["train"]["x"].shape[0]),
        }
        roc_gh_path = os.path.join(pa["out_dir"], sk + "_" + mk + "_roc")
        with open(roc_gh_path + ".json", "wt") as fout:
            fout.write(json.dumps(roc_gh, indent=4))

        fig = irf.summary.figure.figure(fig_1_by_1)
        ax = irf.summary.figure.add_axes(fig, [0.15, 0.15, 0.8, 0.8])
        ax.plot(fpr_gh, tpr_gh, "k")
        ax.set_title("area under curve {:.2f}".format(auc_gh))
        ax.set_xlabel("false positive rate / 1\nproton acceptance")
        ax.set_ylabel("true positive rate / 1\ngamma-ray acceptance")
        plt.savefig(roc_gh_path + ".png")
        plt.close("all")

    """
    y_energy_true = 10 ** MA["test"]["y"][:, 1]
    y_energy_score = 10 ** y_score[:, 1]

    y_maximum_true = 10 ** MA["test"]["y"][:, 2]
    y_maximum_score = 10 ** y_score[:, 2]

    num_bins = 24
    min_number_samples = 100
    energy_bin_edges = np.geomspace(1e-1, 1e3, num_bins + 1)

    cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
        x=y_energy_true,
        y=y_energy_score,
        bin_edges=energy_bin_edges,
        min_exposure_x=min_number_samples,
        default_low_exposure=np.nan,
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
    ax.set_ylabel("reco. energy / GeV")
    ax.loglog()
    ax_h.loglog()
    ax_h.set_xlim([np.min(cm["bin_edges"]), np.max(cm["bin_edges"])])
    ax_h.set_xlabel("true energy / GeV")
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
    plt.savefig(os.path.join(pa["out_dir"], "energy.jpg"))
    plt.close("all")
    """
