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

PARTICLE_IDS = {}
for pk in PARTICLES:
    PARTICLE_IDS[PARTICLES[pk]["particle_id"]] = pk

fig_16_by_9 = sum_config["plot"]["16_by_9"]
fig_1_by_1 = fig_16_by_9.copy()
fig_1_by_1["rows"] = fig_16_by_9["rows"] * (16 / 9)

particle_colors = sum_config["plot"]["particle_colors"]

level_keys = [
    "primary",
    "cherenkovsize",
    "cherenkovpool",
    "core",
    "transformed_features",
]

gamma_hadron_labels = {"gamma": 1, "proton": 0, "helium": 0, "electron": 0}
other_labels = {"electron": 0}

gamma_hadron_feature_keys = [
    "transformed_features.num_photons",
    "transformed_features.image_smallest_ellipse_object_distance",
    "transformed_features.image_smallest_ellipse_solid_angle",
    "transformed_features.image_smallest_ellipse_half_depth",
    "transformed_features.combi_A",
    "transformed_features.combi_B",
    "transformed_features.combi_C",
    "transformed_features.combi_image_infinity_std_density",
    "transformed_features.combi_paxel_intensity_median_hypot",
    "transformed_features.combi_diff_image_and_light_front",
]


def make_train_test_Xy_set_for_particle_classification(
    particle_frames, particle_labels
):
    Xy = {}
    for kk in ["train", "test"]:
        Xy[kk] = {}
        Xy[kk]["idx"] = []
        Xy[kk]["x"] = []
        Xy[kk]["y"] = []
        for pk in particle_labels:
            num_airshower = particle_frames[pk][kk].shape[0]
            Xy[kk]["x"].append(particle_frames[pk][kk])
            Xy[kk]["y"].append(particle_labels[pk] * np.ones(num_airshower))
            Xy[kk]["idx"].append(
                np.array(
                    [
                        particle_frames[pk][kk]["primary.particle_id"].values,
                        particle_frames[pk][kk][spt.IDX].values,
                    ],
                    dtype=np.uint64,
                )
            )
        Xy[kk]["x"] = pandas.concat(Xy[kk]["x"])
        Xy[kk]["y"] = np.concatenate(Xy[kk]["y"])
        Xy[kk]["idx"] = np.hstack(Xy[kk]["idx"])
    return Xy


def make_Xy_train_test_set_with_custom_features(Xy, x_feature_keys):
    MA = {}
    for kk in ["test", "train"]:
        MA[kk] = {}

        MA[kk]["idx"] = Xy[kk]["idx"]

        MA[kk]["x"] = []
        for fk in x_feature_keys:
            MA[kk]["x"].append(Xy[kk]["x"][fk].values)
        MA[kk]["x"] = np.array(MA[kk]["x"]).T

        MA[kk]["y"] = np.array(
            [
                Xy[kk]["y"],
                np.log10(Xy[kk]["x"]["primary.energy_GeV"].values),
                np.log10(Xy[kk]["x"]["cherenkovpool.maximum_asl_m"].values),
            ]
        ).T
    return MA


for sk in SITES:
    particle_frames = {}
    for pk in PARTICLES:
        particle_frames[pk] = irf.summary.read_train_test_frame(
            site_key=sk,
            particle_key=pk,
            run_dir=pa["run_dir"],
            transformed_features_dir=transformed_features_dir,
            trigger_config=trigger_config,
            quality_config=quality_config,
            train_test=train_test,
            level_keys=level_keys,
        )

    # prepare sets
    # ------------
    XyGamHad = make_train_test_Xy_set_for_particle_classification(
        particle_frames=particle_frames, particle_labels=gamma_hadron_labels
    )
    XyOthers = make_train_test_Xy_set_for_particle_classification(
        particle_frames=particle_frames, particle_labels=other_labels
    )

    MA = make_Xy_train_test_set_with_custom_features(
        Xy=XyGamHad, x_feature_keys=gamma_hadron_feature_keys
    )
    OT = make_Xy_train_test_set_with_custom_features(
        Xy=XyOthers, x_feature_keys=gamma_hadron_feature_keys
    )

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

    gammaness = {}
    area_under_curve = {}
    for mk in models:
        gammaness[mk] = {}
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

        area_under_curve[mk] = sklearn.metrics.roc_auc_score(
            y_true=y_gammaness_true, y_score=y_gammaness_score
        )

        roc_gh = {
            "false_positive_rate": fpr_gh.tolist(),
            "true_positive_rate": tpr_gh.tolist(),
            "gamma_hadron_threshold": thresholds_gh.tolist(),
            "area_under_curve": float(area_under_curve[mk]),
            "num_events_for_training": int(MA["train"]["x"].shape[0]),
        }

        for pk in gamma_hadron_labels:
            gammaness[mk][pk] = {}
            particle_mask = (
                MA["test"]["idx"][0, :] == PARTICLES[pk]["particle_id"]
            )
            gammaness[mk][pk][spt.IDX] = MA["test"]["idx"][1, :][particle_mask]
            gammaness[mk][pk]["gammaness"] = y_gammaness_score[particle_mask]

        other_y_score = models[mk].predict(OT["test"]["x"])[:, 0]
        for pk in other_labels:
            gammaness[mk][pk] = {}
            particle_mask = (
                OT["test"]["idx"][0, :] == PARTICLES[pk]["particle_id"]
            )
            gammaness[mk][pk][spt.IDX] = OT["test"]["idx"][1, :][particle_mask]
            gammaness[mk][pk]["gammaness"] = other_y_score[particle_mask]

        roc_gh_path = os.path.join(pa["out_dir"], sk + "_" + mk + "_roc")
        with open(roc_gh_path + ".json", "wt") as fout:
            fout.write(json.dumps(roc_gh, indent=4))

        fig = irf.summary.figure.figure(fig_1_by_1)
        ax = irf.summary.figure.add_axes(fig, [0.15, 0.15, 0.8, 0.8])
        ax.plot(fpr_gh, tpr_gh, "k")
        ax.set_title("area under curve {:.2f}".format(area_under_curve[mk]))
        ax.set_xlabel("false positive rate / 1\nproton acceptance")
        ax.set_ylabel("true positive rate / 1\ngamma-ray acceptance")
        plt.savefig(roc_gh_path + ".png")
        plt.close("all")

    _methods = [mk for mk in area_under_curve]
    _areas = [area_under_curve[mk] for mk in area_under_curve]
    best_method = _methods[np.argmax(_areas)]

    for pk in PARTICLES:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "test.json"),
            gammaness[best_method][pk],
        )
