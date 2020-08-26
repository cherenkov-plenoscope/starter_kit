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

PARTICLE_IDS = {}
for pk in PARTICLES:
    PARTICLE_IDS[PARTICLES[pk]["particle_id"]] = pk

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

        for pk in PARTICLES:
            gammaness[mk][pk] = {}
            particle_mask = (
                ET["test"]["primary.particle_id"].values
                == PARTICLES[pk]["particle_id"]
            )
            gammaness[mk][pk][spt.IDX] = (ET["test"][spt.IDX].values)[
                particle_mask
            ]
            gammaness[mk][pk]["gammaness"] = y_gammaness_score[particle_mask]

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
