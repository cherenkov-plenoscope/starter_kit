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
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils

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
passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

random_seed = sum_config["random_seed"]

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
NON_GAMMA_PARTICLES = dict(PARTICLES)
NON_GAMMA_PARTICLES.pop("gamma")

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


def read_event_frame(
    site_key,
    particle_key,
    run_dir,
    transformed_features_dir,
    passing_trigger,
    passing_quality,
    train_test,
    level_keys,
):
    sk = site_key
    pk = particle_key

    airshower_table = spt.read(
        path=os.path.join(run_dir, "event_table", sk, pk, "event_table.tar",),
        structure=irf.table.STRUCTURE,
    )

    airshower_table["transformed_features"] = spt.read(
        path=os.path.join(
            transformed_features_dir, sk, pk, "transformed_features.tar",
        ),
        structure=irf.features.TRANSFORMED_FEATURE_STRUCTURE,
    )["transformed_features"]

    EXT_STRUCTRURE = dict(irf.table.STRUCTURE)
    EXT_STRUCTRURE[
        "transformed_features"
    ] = irf.features.TRANSFORMED_FEATURE_STRUCTURE["transformed_features"]

    out = {}
    for kk in ["test", "train"]:
        idxs_valid_kk = spt.intersection(
            [
                passing_trigger[sk][pk]["idx"],
                passing_quality[sk][pk]["idx"],
                train_test[sk][pk][kk],
            ]
        )
        table_kk = spt.cut_and_sort_table_on_indices(
            table=airshower_table,
            structure=EXT_STRUCTRURE,
            common_indices=idxs_valid_kk,
            level_keys=level_keys,
        )
        out[kk] = spt.make_rectangular_DataFrame(table_kk)

    return out


def make_x_y_arrays(event_frame):
    f = event_frame
    x = np.array(
        [
            f["transformed_features/num_photons"].values,
            f[
                "transformed_features/image_smallest_ellipse_object_distance"
            ].values,
            f[
                "transformed_features/image_smallest_ellipse_solid_angle"
            ].values,
            f["transformed_features/image_smallest_ellipse_half_depth"].values,
            f["transformed_features/combi_A"].values,
            f["transformed_features/combi_B"].values,
            f["transformed_features/combi_C"].values,
            f["transformed_features/combi_image_infinity_std_density"].values,
            f[
                "transformed_features/combi_paxel_intensity_median_hypot"
            ].values,
            f["transformed_features/combi_diff_image_and_light_front"].values,
        ]
    ).T
    y = np.array(
        [
            np.log10(f["primary/energy_GeV"].values),
            np.log10(f["cherenkovpool/maximum_asl_m"].values),
        ]
    ).T
    return x, y


train_test_gamma_energy = {}
for sk in SITES:
    train_test_gamma_energy[sk] = {}
    for pk in PARTICLES:
        if pk == "gamma":
            train_test_gamma_energy[sk][pk] = {}
            train_test_gamma_energy[sk][pk]["train"] = train_test[sk][pk][
                "train"
            ]
            train_test_gamma_energy[sk][pk]["test"] = train_test[sk][pk][
                "test"
            ]
        else:
            train_test_gamma_energy[sk][pk] = {}
            train_test_gamma_energy[sk][pk]["train"] = []
            train_test_gamma_energy[sk][pk]["test"] = np.concatenate(
                [train_test[sk][pk]["train"], train_test[sk][pk]["test"]]
            )


for sk in SITES:
    particle_frames = {}
    for pk in PARTICLES:
        particle_frames[pk] = read_event_frame(
            site_key=sk,
            particle_key=pk,
            run_dir=pa["run_dir"],
            transformed_features_dir=transformed_features_dir,
            passing_trigger=passing_trigger,
            passing_quality=passing_quality,
            train_test=train_test_gamma_energy,
            level_keys=level_keys,
        )

    # prepare sets
    # ------------
    MA = {}
    for pk in PARTICLES:
        MA[pk] = {}
        for mk in ["test", "train"]:
            MA[pk][mk] = {}
            MA[pk][mk]["x"], MA[pk][mk]["y"] = make_x_y_arrays(
                event_frame=particle_frames[pk][mk]
            )

    # train model on gamma only
    # -------------------------
    num_features = MA["gamma"]["train"]["x"].shape[1]
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
        MA["gamma"]["train"]["x"],
        MA["gamma"]["train"]["y"],
        random_state=random_seed,
    )

    for mk in models:
        models[mk].fit(_X_shuffle, _y_shuffle)

        model_path = os.path.join(pa["out_dir"], mk + ".pkl")
        with open(model_path, "wb") as fout:
            fout.write(pickle.dumps(models[mk]))

        for pk in PARTICLES:

            _y_score = models[mk].predict(MA[pk]["test"]["x"])

            for tk in targets:
                y_true = 10 ** MA[pk]["test"]["y"][:, targets[tk]["idx"]]
                y_score = 10 ** _y_score[:, targets[tk]["idx"]]

                out = {}
                out["comment"] = "Reconstructed from the test-set."
                out["learner"] = mk
                out[tk] = y_score
                out["unit"] = targets[tk]["unit"]
                out["idx"] = np.array(particle_frames[pk]["test"]["idx"])

                site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
                os.makedirs(site_particle_dir, exist_ok=True)
                irf.json_numpy.write(
                    os.path.join(site_particle_dir, tk + ".json"), out
                )