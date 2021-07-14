#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
SOURCE_GEOMETRY = {
    "point": {
        "comment": (
            "Effective area "
            "for a point source, reconstructed in onregion. "
            "VS reconstructed energy-bins VS onregion-radii"
        ),
    },
    "diffuse": {
        "comment": (
            "Effective acceptance (area x solid angle) "
            "for a diffuse source, reconstructed in onregion. "
            "VS reconstructed energy-bins VS onregion-radii"
        ),
    },
}

# prepare energy confusion
# ------------------------
energy_confusion = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

# prepare onregion acceptance vs true energy
# ------------------------------------------
acceptance_true_energy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

num_bins_onregion_radius = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
ONREGION_SIZES = range(num_bins_onregion_radius)

for sk in SITES:
    for pk in PARTICLES:
        CM = energy_confusion[sk][pk]["confusion_matrix"]
        assert CM["ax0_key"] == "true_energy"
        assert CM["ax1_key"] == "reco_energy"
        for gk in SOURCE_GEOMETRY:
            Q_true = np.array(acceptance_true_energy[sk][pk][gk]["mean"])
            Q_true_u = np.array(
                acceptance_true_energy[sk][pk][gk]["relative_uncertainty"]
            )
            Q_reco = np.nan * np.ones(shape=Q_true.shape)
            Q_reco_u = np.nan * np.ones(shape=Q_true_u.shape)

            for oi in ONREGION_SIZES:
                Q_reco[:, oi] = irf.utils.apply_confusion_matrix(
                    x=Q_true[:, oi],
                    confusion_matrix=CM["confusion_bins_normalized_on_ax0"],
                )
                Q_reco_u[:, oi] = irf.utils.apply_confusion_matrix_uncertainty(
                    x_unc=Q_true_u[:, oi],
                    confusion_matrix=CM["confusion_bins_normalized_on_ax0"],
                )

            out_path = os.path.join(pa["out_dir"], sk, pk, gk + ".json")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            json_numpy.write(
                out_path,
                {
                    "comment": SOURCE_GEOMETRY[gk]["comment"],
                    "unit": acceptance_true_energy[sk][pk][gk]["unit"],
                    "mean": Q_reco,
                    "relative_uncertainty": Q_reco_u,
                },
            )
