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
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]

Q = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

M = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)
dMdE = irf.analysis.differential_sensitivity.derive_all_energy_migration(
    energy_migration=M,
    energy_bin_width=energy_bin["width"],
)

dRtdEt = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0530_diffsens_background_diff_rates")
)

Rt = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0535_diffsens_background_diff_rates_sum")
)

detection_threshold_std = sum_config["on_off_measuremnent"][
    "detection_threshold_std"
]
on_over_off_ratio = sum_config["on_off_measuremnent"]["on_over_off_ratio"]
systematic_uncertainty = sum_config["on_off_measuremnent"][
    "systematic_uncertainty"
]

observation_times = irf.utils.make_civil_times_points_in_quasi_logspace()
observation_times = np.array(observation_times)
num_observation_times = len(observation_times)

critical_method = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

for sk in SITES:
    for ok in ONREGION_TYPES:
        os.makedirs(os.path.join(pa["out_dir"], sk, ok), exist_ok=True)


for sk in SITES:
    for ok in ONREGION_TYPES:
        # estimate gamma eff. area
        # ------------------------
        A = Q[sk][ok]["gamma"]["point"]["mean"]
        A_au = Q[sk][ok]["gamma"]["point"]["absolute_uncertainty"]

        for dk in irf.analysis.differential_sensitivity.SCENARIOS:

            critical_dKdE = np.nan * np.ones(
                shape=(energy_bin["num_bins"], num_observation_times)
            )

            for obstix in range(num_observation_times):
                print(sk, ok, dk, obstix)
                critical_rate_per_s = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rt[sk][ok]["mean"],
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                dFdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_area_vs_energy_m2=A,
                    signal_rate_vs_energy_per_s=critical_rate_per_s,
                )

                critical_dKdE[:, obstix] = dFdE

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk + ".json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "observation_times": observation_times,
                    "differential_flux": critical_dKdE,
                    "comment": (
                        "Critical differential flux-sensitivity "
                        "VS energy VS onregion-size VS observation-time"
                    ),
                },
            )
