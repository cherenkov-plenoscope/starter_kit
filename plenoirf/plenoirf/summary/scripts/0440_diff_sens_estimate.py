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

acceptance_after_all_cuts = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

energy_interpretation = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0420_diff_sens_energy_interpretation"),
)
rates_after_all_cuts_in_interpreted_energy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0430_diff_sens_rates"),
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]

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

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")

critical_method = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for dk in irf.analysis.differential_sensitivity.SCENARIOS:
        critical_dFdE = np.nan * np.ones(
            shape=(
                energy_bin["num_bins"],
                num_onregion_sizes,
                num_observation_times
            )
        )

        # estimate cosmic-ray rates
        # -------------------------
        for oridx in range(num_onregion_sizes):

            cosmic_ray_rate_per_s = np.zeros(energy_bin["num_bins"])
            for ck in COSMIC_RAYS:
                cosmic_ray_rate_per_s += (
                    rates_after_all_cuts_in_interpreted_energy[sk][ck][dk]["rate"][:, oridx]
                )

            # estimate gamma eff. area
            # ------------------------
            signal_area_vs_true_energy_m2 = acceptance_after_all_cuts[sk]["gamma"]["point"][
                "mean"
            ][:, oridx]
            _gamma_mm = energy_interpretation[sk]["gamma"][dk]
            signal_area_m2 = np.matmul(_gamma_mm.T, signal_area_vs_true_energy_m2)


            for obstix in range(num_observation_times):
                print(sk, dk, oridx, obstix)
                critical_rate_per_s = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=cosmic_ray_rate_per_s,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                dFdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_area_vs_energy_m2=signal_area_m2,
                    signal_rate_vs_energy_per_s=critical_rate_per_s,
                )

                critical_dFdE[:, oridx, obstix] = dFdE

        json_numpy.write(
            os.path.join(pa["out_dir"], sk, dk + ".json"),
            {
                "energy_bin_edges": energy_bin["edges"],
                "observation_times": observation_times,
                "differential_flux": critical_dFdE,
                "comment": (
                    "Critical differential flux-sensitivity "
                    "VS energy VS onregion-size VS observation-time"
                ),
            },
        )
