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

acceptance_Ereco = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0311_onregion_trigger_acceptance_in_reconstructed_energy",
    )
)

acceptance_Etrue = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

rate_onregion_reco_energy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0326_differential_rates")
)

energy_confusion = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

num_bins_energy = sum_config["energy_binning"]["num_bins"][
    "trigger_acceptance_onregion"
]
energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    num_bins_energy + 1,
)
energy_bin_centers = irf.utils.bin_centers(bin_edges=energy_bin_edges)
energy_bin_widths = irf.utils.bin_width(bin_edges=energy_bin_edges)

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

COSMIC_RAYS = list(irf_config["config"]["particles"].keys())
COSMIC_RAYS.remove("gamma")

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

critical_method = sum_config["on_off_measuremnent"]["method"]

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

SCENARIOS = ["RecoIsTrue", "RecoBroad", "RecoSharp"]


for sk in SITES:
    os.makedirs(os.path.join(pa["out_dir"], sk), exist_ok=True)

    critical_dFdE = {}
    for scenario in SCENARIOS:
        critical_dFdE[scenario] = np.nan * np.ones(
            shape=(num_bins_energy, num_onregion_sizes, num_observation_times)
        )

    assert (
        energy_confusion[sk]["gamma"]["confusion_matrix"]["ax0_key"]
        == "true_energy"
    )
    assert (
        energy_confusion[sk]["gamma"]["confusion_matrix"]["ax1_key"]
        == "reco_energy"
    )
    CM = {}
    for scenario in SCENARIOS:
        CM[
            scenario
        ] = irf.analysis.differential_sensitivity.make_energy_confusion_matrix_for_scenario(
            energy_confusion_matrix=energy_confusion[sk]["gamma"][
                "confusion_matrix"
            ]["confusion_bins_normalized_on_ax0"],
            scenario=scenario,
        )

    for oridx in range(num_onregion_sizes):

        # background rates
        # ----------------
        cosmic_ray_rate_per_s = np.zeros(num_bins_energy)
        for ck in COSMIC_RAYS:
            cosmic_ray_rate_per_s += rate_onregion_reco_energy[sk][ck][
                "rate_in_onregion_and_reconstructed_energy"
            ]["rate"][:, oridx]

        # signal effective area
        # ---------------------
        signal_area_vs_true_energy_m2 = acceptance_Etrue[sk]["gamma"]["point"][
            "mean"
        ][:, oridx]

        signal_area_m2 = {}
        for scenario in SCENARIOS:
            signal_area_m2[scenario] = np.matmul(
                CM[scenario].T, signal_area_vs_true_energy_m2
            )

        for obstix in range(num_observation_times):
            critical_rate_per_s = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                background_rate_in_onregion_vs_energy_per_s=cosmic_ray_rate_per_s,
                onregion_over_offregion_ratio=on_over_off_ratio,
                observation_time_s=observation_times[obstix],
                instrument_systematic_uncertainty=systematic_uncertainty,
                detection_threshold_std=detection_threshold_std,
                method=critical_method,
            )

            for scenario in SCENARIOS:
                dFdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin_edges,
                    signal_area_vs_energy_m2=signal_area_m2[scenario],
                    signal_rate_vs_energy_per_s=critical_rate_per_s,
                )

                critical_dFdE[scenario][:, oridx, obstix] = dFdE

    json_numpy.write(
        os.path.join(pa["out_dir"], sk, "differential_sensitivity" + ".json"),
        {
            "energy_bin_edges": energy_bin_edges,
            "observation_times": observation_times,
            "differential_flux": critical_dFdE,
            "comment": (
                "Critical differential flux-sensitivity "
                "VS energy VS onregion-size VS observation-time"
            ),
        },
    )
