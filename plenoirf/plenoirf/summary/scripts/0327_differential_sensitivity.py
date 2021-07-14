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

acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

acceptance_Ereco = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0311_onregion_trigger_acceptance_in_reconstructed_energy")
)

acceptance_Etrue = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

rate_onregion_reco_energy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0326_differential_rates")
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
systematic_uncertainty = sum_config["on_off_measuremnent"]["systematic_uncertainty"]

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

SCENARIOS = ["true", "reco_broad", "reco_sharp"]

for sk in SITES:
    os.makedirs(os.path.join(pa["out_dir"], sk), exist_ok=True)

    critical_dFdE = np.nan * np.ones(
        shape=(num_bins_energy, num_onregion_sizes, num_observation_times)
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
        gamma_effective_area_Ereco_m2 = acceptance_Ereco[sk]["gamma"]["point"]["mean"][:, oridx]
        gamma_effective_area_Etrue_m2 = acceptance_Etrue[sk]["gamma"]["point"]["mean"][:, oridx]

        for eidx in range(num_bins_energy):

            for obstix in range(num_observation_times):
                if cosmic_ray_rate_per_s[eidx] > 0:
                    critical_rate_per_s = irf.analysis.integral_sensitivity.estimate_critical_rate(
                        background_rate_in_onregion_per_s=cosmic_ray_rate_per_s[
                            eidx
                        ],
                        onregion_over_offregion_ratio=on_over_off_ratio,
                        observation_time_s=observation_times[obstix],
                        instrument_systematic_uncertainty=systematic_uncertainty,
                        detection_threshold_std=detection_threshold_std,
                        method=critical_method,
                    )
                else:
                    critical_rate_per_s = float("nan")

                critical_F_per_m2_per_s = (
                    critical_rate_per_s / gamma_effective_area_Ereco_m2[eidx]
                )
                critical_dFdE_per_m2_per_s_per_GeV = (
                    critical_F_per_m2_per_s / energy_bin_widths[eidx]
                )
                critical_dFdE[eidx, oridx, obstix] = critical_dFdE_per_m2_per_s_per_GeV

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
