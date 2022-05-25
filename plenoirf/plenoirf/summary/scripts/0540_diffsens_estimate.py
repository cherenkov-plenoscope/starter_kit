#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import flux_sensitivity
import propagate_uncertainties as pru
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
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
energy_bin_width_au = np.zeros(energy_bin["num_bins"])

S = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0538_diffsens_signal_area_and_background_rates_for_multiple_scenarios",
    )
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

estimator_statistics = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

# prepare
# -------
for sk in SITES:
    for ok in ONREGION_TYPES:
        os.makedirs(os.path.join(pa["out_dir"], sk, ok), exist_ok=True)

# work
# ----
for sk in SITES:
    for ok in ONREGION_TYPES:
        for dk in flux_sensitivity.differential.SCENARIOS:
            print(sk, ok, dk)

            A_gamma_scenario = S[sk][ok][dk]["gamma"]["area"]["mean"]
            A_gamma_scenario_au = S[sk][ok][dk]["gamma"]["area"][
                "absolute_uncertainty"
            ]

            # Sum up components of background rate in scenario
            # ------------------------------------------------
            R_background_components = []
            R_background_components_au = []
            for ck in COSMIC_RAYS:
                R_background_components.append(
                    S[sk][ok][dk][ck]["rate"]["mean"][:]
                )
                R_background_components_au.append(
                    S[sk][ok][dk][ck]["rate"]["absolute_uncertainty"][:]
                )

            R_background_scenario, R_background_scenario_au = pru.sum_axis0(
                x=R_background_components, x_au=R_background_components_au,
            )

            critical_dVdE = np.nan * np.ones(
                shape=(energy_bin["num_bins"], num_observation_times)
            )
            critical_dVdE_au = np.nan * np.ones(critical_dVdE.shape)
            for obstix in range(num_observation_times):
                (
                    R_gamma_scenario,
                    R_gamma_scenario_au,
                ) = flux_sensitivity.differential.estimate_critical_signal_rate_vs_energy(
                    background_rate_onregion_in_scenario_per_s=R_background_scenario,
                    background_rate_onregion_in_scenario_per_s_au=R_background_scenario_au,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty_relative=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    estimator_statistics=estimator_statistics,
                )

                (
                    dVdE,
                    dVdE_au,
                ) = flux_sensitivity.differential.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_area_in_scenario_m2=A_gamma_scenario,
                    signal_area_in_scenario_m2_au=A_gamma_scenario_au,
                    critical_signal_rate_in_scenario_per_s=R_gamma_scenario,
                    critical_signal_rate_in_scenario_per_s_au=R_gamma_scenario_au,
                )

                critical_dVdE[:, obstix] = dVdE
                critical_dVdE_au[:, obstix] = dVdE_au

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk + ".json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "observation_times": observation_times,
                    "differential_flux": critical_dVdE,
                    "differential_flux_au": critical_dVdE_au,
                    "comment": (
                        "Critical, differential flux-sensitivity "
                        "VS energy VS observation-time"
                    ),
                },
            )
