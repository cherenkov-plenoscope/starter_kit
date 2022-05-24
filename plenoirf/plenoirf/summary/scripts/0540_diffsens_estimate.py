#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
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

critical_method = sum_config["on_off_measuremnent"][
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
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            print(sk, ok, dk)

            Areco = S[sk][ok][dk]["gamma"]["area"]["mean"]
            Areco_au = S[sk][ok][dk]["gamma"]["area"]["absolute_uncertainty"]

            # total background rate in reco energy
            # -------------------------------------
            Rreco_total = np.zeros(energy_bin["num_bins"])
            Rreco_total_au = np.zeros(energy_bin["num_bins"])
            for ereco in range(energy_bin["num_bins"]):
                tmp = []
                tmp_au = []
                for ck in COSMIC_RAYS:
                    tmp.append(S[sk][ok][dk][ck]["rate"]["mean"][ereco])
                    tmp_au.append(
                        S[sk][ok][dk][ck]["rate"]["absolute_uncertainty"][
                            ereco
                        ]
                    )
                (Rreco_total[ereco], Rreco_total_au[ereco],) = pru.sum(
                    x=(tmp, tmp_au)
                )

            Rreco_total_uu = Rreco_total + Rreco_total_au
            Areco_lu = Areco - Areco_au

            critical_dKdE = np.nan * np.ones(
                shape=(energy_bin["num_bins"], num_observation_times)
            )
            critical_dKdE_au = np.nan * np.ones(critical_dKdE.shape)
            for obstix in range(num_observation_times):
                critical_rate = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rreco_total,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty_relative=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                critical_rate_uu = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rreco_total_uu,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty_relative=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                dVdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_effective_area_m2=Areco,
                    signal_rate_per_s=critical_rate,
                )

                dVdE_uu = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_effective_area_m2=Areco_lu,
                    signal_rate_per_s=critical_rate_uu,
                )

                dVdE_au = dVdE_uu - dVdE

                critical_dKdE[:, obstix] = dVdE
                critical_dKdE_au[:, obstix] = dVdE_au

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk + ".json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "observation_times": observation_times,
                    "differential_flux": critical_dKdE,
                    "differential_flux_au": critical_dKdE_au,
                    "comment": (
                        "Critical, differential flux-sensitivity "
                        "VS energy VS observation-time"
                    ),
                },
            )
