#!/usr/bin/python
import sys
import copy
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
energy_bin_width_au = np.zeros(energy_bin["num_bins"])

Q = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

M = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
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
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            print(sk, ok, dk)

            A = copy.deepcopy(Q[sk][ok]["gamma"]["point"]["mean"])
            A_au = copy.deepcopy(
                Q[sk][ok]["gamma"]["point"]["absolute_uncertainty"]
            )

            # Gamma-ray eff. Area
            # -------------------

            M_gamma = M[sk]["gamma"]["confusion_matrix"][
                "counts_normalized_on_ax0"
            ]
            M_gamma_au = M[sk]["gamma"]["confusion_matrix"][
                "counts_normalized_on_ax0_abs_unc"
            ]

            scn = irf.analysis.differential_sensitivity.make_energy_confusion_matrices_for_signal_and_background(
                signal_energy_confusion_matrix=M_gamma,
                signal_energy_confusion_matrix_abs_unc=M_gamma_au,
                scenario_key=dk,
            )

            M_gamma_scenario = scn["signal_matrix"]
            M_gamma_scenario_au = scn["signal_matrix_abs_unc"]

            (
                dMdE_scenario_gamma,
                dMdE_scenario_gamma_au,
            ) = irf.analysis.differential_sensitivity.derive_migration_matrix_by_ax0(
                migration_matrix_counts=M_gamma_scenario,
                migration_matrix_counts_abs_unc=M_gamma_scenario_au,
                ax0_bin_widths=energy_bin["width"],
            )

            A_scenario = np.zeros(energy_bin["num_bins"])
            A_scenario_au = np.zeros(energy_bin["num_bins"])

            for ereco in range(energy_bin["num_bins"]):
                _P = np.zeros(energy_bin["num_bins"])
                _P_au = np.zeros(energy_bin["num_bins"])
                for etrue in range(energy_bin["num_bins"]):

                    (
                        _P[etrue],
                        _P_au[etrue],
                    ) = irf.utils.multiply_elemnetwise_au(
                        x=[
                            dMdE_scenario_gamma[etrue, ereco],
                            A[etrue],
                            energy_bin["width"][etrue],
                        ],
                        x_au=[
                            dMdE_scenario_gamma_au[etrue, ereco],
                            A_au[etrue],
                            energy_bin_width_au[etrue],
                        ],
                    )

                (
                    A_scenario[ereco],
                    A_scenario_au[ereco],
                ) = irf.utils.sum_elemnetwise_au(x=_P, x_au=_P_au,)

            # background rates
            # ----------------
            bg_mask = scn["background_integral_mask"]

            Rt_full = copy.deepcopy(Rt[sk][ok]["mean"])
            Rt_full_au = copy.deepcopy(Rt[sk][ok]["absolute_uncertainty"])

            Rt_scenario = np.zeros(energy_bin["num_bins"])
            Rt_scenario_au = Rt_full_au

            _num_bins = np.zeros(energy_bin["num_bins"])
            for ereco in range(energy_bin["num_bins"]):
                for eck in range(energy_bin["num_bins"]):
                    _num_bins[ereco] += bg_mask[ereco, eck]
                    Rt_scenario[ereco] += bg_mask[ereco, eck] * Rt_full[eck]

            Rt_scenario_uu = Rt_scenario + Rt_scenario_au
            A_scenario_lu = A_scenario - A_scenario_au

            critical_dKdE = np.nan * np.ones(
                shape=(energy_bin["num_bins"], num_observation_times)
            )
            critical_dKdE_au = np.nan * np.ones(critical_dKdE.shape)
            for obstix in range(num_observation_times):
                critical_rate = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rt_scenario,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                critical_rate_uu = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rt_scenario_uu,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                dFdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_area_vs_energy_m2=A_scenario,
                    signal_rate_vs_energy_per_s=critical_rate,
                )

                dFdE_uu = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=energy_bin["edges"],
                    signal_area_vs_energy_m2=A_scenario_lu,
                    signal_rate_vs_energy_per_s=critical_rate_uu,
                )

                dFdE_au = dFdE_uu - dFdE

                critical_dKdE[:, obstix] = dFdE
                critical_dKdE_au[:, obstix] = dFdE_au

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk + ".json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "observation_times": observation_times,
                    "mean": critical_dKdE,
                    "absolute_uncertainty": critical_dKdE_au,
                    "comment": (
                        "Critical differential flux-sensitivity "
                        "VS energy VS onregion-size VS observation-time"
                    ),
                },
            )
