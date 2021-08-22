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

iAcceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0425_diff_sens_acceptance_interpretation")
)

iRate = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0428_diff_sens_rate_interpretation"),
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
        critical_dKdE = np.nan * np.ones(
            shape=(
                energy_bin["num_bins"],
                num_onregion_sizes,
                num_observation_times,
            )
        )

        # estimate cosmic-ray rates
        # -------------------------
        for oridx in range(num_onregion_sizes):

            R = np.zeros(energy_bin["num_bins"])
            R_abs_unc = np.zeros(energy_bin["num_bins"])
            _Rsum = np.zeros((len(COSMIC_RAYS), energy_bin["num_bins"]))
            _Rsum_abs_unc = np.zeros((len(COSMIC_RAYS), energy_bin["num_bins"]))
            for ick, ck in enumerate(COSMIC_RAYS):
                _Rsum[ick, :] = iRate[sk][ck][dk]["rate"]["mean"][:, oridx]
                _Rsum_abs_unc[ick, :] = iRate[sk][ck][dk]["rate"]["absolute_uncertainty"][:, oridx]
            for ee in range(energy_bin["num_bins"]):
                R[ee], R_abs_unc[ee] = irf.utils.sum(x=_Rsum[:, ee], x_au=_Rsum_abs_unc[:, ee])


            # estimate gamma eff. area
            # ------------------------
            A = iAcceptance[sk]["gamma"]["point"][dk]["mean"][:, oridx]

            for obstix in range(num_observation_times):
                print(sk, dk, oridx, obstix)
                critical_rate_per_s = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=R,
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

                critical_dKdE[:, oridx, obstix] = dFdE

        json_numpy.write(
            os.path.join(pa["out_dir"], sk, dk + ".json"),
            {
                "energy_bin_edges": energy_bin["edges"],
                "observation_times": observation_times,
                "differential_flux": critical_dKdE,
                "comment": (
                    "Critical differential flux-sensitivity "
                    "VS energy VS onregion-size VS observation-time"
                ),
            },
        )
