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

iacceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0410_diffsens_interp_acceptance")
)

ienergy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0420_diffsens_interp_energy_migration")
)

cosmic_background_diff_rate = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0430_diffsens_interp_background_diff_rates")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
fenergy_bin = energy_binning["interpolation"]

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
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

critical_method = sum_config["on_off_measuremnent"][
    "estimator_for_critical_signal_rate"
]

for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for ok in ONREGION_TYPES:

        # estimate cosmic-ray rates
        # -------------------------
        Rt = np.zeros(fenergy_bin["num_bins"])
        Rt_au = np.zeros(fenergy_bin["num_bins"])

        _Rtsum = np.zeros((len(COSMIC_RAYS), fenergy_bin["num_bins"]))
        _Rtsum_au = np.zeros((len(COSMIC_RAYS), fenergy_bin["num_bins"]))

        for ick, ck in enumerate(COSMIC_RAYS):
            dRtdEt = cosmic_background_diff_rate[sk][ok][ck]["mean"]
            dRtdEt_au = cosmic_background_diff_rate[sk][ok][ck][
                "absolute_uncertainty"
            ]

            _Rtsum[ick, :] = dRtdEt * fenergy_bin["width"]
            _Rtsum_au[ick, :] = dRtdEt_au * fenergy_bin["width"]

        for ee in range(fenergy_bin["num_bins"]):
            Rt[ee], Rt_au[ee] = irf.utils.sum(
                x=_Rtsum[:, ee], x_au=_Rtsum_au[:, ee]
            )

        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            critical_dKdE = np.nan * np.ones(
                shape=(fenergy_bin["num_bins"], num_observation_times)
            )

            # estimate gamma eff. area
            # ------------------------
            A = iacceptance[sk][ok]["gamma"]["point"]["mean"]
            A_au = iacceptance[sk][ok]["gamma"]["point"]["absolute_uncertainty"]

            for obstix in range(num_observation_times):
                print(sk, dk, oridx, obstix)
                critical_rate_per_s = irf.analysis.differential_sensitivity.estimate_critical_rate_vs_energy(
                    background_rate_in_onregion_vs_energy_per_s=Rt,
                    onregion_over_offregion_ratio=on_over_off_ratio,
                    observation_time_s=observation_times[obstix],
                    instrument_systematic_uncertainty=systematic_uncertainty,
                    detection_threshold_std=detection_threshold_std,
                    method=critical_method,
                )

                dFdE = irf.analysis.differential_sensitivity.estimate_differential_sensitivity(
                    energy_bin_edges_GeV=fenergy_bin["edges"],
                    signal_area_vs_energy_m2=A,
                    signal_rate_vs_energy_per_s=critical_rate_per_s,
                )

                critical_dKdE[:, obstix] = dFdE

        json_numpy.write(
            os.path.join(pa["out_dir"], sk, dk + ".json"),
            {
                "energy_bin_edges": fenergy_bin["edges"],
                "observation_times": observation_times,
                "differential_flux": critical_dKdE,
                "comment": (
                    "Critical differential flux-sensitivity "
                    "VS energy VS observation-time"
                ),
            },
        )
