#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

acceptance = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

rate_onregion_reco_energy = irf.json_numpy.read_tree(
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

detection_threshold_std = sum_config["on_off_measuremnent"][
    "detection_threshold_std"
]
on_over_off_ratio = sum_config["on_off_measuremnent"]["on_over_off_ratio"]
observation_time_s = 300
systematic_uncertainty = 1e-2

COSMIC_RAYS = list(irf_config["config"]["particles"].keys())
COSMIC_RAYS.remove("gamma")

ORIDX = 1

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

# prepare background rates
# ------------------------

cosmic_ray_rate = {}
for sk in SITES:
    cosmic_ray_rate[sk] = np.zeros(num_bins_energy)
    for ck in COSMIC_RAYS:
        cosmic_ray_rate[sk] += np.array(
            rate_onregion_reco_energy[sk][ck][
                "rate_in_onregion_and_reconstructed_energy"
            ]["rate"]
        )[:, ORIDX]


for sk in SITES:

    gamma_effective_area = np.array(acceptance[sk]["gamma"]["point"]["mean"])[
        :, ORIDX
    ]

    for ee in range(num_bins_energy):

        if cosmic_ray_rate[sk][ee] > 0:
            critical_rate_per_s = irf.analysis.integral_sensitivity.estimate_critical_rate(
                background_rate_in_onregion_per_s=cosmic_ray_rate[sk][ee],
                onregion_over_offregion_ratio=on_over_off_ratio,
                observation_time_s=observation_time_s,
                instrument_systematic_uncertainty=systematic_uncertainty,
                detection_threshold_std=detection_threshold_std,
                method="LiMa_eq17",
            )
        else:
            critical_rate_per_s = float("nan")

        min_flux = critical_rate_per_s / gamma_effective_area[ee]

        print(sk, ee, critical_rate_per_s, min_flux)
