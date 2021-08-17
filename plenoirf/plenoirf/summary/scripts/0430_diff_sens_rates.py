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

# prepare energy confusion
# ------------------------
energy_interpretation = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0420_diff_sens_energy_interpretation"),
)

# prepare onregion rates
# ----------------------
_onregion_rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)
diff_rate_per_s_per_GeV = {}
for sk in SITES:
    diff_rate_per_s_per_GeV[sk] = {}
    for pk in PARTICLES:
        diff_rate_per_s_per_GeV[sk][pk] = np.array(
            _onregion_rates[sk][pk]["differential_rate"]["mean"]
        )

# prepare integration-intervalls
fine_energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]
energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]


fine_energy_bin_edge_matches = []
for energy in energy_bin["edges"]:
    idx_near = np.argmin(np.abs(fine_energy_bin["edges"] - energy))
    fine_energy_bin_edge_matches.append(idx_near)

num_bins_onregion_radius = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

for sk in SITES:
    for pk in PARTICLES:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        rate_reco_energy_per_s = np.zeros(
            shape=(energy_bin["num_bins"], num_bins_onregion_radius)
        )
        for ordix in range(num_bins_onregion_radius):

            rate_true_energy_per_s = np.zeros(energy_bin["num_bins"])
            for ee in range(energy_bin["num_bins"]):
                estart = fine_energy_bin_edge_matches[ee]
                estop = fine_energy_bin_edge_matches[ee + 1]

                rate_true_energy_per_s[ee] = np.sum(
                    diff_rate_per_s_per_GeV[sk][pk][:, ordix][estart:estop]
                    * fine_energy_bin["width"][estart:estop]
                )

            for ee in range(energy_bin["num_bins"]):
                rate_reco_energy_per_s[:, ordix] += (
                    energy_confusion[sk][pk][ee] * rate_true_energy_per_s[ee]
                )

        json_numpy.write(
            os.path.join(
                site_particle_dir,
                "rate_in_onregion_and_reconstructed_energy.json",
            ),
            {
                "comment": (
                    "rate in onregion and reconstructed energy "
                    "VS onregion-radius"
                ),
                "unit": "s$^{-1}$",
                "rate": rate_reco_energy_per_s,
            },
        )
