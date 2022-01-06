#!/usr/bin/python
import sys
import copy
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
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

Q = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

M = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)

R = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0530_diffsens_background_diff_rates")
)

# prepare
# -------
for sk in SITES:
    for ok in ONREGION_TYPES:
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            for pk in PARTICLES:
                os.makedirs(
                    os.path.join(pa["out_dir"], sk, ok, dk, pk), exist_ok=True
                )

for sk in SITES:
    M_gamma = M[sk]["gamma"]

    for ok in ONREGION_TYPES:
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            print(sk, ok, dk)

            scenario = irf.analysis.differential_sensitivity.make_energy_confusion_matrices_for_signal_and_background(
                probability_true_given_reco=M_gamma["true_given_reco"],
                probability_true_given_reco_abs_unc=M_gamma["true_given_reco_abs_unc"],
                probability_reco_given_true=M_gamma["reco_given_true"],
                scenario_key=dk,
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk, "gamma", "scenario.json"),
                scenario,
            )

            A_gamma_scenario, A_gamma_scenario_au = irf.analysis.differential_sensitivity.make_area_in_reco_energy(
                area=Q[sk][ok]["gamma"]["point"]["mean"],
                area_au=Q[sk][ok]["gamma"]["point"]["absolute_uncertainty"],
                probability_true_given_reco=scenario["probability_true_given_reco"],
                probability_true_given_reco_au=scenario["probability_true_given_reco_abs_unc"],
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk, "gamma", "area.json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "mean": A_gamma_scenario,
                    "absolute_uncertainty": A_gamma_scenario_au,
                },
            )

            # background rates
            # ----------------
            for ck in COSMIC_RAYS:
                (
                    Rreco_scenario,
                    Rreco_scenario_au,
                ) = irf.analysis.differential_sensitivity.integrate_rates_in_reco_energy_with_mask(
                    Rreco=R[sk][ok][ck]["reco"]["mean"],
                    Rreco_au=R[sk][ok][ck]["reco"]["absolute_uncertainty"],
                    integration_mask=scenario["background_integral_mask"],
                    integration_mask_au=np.zeros(shape=scenario["background_integral_mask"].shape)
                )

                json_numpy.write(
                    os.path.join(pa["out_dir"], sk, ok, dk, ck, "rate.json"),
                    {
                        "energy_binning_key": energy_bin["key"],
                        "mean": Rreco_scenario,
                        "absolute_uncertainty": Rreco_scenario_au,
                    },
                )
