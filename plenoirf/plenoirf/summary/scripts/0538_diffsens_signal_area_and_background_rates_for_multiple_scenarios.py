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

Rt = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0530_diffsens_background_diff_rates")
)


def make_area_in_reco_energy(A, A_au, Mtgr, Mtgr_au):
    num_bins = len(A)
    A_out = np.zeros(num_bins)
    A_out_au = np.zeros(num_bins)

    for er in range(num_bins):
        tmp = np.zeros(num_bins)
        tmp_au = np.zeros(num_bins)
        checksum = 0.0
        for et in range(num_bins):
            tmp[et], tmp_au[et] = irf.utils.multiply_elemnetwise_au(
                x=[Mtgr[et, er], A[et],],
                x_au=[Mtgr_au[et, er], A_au[et],],
            )
            checksum += Mtgr[et, er]
        if checksum > 0.0:
            assert checksum < 1.01
        A_out[er], A_out_au[er] = irf.utils.sum_elemnetwise_au(
            x=tmp, x_au=tmp_au
        )
    return A_out, A_out_au


def integrate_over_background_in_reco_energy(
    Rt, Rt_au, integration_mask, E_bin_width,
):
    num_bins = len(E_bin_width)
    E_bin_width_au = np.zeros(num_bins)

    imask = integration_mask
    imask_au = np.zeros(imask.shape)

    Rt_total = np.zeros(num_bins)
    Rt_total_au = np.zeros(num_bins)

    for er in range(num_bins):
        tmp_sum = np.zeros(num_bins)
        tmp_sum_au = np.zeros(num_bins)
        for ew in range(num_bins):
            tmp_sum[ew], tmp_sum_au[ew] = irf.utils.multiply_elemnetwise_au(
                x=[imask[er, ew], Rt[ew],],
                x_au=[imask_au[er, ew], Rt_au[ew],],
            )
        Rt_total[er], Rt_total_au[er] = irf.utils.sum_elemnetwise_au(
            x=tmp_sum, x_au=tmp_sum_au
        )
    return Rt_total, Rt_total_au


for sk in SITES:
    for ok in ONREGION_TYPES:
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            for pk in PARTICLES:
                os.makedirs(
                    os.path.join(pa["out_dir"], sk, ok, dk, pk), exist_ok=True
                )

for sk in SITES:
    A_gamma = Q[sk][ok]["gamma"]["point"]["mean"]
    A_gamma_au = Q[sk][ok]["gamma"]["point"]["absolute_uncertainty"]

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

            Mtgr_gamma_scenario = scenario["probability_true_given_reco"]
            Mtgr_gamma_scenario_au = scenario["probability_true_given_reco_abs_unc"]

            A_gamma_scenario, A_gamma_scenario_au = make_area_in_reco_energy(
                A=A_gamma,
                A_au=A_gamma_au,
                Mtgr=Mtgr_gamma_scenario,
                Mtgr_au=Mtgr_gamma_scenario_au,
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk, "gamma", "area.json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "mean": A_gamma_scenario,
                    "absolute_uncertainty": A_gamma_scenario_au,
                },
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk, "gamma", "M.json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "mean": Mtgr_gamma_scenario,
                    "absolute_uncertainty": Mtgr_gamma_scenario_au,
                },
            )

            # background rates
            # ----------------
            for ck in COSMIC_RAYS:
                (
                    Rt_scenario,
                    Rt_scenario_au,
                ) = integrate_over_background_in_reco_energy(
                    Rt=Rt[sk][ok][ck]["mean"],
                    Rt_au=Rt[sk][ok][ck]["absolute_uncertainty"],
                    integration_mask=scenario["background_integral_mask"],
                    E_bin_width=energy_bin["width"],
                )

                json_numpy.write(
                    os.path.join(pa["out_dir"], sk, ok, dk, ck, "rate.json"),
                    {
                        "energy_binning_key": energy_bin["key"],
                        "mean": Rt_scenario,
                        "absolute_uncertainty": Rt_scenario_au,
                    },
                )
