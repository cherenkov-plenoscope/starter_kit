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
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")
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

dRtdEt = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0530_diffsens_background_diff_rates")
)


def make_area_in_reco_energy(A, A_au, dMdE, dMdE_au, E_bin_width):
    A = copy.deepcopy(A)
    A_au = copy.deepcopy(A_au)
    dMdE = copy.deepcopy(dMdE)
    dMdE_au = copy.deepcopy(dMdE_au)
    E_bin_width = copy.deepcopy(E_bin_width)

    num_bins = len(E_bin_width)
    A_out = np.zeros(num_bins)
    A_out_au = np.zeros(num_bins)
    E_bin_width_au = np.zeros(num_bins)

    for er in range(num_bins):
        Ar = np.zeros(num_bins)
        Ar_au = np.zeros(num_bins)
        checksum = 0.0
        for et in range(num_bins):
            Ar[et], Ar_au[et] = irf.utils.multiply_elemnetwise_au(
                x=[dMdE[et, er], A[et], E_bin_width[et],],
                x_au=[dMdE_au[et, er], A_au[et], E_bin_width_au[et],],
            )
            checksum += dMdE[et, er] * E_bin_width[et]
        if checksum > 0.0:
            print(checksum)
            assert checksum < 1.01
        A_out[er], A_out_au[er] = irf.utils.sum_elemnetwise_au(
            x=Ar, x_au=Ar_au
        )
    return A_out, A_out_au


def integrate_over_background_in_reco_energy(
    dRtdEt, dRtdEt_au, integration_mask, E_bin_width,
):
    num_bins = len(E_bin_width)
    E_bin_width_au = np.zeros(num_bins)

    imask = integration_mask
    imask_au = np.zeros(imask.shape)

    Rt = np.zeros(num_bins)
    Rt_au = np.zeros(num_bins)

    for er in range(num_bins):
        Rr = np.zeros(num_bins)
        Rr_au = np.zeros(num_bins)
        for ew in range(num_bins):
            Rr[ew], Rr_au[ew] = irf.utils.multiply_elemnetwise_au(
                x=[imask[er, ew], dRtdEt[ew], E_bin_width[ew],],
                x_au=[imask_au[er, ew], dRtdEt_au[ew], E_bin_width_au[ew],],
            )
        Rt[er], Rt_au[er] = irf.utils.sum_elemnetwise_au(x=Rr, x_au=Rr_au)
    return Rt, Rt_au


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
                probability_reco_given_true=M_gamma["reco_given_true"],
                probability_reco_given_true_abs_unc=M_gamma["reco_given_true_abs_unc"],
                probability_true_given_reco=M_gamma["true_given_reco"],
                scenario_key=dk,
            )

            M_gamma_scenario = scenario["probability_reco_given_true"]
            M_gamma_scenario_au = scenario["probability_reco_given_true_abs_unc"]

            (
                dMdE_scenario_gamma,
                dMdE_scenario_gamma_au,
            ) = irf.analysis.differential_sensitivity.derive_migration_matrix_by_ax0(
                migration_matrix_counts=M_gamma_scenario,
                migration_matrix_counts_abs_unc=M_gamma_scenario_au,
                ax0_bin_widths=energy_bin["width"],
            )

            A_gamma_scenario, A_gamma_scenario_au = make_area_in_reco_energy(
                A=A_gamma,
                A_au=A_gamma_au,
                dMdE=dMdE_scenario_gamma,
                dMdE_au=dMdE_scenario_gamma_au,
                E_bin_width=energy_bin["width"],
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
                    "mean": M_gamma_scenario,
                    "absolute_uncertainty": M_gamma_scenario_au,
                },
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, dk, "gamma", "dMdE.json"),
                {
                    "energy_binning_key": energy_bin["key"],
                    "mean": dMdE_scenario_gamma,
                    "absolute_uncertainty": dMdE_scenario_gamma_au,
                },
            )

            # background rates
            # ----------------
            for ck in COSMIC_RAYS:
                (
                    Rt_scenario,
                    Rt_scenario_au,
                ) = integrate_over_background_in_reco_energy(
                    dRtdEt=dRtdEt[sk][ok][ck]["mean"],
                    dRtdEt_au=dRtdEt[sk][ok][ck]["absolute_uncertainty"],
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
