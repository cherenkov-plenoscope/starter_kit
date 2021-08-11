#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import cosmic_fluxes
import os
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

acceptance = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles"
    )
)

energy_bin_edges, num_energy_bins = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["trigger_acceptance"]
)
energy_bin_centers = irf.utils.bin_centers(energy_bin_edges)

fine_energy_bin_edges, num_fine_energy_bins = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["interpolation"]
)
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)
fine_energy_bin_width = irf.utils.bin_width(fine_energy_bin_edges)

trigger_thresholds = np.array(sum_config["trigger"]["ratescan_thresholds_pe"])
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]
num_trigger_thresholds = len(trigger_thresholds)

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
fermi_3fgl = json_numpy.read(
    os.path.join(pa["summary_dir"], "0010_flux_of_cosmic_rays", "gamma_sources.json")
)

(
    gamma_differential_flux_per_m2_per_s_per_GeV,
    gamma_name,
) = irf.summary.make_gamma_ray_reference_flux(
    fermi_3fgl=fermi_3fgl,
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)

comment_differential = (
    "Differential trigger-rate, entire field-of-view. "
    "VS trigger-ratescan-thresholds"
)
comment_integral = (
    "Integral trigger-rate, entire field-of-view. "
    "VS trigger-ratescan-thresholds"
)

for sk in irf_config["config"]["sites"]:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)

    # gamma-ray
    # ---------
    sk_gamma_dir = os.path.join(sk_dir, "gamma")
    os.makedirs(sk_gamma_dir, exist_ok=True)

    _area = np.array(acceptance[sk]["gamma"]["point"]["mean"])

    T = []
    dT_dE = []
    for tt in range(num_trigger_thresholds):
        area_m2 = np.interp(
            x=fine_energy_bin_centers, xp=energy_bin_centers, fp=_area[tt, :]
        )
        gamma_differential_rate_per_s_per_GeV = (
            gamma_differential_flux_per_m2_per_s_per_GeV * area_m2
        )
        gamma_rate_per_s = np.sum(
            gamma_differential_rate_per_s_per_GeV * fine_energy_bin_width
        )
        T.append(gamma_rate_per_s)
        dT_dE.append(gamma_differential_rate_per_s_per_GeV)

    json_numpy.write(
        os.path.join(sk_gamma_dir, "differential_rate.json"),
        {
            "comment": comment_differential + ", " + gamma_name,
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dT_dE,
        },
    )
    json_numpy.write(
        os.path.join(sk_gamma_dir, "integral_rate.json"),
        {
            "comment": comment_integral + ", " + gamma_name,
            "unit": "s$^{-1}$",
            "mean": T,
        },
    )

    # cosmic-rays
    # -----------
    for ck in airshower_fluxes[sk]:
        sk_ck_dir = os.path.join(sk_dir, ck)
        os.makedirs(sk_ck_dir, exist_ok=True)

        T = []
        dT_dE = []
        _acceptance = np.array(
            acceptance[sk][ck]["diffuse"]["mean"]
        )
        for tt in range(num_trigger_thresholds):
            acceptance_m2_sr = np.interp(
                x=fine_energy_bin_centers,
                xp=energy_bin_centers,
                fp=_acceptance[tt, :],
            )
            cosmic_differential_rate_per_s_per_GeV = (
                acceptance_m2_sr
                * airshower_fluxes[sk][ck]["differential_flux"]["values"]
            )
            cosmic_rate_per_s = np.sum(
                cosmic_differential_rate_per_s_per_GeV * fine_energy_bin_width
            )
            T.append(cosmic_rate_per_s)
            dT_dE.append(cosmic_differential_rate_per_s_per_GeV)

        json_numpy.write(
            os.path.join(sk_ck_dir, "differential_rate.json"),
            {
                "comment": comment_differential,
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dT_dE,
            },
        )
        json_numpy.write(
            os.path.join(sk_ck_dir, "integral_rate.json"),
            {"comment": comment_integral, "unit": "s$^{-1}$", "mean": T},
        )
