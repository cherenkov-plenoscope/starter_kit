#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import cosmic_fluxes
import os
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

onregion_acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_numpy.read(
    os.path.join(
        pa["summary_dir"], "0009_flux_of_gamma_rays", "reference_source.json"
    )
)
gamma_differential_flux_per_m2_per_s_per_GeV = gamma_source[
    "differential_flux"
]["values"]


comment_differential = "Differential trigger-rate, reconstructed in onregion."
comment_integral = "Integral trigger-rate, reconstructed in onregion."

for site_key in irf_config["config"]["sites"]:
    site_dir = os.path.join(pa["out_dir"], site_key)
    os.makedirs(site_dir, exist_ok=True)

    # gamma-ray
    # ---------
    site_gamma_dir = os.path.join(site_dir, "gamma")
    os.makedirs(site_gamma_dir, exist_ok=True)

    T = np.zeros(shape=(num_bins_onregion_radius))
    dT_dE = np.zeros(
        shape=(fine_energy_bin["num_bins"], num_bins_onregion_radius)
    )
    for oridx in range(num_bins_onregion_radius):
        _area = np.array(
            onregion_acceptance[site_key]["gamma"]["point"]["mean"]
        )[:, oridx]

        area_m2 = np.interp(
            x=fine_energy_bin["centers"], xp=energy_bin["centers"], fp=_area
        )
        gamma_differential_rate_per_s_per_GeV = (
            gamma_differential_flux_per_m2_per_s_per_GeV * area_m2
        )
        gamma_rate_per_s = np.sum(
            gamma_differential_rate_per_s_per_GeV * fine_energy_bin["width"]
        )
        T[oridx] = gamma_rate_per_s
        dT_dE[:, oridx] = gamma_differential_rate_per_s_per_GeV

    json_numpy.write(
        os.path.join(site_gamma_dir, "differential_rate.json"),
        {
            "comment": comment_differential
            + ", "
            + gamma_source["name"]
            + " VS onregion-radius",
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dT_dE,
        },
    )
    json_numpy.write(
        os.path.join(site_gamma_dir, "integral_rate.json"),
        {
            "comment": comment_integral
            + ", "
            + gamma_source["name"]
            + " VS onregion-radius",
            "unit": "s$^{-1}$",
            "mean": T,
        },
    )

    # cosmic-rays
    # -----------
    for cosmic_key in airshower_fluxes[site_key]:
        site_particle_dir = os.path.join(site_dir, cosmic_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        T = np.zeros(shape=(num_bins_onregion_radius))
        dT_dE = np.zeros(
            shape=(fine_energy_bin["num_bins"], num_bins_onregion_radius)
        )
        for oridx in range(num_bins_onregion_radius):
            _acceptance = np.array(
                onregion_acceptance[site_key][cosmic_key]["diffuse"]["mean"]
            )[:, oridx]

            acceptance_m2_sr = np.interp(
                x=fine_energy_bin["centers"],
                xp=energy_bin["centers"],
                fp=_acceptance,
            )
            cosmic_differential_rate_per_s_per_GeV = (
                acceptance_m2_sr
                * airshower_fluxes[site_key][cosmic_key]["differential_flux"][
                    "values"
                ]
            )
            cosmic_rate_per_s = np.sum(
                cosmic_differential_rate_per_s_per_GeV
                * fine_energy_bin["width"]
            )
            T[oridx] = cosmic_rate_per_s
            dT_dE[:, oridx] = cosmic_differential_rate_per_s_per_GeV

        json_numpy.write(
            os.path.join(site_particle_dir, "differential_rate.json"),
            {
                "comment": comment_differential + " VS onregion-radius",
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dT_dE,
            },
        )
        json_numpy.write(
            os.path.join(site_particle_dir, "integral_rate.json"),
            {
                "comment": comment_integral + " VS onregion-radius",
                "unit": "s$^{-1}$",
                "mean": T,
            },
        )
