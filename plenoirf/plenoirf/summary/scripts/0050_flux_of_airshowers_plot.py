#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import pandas as pd
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

energy_lower = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper = sum_config["energy_binning"]["upper_edge_GeV"]
fine_energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["interpolation"] + 1,
)
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)

particle_colors = sum_config["plot"]["particle_colors"]

# cosmic-ray-flux
# ----------------
airshower_fluxes = irf.summary.read_airshower_differential_flux_zenith_compensated(
    run_dir=pa["run_dir"],
    summary_dir=pa["summary_dir"],
    energy_bin_centers=fine_energy_bin_centers,
    sites=irf_config["config"]["sites"],
    geomagnetic_cutoff_fraction=sum_config["airshower_flux"][
        "fraction_of_flux_below_geomagnetic_cutoff"
    ],
)

for site_key in irf_config["config"]["sites"]:

    fig = seb.figure(seb.FIGURE_16_9)
    ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))
    for particle_key in airshower_fluxes[site_key]:
        ax.plot(
            fine_energy_bin_centers,
            airshower_fluxes[site_key][particle_key]["differential_flux"],
            label=particle_key,
            color=particle_colors[particle_key],
        )
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "differential flux of airshowers / "
        + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
    )
    ax.loglog()
    ax.set_xlim([energy_lower, energy_upper])
    ax.legend()
    ax.set_title("compensated for zenith-distance w.r.t. observation-plane")
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_airshower_differential_flux.jpg".format(site_key),
        )
    )
    seb.close_figure(fig)

    for particle_key in airshower_fluxes[site_key]:
        out_df = pd.DataFrame(
            {
                "energy / GeV": fine_energy_bin_centers,
                "differential flux / m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$": (
                    airshower_fluxes[site_key][particle_key][
                        "differential_flux"
                    ]
                ),
            }
        )
        out_path = os.path.join(
            pa["out_dir"], site_key + "_" + particle_key + ".csv"
        )
        with open(out_path, "wt") as f:
            f.write(out_df.to_csv(index=False))

# gamma-ray-flux of reference source
# ----------------------------------
(
    gamma_differential_flux_per_m2_per_s_per_GeV,
    gamma_name,
) = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa["summary_dir"],
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)
out_df = pd.DataFrame(
    {
        "energy / GeV": fine_energy_bin_centers,
        "differential flux / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$": (
            gamma_differential_flux_per_m2_per_s_per_GeV
        ),
    }
)
out_path = os.path.join(pa["out_dir"], "{:s}.csv".format(gamma_name))
with open(out_path, "wt") as f:
    f.write(out_df.to_csv(index=False))
