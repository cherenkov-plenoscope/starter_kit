#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

cosmic_rates = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0105_trigger_rates_for_cosmic_particles")
)

energy_lower = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper = sum_config["energy_binning"]["upper_edge_GeV"]
fine_energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config["energy_binning"]["num_bins"]["interpolation"] + 1,
)
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)

trigger_thresholds = np.array(sum_config["trigger"]["ratescan_thresholds_pe"])
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]

fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]

_, gamma_name = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa["summary_dir"],
    gamma_ray_reference_source=sum_config["gamma_ray_reference_source"],
    energy_supports_GeV=fine_energy_bin_centers,
)

for site_key in irf_config["config"]["sites"]:

    tt = 0
    for tt, trigger_threshold in enumerate(trigger_thresholds):
        if trigger_threshold == analysis_trigger_threshold:
            break

    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

    text_y = 0.7
    for particle_key in irf_config["config"]["particles"]:

        dT_dE_vs_threshold = np.array(
            cosmic_rates[site_key][particle_key]["differential_rate"]["mean"]
        )

        ax.plot(
            fine_energy_bin_centers,
            dT_dE_vs_threshold[tt, :],
            color=particle_colors[particle_key],
            label=particle_key,
        )
        ax.text(
            0.8,
            0.1 + text_y,
            particle_key,
            color=particle_colors[particle_key],
            transform=ax.transAxes,
        )
        ir = cosmic_rates[site_key][particle_key]["integral_rate"]["mean"][tt]
        ax.text(
            0.9,
            0.1 + text_y,
            "{: 12.1f} s$^{{-1}}$".format(ir),
            color="k",
            family="monospace",
            transform=ax.transAxes,
        )
        text_y += 0.06

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_title("trigger, entire field-of-view, " + gamma_name)
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax.loglog()
    ax.set_xlim([energy_lower, energy_upper])
    ax.set_ylim([1e-3, 1e5])
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_differential_trigger_rate.jpg".format(site_key),
        )
    )
    plt.close(fig)
