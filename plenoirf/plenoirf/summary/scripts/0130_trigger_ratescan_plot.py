#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

cosmic_rates = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = irf.json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0120_trigger_rates_for_night_sky_background"
    )
)

trigger_thresholds = np.array(sum_config["trigger"]["ratescan_thresholds_pe"])
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]

particle_colors = sum_config["plot"]["particle_colors"]

trigger_rates = {}
for site_key in irf_config["config"]["sites"]:
    trigger_rates[site_key] = {}
    trigger_rates[site_key]["night_sky_background"] = np.array(
        nsb_rates[site_key]["night_sky_background_rates"]["mean"]
    )
    for cosmic_key in irf_config["config"]["particles"]:
        trigger_rates[site_key][cosmic_key] = np.array(
            cosmic_rates[site_key][cosmic_key]["integral_rate"]["mean"]
        )

cosmic_ray_keys = list(irf_config["config"]["particles"].keys())
cosmic_ray_keys.remove("gamma")

for site_key in irf_config["config"]["sites"]:
    tr = trigger_rates[site_key]

    fig = seb.figure(seb.FIGURE_16_9)
    ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))
    ax.plot(
        trigger_thresholds,
        tr["night_sky_background"]
        + tr["electron"]
        + tr["proton"]
        + tr["helium"],
        "k",
        label="night-sky + cosmic-rays",
    )
    ax.plot(
        trigger_thresholds, tr["night_sky_background"], "k:", label="night-sky"
    )

    for cosmic_key in cosmic_ray_keys:
        ax.plot(
            trigger_thresholds,
            tr[cosmic_key],
            color=particle_colors[cosmic_key],
            label=cosmic_key,
        )

    ax.semilogy()
    ax.set_xlabel("trigger-threshold / photo-electrons")
    ax.set_ylabel("trigger-rate / s$^{-1}$")
    ax.legend(loc="best", fontsize=10)
    ax.axvline(
        x=analysis_trigger_threshold, color="k", linestyle="-", alpha=0.25
    )
    ax.set_ylim([1e0, 1e7])
    fig.savefig(
        os.path.join(pa["out_dir"], "{:s}_ratescan.jpg".format(site_key))
    )
    seb.close_figure(fig)
