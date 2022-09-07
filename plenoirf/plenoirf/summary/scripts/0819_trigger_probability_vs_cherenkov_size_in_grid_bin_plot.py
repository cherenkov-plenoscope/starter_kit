#!/usr/bin/python
import sys
import plenoirf as irf
import os
import numpy as np
from os.path import join as opj
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_vs_size = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0818_trigger_probability_vs_cherenkov_size_in_grid_bin"
    )
)

grid_bin_area_m2 = irf_config["grid_geometry"]["bin_area"]

particle_colors = sum_config["plot"]["particle_colors"]
key = "trigger_probability_vs_cherenkov_size_in_grid_bin"

for sk in irf_config["config"]["sites"]:

    # all particles together
    # ----------------------
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    text_y = 0
    for pk in irf_config["config"]["particles"]:

        size_bin_edges = trigger_vs_size[sk][pk][
                key
            ]["true_Cherenkov_size_bin_edges_pe"]
        density_bin_edges = size_bin_edges / grid_bin_area_m2

        prob = trigger_vs_size[sk][pk][
                key
            ]["mean"]

        prob_unc = trigger_vs_size[sk][pk][
                key
            ]["relative_uncertainty"]

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=density_bin_edges,
            bincounts=prob,
            linestyle="-",
            linecolor=particle_colors[pk],
            bincounts_upper=prob * (1 + prob_unc),
            bincounts_lower=prob * (1 - prob_unc),
            face_color=particle_colors[pk],
            face_alpha=0.25,
        )
        ax.text(
            0.85,
            0.1 + text_y,
            pk,
            color=particle_colors[pk],
            transform=ax.transAxes,
        )
        text_y += 0.06
    ax.semilogx()
    ax.semilogy()
    ax.set_xlim([np.min(density_bin_edges), np.max(density_bin_edges)])
    ax.set_ylim([1e-6, 1.5e-0])
    ax.set_xlabel("density of Cherenkov-photons at plenoscope / m$^{-2}$")
    ax.set_ylabel("trigger-probability / 1")
    fig.savefig(
        opj(
            pa["out_dir"],
            sk + "_trigger_probability_vs_cherenkov_size_in_grid_bin.jpg",
        )
    )
    seb.close(fig)
