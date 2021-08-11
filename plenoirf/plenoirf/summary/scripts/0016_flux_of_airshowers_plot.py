#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy
import sebastians_matplotlib_addons as seb


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

fine_energy_bin_edges, num_fine_energy_bins = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["interpolation"]
)
energy_lower = fine_energy_bin_edges[0]
energy_upper = fine_energy_bin_edges[-1]
fine_energy_bin_centers = irf.utils.bin_centers(fine_energy_bin_edges)


particle_colors = sum_config["plot"]["particle_colors"]

for sk in irf_config["config"]["sites"]:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in airshower_fluxes[sk]:
        ax.plot(
            fine_energy_bin_centers,
            airshower_fluxes[sk][pk]["differential_flux"]["values"],
            label=pk,
            color=particle_colors[pk],
        )
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel(
        "differential flux of airshowers /\n"
        + "m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$"
    )
    ax.loglog()
    ax.set_xlim([energy_lower, energy_upper])
    ax.legend()
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_airshower_differential_flux.jpg".format(sk),
        )
    )
    seb.close_figure(fig)
