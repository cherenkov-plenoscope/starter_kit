#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_numpy


"""
differential sensitivity w.r.t. energy
======================================

A series (500s) of scripts to estimate the diff. sensitivity.

505) Rebin the diff. flux of cosmic-rays dFdE into the energy-binning used
     for the diff. sensitivity.

530) Estimate the rates of cosmic-ray in reconstructed energy Rreco(E).

"""

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)

# load
# ----
airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)

# prepare
# -------
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]
fine_energy_bin_matches = []
for E in energy_bin["edges"]:
    match = np.argmin(np.abs(fine_energy_bin["edges"] - E))
    fine_energy_bin_matches.append(match)

# work
# ----
diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in COSMIC_RAYS:
        fine_dFdE = airshower_fluxes[sk][pk]["differential_flux"]["values"]
        dFdE = np.zeros(energy_bin["num_bins"])

        for ee in range(energy_bin["num_bins"]):
            fe_start = fine_energy_bin_matches[ee]
            fe_stop = fine_energy_bin_matches[ee + 1]
            dFdE[ee] = np.mean(fine_dFdE[fe_start:fe_stop])

        diff_flux[sk][pk] = dFdE

        json_numpy.write(
            os.path.join(pa["out_dir"], sk, pk + ".json"),
            {
                "energy_binning_key": energy_bin["key"],
                "differential_flux": dFdE,
                "unit": "m$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$",
            },
        )

# plot
# ----
for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        ax.plot(
            fine_energy_bin["centers"],
            airshower_fluxes[sk][pk]["differential_flux"]["values"],
            color=sum_config["plot"]["particle_colors"][pk],
        )
        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=diff_flux[sk][pk],
            linecolor=sum_config["plot"]["particle_colors"][pk],
        )
    ax.set_ylabel(
        "differential flux /\nm$^{-2}$ sr$^{-1}$ s$^{-1}$ (GeV)$^{-1}$"
    )
    ax.set_xlabel("energy / GeV")
    ax.set_ylim([1e-6, 1e2])
    ax.loglog()
    fig.savefig(os.path.join(pa["out_dir"], sk + ".jpg"))
    seb.close(fig)
