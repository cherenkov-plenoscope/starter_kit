#!/usr/bin/python
import sys
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

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

iacceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0410_interpolate_acceptance")
)
acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]

for sk in SITES:
    for gk in ["diffuse", "point"]:
        for ok in range(num_onregion_sizes):

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            for pk in PARTICLES:
                iQ = iacceptance[sk][pk][gk]["mean"][:, ok]
                iQ_au = iacceptance[sk][pk][gk]["absolute_uncertainty"][:, ok]
                Q = acceptance[sk][pk][gk]["mean"][:, ok]

                ax.plot(
                    fine_energy_bin["centers"],
                    iQ,
                    color=sum_config["plot"]["particle_colors"][pk],
                )
                ax.fill_between(
                    x=fine_energy_bin["centers"],
                    y1=iQ - iQ_au,
                    y2=iQ + iQ_au,
                    color=sum_config["plot"]["particle_colors"][pk],
                    alpha=0.2,
                    linewidth=0.0,
                )
                ax.plot(
                    energy_bin["centers"],
                    Q,
                    color=sum_config["plot"]["particle_colors"][pk],
                    linewidth=0.0,
                    marker="o",
                )
            ax.set_xlabel("energy / GeV")
            ax.set_ylabel(
                "{:s} / {:s}".format(
                    irf.summary.figure.SOURCES[gk]["label"],
                    irf.summary.figure.SOURCES[gk]["unit"],
                )
            )
            ax.set_ylim(
                irf.summary.figure.SOURCES[gk]["limits"][
                    "passed_trigger"
                ]
            )
            ax.loglog()
            fig.savefig(
                os.path.join(
                    pa["out_dir"], sk + "_" + gk + "_onregion{:06d}_acceptance_interpolated.jpg".format(ok)
                )
            )
            seb.close_figure(fig)


