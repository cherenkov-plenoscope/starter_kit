#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

iacc = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0425_diff_sens_acceptance_interpretation")
)

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
num_bins_onregion_radius = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
ONREGIONS = range(num_bins_onregion_radius)
particle_colors = sum_config["plot"]["particle_colors"]

GEOMETRIES = ["point", "diffuse"]
ok = 2

for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for gk in GEOMETRIES:
        sk_gk_dir = os.path.join(sk_dir, gk)
        os.makedirs(sk_gk_dir, exist_ok=True)

        for dk in irf.analysis.differential_sensitivity.SCENARIOS:

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            for pk in PARTICLES:

                iQ = iacc[sk][pk][gk][dk]["mean"][:, ok]
                iQ_au = iacc[sk][pk][gk][dk]["absolute_uncertainty"][:, ok]

                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=energy_bin["edges"],
                    bincounts=iQ,
                    bincounts_upper=iQ + iQ_au,
                    bincounts_lower=iQ - iQ_au,
                    face_color=particle_colors[pk],
                    face_alpha=0.25,
                    linestyle="-",
                    linecolor=particle_colors[pk],
                )


            ax.set_xlabel("interpreted energy / GeV")
            ax.set_ylabel(
                "{:s} / {:s}".format(
                    irf.summary.figure.SOURCES[gk]["label"],
                    irf.summary.figure.SOURCES[gk]["unit"],
                )
            )
            ax.loglog()
            ax.set_xlim(energy_bin["limits"])
            ax.set_ylim(
                irf.summary.figure.SOURCES[gk]["limits"][
                    "passed_trigger"
                ]
            )
            fig.savefig(os.path.join(sk_gk_dir, "{:s}.jpg".format(dk)))
            seb.close_figure(fig)
