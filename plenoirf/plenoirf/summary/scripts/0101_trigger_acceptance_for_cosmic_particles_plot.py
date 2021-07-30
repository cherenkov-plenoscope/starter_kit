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

cr = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles"
    )
)

energy_lower = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance"] + 1,
)

trigger_thresholds = np.array(sum_config["trigger"]["ratescan_thresholds_pe"])
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]

particle_colors = sum_config["plot"]["particle_colors"]

for site_key in irf_config["config"]["sites"]:
    for source_key in irf.summary.figure.SOURCES:
        for tt in range(len(trigger_thresholds)):

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            text_y = 0
            for particle_key in irf_config["config"]["particles"]:

                Q = np.array(
                    cr[site_key][particle_key][source_key]["mean"][tt]
                )
                delta_Q = np.array(
                    cr[site_key][particle_key][source_key][
                        "relative_uncertainty"
                    ][tt]
                )
                Q_lower = (1 - delta_Q) * Q
                Q_upper = (1 + delta_Q) * Q

                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=energy_bin_edges,
                    bincounts=Q,
                    linestyle="-",
                    linecolor=particle_colors[particle_key],
                    bincounts_upper=Q_upper,
                    bincounts_lower=Q_lower,
                    face_color=particle_colors[particle_key],
                    face_alpha=0.25,
                )

                ax.text(
                    0.9,
                    0.1 + text_y,
                    particle_key,
                    color=particle_colors[particle_key],
                    transform=ax.transAxes,
                )
                text_y += 0.06

            ax.set_xlabel("energy / GeV")
            ax.set_ylabel(
                "{:s} / {:s}".format(
                    irf.summary.figure.SOURCES[source_key]["label"],
                    irf.summary.figure.SOURCES[source_key]["unit"],
                )
            )
            ax.set_ylim(
                irf.summary.figure.SOURCES[source_key]["limits"][
                    "passed_trigger"
                ]
            )
            ax.loglog()
            ax.set_xlim([energy_bin_edges[0], energy_bin_edges[-1]])

            if trigger_thresholds[tt] == analysis_trigger_threshold:
                fig.savefig(
                    os.path.join(
                        pa["out_dir"],
                        "{:s}_{:s}.jpg".format(site_key, source_key,),
                    )
                )
            ax.set_title(
                "trigger-threshold: {:d} p.e.".format(trigger_thresholds[tt])
            )
            fig.savefig(
                os.path.join(
                    pa["out_dir"],
                    "{:s}_{:s}_{:06d}.jpg".format(site_key, source_key, tt,),
                )
            )
            seb.close_figure(fig)
