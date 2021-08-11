#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_thresholds = sum_config["trigger"]["ratescan_thresholds_pe"]
trigger_threshold = sum_config["trigger"]["threshold_pe"]
idx_trigger_threshold = np.where(
    np.array(trigger_thresholds) == trigger_threshold,
)[0][0]
assert trigger_threshold in trigger_thresholds

IDX_FINAL_ONREGION = 1

# trigger
# -------
A = json_numpy.read_tree(
    opj(pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles")
)
A_energy_bin_edges, _ = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["trigger_acceptance"]
)

# trigger fix onregion
# --------------------
G = json_numpy.read_tree(
    opj(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)
G_energy_bin_edges, _ = irf.utils.power10space_bin_edges(
    binning=sum_config["energy_binning"],
    fine=sum_config["energy_binning"]["fine"]["trigger_acceptance_onregion"]
)

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

particle_colors = sum_config["plot"]["particle_colors"]


for site_key in irf_config["config"]["sites"]:
    for source_key in irf.summary.figure.SOURCES:

        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        text_y = 0
        for particle_key in irf_config["config"]["particles"]:

            Q = np.array(G[site_key][particle_key][source_key]["mean"])[
                :, IDX_FINAL_ONREGION
            ]
            delta_Q = np.array(
                G[site_key][particle_key][source_key]["relative_uncertainty"]
            )[:, IDX_FINAL_ONREGION]

            Q_lower = (1 - delta_Q) * Q
            Q_upper = (1 + delta_Q) * Q

            seb.ax_add_histogram(
                ax=ax,
                bin_edges=G_energy_bin_edges,
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
            irf.summary.figure.SOURCES[source_key]["limits"]["passed_all_cuts"]
        )
        ax.loglog()
        ax.set_xlim([G_energy_bin_edges[0], G_energy_bin_edges[-1]])

        fig.savefig(
            os.path.join(
                pa["out_dir"], "{:s}_{:s}.jpg".format(site_key, source_key,),
            )
        )
        seb.close_figure(fig)


for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        for source_key in irf.summary.figure.SOURCES:

            acc_trg = np.array(
                A[site_key][particle_key][source_key]["mean"][
                    idx_trigger_threshold
                ]
            )
            acc_trg_unc = np.array(
                A[site_key][particle_key][source_key]["relative_uncertainty"][
                    idx_trigger_threshold
                ]
            )

            acc_trg_onregions = np.array(
                G[site_key][particle_key][source_key]["mean"]
            )
            acc_trg_onregions_unc = np.array(
                G[site_key][particle_key][source_key]["relative_uncertainty"]
            )

            for oridx in range(num_bins_onregion_radius):
                acc_trg_onregion = acc_trg_onregions[:, oridx]
                acc_trg_onregion_unc = acc_trg_onregions_unc[:, oridx]

                fig = seb.figure(seb.FIGURE_16_9)
                ax = seb.add_axes(fig=fig, span=(0.1, 0.1, 0.8, 0.8))

                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=A_energy_bin_edges,
                    bincounts=acc_trg,
                    linestyle="-",
                    linecolor="gray",
                    bincounts_upper=acc_trg * (1 + acc_trg_unc),
                    bincounts_lower=acc_trg * (1 - acc_trg_unc),
                    face_color=particle_colors[particle_key],
                    face_alpha=0.05,
                )
                seb.ax_add_histogram(
                    ax=ax,
                    bin_edges=G_energy_bin_edges,
                    bincounts=acc_trg_onregion,
                    linestyle="-",
                    linecolor=particle_colors[particle_key],
                    bincounts_upper=acc_trg_onregion
                    * (1 + acc_trg_onregion_unc),
                    bincounts_lower=acc_trg_onregion
                    * (1 - acc_trg_onregion_unc),
                    face_color=particle_colors[particle_key],
                    face_alpha=0.25,
                )

                ax.set_title(
                    "onregion-radius at 100p.e.: {:.3f}".format(
                        onregion_radii_deg[oridx]
                    )
                    + r"$^{\circ}$"
                )
                ax.set_xlabel("energy / GeV")
                ax.set_ylabel(
                    irf.summary.figure.SOURCES[source_key]["label"]
                    + " / "
                    + irf.summary.figure.SOURCES[source_key]["unit"]
                )
                ax.set_ylim(
                    irf.summary.figure.SOURCES[source_key]["limits"][
                        "passed_trigger"
                    ]
                )
                ax.loglog()
                ax.set_xlim([A_energy_bin_edges[0], A_energy_bin_edges[-1]])
                fig.savefig(
                    opj(
                        pa["out_dir"],
                        "{:s}_{:s}_{:s}_onregion_onr{:06d}.jpg".format(
                            site_key, particle_key, source_key, oridx
                        ),
                    )
                )
                seb.close_figure(fig)
