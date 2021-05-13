#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
from os.path import join as opj

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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

# trigger
# -------
A = irf.json_numpy.read_tree(
    opj(pa["summary_dir"], "0100_trigger_acceptance_for_cosmic_particles")
)
A_energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance"] + 1,
)

# trigger fix onregion
# --------------------
G = irf.json_numpy.read_tree(
    opj(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)
G_energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance_onregion"]
    + 1,
)

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]

sources = {
    "diffuse": {
        "label": "area $\\times$ solid angle",
        "unit": "m$^{2}$ sr",
        "limits": [1e-1, 1e5],
    },
    "point": {"label": "area", "unit": "m$^{2}$", "limits": [1e1, 1e6],},
}

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        for source_key in sources:

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

                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

                irf.summary.figure.ax_add_hist(
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
                irf.summary.figure.ax_add_hist(
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
                    sources[source_key]["label"]
                    + " / "
                    + sources[source_key]["unit"]
                )
                ax.spines["top"].set_color("none")
                ax.spines["right"].set_color("none")
                ax.set_ylim(sources[source_key]["limits"])
                ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
                ax.loglog()
                ax.set_xlim([A_energy_bin_edges[0], A_energy_bin_edges[-1]])
                fig.savefig(
                    opj(
                        pa["out_dir"],
                        "{:s}_{:s}_{:s}_onregion_{:06d}.jpg".format(
                            site_key, particle_key, source_key, oridx
                        ),
                    )
                )
                plt.close(fig)
