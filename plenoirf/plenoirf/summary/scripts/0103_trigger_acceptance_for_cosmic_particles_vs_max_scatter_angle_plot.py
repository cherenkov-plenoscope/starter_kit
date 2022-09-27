#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import copy
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]
PLT = sum_config["plot"]

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

max_scatter_angles_deg = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "max_scatter_angles_deg.json")
)

acceptance = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0102_trigger_acceptance_for_cosmic_particles_vs_max_scatter_angle"
    )
)

source_key = "diffuse"

# find limits for axis
# --------------------
MAX_MAX_SCATTER_ANGLE_DEG = 0
for pk in PARTICLES:
    MAX_MAX_SCATTER_ANGLE_DEG = np.max([
        MAX_MAX_SCATTER_ANGLE_DEG,
        np.max(max_scatter_angles_deg[pk])
    ])

AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for sk in SITES:
    for pk in PARTICLES:
        print("plot 2D", sk, pk)

        acc = acceptance[sk][pk][source_key]

        Q = acc["mean"]
        Q_au = acc["absolute_uncertainty"]

        dQdScatter = np.zeros(shape=(Q.shape[0] - 1, Q.shape[1]))
        for isc in range(len(max_scatter_angles_deg[pk]) - 1):
            dQdScatter[isc, :] = (
                (Q[isc + 1, :] - Q[isc, :]) / (0.5 * (Q[isc + 1, :] + Q[isc, :]))
            )

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.6, 0.7])

        ax_cb = seb.add_axes(
            fig=fig,
            span=[0.85, AXSPAN[1], 0.02, 0.7],
            #style=seb.AXES_BLANK,
        )

        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim([
            0,
            MAX_MAX_SCATTER_ANGLE_DEG,
        ])
        ax.semilogx()

        ax.set_xlabel("energy / GeV")
        ax.set_ylabel("max scatter angle / $1^\\circ$")

        pcm_ratio = ax.pcolormesh(
            energy_bin["edges"],
            max_scatter_angles_deg[pk],
            dQdScatter,
            norm=seb.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-2,
            vmax=1e0,
        )

        seb.plt.colorbar(
            pcm_ratio,
            cax=ax_cb,
            label="rel. gain in acceptance\nw.r.t. max scatter angle / 1",
        )
        seb.ax_add_grid(ax=ax)

        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_{:s}.jpg".format(
                    sk,
                    pk,
                ),
            )
        )
        seb.close(fig)
