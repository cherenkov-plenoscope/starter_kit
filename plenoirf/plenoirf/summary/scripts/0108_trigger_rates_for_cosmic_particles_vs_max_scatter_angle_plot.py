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
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
SITES = irf_config["config"]["sites"]
PLT = sum_config["plot"]

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

max_scatter_angles_deg = json_numpy.read(
    os.path.join(
        pa["summary_dir"], "0005_common_binning", "max_scatter_angles_deg.json"
    )
)

rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0107_trigger_rates_for_cosmic_particles_vs_max_scatter_angle",
    )
)

# find limits for axis
# --------------------
MAX_MAX_SCATTER_ANGLE_DEG = 0
for pk in PARTICLES:
    MAX_MAX_SCATTER_ANGLE_DEG = np.max(
        [MAX_MAX_SCATTER_ANGLE_DEG, np.max(max_scatter_angles_deg[pk])]
    )
MAX_MAX_SCATTER_SOLID_ANGLE_SR = irf.utils.cone_solid_angle(
    np.deg2rad(MAX_MAX_SCATTER_ANGLE_DEG)
)

for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:

        scatt_solid_sr = irf.utils.cone_solid_angle(
            np.deg2rad(max_scatter_angles_deg[pk])
        )
        scatt0_solid_sr = [0,] + list(scatt_solid_sr)
        scatt0_solid_sr = np.array(scatt0_solid_sr)

        R = rates[sk][pk]["integral"]["R"]
        R_au = rates[sk][pk]["integral"]["R_au"]

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=scatt0_solid_sr,
            bincounts=R,
            bincounts_lower=R - R_au,
            bincounts_upper=R + R_au,
            linecolor=PLT["particle_colors"][pk],
            face_color=PLT["particle_colors"][pk],
            face_alpha=0.2,
        )
    ax.set_ylabel("trigger-rate / 1")
    ax.set_xlabel("max. scatter solid angle / sr")
    ax.set_xlim([0, MAX_MAX_SCATTER_SOLID_ANGLE_SR])
    ax.set_ylim([1, 1e5])
    ax.semilogy()
    fig.savefig(os.path.join(pa["out_dir"], sk + ".jpg"))
    seb.close(fig)


for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:

        scatt_solid_sr = irf.utils.cone_solid_angle(
            np.deg2rad(max_scatter_angles_deg[pk])
        )
        scatt0_solid_sr = [0,] + list(scatt_solid_sr)
        scatt0_solid_sr = np.array(scatt0_solid_sr)

        R = rates[sk][pk]["integral"]["R"]
        R_au = rates[sk][pk]["integral"]["R_au"]

        relR = np.zeros(len(scatt_solid_sr) - 1)
        for sc in range(len(scatt_solid_sr) - 1):
            relR[sc] = (R[sc + 1] - R[sc]) / (0.5 * (R[sc + 1] + R[sc]))

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=scatt0_solid_sr[0:-1],
            bincounts=relR,
            linecolor=PLT["particle_colors"][pk],
            face_color=PLT["particle_colors"][pk],
            face_alpha=0.2,
        )

    ax.set_ylabel("rel. change in trigger-rate w.r.t.\nmax scatter angle / 1")
    ax.set_xlabel("max. scatter solid angle / sr")
    ax.set_xlim([0, MAX_MAX_SCATTER_SOLID_ANGLE_SR])
    ax.set_ylim([1e-2, 1e1])
    ax.semilogy()
    fig.savefig(os.path.join(pa["out_dir"], sk + "_diff.jpg"))
    seb.close(fig)


AXSPAN = copy.deepcopy(irf.summary.figure.AX_SPAN)
AXSPAN = [AXSPAN[0], AXSPAN[1], AXSPAN[2], AXSPAN[3]]

for sk in SITES:
    for pk in COSMIC_RAYS:
        print("plot 2D", sk, pk)

        dRdE = rates[sk][pk]["differential"]["dRdE"]
        dRdE_au = rates[sk][pk]["differential"]["dRdE_au"]

        R = np.zeros(shape=dRdE.shape)
        for sc in range(len(max_scatter_angles_deg[pk])):
            R[sc, :] = dRdE[sc, :] * energy_bin["width"]

        fig = seb.figure(style=irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=[AXSPAN[0], AXSPAN[1], 0.6, 0.7])

        ax_cb = seb.add_axes(
            fig=fig,
            span=[0.85, AXSPAN[1], 0.02, 0.7],
            # style=seb.AXES_BLANK,
        )

        ax.set_xlim(energy_bin["limits"])
        ax.set_ylim(
            [0, MAX_MAX_SCATTER_ANGLE_DEG,]
        )
        ax.semilogx()

        ax.set_xlabel("energy / GeV")
        ax.set_ylabel("max scatter angle / $1^\\circ$")

        pcm_ratio = ax.pcolormesh(
            energy_bin["edges"],
            max_scatter_angles_deg[pk],
            R,
            norm=seb.plt_colors.LogNorm(),
            cmap="terrain_r",
            vmin=1e-2,
            vmax=1e6,
        )

        seb.plt.colorbar(
            pcm_ratio, cax=ax_cb, label="trigger-rate / 1",
        )
        seb.ax_add_grid(ax=ax)

        fig.savefig(
            os.path.join(pa["out_dir"], "{:s}_{:s}.jpg".format(sk, pk,),)
        )
        seb.close(fig)
