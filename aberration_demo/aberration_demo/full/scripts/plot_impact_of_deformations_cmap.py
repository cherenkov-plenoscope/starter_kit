#!/usr/bin/python3
import corsika_primary as cpw
import os
import plenoirf
import numpy as np
import plenopy
import scipy
from scipy import spatial
from scipy import stats
import aberration_demo as abe
import json_numpy
import sebastians_matplotlib_addons as sebplt
import argparse

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "work_dir", metavar="WORK_DIR", type=str,
)

args = argparser.parse_args()
work_dir = args.work_dir

plot_iod_dir = os.path.join(
    work_dir, "plots", "impact_of_deformations", "guide_stars"
)

maximum_image_intensity_for_guide_stars = json_numpy.read(
    os.path.join(plot_iod_dir, "maximum_image_intensity_for_guide_stars.json")
)
CMAP_VMAX = np.max(
    [
        maximum_image_intensity_for_guide_stars[instrument_key]
        for instrument_key in maximum_image_intensity_for_guide_stars
    ]
)
CMAPS = abe.full.plots.utils.CMAPS

for cmap_key in CMAPS:

    _fig = sebplt.figure(style={"rows": 100, "cols": 100, "fontsize": 1.0})
    _ax = sebplt.add_axes(fig=_fig, span=[0.15, 0.15, 0.85, 0.85],)
    cmap_psf = _ax.pcolormesh(
        [0, 1],
        [0, 1],
        [[CMAP_VMAX]],
        cmap=cmap_key,
        norm=sebplt.plt_colors.PowerNorm(
            gamma=CMAPS[cmap_key]["gamma"], vmin=0.0, vmax=1.0,
        ),
    )
    sebplt.close(_fig)

    # colormap
    # --------
    fig_psf_cmap = sebplt.figure(
        style={"rows": 120, "cols": 1280, "fontsize": 1}
    )
    ax_cmap = sebplt.add_axes(fig_psf_cmap, [0.1, 0.8, 0.8, 0.15])
    ax_cmap.text(0.5, -4.7, "intensity / 1")
    sebplt.plt.colorbar(
        cmap_psf, cax=ax_cmap, extend="max", orientation="horizontal"
    )
    fig_cmap_filename = "star_cmap_key_{:s}.jpg".format(cmap_key)
    fig_psf_cmap.savefig(os.path.join(plot_iod_dir, fig_cmap_filename))
    sebplt.close(fig_psf_cmap)


"""
# summary plot of time-spread-functions
# -------------------------------------
time_start_ns = -25
time_stop_ns = 25
time_weight_start_perc = 0
time_weight_stop_perc = 10

time_grid_yticks = np.linspace(
    time_weight_start_perc, time_weight_stop_perc, 5
)
time_yticks = np.array([0, 5])
time_xticks_ns = np.linspace(time_start_ns, time_stop_ns, 11)


for isens, pkey in enumerate(coll):

    fig_tsf_filename = "tsf_{:s}_{:s}_hists.jpg".format(mkey, pkey)
    fig_tsf_path = os.path.join(out_dir, fig_tsf_filename)

    if os.path.exists(fig_tsf_path):
        continue

    fig_tsf = sebplt.figure(style={"rows": 440, "cols": 1280, "fontsize": 1})

    for iiofa, ofa in enumerate(OFFAXIS_ANGLE_IDXS):
        akey = abe.offaxis.ANGLE_FMT.format(ofa)
        tcoll = coll[pkey][akey]

        ax_tsf_width = 0.85 / num_offaxis
        ax_tsf = sebplt.add_axes(
            fig=fig_tsf,
            span=[
                0.1 + iiofa * (ax_tsf_width + 0.02),
                0.15,
                ax_tsf_width,
                0.8,
            ],
            style={
                "spines": ["left", "bottom"],
                "axes": ["x", "y"],
                "grid": False,
            },
        )

        time_center_ns = 0.5e9 * (
            tcoll["time"]["fwhm"]["stop"] + tcoll["time"]["fwhm"]["start"]
        )
        time_weights_norm_perc = (
            100 * tcoll["time"]["weights"] / np.sum(tcoll["time"]["weights"])
        )
        sebplt.ax_add_histogram(
            ax=ax_tsf,
            bin_edges=1e9 * tcoll["time"]["bin_edges"] - time_center_ns,
            bincounts=time_weights_norm_perc,
            face_color="k",
            face_alpha=0.1,
            draw_bin_walls=True,
        )
        ax_tsf.set_xlim([time_start_ns, time_stop_ns])
        ax_tsf.set_ylim([time_weight_start_perc, time_weight_stop_perc])
        # ax_pax_off.semilogy()

        sebplt.ax_add_grid_with_explicit_ticks(
            xticks=time_xticks_ns,
            yticks=time_grid_yticks,
            ax=ax_tsf,
            color="k",
            linestyle="-",
            linewidth=0.33,
            alpha=0.33,
        )

        if iiofa == 0:
            ax_tsf.set_yticks(time_yticks)
        else:
            ax_tsf.set_yticks([])

    fig_tsf.savefig(fig_tsf_path)
    sebplt.close(fig_tsf)


# plot psf80 vs off-axes
# ----------------------
NUM_PAXEL_STYLE = {
    "paxel000001": {
        "color": "gray",
        "linestyle": "-",
        "alpha": 1,
        "marker": "P",
    },
    "paxel000003": {
        "color": "gray",
        "linestyle": "-",
        "alpha": 0.3,
        "marker": "s",
    },
    "paxel000009": {
        "color": "black",
        "linestyle": "-",
        "alpha": 1,
        "marker": "o",
    },
}
SOLID_ANGLE_SCALE = 1e6

solid_angle_80_sr_start = 0
solid_angle_80_sr_stop = 20e-6

fig = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.2, 0.72, 0.75],)
ax_deg2 = ax.twinx()
ax_deg2.spines["top"].set_visible(False)


ylabel_name = r"solid angle containing 80%"
label_sep = r"$\,/\,$"
ax.set_ylim(
    SOLID_ANGLE_SCALE
    * np.array([solid_angle_80_sr_start, solid_angle_80_sr_stop])
)
# ax.semilogy()
ax.set_ylabel(ylabel_name + label_sep + r"$\mu$sr")

solid_angle_80_deg2_start = plenoirf.utils.sr2squaredeg(
    solid_angle_80_sr_start
)
solid_angle_80_deg2_stop = plenoirf.utils.sr2squaredeg(solid_angle_80_sr_stop)
ax_deg2.set_ylim(
    np.array([solid_angle_80_deg2_start, solid_angle_80_deg2_stop])
)
# ax_deg2.semilogy()
ax_deg2.set_ylabel(r"(1$^{\circ}$)$^2$")

sebplt.ax_add_grid(ax=ax, add_minor=True)

average_angle80_in_fov = {}
for isens, pkey in enumerate(coll):

    offaxis_angles_deg = config["sources"]["off_axis_angles_deg"]
    angles80_rad = np.zeros(len(offaxis_angles_deg))
    for iang, akey in enumerate(coll[pkey]):
        angles80_rad[iang] = coll[pkey][akey]["image"]["angle80"]

    cone80_solid_angle_sr = np.zeros(len(angles80_rad))
    for iang in range(len(angles80_rad)):
        cone80_solid_angle_sr[iang] = plenoirf.utils.cone_solid_angle(
            cone_radial_opening_angle_rad=angles80_rad[iang]
        )

    off_axis_weight = np.pi * offaxis_angles_deg ** 2
    off_axis_weight /= np.sum(off_axis_weight)
    average_angle80_in_fov[pkey] = np.average(
        angles80_rad, weights=off_axis_weight,
    )

    ax.plot(
        offaxis_angles_deg,
        cone80_solid_angle_sr * SOLID_ANGLE_SCALE,
        color=NUM_PAXEL_STYLE[pkey]["color"],
        linestyle=NUM_PAXEL_STYLE[pkey]["linestyle"],
        alpha=NUM_PAXEL_STYLE[pkey]["alpha"],
    )

    for iang in range(len(angles80_rad)):
        if iang in OFFAXIS_ANGLE_IDXS:
            markersize = 8
        else:
            markersize = 3
        marker = NUM_PAXEL_STYLE[pkey]["marker"]
        ax.plot(
            offaxis_angles_deg[iang],
            cone80_solid_angle_sr[iang] * SOLID_ANGLE_SCALE,
            color=NUM_PAXEL_STYLE[pkey]["color"],
            alpha=NUM_PAXEL_STYLE[pkey]["alpha"],
            marker=marker,
            markersize=markersize,
            linewidth=0,
        )

ax.set_xlabel(r"angle off the mirror's optical axis$\,/\,1^{\circ}$")

fig.savefig(os.path.join(out_dir, "psf_vs_num_paxel_vs_off_axis.jpg"))
sebplt.close(fig)


with open(
    os.path.join(out_dir, "psf_vs_num_paxel_vs_off_axis.txt"), "wt"
) as f:
    f.write("{:>20s},".format("offaxis/deg"))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format(pkey))
    f.write("\n")

    # half angle
    # ==========
    f.write("{:>20s},".format(""))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format("half-angle-80/deg"))
    f.write("\n")

    for iang, akey in enumerate(coll[pkey]):
        offax_deg = config["sources"]["off_axis_angles_deg"][iang]
        f.write("{: 20.2},".format(offax_deg))
        for isens, pkey in enumerate(coll):

            half_angle80_rad = coll[pkey][akey]["image"]["angle80"]
            half_angle80_deg = np.rad2deg(half_angle80_rad)
            f.write("{: 20.2},".format(half_angle80_deg))

        f.write("\n")

    f.write("\n")

    # solid angle
    # ===========
    f.write("{:>20s},".format(""))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format("solid-angle-80/usr"))
    f.write("\n")

    for iang, akey in enumerate(coll[pkey]):
        offax_deg = config["sources"]["off_axis_angles_deg"][iang]
        f.write("{: 20.2},".format(offax_deg))
        for isens, pkey in enumerate(coll):

            half_angle80_rad = coll[pkey][akey]["image"]["angle80"]
            solid_angle_80_sr = plenoirf.utils.cone_solid_angle(
                cone_radial_opening_angle_rad=half_angle80_rad
            )
            f.write("{: 20.2},".format(SOLID_ANGLE_SCALE * solid_angle_80_sr))

        f.write("\n")

    f.write("\n")

    # average
    # =======
    f.write("{:>20s},".format(""))
    for pkey in coll:
        f.write("{:>20s},".format("avg-solid-80/usr"))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        avg_solid_angle_usr = (
            SOLID_ANGLE_SCALE
            * plenoirf.utils.cone_solid_angle(
                cone_radial_opening_angle_rad=average_angle80_in_fov[pkey]
            )
        )
        f.write("{: 20.2},".format(avg_solid_angle_usr))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        f.write("{:>20s},".format("avg-angle-80/deg"))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        average_angle80_in_fov_deg = np.rad2deg(average_angle80_in_fov[pkey])
        f.write("{: 20.2},".format(average_angle80_in_fov_deg))
    f.write("\n")
"""
