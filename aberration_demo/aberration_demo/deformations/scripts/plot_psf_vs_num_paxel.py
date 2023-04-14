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
import sys

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argv = sys.argv
if argv[0] == "ipython" and argv[1] == "-i":
    argv.pop(1)

work_dir = argv[1]
out_dir = os.path.join(work_dir, "plot", "psf_vs_num_paxel")
os.makedirs(out_dir, exist_ok=True)

config = json_numpy.read(os.path.join(work_dir, "config.json"))
coll = abe.deformations.read_analysis(work_dir=work_dir)

# summary plot of point-spread-functions
# --------------------------------------

OFFAXIS_ANGLE_IDXS = [0, 3, 6]
OFF_AXIS_ANGLE_LABEL = r"off-axis-angle / 1$^\circ$"
GRID_ANGLE_DEG = 0.2

CMAPS = {
    "hot": {"gamma": 1, "linecolor": "white",},
    "Blues": {"gamma": 1, "linecolor": "black",},
    "binary": {"gamma": 1, "linecolor": "black",},
    "magma_r": {"gamma": 1, "linecolor": "black",},
}

mkey = "parabola_segmented"


def make_grid_ticks(center, num_pixel, pixel_angel, tick_angle):
    extent = pixel_angel * num_pixel
    num_ticks = int(np.ceil(extent / tick_angle))
    tick_extent = num_ticks * tick_angle

    tick_start = center - 0.5 * tick_extent
    tick_stop = center + 0.5 * tick_extent
    ticks = np.linspace(tick_start, tick_stop, num_ticks + 1)
    return ticks


def _ax_add_paxel_and_off_axis_labels(ax):
    for isens, pkey in enumerate(coll):
        num_paxel_on_diagonal = config["sensor"][
            "num_paxel_on_pixel_diagonal"
        ][isens]

        ax.text(
            0.01,
            0.13
            + ax_vert_start
            + ax_margin_height_rel
            + ax_height_rel * (num_sensors - isens - 1),
            r"{:d}".format(num_paxel_on_diagonal),
            rotation=0,
            fontsize=14,
        )

        for iiofa, ofa in enumerate(OFFAXIS_ANGLE_IDXS):
            off_axis_cx_deg = config["sources"]["off_axis_angles_deg"][ofa]
            off_axis_cy_deg = 0.0

            ax.text(
                0.12
                + ax_hori_start
                + ax_margin_width_rel
                + ax_width_rel * iiofa,
                0.01,
                r"{:1.1f}".format(off_axis_cx_deg) + r"$^\circ$",
                rotation=0,
                fontsize=14,
            )


def make_explicit_cx_cy_ticks(tcoll, tick_angle):
    ticks_cx_deg = make_grid_ticks(
        center=tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
        num_pixel=tcoll["image"]["binning"]["image"]["num_pixel_cx"],
        pixel_angel=tcoll["image"]["binning"]["image"]["pixel_angle_deg"],
        tick_angle=tick_angle,
    )
    ticks_cy_deg = make_grid_ticks(
        center=tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
        num_pixel=tcoll["image"]["binning"]["image"]["num_pixel_cy"],
        pixel_angel=tcoll["image"]["binning"]["image"]["pixel_angle_deg"],
        tick_angle=tick_angle,
    )
    return ticks_cx_deg, ticks_cy_deg


def ax_psf_add_eye(ax, tcoll, bin_edges_cx_deg, bin_edges_cy_deg, linecolor):
    sebplt.ax_add_circle(
        ax=ax,
        x=np.mean(
            [
                tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                bin_edges_cx_deg[0],
            ]
        ),
        y=np.mean(
            [
                tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                bin_edges_cy_deg[0],
            ]
        ),
        r=config["sensor"]["dimensions"]["hex_pixel_FoV_flat2flat_deg"]
        * 0.5
        * 2
        / np.sqrt(3),
        linewidth=0.75,
        linestyle="-",
        color=linecolor,
        alpha=1,
        num_steps=7,
    )


def ax_psf_set_ticks(ax, tcoll, grid_angle_deg, x=True, y=True):
    if x:
        ax.set_xticks(
            [
                tcoll["image"]["binning"]["image"]["center"]["cx_deg"]
                - 3 * grid_angle_deg,
                tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                tcoll["image"]["binning"]["image"]["center"]["cx_deg"]
                + 3 * grid_angle_deg,
            ]
        )
    else:
        ax.set_xticks([])

    if y:
        ax.set_yticks(
            [
                tcoll["image"]["binning"]["image"]["center"]["cy_deg"]
                - 3 * grid_angle_deg,
                tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                tcoll["image"]["binning"]["image"]["center"]["cy_deg"]
                + 3 * grid_angle_deg,
            ]
        )
    else:
        ax.set_yticks([])


# estimate vmax
# -------------
CMAP_VMAX = 0.0
for isens, pkey in enumerate(coll):
    for iiofa, ofa in enumerate(OFFAXIS_ANGLE_IDXS):
        akey = abe.offaxis.ANGLE_FMT.format(ofa)
        tcoll = coll[pkey][akey]
        norm_image = (
            tcoll["image"]["raw"] / tcoll["statistics"]["photons"]["valid"]
        )
        vmax = np.max(norm_image)
        CMAP_VMAX = np.max([CMAP_VMAX, vmax])


for cmapkey in CMAPS:
    num_sensors = len(coll)
    num_offaxis = len(OFFAXIS_ANGLE_IDXS)

    cmap_psf = None
    for isens, pkey in enumerate(coll):

        fig_psf = sebplt.figure(
            style={"rows": 440, "cols": 1280, "fontsize": 1.0}
        )

        for iiofa, ofa in enumerate(OFFAXIS_ANGLE_IDXS):
            akey = abe.offaxis.ANGLE_FMT.format(ofa)

            tcoll = coll[pkey][akey]
            (
                bin_edges_cx,
                bin_edges_cy,
            ) = abe.offaxis.analysis.binning_image_bin_edges(
                binning=tcoll["image"]["binning"]
            )
            bin_edges_cx_deg = np.rad2deg(bin_edges_cx)
            bin_edges_cy_deg = np.rad2deg(bin_edges_cy)
            (ticks_cx_deg, ticks_cy_deg,) = make_explicit_cx_cy_ticks(
                tcoll=tcoll, tick_angle=GRID_ANGLE_DEG
            )

            ax_psf_width = 0.85 / num_offaxis
            ax_psf = sebplt.add_axes(
                fig=fig_psf,
                span=[
                    0.1 + iiofa * (ax_psf_width + 0.02),
                    0.15,
                    ax_psf_width,
                    0.8,
                ],
            )
            ax_psf.set_aspect("equal")
            # ax_psf.tick_params(direction="in")
            norm_image = (
                tcoll["image"]["raw"] / tcoll["statistics"]["photons"]["valid"]
            )

            cmap_psf = ax_psf.pcolormesh(
                bin_edges_cx_deg,
                bin_edges_cy_deg,
                np.transpose(norm_image) / CMAP_VMAX,
                cmap=cmapkey,
                norm=sebplt.plt_colors.PowerNorm(
                    gamma=CMAPS[cmapkey]["gamma"], vmin=0.0, vmax=1.0,
                ),
            )
            sebplt.ax_add_grid_with_explicit_ticks(
                xticks=ticks_cx_deg,
                yticks=ticks_cy_deg,
                ax=ax_psf,
                color=CMAPS[cmapkey]["linecolor"],
                linestyle="-",
                linewidth=0.33,
                alpha=0.33,
            )
            sebplt.ax_add_circle(
                ax=ax_psf,
                x=tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                y=tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                r=np.rad2deg(tcoll["image"]["angle80"]),
                linewidth=1.0,
                linestyle="--",
                color=CMAPS[cmapkey]["linecolor"],
                alpha=0.5,
                num_steps=360,
            )
            ax_psf_set_ticks(
                ax=ax_psf,
                tcoll=tcoll,
                grid_angle_deg=GRID_ANGLE_DEG,
                x=True,
                y=iiofa == 0,
            )
            ax_psf_add_eye(
                ax=ax_psf,
                tcoll=tcoll,
                bin_edges_cx_deg=bin_edges_cx_deg,
                bin_edges_cy_deg=bin_edges_cy_deg,
                linecolor=CMAPS[cmapkey]["linecolor"],
            )

        fig_psf_name = "psf_{:s}_{:s}_{:s}_images.jpg".format(
            mkey, pkey, cmapkey
        )
        fig_psf.savefig(os.path.join(out_dir, fig_psf_name))
        sebplt.close(fig_psf)

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
    fig_cmap_filename = "psf_{:s}_{:s}_cmap_.jpg".format(mkey, cmapkey)
    fig_psf_cmap.savefig(os.path.join(out_dir, fig_cmap_filename))
    sebplt.close(fig_psf_cmap)


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

    fig_tsf.savefig(
        os.path.join(out_dir, "tsf_{:s}_{:s}_hists.jpg".format(mkey, pkey))
    )
    sebplt.close(fig_tsf)
