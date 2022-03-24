import corsika_primary as cpw
import os
import plenoirf
import numpy as np
import plenopy
import scipy
from scipy import spatial
from scipy import stats
import aberration_demo
import json_numpy
import sebastians_matplotlib_addons as sebplt
import sys

sebplt.matplotlib.rcParams.update(plenoirf.summary.figure.MATPLOTLIB_RCPARAMS)

argv = sys.argv
if argv[0] == "ipython" and argv[1] == "-i":
    argv.pop(1)

work_dir = argv[1]

with open(os.path.join(work_dir, "config.json"), "rt") as f:
    config = json_numpy.loads(f.read())


coll = aberration_demo.read_analysis(work_dir)

# summary plot of poin-spread-functions
# -------------------------------------

GRID_ANGLE_DEG = 0.2

def ax_add_grid(ax, xticks, yticks, color="k", linestyle="-", linewidth=0.66, alpha=0.33):
    for ytick in yticks:
        ax.axhline(
            y=ytick,
            xmin=0,
            xmax=1,
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha
        )
    for xtick in xticks:
        ax.axvline(
            x=xtick,
            ymin=0,
            ymax=1,
            color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha
        )


def make_grid_ticks(center, num_pixel, pixel_angel, tick_angle):
    extent = (pixel_angel*num_pixel)
    num_ticks = int(np.ceil(extent / tick_angle))
    tick_extent = num_ticks * tick_angle

    tick_start = center - 0.5*tick_extent
    tick_stop = center + 0.5*tick_extent
    ticks = np.linspace(tick_start, tick_stop, num_ticks + 1)
    return ticks


num_mirrors = len(coll)
for mkey in coll:
    num_sensors = len(coll[mkey])
    for paxkey in coll[mkey]:
        num_offaxis = len(coll[mkey][paxkey])

    ax_hori_start = 0.1
    ax_vert_start = 0.1
    ax_width_rel = (1.0 - ax_hori_start) / num_offaxis
    ax_height_rel = (1.0 - ax_vert_start) / num_sensors
    ax_panel_width_rel = ax_width_rel * 0.9
    ax_panel_height_rel = ax_height_rel * 0.9
    ax_margin_width_rel = ax_width_rel - ax_panel_width_rel
    ax_margin_height_rel = ax_height_rel - ax_panel_height_rel


    fig_psf = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1})
    ax_psf_labels = sebplt.add_axes(
        fig=fig_psf,
        span=[0,0,1,1],
        style={"spines": [], "axes": [], "grid": False},
    )
    ax_psf_labels.text(0.02, 0.55, r"$c_y$ / 1$^\circ$", rotation=90, fontsize=12)
    ax_psf_labels.text(0.55, 0.02, r"$c_x$ / 1$^\circ$", rotation=0, fontsize=12)

    for isens, paxkey in enumerate(coll[mkey]):
        for iofa, ofakey in enumerate(coll[mkey][paxkey]):

            tcoll = coll[mkey][paxkey][ofakey]
            bin_edges_cx, bin_edges_cy = aberration_demo.analysis.binning_image_bin_edges(
                binning=tcoll["image"]["binning"]
            )
            bin_edges_cx_deg = np.rad2deg(bin_edges_cx)
            bin_edges_cy_deg = np.rad2deg(bin_edges_cy)

            ticks_cx_deg = make_grid_ticks(
                center=tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                num_pixel=tcoll["image"]["binning"]["image"]["num_pixel_cx"],
                pixel_angel=tcoll["image"]["binning"]["image"]["pixel_angle_deg"],
                tick_angle=GRID_ANGLE_DEG,
            )
            ticks_cy_deg = make_grid_ticks(
                center=tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                num_pixel=tcoll["image"]["binning"]["image"]["num_pixel_cy"],
                pixel_angel=tcoll["image"]["binning"]["image"]["pixel_angle_deg"],
                tick_angle=GRID_ANGLE_DEG,
            )

            ax_pax_off = sebplt.add_axes(
                fig=fig_psf,
                span=[
                    ax_hori_start + ax_margin_width_rel + ax_width_rel * iofa,
                    ax_vert_start + ax_margin_height_rel + ax_height_rel * (num_sensors - isens - 1),
                    ax_panel_width_rel,
                    ax_panel_height_rel,
                ]
            )

            img_raw_norm = tcoll["image"]["raw"] / tcoll["statistics"]["photons"]["valid"]
            cmap_psf = ax_pax_off.pcolormesh(
                bin_edges_cx_deg,
                bin_edges_cy_deg,
                np.transpose(tcoll["image"]["raw"] ),
                cmap="Greys",
                norm=sebplt.plt_colors.PowerNorm(gamma=0.5),
            )
            ax_add_grid(
                xticks=ticks_cx_deg,
                yticks=ticks_cy_deg,
                ax=ax_pax_off,
                color="k",
                linestyle="-",
                linewidth=0.66,
                alpha=0.33,
            )
            sebplt.ax_add_circle(
                ax=ax_pax_off,
                x=tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                y=tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                r=np.rad2deg(tcoll["image"]["angle80"]),
                linewidth=1.5,
                linestyle='--',
                color='w',
                alpha=1,
                num_steps=360,
            )
            ax_pax_off.set_aspect("equal")

            if isens == num_sensors - 1:
                ax_pax_off.set_xticks([
                    tcoll["image"]["binning"]["image"]["center"]["cx_deg"] - 3 * GRID_ANGLE_DEG,
                    tcoll["image"]["binning"]["image"]["center"]["cx_deg"],
                    tcoll["image"]["binning"]["image"]["center"]["cx_deg"] + 3 * GRID_ANGLE_DEG,
                ])
            else:
                ax_pax_off.set_xticks([])

            if iofa == 0:
                ax_pax_off.set_yticks([
                    tcoll["image"]["binning"]["image"]["center"]["cy_deg"] - 3 * GRID_ANGLE_DEG,
                    tcoll["image"]["binning"]["image"]["center"]["cy_deg"],
                    tcoll["image"]["binning"]["image"]["center"]["cy_deg"] + 3 * GRID_ANGLE_DEG,
                ])
            else:
                ax_pax_off.set_yticks([])

            if True:
                sebplt.ax_add_circle(
                    ax=ax_pax_off,
                    x=np.mean([tcoll["image"]["binning"]["image"]["center"]["cx_deg"], bin_edges_cx_deg[0]]),
                    y=np.mean([tcoll["image"]["binning"]["image"]["center"]["cy_deg"], bin_edges_cy_deg[0]]),
                    r=config["sensor"]["hex_pixel_fov_flat2flat_deg"]*0.5 * 2/np.sqrt(3),
                    linewidth=0.75,
                    linestyle='-',
                    color='k',
                    alpha=1,
                    num_steps=7,
                )


    fig_psf.savefig(os.path.join(work_dir, "plot", "psf_" + mkey + "_images.jpg"))
    sebplt.close(fig_psf)

    fig_psf_cmap = sebplt.figure(style={"rows": 160, "cols": 1280, "fontsize": 1})
    ax_cmap = sebplt.add_axes(fig_psf_cmap, [0.1, 0.3, 0.8, 0.5])
    sebplt.plt.colorbar(cmap_psf, cax=ax_cmap, extend="max", orientation='horizontal')
    fig_psf_cmap.savefig(os.path.join(work_dir, "plot", "psf_" + mkey + "_cmap.jpg"))
    sebplt.close(fig_psf_cmap)




# summary plot of time-spread-functions
# -------------------------------------


time_start_ns = -25
time_stop_ns = 25
time_weight_start_perc = 0
time_weight_stop_perc = 10

time_grid_yticks = np.linspace(time_weight_start_perc, time_weight_stop_perc, 5)
time_yticks = np.array([0, 5])
time_xticks_ns = np.linspace(time_start_ns, time_stop_ns, 11)
for mkey in coll:
    fig_tsf = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1})
    ax_tsf_labels = sebplt.add_axes(
        fig=fig_tsf,
        span=[0,0,1,1],
        style={"spines": [], "axes": [], "grid": False},
    )
    ax_tsf_labels.text(0.06, 0.45, r"rel. intensity / %", rotation=90, fontsize=12)
    ax_tsf_labels.text(0.5, 0.05, r"time / ns", rotation=0, fontsize=12)


    for isens, paxkey in enumerate(coll[mkey]):
        num_paxel_on_diagonal = config["sensor"]["num_paxel_on_diagonal"][isens]

        ax_tsf_labels.text(
            0.01,
            0.1 + ax_vert_start + ax_margin_height_rel + ax_height_rel * (num_sensors - isens - 1),
            r"$N_D$ = " + "{:d}".format(num_paxel_on_diagonal),
            rotation=90,
            fontsize=10,
        )

        for iofa, ofakey in enumerate(coll[mkey][paxkey]):

            off_axis_cx_deg = config["sources"]["off_axis_angles_deg"][iofa][0]
            off_axis_cy_deg = config["sources"]["off_axis_angles_deg"][iofa][1]

            ax_tsf_labels.text(
                0.1 + ax_hori_start + ax_margin_width_rel + ax_width_rel * iofa,
                0.01,
                r"$c_x$ = " + "{:1.1f}".format(off_axis_cx_deg) + r"$^\circ$",
                rotation=0,
                fontsize=10,
            )


    for isens, paxkey in enumerate(coll[mkey]):
        for iofa, ofakey in enumerate(coll[mkey][paxkey]):
            tcoll = coll[mkey][paxkey][ofakey]


            ax_pax_off = sebplt.add_axes(
                fig=fig_tsf,
                span=[
                    ax_hori_start + ax_margin_width_rel + ax_width_rel * iofa,
                    ax_vert_start + ax_margin_height_rel + ax_height_rel * (num_sensors - isens - 1),
                    ax_panel_width_rel,
                    ax_panel_height_rel,
                ],
                style={"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}
            )
            time_center_ns = 0.5e9 * (tcoll["time"]["fwhm"]["stop"] + tcoll["time"]["fwhm"]["start"])
            time_weights_norm_perc = 100 * tcoll["time"]["weights"] / np.sum(tcoll["time"]["weights"])
            sebplt.ax_add_histogram(
                ax=ax_pax_off,
                bin_edges=1e9 * tcoll["time"]["bin_edges"] - time_center_ns,
                bincounts=time_weights_norm_perc,
                face_color="k",
                face_alpha=0.1,
                draw_bin_walls=True,
            )
            ax_pax_off.set_xlim([time_start_ns, time_stop_ns])
            ax_pax_off.set_ylim([time_weight_start_perc, time_weight_stop_perc])
            #ax_pax_off.semilogy()

            ax_add_grid(
                xticks=time_xticks_ns,
                yticks=time_grid_yticks,
                ax=ax_pax_off,
                color="k",
                linestyle="-",
                linewidth=0.66,
                alpha=0.33,
            )

            if isens == num_sensors - 1:
                pass #ax_pax_off.set_xticks(tcoll["time"]["bin_edges"])
            else:
                ax_pax_off.set_xticks([])

            if iofa == 0:
                ax_pax_off.set_yticks(time_yticks)
            else:
                ax_pax_off.set_yticks([])


    fig_tsf.savefig(os.path.join(work_dir, "plot", "tsf_" + mkey + "_hists.jpg"))
    sebplt.close(fig_tsf)