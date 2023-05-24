#!/usr/bin/python
import os
import plenoirf
import numpy as np
import aberration_demo as abe
import json_numpy
import sebastians_matplotlib_addons as sebplt
import argparse

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--work_dir", type=str)
argparser.add_argument("--out_dir", type=str)
argparser.add_argument("--instrument_key", type=str)
argparser.add_argument("--star_key", type=str)
argparser.add_argument("--vmax", type=float)

args = argparser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
instrument_key = args.instrument_key
star_key = args.star_key
cmap_vmax = args.vmax

os.makedirs(out_dir, exist_ok=True)

config = json_numpy.read_tree(os.path.join(work_dir, "config"))
instrument_sensor_key = config["instruments"][instrument_key]["sensor"]

GRID_ANGLE_DEG = 0.1
CMAPS = abe.plot.CMAPS

point_source_report = json_numpy.read(
    os.path.join(
        work_dir, "analysis", instrument_key, "star", star_key + ".json"
    )
)

for cmap_key in CMAPS:
    cmap_dir = os.path.join(out_dir, cmap_key)
    os.makedirs(cmap_dir, exist_ok=True)

    fig_filename = "instrument_{:s}_star_{:s}_cmap_{:s}.jpg".format(
        instrument_key, star_key, cmap_key,
    )
    fig_path = os.path.join(cmap_dir, fig_filename)

    if os.path.exists(fig_path):
        continue

    fig_psf = sebplt.figure(style={"rows": 640, "cols": 640, "fontsize": 1.0})

    (bin_edges_cx, bin_edges_cy,) = abe.analysis.image.binning_image_bin_edges(
        binning=point_source_report["image"]["binning"]
    )
    bin_edges_cx_deg = np.rad2deg(bin_edges_cx)
    bin_edges_cy_deg = np.rad2deg(bin_edges_cy)
    (ticks_cx_deg, ticks_cy_deg,) = abe.plot.make_explicit_cx_cy_ticks(
        image_response=point_source_report, tick_angle=GRID_ANGLE_DEG
    )

    ax_psf = sebplt.add_axes(fig=fig_psf, span=[0.15, 0.15, 0.85, 0.85],)
    ax_psf.set_aspect("equal")

    image_response_norm = abe.analysis.point_source_report.make_norm_image(
        point_source_report=point_source_report
    )

    cmap_psf = ax_psf.pcolormesh(
        bin_edges_cx_deg,
        bin_edges_cy_deg,
        np.transpose(image_response_norm) / cmap_vmax,
        cmap=cmap_key,
        norm=sebplt.plt_colors.PowerNorm(
            gamma=CMAPS[cmap_key]["gamma"], vmin=0.0, vmax=1.0,
        ),
    )
    sebplt.ax_add_grid_with_explicit_ticks(
        xticks=ticks_cx_deg,
        yticks=ticks_cy_deg,
        ax=ax_psf,
        color=CMAPS[cmap_key]["linecolor"],
        linestyle="-",
        linewidth=0.33,
        alpha=0.33,
    )
    sebplt.ax_add_circle(
        ax=ax_psf,
        x=point_source_report["image"]["binning"]["image"]["center"]["cx_deg"],
        y=point_source_report["image"]["binning"]["image"]["center"]["cy_deg"],
        r=np.rad2deg(point_source_report["image"]["angle80"]),
        linewidth=1.0,
        linestyle="--",
        color=CMAPS[cmap_key]["linecolor"],
        alpha=0.5,
        num_steps=360,
    )
    sebplt.ax_add_circle(
        ax=ax_psf,
        x=0.0,
        y=0.0,
        r=0.5
        * config["sensors"][instrument_sensor_key]["max_FoV_diameter_deg"],
        linewidth=1.0,
        linestyle="-",
        color=CMAPS[cmap_key]["linecolor"],
        alpha=0.5,
        num_steps=360 * 5,
    )
    abe.plot.ax_psf_set_ticks(
        ax=ax_psf,
        image_response=point_source_report,
        grid_angle_deg=GRID_ANGLE_DEG,
        x=True,
        y=True,
    )
    abe.plot.ax_psf_add_eye(
        ax=ax_psf,
        image_response=point_source_report,
        bin_edges_cx_deg=bin_edges_cx_deg,
        bin_edges_cy_deg=bin_edges_cy_deg,
        linecolor=CMAPS[cmap_key]["linecolor"],
        eye_FoV_flat2flat_deg=config["sensors"][instrument_sensor_key][
            "hex_pixel_FoV_flat2flat_deg"
        ],
    )
    ccx_deg = point_source_report["image"]["binning"]["image"]["center"][
        "cx_deg"
    ]
    ccy_deg = point_source_report["image"]["binning"]["image"]["center"][
        "cy_deg"
    ]
    ccxr_deg = 0.5 * (
        point_source_report["image"]["binning"]["image"]["pixel_angle_deg"]
        * point_source_report["image"]["binning"]["image"]["num_pixel_cx"]
    )
    ccyr_deg = 0.5 * (
        point_source_report["image"]["binning"]["image"]["pixel_angle_deg"]
        * point_source_report["image"]["binning"]["image"]["num_pixel_cy"]
    )
    ax_psf.set_xlim([ccx_deg - ccxr_deg, ccx_deg + ccxr_deg])
    ax_psf.set_ylim([ccy_deg - ccyr_deg, ccy_deg + ccyr_deg])

    fig_psf.savefig(fig_path)
    sebplt.close(fig_psf)
