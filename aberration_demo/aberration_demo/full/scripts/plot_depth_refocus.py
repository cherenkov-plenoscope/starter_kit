#!/usr/bin/python
import pandas
import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as sebplt
import binning_utils
import confusion_matrix
import plenopy
import plenoirf
import argparse

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--work_dir", type=str)
argparser.add_argument("--out_dir", type=str)
argparser.add_argument("--instrument_key", type=str)

args = argparser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir
instrument_key = args.instrument_key

os.makedirs(out_dir, exist_ok=True)

config = json_numpy.read_tree(os.path.join(work_dir, "config"))
_point_reports = json_numpy.read(
    os.path.join(work_dir, "analysis", instrument_key, "point.json")
)

pixel_pitch_deg = (
    config["analysis"]["point"]["field_of_view_deg"]
    / config["analysis"]["point"]["num_pixel_on_edge"]
)
solid_angle_per_px_sr = np.deg2rad(pixel_pitch_deg) ** 2

instrument_field_of_view_half_angle_deg = 3.25

# rm points far out in the fov
# ----------------------------
point_reports = {}
for point_key in _point_reports:
    report = _point_reports[point_key]
    cc_deg = np.hypot(report["cx_deg"], report["cy_deg"])
    if cc_deg < (3 / 4) * instrument_field_of_view_half_angle_deg:
        point_reports[point_key] = report

# make samples
# ------------
NUM_SAMPLES = 5
samples_depth_m = np.geomspace(
    config["observations"]["point"]["min_object_distance_m"],
    config["observations"]["point"]["max_object_distance_m"],
    2 + NUM_SAMPLES,
)
samples_depth_m = samples_depth_m[1:-1]

# find closest matches
# --------------------
point_depths_m = []
point_keys = []
for point_key in point_reports:
    point_keys.append(point_key)
    point_depths_m.append(point_reports[point_key]["object_distance_m"])
point_depths_m = np.array(point_depths_m)

sample_point_keys = []
for sample_depth_m in samples_depth_m:
    amin = np.argmin(np.abs(point_depths_m - sample_depth_m))
    sample_point_keys.append(point_keys[amin])


ooo = {}
# prepare point reports
# ---------------------
for point_key in sample_point_keys:
    report = point_reports[point_key]

    _ssort = np.argsort(report["depth_m"])
    depth_m = np.array(report["depth_m"])[_ssort]
    _spread_raw = np.array(report["spreads_pixel_per_photon"])[_ssort]
    spread_px = _spread_raw * report["num_photons"]
    spread_usr = 1e6 * spread_px * solid_angle_per_px_sr

    ooo[point_key] = {
        "spread_usr": spread_usr,
        "depth_m": depth_m,
        "true_depth_m": report["object_distance_m"],
        "reco_depth_m": depth_m[np.argmin(spread_usr)],
    }

N = NUM_SAMPLES

ax_xlow = 0.175
ax_ylow = 0.11
ax_height = 0.75
ax_width = 0.82
axn_height = ax_height / N

fig = sebplt.figure({"rows": 300 * N, "cols": 1280, "fontsize": 1.5})

axylabel = sebplt.add_axes(
    fig=fig,
    span=[0.07, ax_ylow, ax_height, ax_width],
    style={"spines": [], "axes": ["y"], "grid": False},
)
axylabel.set_ylabel(r"solid angle containign 95% $\,/\,\mu$sr")
axylabel.set_yticks([])

ymin = 1e1
ymax = 1e2 * np.geomspace(1, 1e1, 3)[1]
my_yticks = np.geomspace(ymin, ymax, 4)
my_yticklabels = [""] * len(my_yticks)
my_yticklabels[0] = (
    "$"
    + plenoirf.utils.latex_scientific(
        my_yticks[0], format_template="{:.1e}", drop_mantisse_if_one=True
    )
    + "$"
)
my_yticklabels[2] = (
    "$"
    + plenoirf.utils.latex_scientific(
        my_yticks[2], format_template="{:.1e}", drop_mantisse_if_one=True
    )
    + "$"
)

for n, point_key in enumerate(ooo):
    uuu = ooo[point_key]
    spread_lim_usr = [np.min(uuu["spread_usr"]), np.max(uuu["spread_usr"])]
    axn = sebplt.add_axes(
        fig=fig,
        span=[
            ax_xlow,
            ax_ylow + (axn_height + 0.03) * n,
            ax_width,
            axn_height,
        ],
    )
    sebplt.ax_add_grid(ax=axn, add_minor=True)
    axn.plot(uuu["depth_m"], uuu["spread_usr"], "ko", alpha=0.33, markersize=2)
    axn.plot(uuu["depth_m"], uuu["spread_usr"], "k-", linewidth=1)
    axn.plot(
        [uuu["true_depth_m"], uuu["true_depth_m"]],
        [ymin, ymax],
        "k--",
        linewidth=1,
        alpha=0.5,
    )

    axn.loglog()
    if n == 0:
        axn.set_xlabel(r"depth$\,/\,$m")
    else:
        axn.set_xticklabels([""] * len(axn.get_xticklabels()))

    axn.set_yticks(my_yticks)
    axn.set_yticklabels(my_yticklabels)
    axn.set_ylim([ymin, ymax])


fig.savefig(os.path.join(out_dir, "refocus_spread_five_samples.jpg"))
sebplt.close(fig)
