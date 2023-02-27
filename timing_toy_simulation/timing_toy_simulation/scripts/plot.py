#!/usr/bin/python
import sys
import plenoirf
import sparse_numeric_table as spt
import os
import numpy as np
import sebastians_matplotlib_addons as seb
import json_numpy


argv = plenoirf.summary.argv_since_py(sys.argv)
work_dir = argv[1]
plot_dir = os.path.join(work_dir, "plot")

with open(os.path.join(work_dir, "config.json"), "rt") as f:
    config = json_numpy.loads(f.read())

#seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(plot_dir, exist_ok=True)

result = spt.read(os.path.join(work_dir, "result.tar"))

assert config["flux"]["azimuth_deg"] == 0.0
assert config["flux"]["zenith_deg"] == 0.0

MIN_NUM_DETECTED_PHOTONS = 15


mask_az = np.logical_and(
    result["base"]["primary_azimuth_rad"] >= -np.deg2rad(360),
    result["base"]["primary_azimuth_rad"] < np.deg2rad(360),
)
idx_az = result["base"]["idx"][mask_az]

mask_inst_x = np.logical_and(
    result["base"]["instrument_x_m"] >= -650,
    result["base"]["instrument_x_m"] < 650,
)
idx_inst_x = result["base"]["idx"][mask_inst_x]


mask_valid = (
    result["cherenkov_detected_size"]["num_photons"] >= MIN_NUM_DETECTED_PHOTONS
)
idx_valid = result["cherenkov_detected_size"]["idx"][mask_valid]


idx_valid = spt.intersection([idx_valid, idx_az, idx_inst_x])


num_valid = idx_valid.shape[0]


zd_num_bins = int(np.floor(num_valid))

zd_bin_edges_rad = plenoirf.utils.cone_opening_angle_space(
    stop_cone_radial_opening_angle_rad=np.deg2rad(config["flux"]["radial_angle_deg"]),
    num=zd_num_bins,
)

zd_bin_edges_deg = np.rad2deg(zd_bin_edges_rad)

res = spt.cut_and_sort_table_on_indices(
    table=result,
    common_indices=idx_valid,
    level_keys=["base", "reconstruction"],
)

r = spt.make_rectangular_DataFrame(res).to_records()

time_true = r["base/primary_time_to_closest_point_to_instrument_s"]
time_reco = r["reconstruction/arrival_time_median_s"]
time_delta = time_reco - time_true

fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(
    np.rad2deg(r["base/primary_zenith_rad"]),
    time_delta * 1e9,
    "xk"
)
ax.set_xlabel("zenith / deg")
ax.set_ylabel("time_delta / ns")

fig.savefig(
    os.path.join(plot_dir, "zd_time.jpg")
)
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(
    np.rad2deg(r["base/primary_azimuth_rad"]),
    time_delta * 1e9,
    "xk"
)
ax.set_xlabel("azimuth / deg")
ax.set_ylabel("time_delta / ns")

fig.savefig(
    os.path.join(plot_dir, "az_time.jpg")
)
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(
    r["base/instrument_x_m"],
    time_delta * 1e9,
    "xk"
)
ax.set_xlabel("instrument_x / m")
ax.set_ylabel("time_delta / ns")

fig.savefig(
    os.path.join(plot_dir, "ins_x_time.jpg")
)
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(
    r["base/instrument_y_m"],
    time_delta * 1e9,
    "xk"
)
ax.set_xlabel("instrument_y / m")
ax.set_ylabel("time_delta / ns")

fig.savefig(
    os.path.join(plot_dir, "ins_y_time.jpg")
)
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(
    r["base/instrument_x_m"],
    r["base/instrument_y_m"],
    "xk"
)
ax.set_xlabel("instrument_x / m")
ax.set_ylabel("instrument_y / m")

fig.savefig(
    os.path.join(plot_dir, "ins_x_y.jpg")
)
seb.close(fig)



fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
seb.hemisphere.ax_add_points(
    ax=ax,
    azimuths_deg=np.rad2deg(r["base/primary_azimuth_rad"]),
    zeniths_deg=np.rad2deg(r["base/primary_zenith_rad"]),
    color="k",
    point_diameter_deg=2.0,
    alpha=0.3,
)
ax.set_xlabel("sky x / m")
ax.set_ylabel("sky y / m")
ax.set_xlim([-1,1])
ax.set_ylim([-1,1])
fig.savefig(
    os.path.join(plot_dir, "az_zd_hemisphere.jpg")
)
seb.close(fig)