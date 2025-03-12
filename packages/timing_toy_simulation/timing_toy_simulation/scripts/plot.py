#!/usr/bin/python
import sys
import plenoirf
import sparse_numeric_table as snt
import os
import numpy as np
import sebastians_matplotlib_addons as seb
import json_utils
import binning_utils as bu
import solid_angle_utils


argv = plenoirf.summary.argv_since_py(sys.argv)
work_dir = argv[1]
plot_dir = os.path.join(work_dir, "plot")

with open(os.path.join(work_dir, "config.json"), "rt") as f:
    config = json_utils.loads(f.read())

# seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])
prng = np.random.Generator(np.random.PCG64(64))
os.makedirs(plot_dir, exist_ok=True)

result = snt.read(os.path.join(work_dir, "result.tar"))

assert config["flux"]["azimuth_deg"] == 0.0
assert config["flux"]["zenith_deg"] == 0.0

MIN_NUM_DETECTED_PHOTONS = 25


mask_az = np.logical_and(
    result["base"]["primary_azimuth_rad"] >= -np.deg2rad(360),
    result["base"]["primary_azimuth_rad"] < np.deg2rad(360),
)
idx_az = result["base"]["uid"][mask_az]

mask_inst_x = np.logical_and(
    result["base"]["instrument_x_m"] >= -650,
    result["base"]["instrument_x_m"] < 650,
)
idx_inst_x = result["base"]["uid"][mask_inst_x]


mask_valid = (
    result["cherenkov_detected_size"]["num_photons"]
    >= MIN_NUM_DETECTED_PHOTONS
)
idx_valid = result["cherenkov_detected_size"]["uid"][mask_valid]


idx_valid = snt.intersection([idx_valid, idx_az, idx_inst_x])


num_valid = idx_valid.shape[0]


zd_num_bins = int(np.floor(num_valid**0.3))

zd_bin_edges_rad = solid_angle_utils.cone.half_angle_space(
    stop_half_angle_rad=np.deg2rad(config["flux"]["radial_angle_deg"]),
    num=zd_num_bins,
)

zd_bin_edges_deg = np.rad2deg(zd_bin_edges_rad)

res = snt.cut_and_sort_table_on_indices(
    table=result,
    common_indices=idx_valid,
    level_keys=["base", "reconstruction"],
)

r = snt.make_rectangular_DataFrame(res).to_records()

time_true = r["base/primary_time_to_closest_point_to_instrument_s"]
time_reco = r["reconstruction/arrival_time_median_s"]
time_delta = time_reco - time_true


# binning
# -------
zenith_bin = bu.Binning(
    bin_edges=solid_angle_utils.cone.half_angle_space(
        stop_half_angle_rad=np.deg2rad(config["flux"]["radial_angle_deg"]),
        num=zd_num_bins,
    )
)
time_delta_bin = bu.Binning(bin_edges=np.linspace(0.0, 100e-9, zd_num_bins))

# simple hist
# -----------
z_bin = bu.Binning(
    bin_edges=solid_angle_utils.cone.half_angle_space(
        stop_half_angle_rad=np.deg2rad(config["flux"]["radial_angle_deg"]),
        num=5,
    )
)
zz_range = np.arange(z_bin["num"] - 1, -1, -1)
time_delta_hists = np.zeros(shape=(z_bin["num"], time_delta_bin["num"]))
time_delta_stddevs = np.zeros(z_bin["num"])
for zz in zz_range:
    z_start = z_bin["edges"][zz]
    z_stop = z_bin["edges"][zz + 1]
    z_mask = np.logical_and(
        r["base/primary_zenith_rad"] >= z_start,
        r["base/primary_zenith_rad"] < z_stop,
    )
    _th = np.histogram(time_delta[z_mask], bins=time_delta_bin["edges"])[0]
    time_delta_hists[zz] = _th
    _ts = np.std(time_delta[z_mask])
    time_delta_stddevs[zz] = _ts
intensity_max = np.max(time_delta_hists)


fig = seb.figure(style={"rows": 1920, "cols": 1080, "fontsize": 1.5})
fig.text(
    x=0.1,
    y=0.97,
    s="gamma-ray energy: {:.1f}GeV - {:.1f}GeV".format(
        config["particle"]["energy_range"]["start_GeV"],
        config["particle"]["energy_range"]["stop_GeV"],
    ),
)
for zz in zz_range:
    z_start = z_bin["edges"][zz]
    z_stop = z_bin["edges"][zz + 1]

    ax_height = 0.85 / z_bin["num"]
    ax = seb.add_axes(
        fig=fig, span=[0.2, 0.1 + zz * ax_height, 0.7, 0.8 * ax_height]
    )
    seb.ax_add_histogram(
        ax=ax,
        bin_edges=(1e9 * time_delta_bin["edges"]),
        bincounts=time_delta_hists[zz],
        linestyle="-",
        linecolor="k",
        linealpha=1.0,
        bincounts_upper=None,
        bincounts_lower=None,
        face_color=None,
        face_alpha=None,
        label=None,
        draw_bin_walls=True,
    )
    ax.set_ylim([1, 1.1 * intensity_max])
    ax.semilogy()
    if zz > 0:
        ax.set_xticklabels([t.set_text("") for t in ax.get_xticklabels()])
    ttt = "zenith: "
    ttt += "{:.1f}".format(np.rad2deg(z_start))
    ttt += "$^{\circ}$"
    ttt += " - "
    ttt += "{:.1f}".format(np.rad2deg(z_stop))
    ttt += "$^{\circ}$"
    ax.text(x=0.3, y=0.9, s=ttt, transform=ax.transAxes)

    ttt = "std: {:.1f}ns".format(1e9 * time_delta_stddevs[zz])
    ax.text(x=0.5, y=0.6, s=ttt, transform=ax.transAxes)


ax.set_ylabel("intensity / 1")
ax.set_xlabel("time (reco - true) / ns")
fig.savefig(os.path.join(plot_dir, "time_reco_minus_true.jpg"))
seb.close(fig)


# zenith
# ------

time_delta_zenith_hist = np.histogram2d(
    r["base/primary_zenith_rad"],
    time_delta,
    bins=(zenith_bin["edges"], time_delta_bin["edges"]),
)[0]

fig = seb.figure(style={"rows": 1080, "cols": 1920, "fontsize": 1.5})
ax = seb.add_axes(fig=fig, span=[0.15, 0.15, 0.5, 0.8])
axc = seb.add_axes(fig=fig, span=[0.7, 0.15, 0.05, 0.8])
_pcm_xy = ax.pcolormesh(
    np.rad2deg(zenith_bin["edges"]),
    1e9 * time_delta_bin["edges"],
    np.transpose(time_delta_zenith_hist),
    cmap="inferno",
    norm=seb.plt_colors.PowerNorm(gamma=0.5),
)
seb.plt.colorbar(_pcm_xy, cax=axc, extend="max")
ax.grid(color="w", linestyle="-", linewidth=0.66, alpha=0.1)
# ax.set_aspect("equal")
ax.set_xlim(np.rad2deg(zenith_bin["limits"]))
ax.set_ylim(1e9 * time_delta_bin["limits"])
ax.set_xlabel("zenith / 1$^{\circ}$")
ax.set_ylabel("time (reco - true) / ns")
fig.savefig(os.path.join(plot_dir, "zd_time.jpg"))
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(np.rad2deg(r["base/primary_azimuth_rad"]), time_delta * 1e9, "xk")
ax.set_xlabel("azimuth / deg")
ax.set_ylabel("time_delta / ns")

fig.savefig(os.path.join(plot_dir, "az_time.jpg"))
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(r["base/instrument_x_m"], time_delta * 1e9, "xk")
ax.set_xlabel("instrument_x / m")
ax.set_ylabel("time_delta / ns")

fig.savefig(os.path.join(plot_dir, "ins_x_time.jpg"))
seb.close(fig)


fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
ax.plot(r["base/instrument_y_m"], time_delta * 1e9, "xk")
ax.set_xlabel("instrument_y / m")
ax.set_ylabel("time_delta / ns")

fig.savefig(os.path.join(plot_dir, "ins_y_time.jpg"))
seb.close(fig)


# x, y
# ----
xy_bin = bu.Binning(
    bin_edges=np.linspace(
        -config["scatter"]["position"]["radius_m"],
        config["scatter"]["position"]["radius_m"],
        int(zd_num_bins),
    )
)
xy_hist = np.histogram2d(
    r["base/instrument_x_m"],
    r["base/instrument_y_m"],
    bins=xy_bin["edges"],
)[0]

fig = seb.figure(style={"rows": 1080, "cols": 1920, "fontsize": 1.5})
ax = seb.add_axes(fig=fig, span=[0.15, 0.15, 0.8, 0.8])
axc = seb.add_axes(fig=fig, span=[0.85, 0.15, 0.05, 0.8])
_pcm_xy = ax.pcolormesh(
    xy_bin["edges"],
    xy_bin["edges"],
    np.transpose(xy_hist),
    cmap="inferno",
    norm=seb.plt_colors.PowerNorm(gamma=0.5),
)
seb.plt.colorbar(_pcm_xy, cax=axc, extend="max")
ax.grid(color="w", linestyle="-", linewidth=0.66, alpha=0.1)
ax.set_aspect("equal")
ax.set_xlim(xy_bin["limits"])
ax.set_ylim(xy_bin["limits"])
ax.set_xlabel("instrument $x$ / m")
ax.set_ylabel("instrument $y$ / m")
fig.savefig(os.path.join(plot_dir, "ins_x_y.jpg"))
seb.close(fig)

# directions in sky
# -----------------
az_lines_deg = np.linspace(0, 360, 6, endpoint=False)
num_points = min([int(1e3), r.shape[0]])
fig = seb.figure(style=seb.FIGURE_1_1)
ax = seb.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8], style=seb.AXES_BLANK)
seb.hemisphere.ax_add_projected_points_with_colors(
    ax=ax,
    azimuths_rad=r["base/primary_azimuth_rad"][0:num_points],
    zeniths_rad=r["base/primary_zenith_rad"][0:num_points],
    color="k",
    half_angle_rad=np.deg2rad(2.0),
    alpha=0.3,
)
seb.hemisphere.ax_add_grid_stellarium_style(ax=ax)
ax.set_xlabel("sky x / m")
ax.set_ylabel("sky y / m")
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
fig.savefig(os.path.join(plot_dir, "az_zd_hemisphere.jpg"))
seb.close(fig)
