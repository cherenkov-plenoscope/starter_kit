#!/usr/bin/python
import sys
import pandas
import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as seb
import binning_utils
import confusion_matrix
import plenopy as pl
import plenoirf as irf

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

res_dir = os.path.join(pa["run_dir"], "benchmarks", "resolution_of_depth")

config = json_numpy.read(os.path.join(res_dir, "config.json"))
result = json_numpy.read(os.path.join(res_dir, "result.json"))

# properties of plenoscope
# ------------------------
lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

plenoscope = {}
plenoscope[
    "focal_length_m"
] = lfg.sensor_plane2imaging_system.expected_imaging_system_focal_length
plenoscope["mirror_diameter_m"] = (
    2
    * lfg.sensor_plane2imaging_system.expected_imaging_system_max_aperture_radius
)
plenoscope["diameter_of_pixel_projected_on_sensor_plane_m"] = (
    np.tan(lfg.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat)
    * plenoscope["focal_length_m"]
)

num_paxel_on_diagonal = (
    lfg.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal
)

paxelscope = {}
for kk in plenoscope:
    paxelscope[kk] = plenoscope[kk] / num_paxel_on_diagonal


# prepare results
# ---------------
res = []
for estimate in result:
    e = {}
    for key in ["cx_deg", "cy_deg", "object_distance_m", "num_photons"]:
        e[key] = estimate[key]
    afocus = np.argmin(estimate["spreads_pixel_per_photon"])
    e["reco_object_distance_m"] = estimate["depth_m"][afocus]
    e["spread_pixel_per_photon"] = estimate["spreads_pixel_per_photon"][afocus]
    res.append(e)

res = pandas.DataFrame(res).to_records()

systematic_reco_over_true = np.median(
    res["reco_object_distance_m"] / res["object_distance_m"]
)
res["reco_object_distance_m"] /= systematic_reco_over_true

# setup binning
# -------------
depth_bin = binning_utils.Binning(
    bin_edges=np.geomspace(
        0.75 * config["min_object_distance_m"],
        1.25 * config["max_object_distance_m"],
        129,
    ),
)
min_number_samples = 1

cm = confusion_matrix.init(
    ax0_key="true_depth_m",
    ax0_values=res["object_distance_m"],
    ax0_bin_edges=depth_bin["edges"],
    ax1_key="reco_depth_m",
    ax1_values=res["reco_object_distance_m"],
    ax1_bin_edges=depth_bin["edges"],
    min_exposure_ax0=min_number_samples,
    default_low_exposure=0.0,
)

# theory curve
# ------------
theory_depth_m = depth_bin["edges"]
theory_depth_minus_m = []
theory_depth_plus_m = []
for g in theory_depth_m:
    g_p, g_m = pl.thin_lens.resolution_of_depth(
        object_distance_m=g, **plenoscope,
    )
    theory_depth_minus_m.append(g_m)
    theory_depth_plus_m.append(g_p)
theory_depth_minus_m = np.array(theory_depth_minus_m)
theory_depth_plus_m = np.array(theory_depth_plus_m)

theory_depth_paxel_minus_m = []
theory_depth_paxel_plus_m = []
for g in theory_depth_m:
    g_p, g_m = pl.thin_lens.resolution_of_depth(
        object_distance_m=g, **paxelscope,
    )
    theory_depth_paxel_minus_m.append(g_m)
    theory_depth_paxel_plus_m.append(g_p)
theory_depth_paxel_minus_m = np.array(theory_depth_paxel_minus_m)
theory_depth_paxel_plus_m = np.array(theory_depth_paxel_plus_m)

# plot
# ====

SCALE = 1e-3
xticks = [3, 10, 30]
xlabels = ["${:.0f}$".format(_x) for _x in xticks]

# statistics
fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax_h = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
seb.ax_add_grid(ax=ax_h, add_minor=True)
ax_h.semilogx()
ax_h.set_xlim(
    [np.min(cm["ax0_bin_edges"]) * SCALE, np.max(cm["ax1_bin_edges"]) * SCALE]
)
ax_h.set_xlabel(r"true depth$\,/\,$km")
ax_h.set_ylabel("statistics")
ax_h.axhline(cm["min_exposure_ax0"], linestyle=":", color="k")
seb.ax_add_histogram(
    ax=ax_h,
    bin_edges=cm["ax0_bin_edges"] * SCALE,
    bincounts=cm["exposure_ax0"],
    linestyle="-",
    linecolor="k",
)
ax_h.set_xticks(xticks)
ax_h.set_xticklabels(xlabels)
fig.savefig(os.path.join(pa["out_dir"], "depth_statistics.jpg"))
seb.close(fig)


# absolute
# --------

linewidth = 1.0
fig = seb.figure(style={"rows": 960, "cols": 1280, "fontsize": 1.0})
ax_c = seb.add_axes(fig=fig, span=[0.05, 0.14, 0.85, 0.85])
ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.14, 0.02, 0.85])

ax_c.plot(
    theory_depth_m * SCALE, theory_depth_m * SCALE, "k--", linewidth=linewidth
)
ax_c.plot(
    theory_depth_m * SCALE,
    theory_depth_minus_m * SCALE,
    "k:",
    linewidth=linewidth,
)
ax_c.plot(
    theory_depth_m * SCALE,
    theory_depth_plus_m * SCALE,
    "k:",
    linewidth=linewidth,
)

_pcm_confusion = ax_c.pcolormesh(
    cm["ax0_bin_edges"] * SCALE,
    cm["ax1_bin_edges"] * SCALE,
    np.transpose(cm["counts_normalized_on_ax0"]) * np.mean(cm["counts_ax0"]),
    cmap="Greys",
    norm=seb.plt_colors.PowerNorm(gamma=0.5),
)
seb.ax_add_grid(ax=ax_c, add_minor=True)
seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
# ax_cb.set_ylabel("trials / 1")
ax_c.set_aspect("equal")
ax_c.set_ylabel(r"reconstructed depth$\,/\,$km")
ax_c.set_xlabel(r"true depth$\,/\,$km")
ax_c.loglog()
ax_c.set_xlim(depth_bin["limits"] * SCALE)
ax_c.set_ylim(depth_bin["limits"] * SCALE)

ax_c.set_xticks(xticks)
ax_c.set_xticklabels(xlabels)
ax_c.set_yticks(xticks)
ax_c.set_yticklabels(xlabels)

fig.savefig(os.path.join(pa["out_dir"], "depth_reco_vs_true.jpg"))
seb.close(fig)


# relative
# --------
rel_bin = binning_utils.Binning(
    bin_edges=np.linspace(1 / np.sqrt(2), np.sqrt(2), depth_bin["num"] + 1)
)

cm = confusion_matrix.init(
    ax0_key="true_depth_m",
    ax0_values=res["object_distance_m"],
    ax0_bin_edges=depth_bin["edges"],
    ax1_key="reco_depth_over_true_depth",
    ax1_values=res["reco_object_distance_m"] / res["object_distance_m"],
    ax1_bin_edges=rel_bin["edges"],
    min_exposure_ax0=min_number_samples,
    default_low_exposure=0.0,
)

fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
ax_c = seb.add_axes(
    fig=fig, span=irf.summary.figure.AX_SPAN_WITH_COLORBAR_PAYLOAD
)
ax_cb = seb.add_axes(
    fig=fig, span=irf.summary.figure.AX_SPAN_WITH_COLORBAR_COLORBAR
)
ax_c.plot(
    theory_depth_m, theory_depth_m / theory_depth_m, "k--", linewidth=linewidth
)
ax_c.plot(
    theory_depth_m * SCALE,
    theory_depth_minus_m / theory_depth_m,
    "k:",
    linewidth=linewidth,
)
ax_c.plot(
    theory_depth_m * SCALE,
    theory_depth_plus_m / theory_depth_m,
    "k:",
    linewidth=linewidth,
)

_pcm_confusion = ax_c.pcolormesh(
    cm["ax0_bin_edges"] * SCALE,
    cm["ax1_bin_edges"],
    np.transpose(cm["counts_normalized_on_ax0"]) * np.mean(cm["counts_ax0"]),
    cmap="Greys",
    norm=seb.plt_colors.PowerNorm(gamma=0.5),
)
seb.ax_add_grid(ax=ax_c, add_minor=True)
seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
ax_c.set_aspect("equal")
ax_c.set_ylabel(r"(reconstructed depth) (true depth)$^{-1}$ $\,/\,$1")
ax_c.set_xlabel(r"true depth$\,/\,$km")
ax_c.semilogx()
ax_c.set_xticklabels([])
ax_c.set_xlim(depth_bin["limits"] * SCALE)
ax_c.set_ylim(rel_bin["limits"])

ax_c.set_xticks(xticks)
ax_c.set_xticklabels(xlabels)

fig.savefig(os.path.join(pa["out_dir"], "relative_depth_reco_vs_true.jpg"))
seb.close(fig)


depth_coarse_bin = binning_utils.Binning(
    bin_edges=np.geomspace(
        0.95 * config["min_object_distance_m"],
        1.05 * config["max_object_distance_m"],
        41,
    ),
)

deltas = [[] for i in range(depth_coarse_bin["num"])]

for i in range(len(res["object_distance_m"])):
    delta_depth = (
        res["object_distance_m"][i] - res["reco_object_distance_m"][i]
    )
    depth = res["object_distance_m"][i]
    b = np.digitize(depth, bins=depth_coarse_bin["edges"]) - 1
    deltas[b].append(delta_depth)

deltas_std = np.array([np.std(ll) for ll in deltas])
deltas_std_ru = np.array([np.sqrt(len(ll)) / len(ll) for ll in deltas])
deltas_std_au = deltas_std * deltas_std_ru

G_PLUS_G_MINUS_SCALE = 1e-1

fig = seb.figure({"rows": 960, "cols": 1920, "fontsize": 1.5})
ax_h = seb.add_axes(fig=fig, span=[0.12, 0.175, 0.87, 0.8])
seb.ax_add_grid(ax=ax_h, add_minor=True)
ax_h.loglog()
ax_h.set_xlim(depth_coarse_bin["limits"] * SCALE)
ax_h.set_ylim([5, 5e3])
ax_h.set_xlabel(r"true depth$\,/\,$km")
ax_h.set_ylabel(r"resolution$\,/\,$m")
ax_h.plot(
    theory_depth_m * SCALE,
    (theory_depth_plus_m - theory_depth_minus_m),
    "k:",
    alpha=0.3,
    linewidth=linewidth,
)
ax_h.plot(
    theory_depth_m * SCALE,
    (theory_depth_plus_m - theory_depth_minus_m) * G_PLUS_G_MINUS_SCALE,
    "k--",
    linewidth=linewidth,
)
seb.ax_add_histogram(
    ax=ax_h,
    bin_edges=depth_coarse_bin["edges"] * SCALE,
    bincounts=deltas_std,
    bincounts_lower=deltas_std - deltas_std_au,
    bincounts_upper=deltas_std + deltas_std_au,
    linestyle="-",
    linecolor="k",
    face_color="k",
    face_alpha=0.4,
)

ax_h.set_xticks(xticks)
ax_h.set_xticklabels(xlabels)
fig.savefig(os.path.join(pa["out_dir"], "depth_resolution.jpg"))
seb.close(fig)
