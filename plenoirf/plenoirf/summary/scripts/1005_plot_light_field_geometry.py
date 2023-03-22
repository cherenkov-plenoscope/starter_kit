#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

FIGSTY = {"rows": 720, "cols": 1920, "fontsize": 1.5}
AXSPAN = [0.1, 0.25, 0.85, 0.7]

YLIM = np.array([1, 1e6])
YLABEL = "intensity / 1"


def make_histogram(v, v_bin_edges):
    return np.histogram(v, bins=v_bin_edges)[0]


def make_percentile_mask(v, v_bin_edges, mask_percentile=90):
    v_bin_counts = make_histogram(v=v, v_bin_edges=v_bin_edges)
    v_total_counts = np.sum(v_bin_counts)

    # watershed
    assert 0 <= mask_percentile <= 100
    target_fraction = mask_percentile / 100

    fraction = 0.0
    num_bins = v_bin_counts.shape[0]
    mask = np.zeros(num_bins)
    v_bin_counts_fraction = v_bin_counts.copy()
    while fraction < target_fraction:
        a = np.argmax(v_bin_counts_fraction)
        mask[a] = 1
        v_bin_counts_fraction[a] = 0
        fraction_part = v_bin_counts[a] / v_total_counts
        fraction += fraction_part
    return mask


def save_histogram(
    path, v_bin_edges, v_bin_counts, v_median, percentile_mask, xlabel, xscale, ylim, yscale=1, ylabel="intensity / 1", semilogy=True,
):
    num_bins = len(v_bin_counts)
    fig = seb.figure(FIGSTY)
    ax = seb.add_axes(fig=fig, span=AXSPAN)
    ylim = np.array(ylim)

    seb.ax_add_histogram(
        ax=ax,
        bin_edges=v_bin_edges * xscale,
        bincounts=np.ones(num_bins) * yscale,
        bincounts_upper=percentile_mask * v_bin_counts * yscale,
        bincounts_lower=np.ones(num_bins) * yscale,
        linestyle=None,
        linecolor=None,
        linealpha=0.0,
        face_color="k",
        face_alpha=0.25,
        label=None,
        draw_bin_walls=False,
    )

    seb.ax_add_histogram(
        ax=ax,
        bin_edges=v_bin_edges * xscale,
        bincounts=v_bin_counts * yscale,
        linestyle="-",
        linecolor="k",
        linealpha=1.0,
        draw_bin_walls=True,
    )
    ax.vlines(
        x=xscale * v_median,
        ymin=ylim[0] * yscale,
        ymax=ylim[1] * yscale,
        color="gray",
        linestyle="--",
    )
    ax.set_ylim(ylim * yscale)
    if semilogy:
        ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.savefig(path)
    seb.close(fig)


hists = {}
hists["solid_angles"] = {
    "v": 4 * np.pi * lfg.cx_std * lfg.cy_std,
    "v_bin_edges": np.linspace(0, 4 * 4e-6, 101),
    "xscale": 1e6,
    "xlabel": "solid angle of beams $\Omega$ / $\mu$sr"
}
hists["areas"] = {
    "v": 4 * np.pi * lfg.x_std * lfg.y_std,
    "v_bin_edges": np.linspace(0, 4 * 300, 101),
    "xscale": 1,
    "xlabel": "area of beams $A$ / m$^{2}$"
}
hists["time_spreads"] = {
    "v": lfg.time_delay_wrt_principal_aperture_plane_std,
    "v_bin_edges": np.linspace(0, 2.5e-9, 101),
    "xscale": 1e9,
    "xlabel": "time-spread of beams $T$ / ns"
}
hists["efficiencies"] = {
    "v": lfg.efficiency / np.median(lfg.efficiency),
    "v_bin_edges": np.linspace(0, 1.2, 101),
    "xscale": 1,
    "xlabel": "relative efficiency of beams $E$ / 1"
}

for key in hists:
    hists[key]["v_bin_counts"] = make_histogram(
        v=hists[key]["v"],
        v_bin_edges=hists[key]["v_bin_edges"],
    )
    hists[key]["percentile_mask"] = make_percentile_mask(
        v=hists[key]["v"],
        v_bin_edges=hists[key]["v_bin_edges"],
    )
    hists[key]["v_median"] = np.median(hists[key]["v"])

max_bin_count = 0
for key in hists:
    if np.max(hists[key]["v_bin_counts"]) > max_bin_count:
        max_bin_count = np.max(hists[key]["v_bin_counts"])

ylim_lin = [0, 1.1 * max_bin_count]
ylim_log = [1, 10 ** np.ceil(np.log10(max_bin_count))]

for key in hists:
    save_histogram(
        path=os.path.join(pa["out_dir"], key + "_log.jpg"),
        ylim=ylim_log,
        semilogy=True,
        v_bin_edges=hists[key]["v_bin_edges"],
        v_bin_counts=hists[key]["v_bin_counts"],
        v_median=hists[key]["v_median"],
        percentile_mask=hists[key]["percentile_mask"],
        xlabel=hists[key]["xlabel"],
        xscale=hists[key]["xscale"],
        ylabel="intensity / 1",
    )
    save_histogram(
        path=os.path.join(pa["out_dir"], key + "_lin.jpg"),
        ylim=ylim_lin,
        semilogy=False,
        v_bin_edges=hists[key]["v_bin_edges"],
        v_bin_counts=hists[key]["v_bin_counts"],
        v_median=hists[key]["v_median"],
        percentile_mask=hists[key]["percentile_mask"],
        xlabel=hists[key]["xlabel"],
        xscale=hists[key]["xscale"],
        ylabel="intensity / 1k",
        yscale=1e-3,
    )