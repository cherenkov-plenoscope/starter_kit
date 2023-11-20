#!/usr/bin/python
import argparse
import numpy as np
import plenoirf
import os
import plenopy
import sebastians_matplotlib_addons as sebplt

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--light_field_geometry_path", type=str)
argparser.add_argument("--out_dir", type=str)

args = argparser.parse_args()

light_field_geometry_path = args.light_field_geometry_path
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

lfg = plenopy.LightFieldGeometry(light_field_geometry_path)

FIGSTY = {"rows": 960, "cols": 1920, "fontsize": 2.0}
AXSPAN = [0.12, 0.23, 0.87, 0.74]

YLIM = np.array([1, 1e6])
YLABEL = r"intensity$\,/\,$1"


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


def find_start_stop(bin_edges, mask):
    start = float("nan")
    found_start = False
    stop = float("nan")
    for i in range(len(mask)):
        if mask[i] and not found_start:
            found_start = True
            start = bin_edges[i]
        if mask[i]:
            stop = bin_edges[i + 1]
    return start, stop


def save_histogram(
    path,
    v_bin_edges,
    v_bin_counts,
    v_median,
    percentile_mask,
    xlabel,
    xscale,
    ylim,
    yscale=1,
    ylabel=YLABEL,
    semilogy=True,
):
    num_bins = len(v_bin_counts)
    fig = sebplt.figure(FIGSTY)
    ax = sebplt.add_axes(fig=fig, span=AXSPAN)
    ylim = np.array(ylim)

    sebplt.ax_add_histogram(
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

    sebplt.ax_add_histogram(
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
    sebplt.close(fig)


NUM_BINS = int(100 * np.sqrt(lfg.number_lixel) / np.sqrt(8443 * 61))
NUM_BINS = np.max([NUM_BINS, 3])
NUM_BIN_EDGES = NUM_BINS + 1

RANGES = ["fix", "sug"]

hists = {}
hists["solid_angles"] = {
    "v": 4 * np.pi * lfg.cx_std * lfg.cy_std,
    "fix": {"v_bin_edges": np.linspace(0, 4 * 4e-6, NUM_BIN_EDGES)},
    "sug": {"v_bin_edges": np.linspace(0, 6e-6, NUM_BIN_EDGES)},
    "xscale": 1e6,
    "xlabel": r"solid angle of beams $\Omega\,/\,\mu$sr",
}
hists["areas"] = {
    "v": 4 * np.pi * lfg.x_std * lfg.y_std,
    "fix": {"v_bin_edges": np.linspace(0, 4 * 300, NUM_BIN_EDGES)},
    "sug": {"v_bin_edges": np.linspace(0, 300, NUM_BIN_EDGES)},
    "xscale": 1,
    "xlabel": r"area of beams $A\,/\,$m$^{2}$",
}
hists["time_spreads"] = {
    "v": lfg.time_delay_wrt_principal_aperture_plane_std,
    "fix": {"v_bin_edges": np.linspace(0, 2.5e-9, NUM_BIN_EDGES)},
    "sug": {"v_bin_edges": np.linspace(0, 1e-9, NUM_BIN_EDGES)},
    "xscale": 1e9,
    "xlabel": r"time-spread of beams $T\,/\,$ns",
}
hists["efficiencies"] = {
    "v": lfg.efficiency / np.median(lfg.efficiency),
    "fix": {"v_bin_edges": np.linspace(0, 1.2, NUM_BIN_EDGES)},
    "sug": {"v_bin_edges": np.linspace(0, 1.2, NUM_BIN_EDGES)},
    "xscale": 1,
    "xlabel": r"relative efficiency of beams $E\,/\,$1",
}
for key in hists:
    hists[key]["v_median"] = np.median(hists[key]["v"])
    for met in RANGES:
        hists[key][met]["v_bin_counts"] = make_histogram(
            v=hists[key]["v"],
            v_bin_edges=hists[key][met]["v_bin_edges"],
        )
        hists[key][met]["percentile_mask"] = make_percentile_mask(
            v=hists[key]["v"],
            v_bin_edges=hists[key][met]["v_bin_edges"],
        )


rrr = {"fix": {"max_bin_count": 0}, "sug": {"max_bin_count": 0}}
for met in RANGES:
    for key in hists:
        if np.max(hists[key][met]["v_bin_counts"]) > rrr[met]["max_bin_count"]:
            rrr[met]["max_bin_count"] = np.max(hists[key][met]["v_bin_counts"])

    rrr[met]["ylim_lin"] = [0, 1.1 * rrr[met]["max_bin_count"]]
    rrr[met]["ylim_log"] = [
        1,
        10 ** np.ceil(np.log10(rrr[met]["max_bin_count"])),
    ]

    for key in hists:
        save_histogram(
            path=os.path.join(out_dir, key + "_log_{:s}.jpg".format(met)),
            ylim=rrr[met]["ylim_log"],
            semilogy=True,
            v_bin_edges=hists[key][met]["v_bin_edges"],
            v_bin_counts=hists[key][met]["v_bin_counts"],
            v_median=hists[key]["v_median"],
            percentile_mask=hists[key][met]["percentile_mask"],
            xlabel=hists[key]["xlabel"],
            xscale=hists[key]["xscale"],
            ylabel=r"intensity$\,/\,$1",
        )
        save_histogram(
            path=os.path.join(out_dir, key + "_lin_{:s}.jpg".format(met)),
            ylim=rrr[met]["ylim_lin"],
            semilogy=False,
            v_bin_edges=hists[key][met]["v_bin_edges"],
            v_bin_counts=hists[key][met]["v_bin_counts"],
            v_median=hists[key]["v_median"],
            percentile_mask=hists[key][met]["percentile_mask"],
            xlabel=hists[key]["xlabel"],
            xscale=hists[key]["xscale"],
            ylabel=r"intensity$\,/\,$1k",
            yscale=1e-3,
        )
