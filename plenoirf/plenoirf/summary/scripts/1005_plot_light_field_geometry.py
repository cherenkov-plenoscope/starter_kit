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


def save_histogram(
    path, v, v_bin_edges, xlabel, xscale=1.0, mask_percentile=90,
):
    v_bin_counts = np.histogram(v, bins=v_bin_edges)[0]
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

    fig = seb.figure(FIGSTY)
    ax = seb.add_axes(fig=fig, span=AXSPAN)

    seb.ax_add_histogram(
        ax=ax,
        bin_edges=v_bin_edges * xscale,
        bincounts=np.ones(num_bins),
        bincounts_upper=mask * v_bin_counts,
        bincounts_lower=np.ones(num_bins),
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
        bincounts=v_bin_counts,
        linestyle="-",
        linecolor="k",
        linealpha=1.0,
        draw_bin_walls=True,
    )
    ax.vlines(
        x=xscale * np.median(v),
        ymin=YLIM[0],
        ymax=YLIM[1],
        color="gray",
        linestyle="--",
    )
    ax.set_ylim(YLIM)
    ax.semilogy()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(YLABEL)
    fig.savefig(path)
    seb.close(fig)


# direction
# ---------
beam_solid_angles_sr = 4 * np.pi * lfg.cx_std * lfg.cy_std

save_histogram(
    path=os.path.join(pa["out_dir"], "directions.jpg"),
    v=beam_solid_angles_sr,
    v_bin_edges=np.linspace(0, 4 * 4e-6, 101),
    xscale=1e6,
    xlabel="solid angle of beams $\Omega$ / $\mu$sr",
)

# position
# ---------
beam_area_m2 = 4 * np.pi * lfg.x_std * lfg.y_std

save_histogram(
    path=os.path.join(pa["out_dir"], "areas.jpg"),
    v=beam_area_m2,
    v_bin_edges=np.linspace(0, 4 * 300, 101),
    xscale=1,
    xlabel="area of beams $A$ / m$^{2}$",
)

# time
# ----
beam_time_spread = lfg.time_delay_wrt_principal_aperture_plane_std

save_histogram(
    path=os.path.join(pa["out_dir"], "time_spreads.jpg"),
    v=beam_time_spread,
    v_bin_edges=np.linspace(0, 2.5e-9, 101),
    xscale=1e9,
    xlabel="time-spread of beams $T$ / ns",
)

# relative efficiency
# -------------------
beam_efficiency = lfg.efficiency / np.median(lfg.efficiency)

save_histogram(
    path=os.path.join(pa["out_dir"], "efficiencies.jpg"),
    v=beam_efficiency,
    v_bin_edges=np.linspace(0, 1.2, 101),
    xscale=1,
    xlabel="relative efficiency of beams $E$ / 1",
)
