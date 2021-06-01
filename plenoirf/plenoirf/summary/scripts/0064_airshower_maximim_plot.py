#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
from os.path import join as opj
import numpy as np
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

weights_thrown2expected = irf.json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

num_energy_bins = 32
min_number_samples = 100

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

distance_bin_edges = np.geomspace(5e3, 25e3, num_energy_bins + 1)

for sk in irf_config["config"]["sites"]:
    pk = "gamma"
    site_particle_prefix = "{:s}_{:s}".format(sk, pk)

    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"], "event_table", sk, pk, "event_table.tar",
        ),
        structure=irf.table.STRUCTURE,
    )

    idx_common = spt.intersection(
        [
            passing_trigger[sk][pk]["passed_trigger"]["idx"],
            passing_quality[sk][pk]["passed_quality"]["idx"],
        ]
    )

    table = spt.cut_and_sort_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=idx_common,
        level_keys=[
            "primary",
            "cherenkovsize",
            "cherenkovpool",
            "core",
            "trigger",
            "features",
        ],
    )

    fk = "image_smallest_ellipse_object_distance"
    true_airshower_maximum_altitude = table["cherenkovpool"]["maximum_asl_m"]
    image_smallest_ellipse_object_distance = table["features"][fk]

    event_weights = np.interp(
        x=table["primary"]["energy_GeV"],
        fp=weights_thrown2expected[sk][pk]["weights_vs_energy"]["mean"],
        xp=weights_thrown2expected[sk][pk]["weights_vs_energy"]["energy_GeV"],
    )

    cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
        x=true_airshower_maximum_altitude,
        y=image_smallest_ellipse_object_distance,
        x_bin_edges=distance_bin_edges,
        y_bin_edges=distance_bin_edges,
        weights_x=event_weights,
        min_exposure_x=min_number_samples,
        default_low_exposure=0.0,
    )

    fig = seb.figure(style=seb.FIGURE_1_1)
    ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
    ax_h = seb.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
    ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax_c.pcolormesh(
        cm["x_bin_edges"],
        cm["y_bin_edges"],
        np.transpose(cm["confusion_bins_normalized_columns"]),
        cmap="Greys",
        norm=seb.plt_colors.PowerNorm(gamma=0.5),
    )
    seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    irf.summary.figure.mark_ax_airshower_spectrum(ax=ax_c)
    ax_c.set_aspect("equal")
    ax_c.set_title("normalized for each column")
    ax_c.set_ylabel(fk + " / " + irf.table.STRUCTURE["features"][fk]["unit"])
    ax_c.loglog()
    seb.ax_add_grid(ax_c)

    ax_h.semilogx()
    ax_h.set_xlim([np.min(cm["x_bin_edges"]), np.max(cm["y_bin_edges"])])
    ax_h.set_xlabel("true maximum of airshower / m")
    ax_h.set_ylabel("num. events / 1")
    irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
    ax_h.axhline(min_number_samples, linestyle=":", color="k")
    seb.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["x_bin_edges"],
        bincounts=cm["exposure_bins_x_no_weights"],
        linestyle="-",
        linecolor="k",
    )
    fig.savefig(opj(pa["out_dir"], site_particle_prefix + "_maximum.jpg"))
    seb.close_figure(fig)
