#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
from os.path import join as opj
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


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

trigger_threshold = sum_config["trigger"]["threshold_pe"]
trigger_modus = sum_config["trigger"]["modus"]

num_energy_bins = 32
min_number_samples = 100

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

distance_bin_edges = np.geomspace(3e3, 3e4, num_energy_bins + 1)

fig_16_by_9 = sum_config["plot"]["16_by_9"]
fig_1_by_1 = fig_16_by_9.copy()
fig_1_by_1["rows"] = fig_16_by_9["rows"] * (16 / 9)

for site_key in irf_config["config"]["sites"]:
    particle_key = "gamma"
    site_particle_prefix = "{:s}_{:s}".format(site_key, particle_key)

    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"],
            "event_table",
            site_key,
            particle_key,
            "event_table.tar",
        ),
        structure=irf.table.STRUCTURE,
    )

    idx_triggered = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=event_table["trigger"],
        threshold=trigger_threshold,
        modus=trigger_modus,
    )
    idx_quality = irf.analysis.cuts.cut_quality(
        feature_table=event_table["features"],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )
    idx_features = event_table["features"][spt.IDX]

    idx_common = spt.intersection([idx_triggered, idx_quality, idx_features])

    table = spt.cut_table_on_indices(
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
    table = spt.sort_table_on_common_indices(
        table=table, common_indices=idx_common
    )

    fk = "image_smallest_ellipse_object_distance"
    true_airshower_maximum_altitude = table["cherenkovpool"]["maximum_asl_m"]
    image_smallest_ellipse_object_distance = table["features"][fk]

    event_weights = np.interp(
        x=table["primary"]["energy_GeV"],
        fp=weights_thrown2expected[site_key][particle_key][
            "weights_vs_energy"
        ]["mean"],
        xp=weights_thrown2expected[site_key][particle_key][
            "weights_vs_energy"
        ]["energy_GeV"],
    )

    cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
        x=true_airshower_maximum_altitude,
        y=image_smallest_ellipse_object_distance,
        bin_edges=distance_bin_edges,
        weights_x=event_weights,
        min_exposure_x=min_number_samples,
        default_low_exposure=np.nan,
    )

    fig = irf.summary.figure.figure(fig_1_by_1)
    ax = irf.summary.figure.add_axes(fig, [0.1, 0.23, 0.7, 0.7])
    ax_h = irf.summary.figure.add_axes(fig, [0.1, 0.07, 0.7, 0.1])
    ax_cb = fig.add_axes([0.85, 0.23, 0.02, 0.7])
    _pcm_confusion = ax.pcolormesh(
        cm["bin_edges"],
        cm["bin_edges"],
        np.transpose(cm["confusion_bins_normalized_columns"]),
        cmap="Greys",
        norm=plt_colors.PowerNorm(gamma=0.5),
    )
    plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    irf.summary.figure.mark_ax_airshower_spectrum(ax=ax)
    ax.set_aspect("equal")
    ax.set_title("normalized for each column")
    ax.set_ylabel(fk + " / " + irf.table.STRUCTURE["features"][fk]["unit"])
    ax.loglog()
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_h.loglog()
    ax_h.set_xlim([np.min(cm["bin_edges"]), np.max(cm["bin_edges"])])
    ax_h.set_xlabel("true maximum of airshower / m")
    ax_h.set_ylabel("num. events / 1")
    irf.summary.figure.mark_ax_thrown_spectrum(ax_h)
    ax_h.axhline(min_number_samples, linestyle=":", color="k")
    irf.summary.figure.ax_add_hist(
        ax=ax_h,
        bin_edges=cm["bin_edges"],
        bincounts=cm["exposure_bins_x_no_weights"],
        linestyle="-",
        linecolor="k",
    )
    plt.savefig(opj(pa["out_dir"], site_particle_prefix + "_maximum.jpg"))
    plt.close("all")
