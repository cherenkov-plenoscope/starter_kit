#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


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

num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "trigger_acceptance"
]
energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    num_energy_bins + 1,
)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

fig_16_by_9 = sum_config["plot"]["16_by_9"]
particle_colors = sum_config["plot"]["particle_colors"]

# Read features
# =============

tables = {}
for sk in SITES:
    tables[sk] = {}
    for pk in PARTICLES:

        _table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_triggered = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=_table["trigger"],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )
        idx_features = _table["features"][spt.IDX]

        idx_common = spt.intersection([idx_triggered, idx_features,])

        table = spt.cut_table_on_indices(
            table=_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_common,
            level_keys=["primary", "features"],
        )
        tables[sk][pk] = spt.sort_table_on_common_indices(
            table=table, common_indices=idx_common
        )

# guess bin edges
lims = {}
Sfeatures = irf.table.STRUCTURE["features"]

for fk in Sfeatures:
    lims[fk] = {}
    for sk in SITES:
        lims[fk][sk] = {}
        for pk in PARTICLES:
            lims[fk][sk][pk] = {}
            features = tables[sk][pk]["features"]
            num_bins = int(np.sqrt(features.shape[0]))
            num_bin_edges = num_bins + 1
            lims[fk][sk][pk]["bin_edges"] = {}
            lims[fk][sk][pk]["bin_edges"]["num"] = num_bin_edges

            start, stop = irf.features.find_values_quantile_range(
                values=features[fk], quantile_range=[0.01, 0.99]
            )
            if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
                start = 10 ** np.floor(np.log10(start))
                stop = 10 ** np.ceil(np.log10(stop))
            else:
                if start >= 0.0:
                    start = 0.9 * start
                else:
                    start = 1.1 * start
                if stop >= 0.0:
                    stop = 1.1 * stop
                else:
                    stop = 0.9 * stop

            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop

# find same bin-edges for all particles
for fk in Sfeatures:
    for sk in SITES:
        starts = [lims[fk][sk][pk]["bin_edges"]["start"] for pk in PARTICLES]
        stops = [lims[fk][sk][pk]["bin_edges"]["stop"] for pk in PARTICLES]
        nums = [lims[fk][sk][pk]["bin_edges"]["num"] for pk in PARTICLES]
        start = np.min(starts)
        stop = np.max(stops)
        num = np.max(nums)
        for pk in PARTICLES:
            lims[fk][sk][pk]["bin_edges"]["stop"] = stop
            lims[fk][sk][pk]["bin_edges"]["start"] = start
            lims[fk][sk][pk]["bin_edges"]["num"] = num

for fk in Sfeatures:
    for sk in SITES:

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))

        for pk in PARTICLES:

            reweight_spectrum = np.interp(
                x=tables[sk][pk]["primary"]["energy_GeV"],
                xp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                    "energy_GeV"
                ],
                fp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                    "mean"
                ],
            )

            if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
                myspace = np.geomspace
            else:
                myspace = np.linspace

            bin_edges_fk = myspace(
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
                lims[fk][sk][pk]["bin_edges"]["num"],
            )
            bin_counts_fk = np.histogram(
                tables[sk][pk]["features"][fk], bins=bin_edges_fk
            )[0]
            bin_counts_weight_fk = np.histogram(
                tables[sk][pk]["features"][fk],
                weights=reweight_spectrum,
                bins=bin_edges_fk,
            )[0]

            bin_counts_unc_fk = irf.analysis.effective_quantity._divide_silent(
                numerator=np.sqrt(bin_counts_fk),
                denominator=bin_counts_fk,
                default=np.nan,
            )
            bin_counts_weight_norm_fk = irf.analysis.effective_quantity._divide_silent(
                numerator=bin_counts_weight_fk,
                denominator=np.sum(bin_counts_weight_fk),
                default=0,
            )

            irf.summary.figure.ax_add_hist(
                ax=ax,
                bin_edges=bin_edges_fk,
                bincounts=bin_counts_weight_norm_fk,
                linestyle="-",
                linecolor=particle_colors[pk],
                linealpha=1.0,
                bincounts_upper=bin_counts_weight_norm_fk
                * (1 + bin_counts_unc_fk),
                bincounts_lower=bin_counts_weight_norm_fk
                * (1 - bin_counts_unc_fk),
                face_color=particle_colors[pk],
                face_alpha=0.3,
            )

        if "log(x)" in Sfeatures[fk]["transformation"]["function"]:
            ax.loglog()
        else:
            ax.semilogy()

        irf.summary.figure.mark_ax_airshower_spectrum(ax=ax)
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.set_xlabel("{:s} / {:s}".format(fk, Sfeatures[fk]["unit"]))
        ax.set_ylabel("relative intensity / 1")
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        ax.set_xlim(
            [
                lims[fk][sk][pk]["bin_edges"]["start"],
                lims[fk][sk][pk]["bin_edges"]["stop"],
            ]
        )
        ax.set_ylim([1e-5, 1.0])
        fig.savefig(
            os.path.join(pa["out_dir"], "{:s}_{:s}.jpg".format(sk, fk))
        )
        plt.close(fig)
