#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
import numpy as np
import sklearn
import pickle
import json
from sklearn import neural_network
from sklearn import ensemble
from sklearn import model_selection
from sklearn import utils
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0059_passing_trajectory_quality")
)
reconstructed_energy = irf.json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0065_learning_airshower_maximum_and_energy"
    ),
)

energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance_onregion"]
    + 1,
)

min_number_samples = 10

def align_on_idx(input_idx, input_values, target_idxs):
    Q = {}
    for ii in range(len(input_idx)):
        Q[input_idx[ii]] = input_values[ii]
    aligned_values = np.nan * np.ones(target_idxs.shape[0])
    for ii in range(target_idxs.shape[0]):
        aligned_values[ii] = Q[target_idxs[ii]]
    return aligned_values



for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_valid = spt.intersection([
            passing_trigger[sk][pk]["idx"],
            passing_quality[sk][pk]["idx"],
            passing_trajectory_quality[sk][pk]["idx"],
            reconstructed_energy[sk][pk]["energy"]["idx"],
        ])

        valid_event_table = spt.cut_and_sort_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_valid,
        )

        true_energy = valid_event_table["primary"]["energy_GeV"]
        reco_energy = align_on_idx(
            input_idx=reconstructed_energy[sk][pk]["energy"]["idx"],
            input_values=reconstructed_energy[sk][pk]["energy"]["energy"],
            target_idxs=valid_event_table["primary"]["idx"]
        )

        cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
            x=true_energy,
            y=reco_energy,
            x_bin_edges=energy_bin_edges,
            y_bin_edges=energy_bin_edges,
            min_exposure_x=min_number_samples,
            default_low_exposure=0.0,
        )

        fig = seb.figure(seb.FIGURE_1_1)
        ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_h = seb.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
        ax_cb = seb.add_axes(
            fig=fig, span=[0.85, 0.27, 0.02, 0.65]
        )
        _pcm_confusion = ax_c.pcolormesh(
            cm["x_bin_edges"],
            cm["y_bin_edges"],
            np.transpose(cm["confusion_bins_normalized_columns"]),
            cmap="Greys",
            norm=seb.plt_colors.PowerNorm(gamma=0.5),
        )
        ax_c.grid(
            color="k", linestyle="-", linewidth=0.66, alpha=0.1
        )
        seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        irf.summary.figure.mark_ax_thrown_spectrum(ax=ax_c)
        ax_c.set_aspect("equal")
        ax_c.set_title("normalized for each column")
        ax_c.set_ylabel("reco. energy / GeV")
        ax_c.loglog()
        ax_h.semilogx()
        ax_h.set_xlim(
            [np.min(cm["x_bin_edges"]), np.max(cm["y_bin_edges"])]
        )
        ax_h.set_xlabel("true energy / GeV")
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
        fig.savefig(
            os.path.join(
                pa["out_dir"], sk + "_" + pk + ".jpg"
            )
        )
        seb.close_figure(fig)
