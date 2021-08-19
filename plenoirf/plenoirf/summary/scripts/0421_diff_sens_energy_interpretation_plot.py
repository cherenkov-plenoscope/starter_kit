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
import json_numpy
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

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

iEnergy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0420_diff_sens_energy_interpretation")
)

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

for sk in SITES:
    for pk in PARTICLES:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:

            cm = iEnergy[sk][pk][dk]

            site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
            os.makedirs(site_particle_dir, exist_ok=True)

            fig = seb.figure(seb.FIGURE_1_1)
            ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
            ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
            _pcm_confusion = ax_c.pcolormesh(
                energy_bin["edges"],
                energy_bin["edges"],
                np.transpose(cm["counts_normalized_on_ax0"]),
                cmap="Greys",
                norm=seb.plt_colors.PowerNorm(gamma=0.5),
            )
            ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
            ax_c.set_aspect("equal")
            ax_c.set_title("normalized for each column")
            ax_c.set_ylabel("reco. energy / GeV")
            ax_c.set_xlabel("true energy / GeV")
            ax_c.loglog()

            fig.savefig(os.path.join(sk_pk_dir, dk + ".jpg"))
            seb.close_figure(fig)

