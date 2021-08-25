#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

cosmic_diff_rate = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0430_interpolate_background_diff_rates")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
fenergy_bin = energy_binning["interpolation"]

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")

ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

for sk in SITES:
    for ok in ONREGION_TYPES:
        sk_dir = os.path.join(pa["out_dir"], sk)
        os.makedirs(sk_dir, exist_ok=True)

        Rt = np.zeros(fenergy_bin["num_bins"])
        Rt_au = np.zeros(fenergy_bin["num_bins"])

        _Rtsum = np.zeros((len(COSMIC_RAYS), fenergy_bin["num_bins"]))
        _Rtsum_au = np.zeros((len(COSMIC_RAYS), fenergy_bin["num_bins"]))

        for ick, ck in enumerate(COSMIC_RAYS):
            dRtdEt = cosmic_diff_rate[sk][ok][ck]["mean"]
            dRtdEt_au = cosmic_diff_rate[sk][ok][ck]["absolute_uncertainty"]

            _Rtsum[ick, :] = dRtdEt * fenergy_bin["width"]
            _Rtsum_au[ick, :] = dRtdEt_au * fenergy_bin["width"]

        for ee in range(fenergy_bin["num_bins"]):
            Rt[ee], Rt_au[ee] = irf.utils.sum(
                x=_Rtsum[:, ee], x_au=_Rtsum_au[:, ee]
            )

        json_numpy.write(
            os.path.join(pa["out_dir"], sk, ok + ".json"),
            {
                "comment": "rate of all cosmic-ray-background VS reco. energy",
                "unit": "s$^{-1}$",
                "mean": Rt,
                "absolute_uncertainty": Rt_au,
            }
        )

Rt = json_numpy.read_tree(os.path.join(pa["out_dir"]))

for sk in SITES:
    for ok in ONREGION_TYPES:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        for ck in COSMIC_RAYS:

            ck_dRtdEt = cosmic_diff_rate[sk][ok][ck]["mean"]
            ck_dRtdEt_au = cosmic_diff_rate[sk][ok][ck]["absolute_uncertainty"]

            ck_Rt = ck_dRtdEt * fenergy_bin["width"]
            ck_Rt_au = ck_dRtdEt_au * fenergy_bin["width"]

            ck_alpha = 0.5

            ax.plot(
                fenergy_bin["centers"],
                ck_Rt,
                color=sum_config["plot"]["particle_colors"][ck],
                alpha=ck_alpha
            )
            ax.fill_between(
                x=fenergy_bin["centers"],
                y1=ck_Rt - ck_Rt_au,
                y2=ck_Rt + ck_Rt_au,
                color=sum_config["plot"]["particle_colors"][ck],
                alpha=ck_alpha*0.2,
                linewidth=0.0,
            )

        ax.plot(
            fenergy_bin["centers"],
            Rt[sk][ok]["mean"],
            ":k",
        )
        ax.fill_between(
            x=fenergy_bin["centers"],
            y1=Rt[sk][ok]["mean"] - Rt[sk][ok]["absolute_uncertainty"],
            y2=Rt[sk][ok]["mean"] + Rt[sk][ok]["absolute_uncertainty"],
            color="k",
            alpha=0.2,
            linewidth=0.0,
        )

        ax.set_ylabel("rate / s$^{-1}$")
        ax.set_xlabel("reco. energy / GeV")

        ax.set_ylim([1e-5, 1e3])

        ax.loglog()
        fig.savefig(
            os.path.join(
                pa["out_dir"], sk + "_" + ok + "_rate_vs_reco_energy.jpg",
            )
        )
        seb.close_figure(fig)
