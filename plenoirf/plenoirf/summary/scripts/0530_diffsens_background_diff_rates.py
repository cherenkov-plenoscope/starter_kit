#!/usr/bin/python
import sys
import copy
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

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]

energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)

acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0505_diffsens_rebin_flux_of_airshowers")
)


def assert_energy_migration_is_valid(M):
    Mshape = M["counts"].shape
    assert Mshape[0] == Mshape[1]
    num_energy_bins = Mshape[0]

    assert M["ax0_key"] == "true_energy"
    assert M["ax1_key"] == "reco_energy"

    for ereco in range(num_energy_bins):
        check = np.sum(M["true_given_reco"][:, ereco])
        if check > 0:
            assert 0.99 < check < 1.01

# prepare diff-flux
# -----------------
diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"]


gk = "diffuse"

Rt = {}
Rt_au = {}

R = {}
R_au = {}

for sk in SITES:
    Rt[sk] = {}
    Rt_au[sk] = {}
    R[sk] = {}
    R_au[sk] = {}
    for ok in ONREGION_TYPES:
        Rt[sk][ok] = {}
        Rt_au[sk][ok] = {}
        R[sk][ok] = {}
        R_au[sk][ok] = {}
        for pk in COSMIC_RAYS:
            Rt[sk][ok][pk] = np.zeros(energy_bin["num_bins"])
            R[sk][ok][pk] = np.zeros(energy_bin["num_bins"])

            Rt_au[sk][ok][pk] = np.zeros(energy_bin["num_bins"])
            R_au[sk][ok][pk] = np.zeros(energy_bin["num_bins"])

            print("apply", sk, pk, ok)
            dFdE = diff_flux[sk][pk]
            dFdE_au = np.zeros(dFdE.shape)

            M = copy.deepcopy(energy_migration[sk][pk])
            assert_energy_migration_is_valid(M=M)

            Q = acceptance[sk][ok][pk][gk]["mean"]
            Q_au = acceptance[sk][ok][pk][gk]["absolute_uncertainty"]

            energy_bin__width__au = np.zeros(energy_bin["num_bins"])
            for ereco in range(energy_bin["num_bins"]):
                _tmp_sum = np.zeros(energy_bin["num_bins"])
                _tmp_sum_au = np.zeros(energy_bin["num_bins"])
                checksum = 0.0
                for etrue in range(energy_bin["num_bins"]):

                    (
                        _tmp_sum[etrue],
                        _tmp_sum_au[etrue],
                    ) = irf.utils.multiply_elemnetwise_au(
                        x=[
                            dFdE[etrue],
                            M["true_given_reco"][etrue, ereco],
                            Q[etrue],
                            energy_bin["width"][etrue],
                        ],
                        x_au=[
                            dFdE_au[etrue],
                            M["true_given_reco_abs_unc"][etrue, ereco],
                            Q_au[etrue],
                            energy_bin__width__au[etrue],
                        ],
                    )
                    checksum += M["true_given_reco"][etrue, ereco]
                if checksum > 0:
                    assert 0.99 < checksum < 1.01
                (
                    Rt[sk][ok][pk][ereco],
                    Rt_au[sk][ok][pk][ereco],
                ) = irf.utils.sum_elemnetwise_au(x=_tmp_sum, x_au=_tmp_sum_au,)

            for etrue in range(energy_bin["num_bins"]):
                (
                    R[sk][ok][pk][etrue],
                    R_au[sk][ok][pk][etrue],
                ) = irf.utils.multiply_elemnetwise_au(
                    x=[dFdE[etrue], Q[etrue], energy_bin["width"][etrue]],
                    x_au=[dFdE_au[etrue], Q_au[etrue], 0.0],
                )

            # cross check
            # -----------
            # total rate must not change under energy migration
            total_R = np.sum(R[sk][ok][pk][:])
            total_Rt = np.sum(Rt[sk][ok][pk][:])
            print("total_R", total_R, "total_Rt", total_Rt)
            assert 0.7 < total_R / total_Rt < 1.3

for sk in SITES:
    for ok in ONREGION_TYPES:
        os.makedirs(os.path.join(pa["out_dir"], sk, ok), exist_ok=True)

for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in COSMIC_RAYS:
            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, pk + ".json"),
                {
                    "comment": (
                        "rate after all cuts " "VS reco. energy"
                    ),
                    "unit": "s$^{-1}$",
                    "mean": Rt[sk][ok][pk],
                    "absolute_uncertainty": Rt_au[sk][ok][pk],
                    "energy_binning_key": energy_bin["key"],
                },
            )


for sk in SITES:
    for ok in ONREGION_TYPES:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        for pk in COSMIC_RAYS:

            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Rt[sk][ok][pk],
                bincounts_upper=Rt[sk][ok][pk] - Rt_au[sk][ok][pk],
                bincounts_lower=Rt[sk][ok][pk] + Rt_au[sk][ok][pk],
                linestyle="-",
                linecolor=sum_config["plot"]["particle_colors"][pk],
                face_color=sum_config["plot"]["particle_colors"][pk],
                face_alpha=0.25,
            )

            alpha = 0.25
            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=R[sk][ok][pk],
                bincounts_upper=R[sk][ok][pk] - R_au[sk][ok][pk],
                bincounts_lower=R[sk][ok][pk] + R_au[sk][ok][pk],
                linecolor=sum_config["plot"]["particle_colors"][pk],
                linealpha=alpha,
                linestyle=":",
                face_color=sum_config["plot"]["particle_colors"][pk],
                face_alpha=alpha * 0.25,
            )

        ax.set_ylabel("rate / s$^{-1}$")
        ax.set_xlabel("reco. energy / GeV")
        ax.set_ylim([1e-6, 1e4])
        ax.loglog()
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                sk + "_" + ok + "_differential_rates_vs_reco_energy.jpg",
            )
        )
        seb.close(fig)
