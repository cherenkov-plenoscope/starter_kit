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
COSMIC_RAYS = irf.utils.filter_particles_with_electric_charge(PARTICLES)
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

# load
# ----
energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
energy_bin__width__au = np.zeros(energy_bin["num_bins"])  # abs. uncertainty

energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)

acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0505_diffsens_rebin_flux_of_airshowers")
)

# prepare
# -------
def assert_energy_migration_is_valid(M):
    assert M["ax0_key"] == "true_energy"
    assert M["ax1_key"] == "reco_energy"

    assert M["counts"].shape[0] == M["counts"].shape[1]
    num_energy_bins = M["counts"].shape[0]

    for etrue in range(num_energy_bins):
        check = np.sum(M["reco_given_true"][etrue, :])
        if check > 0:
            assert 0.99 < check < 1.01, "sum(P(reco|true)) = {:f}".format(
                check
            )


diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"]

# work
# ----
gk = "diffuse"  # geometry-key (gk) for source.

# cosmic-ray-rate
# in reconstructed energy
Rreco = {}
Rreco_au = {}  # absolute uncertainty

# in true energy
Rtrue = {}
Rtrue_au = {}

for sk in SITES:
    Rreco[sk] = {}
    Rreco_au[sk] = {}
    Rtrue[sk] = {}
    Rtrue_au[sk] = {}
    for ok in ONREGION_TYPES:
        Rreco[sk][ok] = {}
        Rreco_au[sk][ok] = {}
        Rtrue[sk][ok] = {}
        Rtrue_au[sk][ok] = {}
        for pk in COSMIC_RAYS:
            print(sk, pk, ok)

            # init rates to zero
            Rreco[sk][ok][pk] = np.zeros(energy_bin["num_bins"])
            Rtrue[sk][ok][pk] = np.zeros(energy_bin["num_bins"])

            Rreco_au[sk][ok][pk] = np.zeros(energy_bin["num_bins"])
            Rtrue_au[sk][ok][pk] = np.zeros(energy_bin["num_bins"])

            # Get cosmic-ray's diff. flux dFdE
            dFdE = diff_flux[sk][pk]
            dFdE_au = np.zeros(dFdE.shape)

            # Get instrument's energy migration M
            M = copy.deepcopy(energy_migration[sk][pk])
            assert_energy_migration_is_valid(M=M)

            # Get instrument's acceptance Q
            Q = acceptance[sk][ok][pk][gk]["mean"]
            Q_au = acceptance[sk][ok][pk][gk]["absolute_uncertainty"]

            # Compute cosmic-ray-rate in reco energy Rreco
            for ereco in range(energy_bin["num_bins"]):
                _tmp_sum = np.zeros(energy_bin["num_bins"])
                _tmp_sum_au = np.zeros(energy_bin["num_bins"])
                for etrue in range(energy_bin["num_bins"]):
                    (
                        _tmp_sum[etrue],
                        _tmp_sum_au[etrue],
                    ) = irf.utils.multiply_elemnetwise_au(
                        x=[
                            dFdE[etrue],
                            M["reco_given_true"][etrue, ereco],
                            Q[etrue],
                            energy_bin["width"][etrue],
                        ],
                        x_au=[
                            dFdE_au[etrue],
                            M["reco_given_true_abs_unc"][etrue, ereco],
                            Q_au[etrue],
                            energy_bin__width__au[etrue],
                        ],
                    )
                (
                    Rreco[sk][ok][pk][ereco],
                    Rreco_au[sk][ok][pk][ereco],
                ) = irf.utils.sum_elemnetwise_au(x=_tmp_sum, x_au=_tmp_sum_au,)

            # Compute cosmic-ray-rate in true energy Rtrue
            for etrue in range(energy_bin["num_bins"]):
                (
                    Rtrue[sk][ok][pk][etrue],
                    Rtrue_au[sk][ok][pk][etrue],
                ) = irf.utils.multiply_elemnetwise_au(
                    x=[dFdE[etrue], Q[etrue], energy_bin["width"][etrue]],
                    x_au=[dFdE_au[etrue], Q_au[etrue], 0.0],
                )

            # cross check
            # -----------
            # Integral rate over all energy-bins must not change (much) under
            # energy migration.
            total_Rtrue = np.sum(Rtrue[sk][ok][pk][:])
            total_Rreco = np.sum(Rreco[sk][ok][pk][:])
            assert 0.7 < total_Rtrue / total_Rreco < 1.3

# export
# ------
for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in COSMIC_RAYS:
            os.makedirs(os.path.join(pa["out_dir"], sk, ok, pk), exist_ok=True)

for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in COSMIC_RAYS:
            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, pk, "reco" + ".json"),
                {
                    "comment": "Rate after all cuts VS reco energy",
                    "unit": "s$^{-1}$",
                    "mean": Rreco[sk][ok][pk],
                    "absolute_uncertainty": Rreco_au[sk][ok][pk],
                    "energy_binning_key": energy_bin["key"],
                    "symbol": "Rreco",
                },
            )

            json_numpy.write(
                os.path.join(pa["out_dir"], sk, ok, pk, "true" + ".json"),
                {
                    "comment": "Rate after all cuts VS true energy",
                    "unit": "s$^{-1}$",
                    "mean": Rtrue[sk][ok][pk],
                    "absolute_uncertainty": Rtrue[sk][ok][pk],
                    "energy_binning_key": energy_bin["key"],
                    "symbol": "Rtrue",
                },
            )

# plot
# ----
for sk in SITES:
    for ok in ONREGION_TYPES:
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
        for pk in COSMIC_RAYS:

            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Rreco[sk][ok][pk],
                bincounts_upper=Rreco[sk][ok][pk] - Rreco_au[sk][ok][pk],
                bincounts_lower=Rreco[sk][ok][pk] + Rreco_au[sk][ok][pk],
                linestyle="-",
                linecolor=sum_config["plot"]["particle_colors"][pk],
                face_color=sum_config["plot"]["particle_colors"][pk],
                face_alpha=0.25,
            )

            alpha = 0.25
            seb.ax_add_histogram(
                ax=ax,
                bin_edges=energy_bin["edges"],
                bincounts=Rtrue[sk][ok][pk],
                bincounts_upper=Rtrue[sk][ok][pk] - Rtrue_au[sk][ok][pk],
                bincounts_lower=Rtrue[sk][ok][pk] + Rtrue_au[sk][ok][pk],
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
