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

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
ONREGIONS = range(num_onregion_sizes)

_energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality")
)

_acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0405_average_flux_of_airshowers")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]


# prepare energy_migration
# ------------------------
energy_migration = {}
for sk in SITES:
    energy_migration[sk] = {}
    for pk in COSMIC_RAYS:
        energy_migration[sk][pk] = {}
        M = _energy_migration[sk][pk]["confusion_matrix"]
        dMdE = {}
        dMdE["ax0_key"] = M["ax0_key"]
        dMdE["ax1_key"] = M["ax1_key"]

        counts = M["counts"]
        counts_abs_unc = M["counts_abs_unc"]

        dMdE["counts"], dMdE["counts_abs_unc"] = irf.analysis.differential_sensitivity.derive_migration_matrix_by_ax0(
            migration_matrix_counts=M["counts_normalized_on_ax0"],
            migration_matrix_counts_abs_unc=M["counts_normalized_on_ax0_abs_unc"],
            ax0_bin_widths=energy_bin["width"],
        )
        energy_migration[sk][pk] = dMdE


# prepare diff-flux
# -----------------
diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"]


gk = "diffuse"
# prepare acceptance
# ------------------
acceptance = {}
for sk in SITES:
    acceptance[sk] = {}
    for pk in COSMIC_RAYS:
        acceptance[sk][pk] = {gk: {}}
        Q = _acceptance[sk][pk][gk]["mean"]
        Q_ru = _acceptance[sk][pk][gk]["relative_uncertainty"]
        Q_ru[np.isnan(Q_ru)] = 0.0
        Q_au = Q * Q_ru
        acceptance[sk][pk][gk]["mean"] = Q
        acceptance[sk][pk][gk]["absolute_uncertainty"] = Q_au


dRtdEt = {}
dRtdEt_au = {}

dRdE = {}
dRdE_au = {}

for sk in SITES:
    dRtdEt[sk] = {}
    dRtdEt_au[sk] = {}
    dRdE[sk] = {}
    dRdE_au[sk] = {}
    for pk in COSMIC_RAYS:
        dRtdEt[sk][pk] = np.zeros(
            (energy_bin["num_bins"], num_onregion_sizes)
        )
        dRdE[sk][pk] = np.zeros(
            (energy_bin["num_bins"], num_onregion_sizes)
        )

        dRtdEt_au[sk][pk] = np.zeros(
            (energy_bin["num_bins"], num_onregion_sizes)
        )
        dRdE_au[sk][pk] = np.zeros(
            (energy_bin["num_bins"], num_onregion_sizes)
        )

        for ok in range(num_onregion_sizes):
            print("apply", sk, pk, ok)
            dFdE = diff_flux[sk][pk]
            dFdE_au = np.zeros(dFdE.shape)

            assert energy_migration[sk][pk]["ax0_key"] == "true_energy"
            assert energy_migration[sk][pk]["ax1_key"] == "reco_energy"
            dMdE = energy_migration[sk][pk]["counts"]
            dMdE_au = energy_migration[sk][pk]["counts_abs_unc"]

            Q = acceptance[sk][pk][gk]["mean"][:, ok]
            Q_au = acceptance[sk][pk][gk]["absolute_uncertainty"][:, ok]

            energy_bin_width_au = np.zeros(energy_bin["width"].shape)

            for ereco in range(energy_bin["num_bins"]):
                _P = np.zeros(energy_bin["num_bins"])
                _P_au = np.zeros(energy_bin["num_bins"])
                for etrue in range(energy_bin["num_bins"]):

                    (
                        _P[etrue],
                        _P_au[etrue],
                    ) = irf.utils.multiply_elemnetwise_au(
                        x=[
                            dFdE[etrue],
                            dMdE[etrue, ereco],
                            Q[etrue],
                            energy_bin["width"][etrue],
                        ],
                        x_au=[
                            dFdE_au[etrue],
                            dMdE_au[etrue, ereco],
                            Q_au[etrue],
                            energy_bin_width_au[etrue],
                        ],
                    )

                (
                    dRtdEt[sk][pk][ereco, ok],
                    dRtdEt_au[sk][pk][ereco, ok],
                ) = irf.utils.sum_elemnetwise_au(x=_P, x_au=_P_au,)

            for ee in range(energy_bin["num_bins"]):
                (
                    dRdE[sk][pk][ee, ok],
                    dRdE_au[sk][pk][ee, ok],
                ) = irf.utils.multiply_elemnetwise_au(
                    x=[dFdE[ee], Q[ee]], x_au=[dFdE_au[ee], Q_au[ee]],
                )

            # cross check
            # -----------
            # total rate must not change under energy migration
            total_R = np.sum(dRdE[sk][pk][:, ok] * energy_bin["width"][:])
            total_Rt = np.sum(
                dRtdEt[sk][pk][:, ok] * energy_bin["width"][:]
            )

            assert 0.9 < total_R / total_Rt < 1.1


for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in COSMIC_RAYS:
        json_numpy.write(
            os.path.join(sk_dir, pk + ".json"),
            {
                "comment": (
                    "differential rate after all cuts "
                    "VS reco. energy VS onregion"
                ),
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dRtdEt[sk][pk],
                "absolute_uncertainty": dRtdEt_au[sk][pk],
                "energy_binning_key": energy_bin["key"],
            },
        )


Ok = 2
for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:

        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=dRtdEt[sk][pk][:, Ok],
            bincounts_upper=dRtdEt[sk][pk][:, Ok] - dRtdEt_au[sk][pk][:, Ok],
            bincounts_lower=dRtdEt[sk][pk][:, Ok] + dRtdEt_au[sk][pk][:, Ok],
            linestyle="-",
            linecolor=sum_config["plot"]["particle_colors"][pk],
            face_color=sum_config["plot"]["particle_colors"][pk],
            face_alpha=0.25,
        )

        alpha = 0.25
        seb.ax_add_histogram(
            ax=ax,
            bin_edges=energy_bin["edges"],
            bincounts=dRdE[sk][pk][:, Ok],
            bincounts_upper=dRdE[sk][pk][:, Ok] - dRdE_au[sk][pk][:, Ok],
            bincounts_lower=dRdE[sk][pk][:, Ok] + dRdE_au[sk][pk][:, Ok],
            linecolor=sum_config["plot"]["particle_colors"][pk],
            linealpha=alpha,
            linestyle=":",
            face_color=sum_config["plot"]["particle_colors"][pk],
            face_alpha=alpha*0.25,
        )

    ax.set_ylabel("differential rate / s$^{-1}$ (GeV)$^{-1}$")
    ax.set_xlabel("reco. energy / GeV")
    ax.set_ylim([1e-6, 1e4])
    ax.loglog()
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            sk + "_" + pk + "_differential_rates_vs_reco_energy.jpg",
        )
    )
    seb.close_figure(fig)
