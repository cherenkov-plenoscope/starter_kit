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

gk = "diffuse"

_energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

_acceptance = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0300_onregion_trigger_acceptance"
    )
)

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]
fine_energy_bin_edge_matches = []
for energy in energy_bin["edges"]:
    idx_near = np.argmin(np.abs(fine_energy_bin["edges"] - energy))
    fine_energy_bin_edge_matches.append(idx_near)


# prepare acceptances
# -------------------
acceptance = {}
acceptance_au = {}
for sk in SITES:
    acceptance[sk] = {}
    acceptance_au[sk] = {}
    for pk in COSMIC_RAYS:
        acceptance[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))
        acceptance_au[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))
        for ok in range(num_onregion_sizes):
            print("acceptance", sk, pk, ok)
            _Q = _acceptance[sk][pk][gk]["mean"][:, ok]
            _Q_ru = _acceptance[sk][pk][gk]["relative_uncertainty"][:, ok]
            _Q_ru[np.isnan(_Q_ru)] = 0.0
            _Q_au = _Q * _Q_ru
            acceptance[sk][pk][:, ok] = irf.utils.log10interp(
                x=fine_energy_bin["centers"],
                xp=energy_bin["centers"],
                fp=_Q,
            )
            acceptance_au[sk][pk][:, ok] = irf.utils.log10interp(
                x=fine_energy_bin["centers"],
                xp=energy_bin["centers"],
                fp=_Q_au,
            )

    Ok  = 2
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        ax.plot(
            fine_energy_bin["centers"],
            acceptance[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
        )
        ax.plot(
            energy_bin["centers"],
            _acceptance[sk][pk][gk]["mean"][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
            marker="o"
        )
        ax.fill_between(
            x=fine_energy_bin["centers"],
            y1=acceptance[sk][pk][:, Ok] - acceptance_au[sk][pk][:, Ok],
            y2=acceptance[sk][pk][:, Ok] + acceptance_au[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
            alpha=0.2,
            linewidth=0.0,
        )
    ax.set_ylabel("acceptance / m$^{2}$ sr")
    ax.set_xlabel("energy / GeV")
    ax.loglog()
    fig.savefig(os.path.join(pa["out_dir"], sk + "_" + pk + "_acceptance_interpolated.jpg"))
    seb.close_figure(fig)


# prepare energy_migration
# ------------------------
diff_energy_migration = {}
diff_energy_migration_au = {}
for sk in SITES:
    diff_energy_migration[sk] = {}
    diff_energy_migration_au[sk] = {}
    for pk in COSMIC_RAYS:
        print("energy_migration", sk, pk)

        M = irf.utils.log10interp2d(
            xp=energy_bin["centers"],
            x=fine_energy_bin["centers"],
            yp=energy_bin["centers"],
            y=fine_energy_bin["centers"],
            fp=_energy_migration[sk][pk]["confusion_matrix"]["counts"],
        )

        _, counts_abs_unc = irf.utils.estimate_rel_abs_uncertainty_in_counts(
            counts=_energy_migration[sk][pk]["confusion_matrix"]["counts"]
        )

        M_au = irf.utils.log10interp2d(
            xp=energy_bin["centers"],
            x=fine_energy_bin["centers"],
            yp=energy_bin["centers"],
            y=fine_energy_bin["centers"],
            fp=counts_abs_unc,
        )
        # account for lower statistics in smaller bins
        # --------------------------------------------
        M_au *= fine_energy_bin["num_bins"] / energy_bin["num_bins"]

        # normalize probability
        # ---------------------
        for etrue in range(fine_energy_bin["num_bins"]):
            sumetru = np.sum(M[etrue, :])
            if sumetru > 0.0:
                M[etrue, :] /= sumetru
                M_au[etrue, :] /= sumetru

        # differentiate
        #--------------
        dMdE = np.zeros(M.shape)
        dMdE_au = np.zeros(M.shape)
        for etrue in range(fine_energy_bin["num_bins"]):
            sumetru = np.sum(M[etrue, :])
            if sumetru > 0.0:
                dMdE[etrue, :] = M[etrue, :] / fine_energy_bin["width"][:]
                dMdE_au[etrue, :] = M_au[etrue, :] / fine_energy_bin["width"][:]

        diff_energy_migration[sk][pk] = dMdE
        diff_energy_migration_au[sk][pk] = dMdE_au

        M_back = np.array(diff_energy_migration[sk][pk])
        for etrue in range(fine_energy_bin["num_bins"]):
            M_back[etrue, :] *= fine_energy_bin["width"]

        fig = seb.figure(seb.FIGURE_1_1)
        ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
        ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
        _pcm_confusion = ax_c.pcolormesh(
            fine_energy_bin["edges"],
            fine_energy_bin["edges"],
            np.transpose(M_back),
            cmap="Greys",
            norm=seb.plt_colors.PowerNorm(gamma=0.5),
        )
        ax_c.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
        ax_c.set_aspect("equal")
        ax_c.set_title("normalized in each column")
        ax_c.set_ylabel("reco. energy / GeV")
        ax_c.set_xlabel("energy / GeV")
        ax_c.loglog()
        fig.savefig(os.path.join(pa["out_dir"], sk + "_" + pk + "_energy_migration_interpolated.jpg"))
        seb.close_figure(fig)


# prepare diff-flux
# -----------------
diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"]["values"]

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
        dRtdEt[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))
        dRdE[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))

        dRtdEt_au[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))
        dRdE_au[sk][pk] = np.zeros((fine_energy_bin["num_bins"], num_onregion_sizes))

        for ok in range(num_onregion_sizes):
            print("apply", sk, pk, ok)
            dFdE = diff_flux[sk][pk]
            dFdE_au = np.zeros(dFdE.shape)

            dMdE = diff_energy_migration[sk][pk]
            dMdE_au = diff_energy_migration_au[sk][pk]

            Q = acceptance[sk][pk][:, ok]
            Q_au = acceptance_au[sk][pk][:, ok]

            fine_energy_bin_width_au = np.zeros(fine_energy_bin["width"].shape)

            for ereco in range(fine_energy_bin["num_bins"]):
                _P = np.zeros(fine_energy_bin["num_bins"])
                _P_au = np.zeros(fine_energy_bin["num_bins"])
                for etrue in range(fine_energy_bin["num_bins"]):

                    _P[etrue], _P_au[etrue] = irf.utils.multiply_elemnetwise_au(
                        x=[
                            dFdE[etrue],
                            dMdE[etrue, ereco],
                            Q[etrue],
                            fine_energy_bin["width"][etrue],
                        ],
                        x_au=[
                            dFdE_au[etrue],
                            dMdE_au[etrue, ereco],
                            Q_au[etrue],
                            fine_energy_bin_width_au[etrue],
                        ],
                    )

                (
                    dRtdEt[sk][pk][ereco, ok],
                    dRtdEt_au[sk][pk][ereco, ok]
                )= irf.utils.sum_elemnetwise_au(
                    x=_P, x_au=_P_au,
                )

            for ee in range(fine_energy_bin["num_bins"]):
                (
                    dRdE[sk][pk][ee, ok], dRdE_au[sk][pk][ee, ok]
                ) = irf.utils.multiply_elemnetwise_au(
                        x=[dFdE[ee], Q[ee]],
                        x_au=[dFdE_au[ee], Q_au[ee]],
                    )

            # cross check
            # -----------
            # total rate must not change under energy migration
            total_R = np.sum(dRdE[sk][pk][:, ok] * fine_energy_bin["width"][:])
            total_Rt = np.sum(dRtdEt[sk][pk][:, ok] * fine_energy_bin["width"][:])

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
            },
        )


Ok = 2
for sk in SITES:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)
    for pk in COSMIC_RAYS:
        ax.plot(
            fine_energy_bin["centers"],
            dRtdEt[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
        )

        ax.fill_between(
            x=fine_energy_bin["centers"],
            y1=dRtdEt[sk][pk][:, Ok] - dRtdEt_au[sk][pk][:, Ok],
            y2=dRtdEt[sk][pk][:, Ok] + dRtdEt_au[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
            alpha=0.2,
            linewidth=0.0,
        )

        ax.plot(
            fine_energy_bin["centers"],
            dRdE[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
            linestyle=":",
            alpha=0.1,
        )
        ax.fill_between(
            x=fine_energy_bin["centers"],
            y1=dRdE[sk][pk][:, Ok] - dRdE_au[sk][pk][:, Ok],
            y2=dRdE[sk][pk][:, Ok] + dRdE_au[sk][pk][:, Ok],
            color=sum_config["plot"]["particle_colors"][pk],
            alpha=0.05,
            linewidth=0.0,
        )

    ax.set_ylabel("differential rate / s$^{-1}$ (GeV)$^{-1}$")
    ax.set_xlabel("reco. energy / GeV")
    ax.loglog()
    fig.savefig(os.path.join(pa["out_dir"], sk + "_" + pk + "_differential_rates_vs_reco_energy.jpg"))
    seb.close_figure(fig)
