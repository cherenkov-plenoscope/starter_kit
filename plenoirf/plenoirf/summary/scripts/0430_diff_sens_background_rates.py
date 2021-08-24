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

iacceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0410_interpolate_acceptance")
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


def interpolate_migration_matrix(
    migration_matrix_counts,
    migration_matrix_counts_abs_unc,
    bin_centers,
    new_bin_centers,
):
    M = irf.utils.log10interp2d(
        xp=bin_centers,
        yp=bin_centers,
        x=new_bin_centers,
        y=new_bin_centers,
        fp=migration_matrix_counts,
    )

    M_abs_unc = irf.utils.log10interp2d(
        xp=bin_centers,
        yp=bin_centers,
        x=new_bin_centers,
        y=new_bin_centers,
        fp=migration_matrix_counts_abs_unc,
    )
    # account for lower statistics in smaller bins
    # --------------------------------------------
    M_abs_unc *= np.sqrt(len(new_bin_centers) / len(bin_centers))

    # normalize probability
    # ---------------------
    for i_ax0 in range(len(new_bin_centers)):
        _sum = np.sum(M[i_ax0, :])
        if _sum > 0.0:
            M[i_ax0, :] /= _sum
            M_abs_unc[i_ax0, :] /= _sum

    return M, M_abs_unc


def derive_migration_matrix_by_ax0(
    migration_matrix_counts,
    migration_matrix_counts_abs_unc,
    ax0_bin_widths,
):
    M = migration_matrix_counts
    M_au = migration_matrix_counts_abs_unc

    dMdE = np.zeros(M.shape)
    dMdE_au = np.zeros(M.shape)
    for i_ax0 in range(len(ax0_bin_widths)):
        _sum = np.sum(M[i_ax0, :])
        if _sum > 0.0:
            dMdE[i_ax0, :] = M[i_ax0, :] / ax0_bin_widths[:]
            dMdE_au[i_ax0, :] = M_au[i_ax0, :] / ax0_bin_widths[:]
    return dMdE, dMdE_au


def write_figure_diff_energy_migration_matrix(
    dMdE,
    dMdE_au,
    energy_bin_edges,
    path,
):
    energy_bin_width = np.diff(energy_bin_edges)

    # integrate probability along ax1
    # -------------------------------
    M_back = np.array(dMdE)
    for i_ax0 in range(len(energy_bin_width)):
        M_back[i_ax0, :] *= energy_bin_width

    fig = seb.figure(seb.FIGURE_1_1)
    ax_c = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
    ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax_c.pcolormesh(
        energy_bin_edges,
        energy_bin_edges,
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
    fig.savefig(path)
    seb.close_figure(fig)



# prepare energy_migration
# ------------------------
diff_energy_migration = {}
diff_energy_migration_au = {}
for sk in SITES:
    diff_energy_migration[sk] = {}
    diff_energy_migration_au[sk] = {}
    for pk in PARTICLES:
        print("energy_migration", sk, pk)

        M, M_au = interpolate_migration_matrix(
            migration_matrix_counts=_energy_migration[sk][pk]["confusion_matrix"]["counts"],
            migration_matrix_counts_abs_unc=_energy_migration[sk][pk]["confusion_matrix"]["counts_abs_unc"],
            bin_centers=energy_bin["centers"],
            new_bin_centers=fine_energy_bin["centers"],
        )

        dMdE, dMdE_au = derive_migration_matrix_by_ax0(
            migration_matrix_counts=M,
            migration_matrix_counts_abs_unc=M_au,
            ax0_bin_widths=fine_energy_bin["width"],
        )

        write_figure_diff_energy_migration_matrix(
            dMdE=dMdE,
            dMdE_au=dMdE_au,
            energy_bin_edges=fine_energy_bin["edges"],
            path=os.path.join(
                pa["out_dir"],
                sk + "_" + pk + "_energy_migration_interpolated.jpg",
            ),
        )

        json_numpy.write(
            os.path.join(
                pa["out_dir"],
                sk + "_" + pk + "_differential_energy_migration_interpolated.json",
            ),
            {
                "unit": "(GeV)$^{-1}$",
                "mean": dMdE,
                "absolute_uncertainty": dMdE_au,
            },
        )

        diff_energy_migration[sk][pk] = dMdE
        diff_energy_migration_au[sk][pk] = dMdE_au



# prepare diff-flux
# -----------------
diff_flux = {}
for sk in SITES:
    diff_flux[sk] = {}
    for pk in COSMIC_RAYS:
        diff_flux[sk][pk] = airshower_fluxes[sk][pk]["differential_flux"][
            "values"
        ]

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
            (fine_energy_bin["num_bins"], num_onregion_sizes)
        )
        dRdE[sk][pk] = np.zeros(
            (fine_energy_bin["num_bins"], num_onregion_sizes)
        )

        dRtdEt_au[sk][pk] = np.zeros(
            (fine_energy_bin["num_bins"], num_onregion_sizes)
        )
        dRdE_au[sk][pk] = np.zeros(
            (fine_energy_bin["num_bins"], num_onregion_sizes)
        )

        for ok in range(num_onregion_sizes):
            print("apply", sk, pk, ok)
            dFdE = diff_flux[sk][pk]
            dFdE_au = np.zeros(dFdE.shape)

            dMdE = diff_energy_migration[sk][pk]
            dMdE_au = diff_energy_migration_au[sk][pk]

            Q = iacceptance[sk][pk][gk]["mean"][:, ok]
            Q_au = iacceptance[sk][pk][gk]["absolute_uncertainty"][:, ok]

            fine_energy_bin_width_au = np.zeros(fine_energy_bin["width"].shape)

            for ereco in range(fine_energy_bin["num_bins"]):
                _P = np.zeros(fine_energy_bin["num_bins"])
                _P_au = np.zeros(fine_energy_bin["num_bins"])
                for etrue in range(fine_energy_bin["num_bins"]):

                    (
                        _P[etrue],
                        _P_au[etrue],
                    ) = irf.utils.multiply_elemnetwise_au(
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
                    dRtdEt_au[sk][pk][ereco, ok],
                ) = irf.utils.sum_elemnetwise_au(x=_P, x_au=_P_au,)

            for ee in range(fine_energy_bin["num_bins"]):
                (
                    dRdE[sk][pk][ee, ok],
                    dRdE_au[sk][pk][ee, ok],
                ) = irf.utils.multiply_elemnetwise_au(
                    x=[dFdE[ee], Q[ee]], x_au=[dFdE_au[ee], Q_au[ee]],
                )

            # cross check
            # -----------
            # total rate must not change under energy migration
            total_R = np.sum(dRdE[sk][pk][:, ok] * fine_energy_bin["width"][:])
            total_Rt = np.sum(
                dRtdEt[sk][pk][:, ok] * fine_energy_bin["width"][:]
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
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            sk + "_" + pk + "_differential_rates_vs_reco_energy.jpg",
        )
    )
    seb.close_figure(fig)
