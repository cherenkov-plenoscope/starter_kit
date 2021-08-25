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

energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]


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


def write_figure_diffenergy_migration_matrix(
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


for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in PARTICLES:

        M, M_au = interpolate_migration_matrix(
            migration_matrix_counts=energy_migration[sk][pk]["confusion_matrix"]["counts"],
            migration_matrix_counts_abs_unc=energy_migration[sk][pk]["confusion_matrix"]["counts_abs_unc"],
            bin_centers=energy_bin["centers"],
            new_bin_centers=fine_energy_bin["centers"],
        )

        dMdE, dMdE_au = irf.analysis.differential_sensitivity.derive_migration_matrix_by_ax0(
            migration_matrix_counts=M,
            migration_matrix_counts_abs_unc=M_au,
            ax0_bin_widths=fine_energy_bin["width"],
        )

        write_figure_diffenergy_migration_matrix(
            dMdE=dMdE,
            dMdE_au=dMdE_au,
            energy_bin_edges=fine_energy_bin["edges"],
            path=os.path.join(sk_dir, pk + ".jpg"),
        )

        json_numpy.write(
            os.path.join(sk_dir, pk + ".json"),
            {
                "comment": "dM(E'|E)/dE, diff. energy-migration-matrix.",
                "ax0_key": energy_migration[sk][pk]["confusion_matrix"]["ax0_key"],
                "ax1_key": energy_migration[sk][pk]["confusion_matrix"]["ax1_key"],
                "unit": "(GeV)$^{-1}$",
                "counts": dMdE,
                "counts_abs_unc": dMdE_au,
                "energy_binning_key": "interpolation",
            },
        )
