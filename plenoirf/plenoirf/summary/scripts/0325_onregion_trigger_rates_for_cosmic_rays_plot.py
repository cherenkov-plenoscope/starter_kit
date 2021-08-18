#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import lima1983analysis
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

onregion_rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)

gamma_source = json_numpy.read(
    os.path.join(
        pa["summary_dir"], "0009_flux_of_gamma_rays", "reference_source.json"
    )
)

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

fine_energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["interpolation"]

particle_colors = sum_config["plot"]["particle_colors"]

cosmic_ray_keys = list(irf_config["config"]["particles"].keys())
cosmic_ray_keys.remove("gamma")

LiMa_alpha = sum_config["on_off_measuremnent"]["on_over_off_ratio"]

observation_time_s = 60 * 5

for sk in irf_config["config"]["sites"]:
    fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
    ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

    signal_rate_in_onregion = np.array(
        onregion_rates[sk]["gamma"]["integral_rate"]["mean"]
    )

    background_rate_in_single_off_region = np.zeros(num_bins_onregion_radius)
    LiMa_S = np.zeros(num_bins_onregion_radius)
    for cosmic_ray_key in cosmic_ray_keys:
        background_rate_in_single_off_region += np.array(
            onregion_rates[sk][cosmic_ray_key]["integral_rate"]["mean"]
        )

    background_rate_in_all_off_regions = (
        1.0 / LiMa_alpha * background_rate_in_single_off_region
    )

    for onr in range(num_bins_onregion_radius):
        N_s = signal_rate_in_onregion[onr] * observation_time_s
        N_on = (
            N_s
            + background_rate_in_single_off_region[onr] * observation_time_s
        )
        N_off = background_rate_in_all_off_regions[onr] * observation_time_s

        LiMa_S[onr] = lima1983analysis.estimate_S_eq17(
            N_on=N_on, N_off=N_off, alpha=LiMa_alpha
        )

    ax.plot(onregion_radii_deg, LiMa_S, "ko")
    ax.plot(onregion_radii_deg, LiMa_S, "k")
    ax.set_title(
        gamma_source["name"]
        + ", observation-time: {:d}s, ".format(int(observation_time_s))
        + r"Li and Ma Eq.17, $\alpha$: "
        + "{:.2f}".format(LiMa_alpha)
    )
    ax.set_xlabel(r"onregion-radius at 100p.e. / $^{\circ}$")
    ax.set_ylabel(r"Significance / 1")
    ax.set_ylim([0, 10])
    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "{:s}_LiMaEq17_significance_vs_onregion_radius.jpg".format(sk),
        )
    )
    seb.close_figure(fig)

mean_key = "mean"
unc_key = "absolute_uncertainty"

for sk in irf_config["config"]["sites"]:
    for ok in range(num_bins_onregion_radius):
        fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
        ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

        text_y = 0.7
        for pk in irf_config["config"]["particles"]:
            dRdE = onregion_rates[sk][pk]["differential_rate"][mean_key][:, ok]
            dRdE_au = onregion_rates[sk][pk]["differential_rate"][unc_key][
                :, ok
            ]
            ax.plot(
                fine_energy_bin["centers"],
                dRdE,
                color=sum_config["plot"]["particle_colors"][pk],
            )
            ax.fill_between(
                x=fine_energy_bin["centers"],
                y1=dRdE - dRdE_au,
                y2=dRdE + dRdE_au,
                facecolor=sum_config["plot"]["particle_colors"][pk],
                alpha=0.2,
                linewidth=0.0,
            )
            ax.text(
                0.6,
                0.1 + text_y,
                pk,
                color=particle_colors[pk],
                transform=ax.transAxes,
            )
            ir = onregion_rates[sk][pk]["integral_rate"][mean_key][ok]
            ir_abs_unc = onregion_rates[sk][pk]["integral_rate"][unc_key][ok]
            ax.text(
                0.7,
                0.1 + text_y,
                r"{: 8.1f}$\pm{:.1f}$ s$^{{-1}}$".format(ir, ir_abs_unc),
                color="k",
                family="monospace",
                transform=ax.transAxes,
            )
            text_y += 0.06

        ax.set_xlim(fine_energy_bin["limits"])
        ax.set_ylim([1e-5, 1e3])
        ax.loglog()
        ax.set_xlabel("Energy / GeV")
        ax.set_ylabel("differential rate /\ns$^{-1}$ (GeV)$^{-1}$")
        fig.savefig(
            os.path.join(
                pa["out_dir"],
                "{:s}_differential_event_rates_in_onregion_onr{:06d}.jpg".format(
                    sk, ok
                ),
            )
        )
        seb.close_figure(fig)
