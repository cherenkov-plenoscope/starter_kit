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

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")

num_bins_onregion_radius = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
ONREGIONS = range(num_bins_onregion_radius)


interpreted_rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0428_diff_sens_rate_interpretation"
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


mean_key = "mean"
unc_key = "absolute_uncertainty"

for sk in SITES:
    for ok in ONREGIONS:
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:

            fig = seb.figure(irf.summary.figure.FIGURE_STYLE)
            ax = seb.add_axes(fig=fig, span=irf.summary.figure.AX_SPAN)

            text_y = 0.7
            for pk in COSMIC_RAYS:
                dRdE = interpreted_rates[sk][pk][dk]["differential_rate"][mean_key][:, ok]
                dRdE_au = interpreted_rates[sk][pk][dk]["differential_rate"][unc_key][
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
                text_y += 0.06

            ax.set_xlim(fine_energy_bin["limits"])
            ax.set_ylim([1e-5, 1e3])
            ax.loglog()
            ax.set_xlabel("interpreted energy / GeV")
            ax.set_ylabel("differential rate /\ns$^{-1}$ (GeV)$^{-1}$")
            fig.savefig(
                os.path.join(
                    pa["out_dir"],
                    "{:s}_{:s}_differential_event_rates_in_onregion_onr{:06d}.jpg".format(
                        sk, dk, ok
                    ),
                )
            )
            seb.close_figure(fig)
