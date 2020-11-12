#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

psf = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0208_fitting_light_field_benchmark")
)

num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "point_spread_function"
]
energy_lower_edge = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper_edge = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower_edge, energy_upper_edge, num_energy_bins + 1
)

fov_radius_deg = (
    0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
)

fc16by9 = sum_config["plot"]["16_by_9"]

for site_key in psf:
    particle_key = "gamma"

    psf_68_radius_deg = np.array(
        psf[site_key][particle_key]["containment_angle_vs_energy"]["mean"]
    )
    psf_68_radius_deg_unc = np.array(
        psf[site_key][particle_key]["containment_angle_vs_energy"][
            "relative_uncertainty"
        ]
    )

    fig = irf.summary.figure.figure(fc16by9)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=energy_bin_edges,
        bincounts=psf_68_radius_deg,
        linestyle="-",
        linecolor="k",
        bincounts_upper=psf_68_radius_deg * (1 + psf_68_radius_deg_unc),
        bincounts_lower=psf_68_radius_deg * (1 - psf_68_radius_deg_unc),
        face_color="k",
        face_alpha=0.25,
    )
    ax.semilogx()
    ax.set_xlabel("energy / GeV")
    ax.set_ylabel("$\\theta_{68\\%}$ / $1^\\circ$")
    ax.set_xlim([energy_lower_edge, energy_upper_edge])
    ax.set_ylim([0, fov_radius_deg])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            pa["out_dir"], "{:s}_gamma_psf_radial.jpg".format(site_key)
        )
    )
    plt.close(fig)
