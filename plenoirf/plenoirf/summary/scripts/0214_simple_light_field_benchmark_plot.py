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
    os.path.join(pa["summary_dir"], "0213_simple_light_field_benchmark")
)

fov_radius_deg = (
    0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
)

fc16by9 = sum_config["plot"]["16_by_9"]

theta_labels = {
    "theta": r"\theta{}",
    "theta_para": r"\theta{}_\mathrm{parallel}",
    "theta_perp": r"\theta{}_\mathrm{perpendicular}",
}

for site_key in psf:
    for particle_key in ["gamma"]:

        print(site_key, particle_key)

        t2psf = psf[site_key][particle_key][
            "theta_square_histogram_vs_energy_vs_core_radius"
        ]

        num_radius_bins = len(t2psf["core_radius_square_bin_edges_m2"]) - 1
        num_energy_bins = len(t2psf["energy_bin_edges_GeV"]) - 1

        for theta_key in ["theta", "theta_para", "theta_perp"]:

            for rad_idx in range(num_radius_bins):
                for ene_idx in range(num_energy_bins):

                    ene_start = t2psf["energy_bin_edges_GeV"][ene_idx]
                    ene_stop = t2psf["energy_bin_edges_GeV"][ene_idx + 1]

                    rad_start = np.sqrt(
                        t2psf["core_radius_square_bin_edges_m2"][rad_idx]
                    )
                    rad_stop = np.sqrt(
                        t2psf["core_radius_square_bin_edges_m2"][rad_idx + 1]
                    )

                    theta_square_bin_edges_deg2 = np.array(
                        t2psf["theta_square_bin_edges_deg2"]
                    )

                    bin_count = t2psf[theta_key]["mean"][ene_idx][rad_idx]
                    bin_count = np.array(bin_count)
                    bin_count = bin_count / np.max(bin_count)
                    bin_count_relunc = t2psf[theta_key][
                        "relative_uncertainty"
                    ][ene_idx][rad_idx]
                    bin_count_relunc = np.array(bin_count_relunc)

                    fig = irf.summary.figure.figure(fc16by9)
                    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
                    irf.summary.figure.ax_add_hist(
                        ax=ax,
                        bin_edges=theta_square_bin_edges_deg2,
                        bincounts=bin_count,
                        bincounts_upper=bin_count * (1 + bin_count_relunc),
                        bincounts_lower=bin_count * (1 - bin_count_relunc),
                        linestyle="-",
                        linecolor="k",
                        face_color="k",
                        face_alpha=0.25,
                    )
                    ax.set_xlabel(
                        r"$("
                        + theta_labels[theta_key]
                        + r")^{2}$ / $(1^\circ{})^{2}$"
                    )
                    ax.set_ylabel("relative intensity / 1")

                    ene_info = "energy      {: 7.1f} - {: 7.1f} GeV".format(
                        ene_start, ene_stop
                    )
                    rad_info = "core-radius {: 7.1f} - {: 7.1f} m".format(
                        rad_start, rad_stop
                    )

                    ax.set_title(
                        " {:s}\n{:s}".format(ene_info, rad_info),
                        family="monospace",
                    )

                    ax.set_ylim([0.0, 1.25])
                    ax.set_xlim(
                        [
                            np.min(theta_square_bin_edges_deg2),
                            np.max(theta_square_bin_edges_deg2),
                        ]
                    )
                    ax.spines["top"].set_color("none")
                    ax.spines["right"].set_color("none")
                    ax.grid(
                        color="k", linestyle="-", linewidth=0.66, alpha=0.1
                    )
                    fig.savefig(
                        os.path.join(
                            pa["out_dir"],
                            "{:s}_{:s}_{:s}_rad{:06d}_ene{:06d}.jpg".format(
                                theta_key,
                                site_key,
                                particle_key,
                                rad_idx,
                                ene_idx,
                            ),
                        )
                    )
                    plt.close(fig)
