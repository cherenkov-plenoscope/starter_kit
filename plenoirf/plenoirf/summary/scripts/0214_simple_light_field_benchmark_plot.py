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

def write_theta_square_figure(
    path,
    theta_square_bin_edges_deg2,
    bin_count,
    bin_count_relunc,
    info_title,
    theta_label,
    square=True,
    ylim_stop=1.25
):
    if square:
        tts_label = r"$(" + theta_label + r")^{2}$ / $(1^\circ{})^{2}$"
        tts = theta_square_bin_edges_deg2
    else:
        tts_label = r"$" + theta_label + r"$ / $1^\circ{}$"
        tts = np.sqrt(theta_square_bin_edges_deg2)

    fig = irf.summary.figure.figure(fc16by9)
    ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=tts,
        bincounts=bin_count,
        bincounts_upper=bin_count * (1 + bin_count_relunc),
        bincounts_lower=bin_count * (1 - bin_count_relunc),
        linestyle="-",
        linecolor="k",
        face_color="k",
        face_alpha=0.25,
    )
    ax.plot(
        0.5 * (tts[0:-1] + tts[1:]),
        bin_count,
        "k-",
        alpha=0.5
    )

    ax.set_xlabel(tts_label)
    ax.set_ylabel("relative intensity / 1")
    fig.suptitle(info_title, family="monospace")
    ax.set_ylim([0.0, ylim_stop])
    ax.set_xlim([0.0, np.max(tts)])
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.grid(
        color="k", linestyle="-", linewidth=0.66, alpha=0.1
    )

    fig.savefig(path)
    plt.close(fig)


for site_key in psf:
    for particle_key in ["gamma"]:

        # theta-square vs energy vs core-radius
        # -------------------------------------

        for theta_key in ["theta", "theta_para", "theta_perp"]:

            t2 = psf[site_key][particle_key][
                "{theta_key:s}_square_histogram_vs_energy_vs_core_radius".format(
                    theta_key=theta_key
                )
            ]

            num_radius_bins = len(t2["core_radius_square_bin_edges_m2"]) - 1
            num_energy_bins = len(t2["energy_bin_edges_GeV"]) - 1

            for ene in range(num_energy_bins):

                ene_start = t2["energy_bin_edges_GeV"][ene]
                ene_stop = t2["energy_bin_edges_GeV"][ene + 1]

                ene_info = "energy      {: 7.1f} - {: 7.1f} GeV".format(
                    ene_start, ene_stop
                )

                for rad in range(num_radius_bins):

                    t2_ene_rad = t2["histogram"][ene][rad]

                    print(site_key, particle_key, theta_key, ene, rad)

                    rad_start = np.sqrt(
                        t2["core_radius_square_bin_edges_m2"][rad]
                    )
                    rad_stop = np.sqrt(
                        t2["core_radius_square_bin_edges_m2"][rad + 1]
                    )

                    rad_info = "core-radius {: 7.1f} - {: 7.1f} m".format(
                        rad_start, rad_stop
                    )

                    bin_count = t2_ene_rad["intensity"]
                    bin_count = np.array(bin_count)
                    bin_count = bin_count / np.max(bin_count)

                    bin_count_relunc = t2_ene_rad["intensity_relative_uncertainty"]
                    bin_count_relunc = np.array(bin_count_relunc)

                    write_theta_square_figure(
                        path=os.path.join(
                            pa["out_dir"],
                            "{:s}_{:s}_{:s}_rad{:06d}_ene{:06d}.jpg".format(
                                theta_key,
                                site_key,
                                particle_key,
                                rad,
                                ene,
                            ),
                        ),
                        theta_square_bin_edges_deg2=np.array(
                            t2_ene_rad["theta_square_bin_edges_deg2"]
                        ),
                        bin_count=bin_count,
                        bin_count_relunc=bin_count_relunc,
                        info_title=" {:s}\n{:s}".format(ene_info, rad_info),
                        theta_label=theta_labels[theta_key],
                        square=False,
                    )
