#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import pandas
import plenopy as pl
import iminuit
import scipy

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

"""
Objective
=========

Quantify the angular resolution of the plenoscope.

Input
-----
- List of reconstructed gamma-ray-directions
- List of true gamma-ray-directions, energy, and more...

Quantities
----------
- theta
- theta parallel component
- theta perpendicular component

histogram theta2
----------------
- in energy
- in core-radius

"""


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

fig_16_by_9 = dict(sum_config["plot"]["16_by_9"])

# energy
# ------
num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "point_spread_function"
]
energy_lower_edge = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper_edge = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower_edge, energy_upper_edge, num_energy_bins + 1
)
energy_bin_centers = irf.utils.bin_centers(energy_bin_edges)

# core-radius bins
# ----------------
core_radius_square_bin_edges_m2 = np.linspace(
    start=0.0,
    stop=sum_config["point_spread_function"]["core_radius"]["max_radius_m"]
    ** 2,
    num=sum_config["point_spread_function"]["core_radius"]["num_bins"] + 1,
)
num_core_radius_bins = core_radius_square_bin_edges_m2.shape[0] - 1

# theta square bins
# -----------------
theta_square_bin_edges_deg2 = np.linspace(
    start=0.0,
    stop=sum_config["point_spread_function"]["theta_square"]["max_angle_deg"]
    ** 2,
    num=sum_config["point_spread_function"]["theta_square"]["num_bins"] + 1,
)

psf_containment_factor = sum_config["point_spread_function"][
    "containment_factor"
]
pivot_energy_GeV = sum_config["point_spread_function"]["pivot_energy_GeV"]

# psf image bins
# --------------
num_c_bins = 32
fov_radius_deg = 3.05
fov_radius_fine_deg = (1.0 / 5.0) * fov_radius_deg
c_bin_edges_deg = np.linspace(
    -fov_radius_deg,
    fov_radius_deg,
    num_c_bins
)
c_bin_edges_fine_deg = np.linspace(
    -fov_radius_fine_deg,
    fov_radius_fine_deg,
    num_c_bins
)

theta_square_max_deg = sum_config[
    "point_spread_function"][
    "theta_square"][
    "max_angle_deg"
] ** 2

num_containment_fractions = 20
containment_fractions = np.linspace(
    0.0,
    1.0,
    num_containment_fractions + 1
)[1:-1]

# READ reconstruction
# ===================
reconstruction = {}
for sk in irf_config["config"]["sites"]:
    reconstruction[sk] = {}
    for pk in irf_config["config"]["particles"]:
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        reconstruction[sk][pk] = event_table["reconstructed_trajectory"]

level_keys = [
    "primary",
    "cherenkovsize",
    "grid",
    "cherenkovpool",
    "cherenkovsizepart",
    "cherenkovpoolpart",
    "core",
    "trigger",
    "pasttrigger",
    "cherenkovclassification",
]


def make_rectangular_table(
    event_table, reconstruction_table, plenoscope_pointing
):
    rec_evt_tab = spt.cut_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=reconstruction_table[spt.IDX],
        level_keys=level_keys,
    )
    rec_evt_tab = spt.sort_table_on_common_indices(
        table=rec_evt_tab, common_indices=reconstruction_table[spt.IDX],
    )
    rec_evt_df = spt.make_rectangular_DataFrame(rec_evt_tab)

    et_df = pandas.merge(
        left=pandas.DataFrame(
            {
                spt.IDX: reconstruction_table[spt.IDX],
                "reco_cx": reconstruction_table["cx_rad"],
                "reco_cy": reconstruction_table["cy_rad"],
                "reco_x": reconstruction_table["x_m"],
                "reco_y": reconstruction_table["y_m"],
            }
        ),
        right=rec_evt_df,
        on=spt.IDX,
    )

    (
        _true_cx,
        _true_cy,
    ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
        momentum_x_GeV_per_c=et_df["primary.momentum_x_GeV_per_c"],
        momentum_y_GeV_per_c=et_df["primary.momentum_y_GeV_per_c"],
        momentum_z_GeV_per_c=et_df["primary.momentum_z_GeV_per_c"],
        plenoscope_pointing=plenoscope_pointing,
    )
    et_df["true_cx"] = _true_cx
    et_df["true_cy"] = _true_cy
    et_df["true_x"] = -et_df["core.core_x_m"]
    et_df["true_y"] = -et_df["core.core_y_m"]
    et_df["true_r"] = np.hypot(et_df["true_x"], et_df["true_y"])

    # w.r.t. source
    # -------------
    c_para, c_perp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=et_df["reco_cx"],
        cer_cy_rad=et_df["reco_cy"],
        cer_x_m=0.0,
        cer_y_m=0.0,
        primary_cx_rad=et_df["true_cx"],
        primary_cy_rad=et_df["true_cy"],
        primary_core_x_m=et_df["true_x"],
        primary_core_y_m=et_df["true_y"],
    )

    et_df["theta_para"] = c_para
    et_df["theta_perp"] = c_perp

    et_df["theta"] = np.hypot(
        et_df["reco_cx"] - et_df["true_cx"],
        et_df["reco_cy"] - et_df["true_cy"],
    )

    return et_df.to_records(index=False)


def empty_dim2(dim0, dim1):
    return [[None for ii in range(dim1)] for jj in range(dim0)]


def estimate_containments_theta_deg(
    containment_fractions,
    theta_deg,
):
    conta_deg = np.nan * np.ones(containment_fractions.shape[0])
    conta_deg_relunc = np.nan * np.ones(containment_fractions.shape[0])
    for con in range(containment_fractions.shape[0]):
        ca = irf.analysis.gamma_direction.estimate_containment_radius(
            theta_deg=theta_deg,
            psf_containment_factor=containment_fractions[con],
        )
        conta_deg[con] = ca[0]
        conta_deg_relunc[con] = ca[1]
    return conta_deg, conta_deg_relunc


def guess_theta_square_bin_edges_deg(
    theta_square_max_deg,
    theta_deg,
    num_min=10,
    num_max=2048,
):
    num_t2_bins = int(np.sqrt(theta_deg.shape[0]))
    num_t2_bins = np.max([num_min, num_t2_bins])

    it = 0
    while True:
        it += 1
        if it > 64:
            break

        theta_square_bin_edges_deg2 = np.linspace(
            start=0.0,
            stop=theta_square_max_deg,
            num=num_t2_bins + 1,
        )

        bc = irf.analysis.gamma_direction.histogram_theta_square(
            theta_deg=theta_deg,
            theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
        )[0]

        # print("it1", it, "bc", bc[0:num_min])

        if np.argmax(bc) != 0:
            break

        if bc[0] > 2.0 * bc[1] and bc[1] > 0:
            num_t2_bins = int(1.1 * num_t2_bins)
            num_t2_bins = np.min([num_max, num_t2_bins])
        else:
            break

    it2 = 0
    while True:
        it2 += 1
        if it2 > 64:
            break

        # print("it2", it, "bc", bc[0:num_min])

        theta_square_bin_edges_deg2 = np.linspace(
            start=0.0,
            stop=theta_square_max_deg,
            num=num_t2_bins + 1,
        )

        bc = irf.analysis.gamma_direction.histogram_theta_square(
            theta_deg=theta_deg,
            theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
        )[0]

        if np.sum(bc[0:num_min] == 0) > 0.33 * num_min:
            num_t2_bins = int(0.8 * num_t2_bins)
            num_t2_bins = np.max([num_min, num_t2_bins])
        else:
            break

    return theta_square_bin_edges_deg2


for sk in reconstruction:
    for pk in reconstruction[sk]:

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        _event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )

        reconstructed_event_table = make_rectangular_table(
            event_table=_event_table,
            reconstruction_table=reconstruction[sk][pk],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        rectab = reconstructed_event_table

        # theta-square vs energy vs core-radius
        # -------------------------------------

        hist_ene_rad = {
            "energy_bin_edges_GeV": energy_bin_edges,
            "core_radius_square_bin_edges_m2": core_radius_square_bin_edges_m2,
            "histogram": empty_dim2(
                num_energy_bins,
                num_core_radius_bins
            ),
        }

        hist_ene = {
            "energy_bin_edges_GeV": energy_bin_edges,
            "histogram": [None for ii in range(num_energy_bins)],
        }

        cont_ene_rad = {
            "energy_bin_edges_GeV": energy_bin_edges,
            "core_radius_square_bin_edges_m2": core_radius_square_bin_edges_m2,
            "containment_fractions": containment_fractions,
            "containment": empty_dim2(
                num_energy_bins,
                num_core_radius_bins
            ),
        }
        cont_ene = {
            "energy_bin_edges_GeV": energy_bin_edges,
            "containment_fractions": containment_fractions,
            "containment": [None for ii in range(num_energy_bins)],
        }

        for the in ["theta", "theta_para", "theta_perp"]:

            h_ene_rad = dict(hist_ene_rad)
            h_ene = dict(hist_ene)

            c_ene_rad = dict(cont_ene_rad)
            c_ene = dict(cont_ene)

            for ene in range(num_energy_bins):

                energy_start = energy_bin_edges[ene]
                energy_stop = energy_bin_edges[ene + 1]
                ene_mask = np.logical_and(
                    rectab["primary.energy_GeV"] >= energy_start,
                    rectab["primary.energy_GeV"] < energy_stop,
                )

                ene_theta_deg = np.rad2deg(rectab[the][ene_mask])
                ene_theta_deg = np.abs(ene_theta_deg)

                ene_theta_square_bin_edges_deg2 = guess_theta_square_bin_edges_deg(
                    theta_square_max_deg=theta_square_max_deg,
                    theta_deg=ene_theta_deg,
                    num_min=10,
                    num_max=2 ** 12,
                )

                ene_hi = irf.analysis.gamma_direction.histogram_theta_square(
                    theta_deg=ene_theta_deg,
                    theta_square_bin_edges_deg2=ene_theta_square_bin_edges_deg2,
                )
                h_ene["histogram"][ene] = {
                    "theta_square_bin_edges_deg2": ene_theta_square_bin_edges_deg2,
                    "intensity": ene_hi[0],
                    "intensity_relative_uncertainty": ene_hi[1]
                }

                ene_co = estimate_containments_theta_deg(
                    containment_fractions=containment_fractions,
                    theta_deg=ene_theta_deg,
                )
                c_ene["containment"][ene] = {
                    "theta_deg": ene_co[0],
                    "theta_deg_relative_uncertainty": ene_co[1],
                }

                for rad in range(num_core_radius_bins):

                    radius_sq_start = core_radius_square_bin_edges_m2[rad]
                    radius_sq_stop = core_radius_square_bin_edges_m2[rad + 1]

                    rad_mask = np.logical_and(
                        rectab["true_r"] ** 2 >= radius_sq_start,
                        rectab["true_r"] ** 2 < radius_sq_stop,
                    )

                    ene_rad_mask = np.logical_and(ene_mask, rad_mask)
                    ene_rad_theta_deg = np.rad2deg(rectab[the][ene_rad_mask])
                    ene_rad_theta_deg = np.abs(ene_rad_theta_deg)

                    ene_rad_theta_square_bin_edges_deg2 = guess_theta_square_bin_edges_deg(
                        theta_square_max_deg=theta_square_max_deg,
                        theta_deg=ene_rad_theta_deg,
                        num_min=10,
                        num_max=2 ** 12,
                    )

                    ene_rad_hi = irf.analysis.gamma_direction.histogram_theta_square(
                        theta_deg=ene_rad_theta_deg,
                        theta_square_bin_edges_deg2=ene_rad_theta_square_bin_edges_deg2,
                    )
                    h_ene_rad["histogram"][ene][rad] = {
                        "theta_square_bin_edges_deg2": ene_rad_theta_square_bin_edges_deg2,
                        "intensity": ene_rad_hi[0],
                        "intensity_relative_uncertainty": ene_rad_hi[1]
                    }

                    ene_rad_co = estimate_containments_theta_deg(
                        containment_fractions=containment_fractions,
                        theta_deg=ene_rad_theta_deg,
                    )
                    c_ene_rad["containment"][ene][rad] = {
                        "theta_deg": ene_rad_co[0],
                        "theta_deg_relative_uncertainty": ene_rad_co[1],
                    }

            irf.json_numpy.write(
                os.path.join(
                    site_particle_dir,
                    "{theta_key:s}_square_histogram_vs_energy_vs_core_radius.json".format(
                        theta_key=the
                    ),
                ),
                h_ene_rad,
            )

            irf.json_numpy.write(
                os.path.join(
                    site_particle_dir,
                    "{theta_key:s}_square_histogram_vs_energy.json".format(
                        theta_key=the
                    ),
                ),
                h_ene,
            )

            irf.json_numpy.write(
                os.path.join(
                    site_particle_dir,
                    "{theta_key:s}_containment_vs_energy_vs_core_radius.json".format(
                        theta_key=the
                    ),
                ),
                c_ene_rad,
            )

            irf.json_numpy.write(
                os.path.join(
                    site_particle_dir,
                    "{theta_key:s}_containment_vs_energy.json".format(
                        theta_key=the
                    ),
                ),
                c_ene,
            )

        # image of point-spread-function
        # -------------------------------

        delta_cx_deg = np.rad2deg(rectab["reco_cx"] - rectab["true_cx"])
        delta_cy_deg = np.rad2deg(rectab["reco_cy"] - rectab["true_cy"])

        for ene in range(num_energy_bins):

            ene_start = energy_bin_edges[ene]
            ene_stop = energy_bin_edges[ene + 1]

            ene_mask = np.logical_and(
                rectab["primary.energy_GeV"] >= ene_start,
                rectab["primary.energy_GeV"] < ene_stop,
            )

            ene_delta_cx_deg = delta_cx_deg[ene_mask]
            ene_delta_cy_deg = delta_cy_deg[ene_mask]

            ene_num_thrown = np.sum(ene_mask)
            ene_psf_image = np.histogram2d(
                ene_delta_cx_deg,
                ene_delta_cy_deg,
                bins=(c_bin_edges_deg, c_bin_edges_deg)
            )[0]
            ene_psf_image_fine = np.histogram2d(
                ene_delta_cx_deg,
                ene_delta_cy_deg,
                bins=(c_bin_edges_fine_deg, c_bin_edges_fine_deg)
            )[0]

            fig = irf.summary.figure.figure(fig_16_by_9)
            ax1 = fig.add_axes([0.075, 0.1, 0.4, 0.8])
            ax1.pcolor(
                c_bin_edges_deg,
                c_bin_edges_deg,
                ene_psf_image,
                cmap="Blues",
                vmax=None,
            )
            ax1.set_title(
                "energy {: 7.1f} - {: 7.1f} GeV".format(
                    ene_start,
                    ene_stop,
                ),
                family="monospace",
            )
            ax1.set_aspect("equal")
            ax1.set_xlim([-1.01 * fov_radius_deg, 1.01 * fov_radius_deg])
            ax1.set_ylim([-1.01 * fov_radius_deg, 1.01 * fov_radius_deg])
            ax1.set_xlabel(r"$\Delta{} c_x$ / $1^\circ{}$")
            ax1.set_ylabel(r"$\Delta{} c_y$ / $1^\circ{}$")
            ax1.spines["top"].set_color("none")
            ax1.spines["right"].set_color("none")
            ax1.spines["bottom"].set_color("none")
            ax1.spines["left"].set_color("none")
            ax1.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

            ax1.plot(
                [-fov_radius_fine_deg, fov_radius_fine_deg],
                [fov_radius_fine_deg, fov_radius_fine_deg],
                "k-",
                linewidth=2 * 0.66,
                alpha=0.15
            )
            ax1.plot(
                [-fov_radius_fine_deg, fov_radius_fine_deg],
                [-fov_radius_fine_deg, -fov_radius_fine_deg],
                "k-",
                linewidth=2 * 0.66,
                alpha=0.15
            )
            ax1.plot(
                [fov_radius_fine_deg, fov_radius_fine_deg],
                [-fov_radius_fine_deg, fov_radius_fine_deg],
                "k-",
                linewidth=2 * 0.66,
                alpha=0.15
            )
            ax1.plot(
                [-fov_radius_fine_deg, -fov_radius_fine_deg],
                [-fov_radius_fine_deg, fov_radius_fine_deg],
                "k-",
                linewidth=2 * 0.66,
                alpha=0.15
            )

            ax2 = fig.add_axes([0.575, 0.1, 0.4, 0.8])
            ax2.set_aspect("equal")
            ax2.spines["top"].set_color("none")
            ax2.spines["right"].set_color("none")
            ax2.spines["bottom"].set_color("none")
            ax2.spines["left"].set_color("none")

            ax2.pcolor(
                c_bin_edges_fine_deg,
                c_bin_edges_fine_deg,
                ene_psf_image_fine,
                cmap="Blues",
                vmax=None,
            )
            ax2.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

            fig.savefig(
                os.path.join(
                    pa["out_dir"],
                    "{:s}_{:s}_psf_image_ene{:06d}.jpg".format(
                        sk,
                        pk,
                        ene
                    )
                )
            )
            plt.close(fig)
