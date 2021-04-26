#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import airshower_template_generator as atg
import plenopy as pl
import glob
from iminuit import Minuit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

loph_chunk_base_dir = os.path.join(
    pa["summary_dir"], "0068_prepare_loph_passed_trigger_and_quality"
)

onregion_angle_vs_num_photons = {}
onregion_angle_vs_num_photons["num_photons_pe"] =  [1e1, 1e2, 1e3, 1e4, 1e5]
onregion_angle_vs_num_photons["opening_angle_deg"]=[1.6, 0.8, 0.4, 0.2, 0.1]
core_radius_uncertainty_doubling_m = 2.5e2


# READ light-field-geometry
# =========================
lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

fov_radius_deg = np.rad2deg(
    0.5 * lfg.sensor_plane2imaging_system.max_FoV_diameter
)


def add_axes_fuzzy_debug(ax, ring_binning, fuzzy_result, fuzzy_debug):
    azi = fuzzy_result["main_axis_azimuth"]
    ax.plot(
        np.rad2deg(ring_binning["bin_edges"]),
        fuzzy_debug["azimuth_ring_smooth"],
        "k",
    )
    ax.plot(np.rad2deg(azi), 1.0, "or")

    unc = 0.5 * fuzzy_result["main_axis_azimuth_uncertainty"]
    ax.plot(np.rad2deg([azi - unc, azi + unc]), [0.5, 0.5], "-r")

    ax.set_xlim([0, 360])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("main-axis-azimuth / deg")
    ax.set_ylabel("probability density / 1")


fuzzy_config = irf.reconstruction.fuzzy_method.compile_user_config(
    user_config=irf_config["config"]["reconstruction"]["trajectory"]["fuzzy_method"]
)

long_fit_cfg = irf.reconstruction.model_fit.compile_user_config(
    user_config=irf_config["config"]["reconstruction"]["trajectory"]["core_axis_fit"]
)

fig_16_by_9 = sum_config["plot"]["16_by_9"]

truth_by_index = {}
for sk in irf_config["config"]["sites"]:
    truth_by_index[sk] = {}
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:
        truth_by_index[sk][pk] = {}

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        passed_trigger_idx = np.array(
            _passed_trigger_indices[sk][pk]["passed_trigger"][spt.IDX]
        )

        all_truth = spt.cut_table_on_indices(
            event_table,
            irf.table.STRUCTURE,
            common_indices=passed_trigger_idx,
            level_keys=[
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
            ],
        )
        all_truth = spt.sort_table_on_common_indices(
            table=all_truth, common_indices=passed_trigger_idx
        )
        (
            true_cx,
            true_cy,
        ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
            momentum_x_GeV_per_c=all_truth["primary"]["momentum_x_GeV_per_c"],
            momentum_y_GeV_per_c=all_truth["primary"]["momentum_y_GeV_per_c"],
            momentum_z_GeV_per_c=all_truth["primary"]["momentum_z_GeV_per_c"],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        for ii in range(all_truth["primary"].shape[0]):
            airshower_id = all_truth["primary"][spt.IDX][ii]
            truth_by_index[sk][pk][airshower_id] = {
                "cx": true_cx[ii],
                "cy": true_cy[ii],
                "x": -all_truth["core"]["core_x_m"][ii],
                "y": -all_truth["core"]["core_y_m"][ii],
                "energy_GeV": all_truth["primary"]["energy_GeV"][ii],
            }


def my_axes_look(ax):
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    return ax


def read_shower_maximum_object_distance(
    site_key, particle_key, key="image_smallest_ellipse_object_distance"
):
    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"],
            "event_table",
            site_key,
            particle_key,
            "event_table.tar",
        ),
        structure=irf.table.STRUCTURE,
    )

    return spt.get_column_as_dict_by_index(
        table=event_table, level_key="features", column_key=key
    )


PLOT_RING = False
PLOT_OVERVIEW = True
PLOT_ONREGION = True


for sk in irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:

        reco_obj = read_shower_maximum_object_distance(
            site_key=sk, particle_key=pk
        )

        loph_chunk_paths = glob.glob(
            os.path.join(loph_chunk_base_dir, sk, pk, "chunks", "*.tar")
        )

        run = pl.photon_stream.loph.LopfTarReader(loph_chunk_paths[1])

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        for event in run:
            airshower_id, loph_record = event

            truth = dict(truth_by_index[sk][pk][airshower_id])

            fit, debug = irf.reconstruction.trajectory.estimate(
                loph_record=loph_record,
                light_field_geometry=lfg,
                shower_maximum_object_distance=reco_obj[airshower_id],
                fuzzy_config=fuzzy_config,
                model_fit_config=long_fit_cfg,
            )

            if not irf.reconstruction.trajectory.is_valid_estimate(fit):
                print(
                    "airshower_id",
                    airshower_id,
                    " Can not reconstruct trajectory",
                )

            # true response
            # -------------

            true_response = irf.reconstruction.trajectory.model_response_for_true_trajectory(
                true_cx=truth["cx"],
                true_cy=truth["cy"],
                true_x=truth["x"],
                true_y=truth["y"],
                loph_record=loph_record,
                light_field_geometry=lfg,
                model_fit_config=long_fit_cfg,
            )

            if PLOT_RING:
                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax = my_axes_look(ax=ax)
                add_axes_fuzzy_debug(
                    ax=ax,
                    ring_binning=fuzzy_config["azimuth_ring"],
                    fuzzy_result=debug["fuzzy_result"],
                    fuzzy_debug=debug["fuzzy_debug"],
                )
                path = os.path.join(
                    pa["out_dir"],
                    sk,
                    pk,
                    "{:09d}_ring.jpg".format(airshower_id,),
                )
                fig.savefig(path)
                plt.close(fig)

            if PLOT_OVERVIEW:

                split_light_field = pl.fuzzy.direction.SplitLightField(
                    loph_record=loph_record, light_field_geometry=lfg
                )

                fit_cx_deg = np.rad2deg(fit["primary_particle_cx"])
                fit_cy_deg = np.rad2deg(fit["primary_particle_cy"])
                fit_x = fit["primary_particle_x"]
                fit_y = fit["primary_particle_y"]

                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.075, 0.1, 0.4, 0.8])
                ax_core = fig.add_axes([0.575, 0.1, 0.4, 0.8])
                for pax in range(split_light_field.number_paxel):
                    ax.plot(
                        np.rad2deg(
                            split_light_field.image_sequences[pax][:, 0]
                        ),
                        np.rad2deg(
                            split_light_field.image_sequences[pax][:, 1]
                        ),
                        "xb",
                        alpha=0.03,
                    )
                ax.pcolor(
                    np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                    np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                    debug["fuzzy_debug"]["fuzzy_image_smooth"],
                    cmap="Reds",
                )
                phi = np.linspace(0, 2 * np.pi, 1000)
                ax.plot(
                    fov_radius_deg * np.cos(phi),
                    fov_radius_deg * np.sin(phi),
                    "k",
                )
                ax.plot(
                    [
                        np.rad2deg(fit["main_axis_support_cx"]),
                        np.rad2deg(fit["main_axis_support_cx"])
                        + 100 * np.cos(fit["main_axis_azimuth"]),
                    ],
                    [
                        np.rad2deg(fit["main_axis_support_cy"]),
                        np.rad2deg(fit["main_axis_support_cy"])
                        + 100 * np.sin(fit["main_axis_azimuth"]),
                    ],
                    ":c",
                )

                ax.plot(
                    np.rad2deg(debug["fuzzy_result"]["reco_cx"]),
                    np.rad2deg(debug["fuzzy_result"]["reco_cy"]),
                    "og",
                )
                ax.plot(fit_cx_deg, fit_cy_deg, "oc")
                ax.plot(np.rad2deg(truth["cx"]), np.rad2deg(truth["cy"]), "xk")

                if PLOT_ONREGION:

                    onregion = irf.reconstruction.onregion.estimate_onregion(
                        reco_cx=fit["primary_particle_cx"],
                        reco_cy=fit["primary_particle_cy"],
                        reco_main_axis_azimuth=fit["main_axis_azimuth"],
                        reco_num_photons=len(
                            loph_record["photons"]["arrival_time_slices"]
                        ),
                        reco_core_radius=np.hypot(
                            fit["primary_particle_x"],
                            fit["primary_particle_y"]
                        ),
                        core_radius_uncertainty_doubling=core_radius_uncertainty_doubling_m,
                        opening_angle_vs_reco_num_photons=onregion_angle_vs_num_photons,
                    )

                    ellx, elly = irf.reconstruction.onregion.make_polygon(
                        onregion=onregion
                    )

                    hit = irf.reconstruction.onregion.is_direction_inside(
                        cx=truth["cx"],
                        cy=truth["cy"],
                        onregion=onregion
                    )

                    if hit:
                        look = "c"
                    else:
                        look = ":c"

                    ax.plot(
                        np.rad2deg(ellx),
                        np.rad2deg(elly),
                        look,
                    )


                info_str = "Energy: {: .1f}GeV, reco. Cherenkov: {: 4d}p.e.\n response of shower-model: {:.4f} ({:.4f})".format(
                    truth["energy_GeV"],
                    loph_record["photons"]["channels"].shape[0],
                    fit["shower_model_response"],
                    true_response,
                )

                ax.set_title(info_str)

                ax.set_xlim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_ylim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_aspect("equal")
                ax.set_xlabel("cx / deg")
                ax.set_ylabel("cy / deg")
                ax = my_axes_look(ax=ax)

                ax_core.plot(fit_x, fit_y, "oc")
                ax_core.plot([0, fit_x], [0, fit_y], "c", alpha=0.5)

                ax_core.plot(truth["x"], truth["y"], "xk")
                ax_core.plot([0, truth["x"]], [0, truth["y"]], "k", alpha=0.5)

                ax_core.set_xlim([-640, 640])
                ax_core.set_ylim([-640, 640])
                ax_core.set_aspect("equal")
                ax_core.set_xlabel("x / m")
                ax_core.set_ylabel("y / m")
                ax_core = my_axes_look(ax=ax_core)
                path = os.path.join(
                    pa["out_dir"], sk, pk, "{:09d}.jpg".format(airshower_id,),
                )

                fig.savefig(path)
                plt.close(fig)
