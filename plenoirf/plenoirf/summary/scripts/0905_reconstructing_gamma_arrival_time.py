#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
import plenopy as pl
import iminuit
import scipy
import sebastians_matplotlib_addons as seb
import json_numpy


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

passing_trigger = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0059_passing_trajectory_quality")
)

#lfg = pl.LightFieldGeometry(path="")

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
SPEED_OF_LIGHT = 299792458

def time_it_takes_for_gamma_ray_to_cross_instrument_plane(
    instrument_position,
    particle_position,
    particle_direction,
):
    """

    """
    return 1

tds = {}
for sk in SITES:
    tds[sk] = {}
    for pk in ["gamma"]:
        tds[sk][pk] = {}
        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        valid_idx = spt.intersection([
            passing_trigger[sk][pk][spt.IDX],
            passing_quality[sk][pk][spt.IDX],
            passing_trajectory_quality[sk][pk][spt.IDX],
        ])

        event_table = spt.cut_table_on_indices(
            table=event_table,
            common_indices=valid_idx,
            level_keys=["primary", "core", "instrument", "features", "reconstructed_trajectory"],
        )

        et = spt.make_rectangular_DataFrame(event_table).to_records()


        tds[sk][pk]["t_delta"] = []
        tds[sk][pk]["t_corr"] = []

        for i in range(len(et)):

            if et["primary/energy_GeV"][i] < 100:
                continue

            particle_direction = np.array([
                et["primary/momentum_x_GeV_per_c"][i],
                et["primary/momentum_y_GeV_per_c"][i],
                et["primary/momentum_z_GeV_per_c"][i],
            ])
            particle_direction = particle_direction / np.linalg.norm(particle_direction)

            particle_position = np.array([
                et["core/core_x_m"][i],
                et["core/core_y_m"][i],
                et["primary/starting_height_asl_m"][i],
            ])

            instrument_position = np.array([
                0,
                0,
                irf_config["config"]["sites"][sk]["observation_level_asl_m"]
            ])

            p = irf.utils.ray_parameter_for_closest_distance_to_point(
                ray_support=particle_position,
                ray_direction=particle_direction,
                point=instrument_position,
            )

            """
            gamma_core = irf.utils.ray_at(
                ray_support=particle_position,
                ray_direction=particle_direction,
                parameter=p,
            )
            """

            t_corsika_start_to_instrument_plane = p / SPEED_OF_LIGHT

            #print("pos", particle_position, "dir", particle_direction, "p", p, "core", gamma_core)


            reco_dir = np.array([
                et["reconstructed_trajectory/cx_rad"][i],
                et["reconstructed_trajectory/cy_rad"][i],
                -np.sqrt(
                    1.0 -
                    et["reconstructed_trajectory/cx_rad"][i] ** 2 -
                    et["reconstructed_trajectory/cy_rad"][i] ** 2
                )
            ])

            reco_support_observation_level = np.array([
                et["reconstructed_trajectory/x_m"][i],
                et["reconstructed_trajectory/y_m"][i],
                irf_config["config"]["sites"][sk]["observation_level_asl_m"],
            ])
            core_r_obs = np.linalg.norm(reco_support_observation_level[0:1])

            reco_max_altitde = et["features/image_smallest_ellipse_object_distance"]


            instrument_position_2 = np.array([
                0,
                0,
                irf_config["config"]["sites"][sk]["observation_level_asl_m"]
            ])

            p_reco = irf.utils.ray_parameter_for_closest_distance_to_point(
                ray_support=reco_support_observation_level,
                ray_direction=reco_dir,
                point=instrument_position,
            )
            t_corr = p_reco / SPEED_OF_LIGHT

            t_delta = (
                 et["instrument/start_time_of_exposure_s"][i] - t_corsika_start_to_instrument_plane
            )
            print(t_delta*1e9, t_corr*1e9, core_r_obs)


            tds[sk][pk]["t_corr"].append(t_corr)
            tds[sk][pk]["t_delta"].append(t_delta)

            #print(t_delta * 1e9)