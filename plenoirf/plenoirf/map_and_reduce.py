from . import table
from . import random_seed
from . import grid
from . import merlict
from . import logging
from . import network_file_system as nfs
from . import utils
from . import production

import sys
import numpy as np
import os
from os import path as op
import shutil

import tempfile
import pandas
import json
import tarfile
import corsika_primary_wrapper as cpw
import plenopy as pl
import sparse_numeric_table as spt


"""
I think I have an efficient and very simple algorithm

0) Pick a threshold photon number T1 where trigger curve starts rising
(for a given type of primary)

1) Generate shower such that particle direction hits ground at 0,0;
shower direction spread over large solid angle Omega (energy-dep.)
(for charged particles)
{could also pick (0,0) at some height, but I believe for z=0 the photon
scatter is smallest}

2) Divide ground in grid of spacing = mirror diameter; could e.g. without
too much trouble use up to M x M = 1000 x 1000 grid cells = 70 x 70 km^2;
grid area is A, grid centered on (0,0)

3) Reset photon counter for each cell

3) For each shower, shift grid randomly in x,y by 1/2 mirror diameter

4) Loop over shower photons
   4.1) reject photon if angle outside FOV
   4.2) for each photon, calculate grid cell index ix, iy
        {easy since square grid}
   4.3) calculate distance of photon from cell center;
        keep photon if distance < R_Mirror
   4.4) increment photon counter for cell
   4.5) optionally save photon in a buffer

5) Loop over grid cells
   5.1) count cells with photons > T1: N_1
   5.2) using trigger curve for given particle type;
        calculate trigger prob. for (real) trigger
        and randomly reject events: keep N_2
        {or simply use a 2nd threshold where trigger prob=0.5}
   5.3) Increment event counters by N_1, N_2
        Increment error counters by N_1^2, N_2^2

6) For detailed simulation, optionally output photons for
   few randomly selected T1-triggered cells
   (up to 10 should be fine, given that
   probably only one of 10 triggers the detailed simulation)

7) Toy effective area (x solid angle): (N_1 event counter/M^2/Nevent)*A*Omega
   error = sqrt(error counter) ...
   Somewhat better effective area: N_2 event counter ...
   Final eff. area: N1_eff area x fraction of events kept in detailed sim.

Cheers
Werner



Coordinate system
=================
                                  | z
                                  |                               starting pos.
                                  |                                  ___---O
                                  |                            ___---    / |
                                  |                      ___---     n  /   |
                                  |                ___---         io /     |
                                  |          ___---             ct /       |
                                  |    ___---                 re /         |
              starting altitude __|_---                     di /           |
                                  |                       y- /             |
                                  | _-------__          ar /               |
                                  |-    th    |_      im /                 |
                                  |   ni        |_  pr /                   |
                                  | ze            |  /                     |
                                  |               |/                       |
                      ____________|______________/________________________ |
                     /            |            /            /            / |
                    /            /|          //            /            /  |
                  3/            / |        / /            /            /   |
                  /            /  |      /  /            /            /    |
                 /____________/___|____/___/____________/____________/     |
                /            /    |  /    /            /            /      |
obs. level     /            /     |/     /    grid    /            /       |
altitude -  -2/-  -  -  -  /  -  -X-----/  <-shift y /            /        |
             /            /      /|    /            /            /         |
            /____________/______/_____/____________/____________/          |
           /            /     -|  |  /            /            /           |
          /            /      /   | /            /            /            |
        1/            /  grid     |/            /            /             |
        /            /  shift x   /            /            /              |
       /____________/____________/____________/____________/               |
      /            /            / |          /            /                |
     /            /            /  |         /            /                 |
   0/            /            /   |        /            /                  |
   /            /            /    |       /            /                   |
  /____________/____________/____________/____________/                    |
        0            1           2|             3                          |
                                  |                                  ___---O
                                  |                            ___---
                                  |                      ___--- |
                                  |                ___---        |
                                  |          ___---               |
                                  |    ___---       azimuth       |
                sea leavel z=0    |_---__________________________/______ x
                                  /
                                 /
                                /
                               /
                              /
                             /
                            /
                           /
                          /
                         /
                        / y
Drawn by Sebastian
"""


def _append_bunch_ssize(cherenkovsise_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = cherenkovsise_dict
    ase["num_bunches"] = cb.shape[0]
    ase["num_photons"] = np.sum(cb[:, cpw.IBSIZE])
    return ase


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    assert cb.shape[0] > 0
    ase["maximum_asl_m"] = cpw.CM2M * np.median(cb[:, cpw.IZEM])
    ase["wavelength_median_nm"] = np.abs(np.median(cb[:, cpw.IWVL]))
    ase["cx_median_rad"] = np.median(cb[:, cpw.ICX])
    ase["cy_median_rad"] = np.median(cb[:, cpw.ICY])
    ase["x_median_m"] = cpw.CM2M * np.median(cb[:, cpw.IX])
    ase["y_median_m"] = cpw.CM2M * np.median(cb[:, cpw.IY])
    ase["bunch_size_median"] = np.median(cb[:, cpw.IBSIZE])
    return ase


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir + ".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


def _run_id_str(job):
    form = '{:0' + str(random_seed.STRUCTURE.NUM_DIGITS_RUN_ID) + 'd}'
    return form.format(job["run_id"])


def _run_corsika_and_grid_and_output_to_tmp_dir(
    job,
    prng,
    logger,
    tmp_dir,
    corsika_primary_steering,
    tabrec,
):
    # set up grid geometry
    # --------------------
    assert job["plenoscope_pointing"]["zenith_deg"] == 0.0
    assert job["plenoscope_pointing"]["azimuth_deg"] == 0.0
    plenoscope_pointing_direction = np.array([0, 0, 1])  # For now this is fix.

    _scenery_path = op.join(job["plenoscope_scenery_path"], "scenery.json")
    _light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        merlict_scenery_path=_scenery_path
    )
    plenoscope_diameter = (
        2.0
        * _light_field_sensor_geometry[
            "expected_imaging_system_aperture_radius"
        ]
    )
    plenoscope_radius = 0.5 * plenoscope_diameter
    plenoscope_field_of_view_radius_deg = (
        0.5 * _light_field_sensor_geometry["max_FoV_diameter_deg"]
    )

    grid_geometry = grid.init_geometry(
        instrument_aperture_outer_diameter=plenoscope_diameter,
        bin_width_overhead=job["grid"]["bin_width_overhead"],
        instrument_field_of_view_outer_radius_deg=(
            plenoscope_field_of_view_radius_deg
        ),
        instrument_pointing_direction=plenoscope_pointing_direction,
        field_of_view_overhead=job["grid"]["field_of_view_overhead"],
        num_bins_radius=job["grid"]["num_bins_radius"],
    )
    logger.log("init_grid_geometry")

    # loop over air-showers
    # ---------------------
    for level_key in table.STRUCTURE:
        tabrec[level_key] = []
    reuse_run_path = op.join(tmp_dir, _run_id_str(job) + "_reuse.tar")
    grid_histogram_filename = _run_id_str(job) + "_grid.tar"
    tmp_grid_histogram_path = op.join(tmp_dir, grid_histogram_filename)

    with tarfile.open(reuse_run_path, "w") as tarout, tarfile.open(
        tmp_grid_histogram_path, "w"
    ) as imgtar:

        corsika_run = cpw.CorsikaPrimary(
            corsika_path=job["corsika_primary_path"],
            steering_dict=corsika_primary_steering,
            stdout_path=op.join(tmp_dir, "corsika.stdout"),
            stderr_path=op.join(tmp_dir, "corsika.stderr"),
        )
        logger.log("corskia_startup")

        utils.tar_append(
            tarout=tarout,
            file_name=cpw.TARIO_RUNH_FILENAME,
            file_bytes=corsika_run.runh.tobytes()
        )
        for event_idx, corsika_airshower in enumerate(corsika_run):
            event_header, cherenkov_bunches = corsika_airshower

            # assert match
            run_id = int(event_header[cpw.I_EVTH_RUN_NUMBER])
            assert run_id == corsika_primary_steering["run"]["run_id"]
            event_id = event_idx + 1
            assert event_id == event_header[cpw.I_EVTH_EVENT_NUMBER]
            primary = corsika_primary_steering["primaries"][event_idx]
            event_seed = primary["random_seed"][0]["SEED"]
            ide = {spt.IDX: event_seed}
            assert event_seed == random_seed.STRUCTURE.random_seed_based_on(
                run_id=run_id, airshower_id=event_id
            )

            # export primary table
            # --------------------
            prim = ide.copy()
            prim["particle_id"] = primary["particle_id"]
            prim["energy_GeV"] = primary["energy_GeV"]
            prim["azimuth_rad"] = primary["azimuth_rad"]
            prim["zenith_rad"] = primary["zenith_rad"]
            prim["max_scatter_rad"] = primary["max_scatter_rad"]
            prim["solid_angle_thrown_sr"] = utils.cone_solid_angle(
                prim["max_scatter_rad"]
            )
            prim["depth_g_per_cm2"] = primary["depth_g_per_cm2"]
            prim["momentum_x_GeV_per_c"] = event_header[
                cpw.I_EVTH_PX_MOMENTUM_GEV_PER_C
            ]
            prim["momentum_y_GeV_per_c"] = event_header[
                cpw.I_EVTH_PY_MOMENTUM_GEV_PER_C
            ]
            prim["momentum_z_GeV_per_c"] = (
                -1.0 * event_header[cpw.I_EVTH_PZ_MOMENTUM_GEV_PER_C]
            )
            prim["first_interaction_height_asl_m"] = (
                -1.0
                * cpw.CM2M
                * event_header[cpw.I_EVTH_Z_FIRST_INTERACTION_CM]
            )
            prim["starting_height_asl_m"] = (
                cpw.CM2M * event_header[cpw.I_EVTH_STARTING_HEIGHT_CM]
            )
            obs_lvl_intersection = utils.ray_plane_x_y_intersection(
                support=[0, 0, prim["starting_height_asl_m"]],
                direction=[
                    prim["momentum_x_GeV_per_c"],
                    prim["momentum_y_GeV_per_c"],
                    prim["momentum_z_GeV_per_c"],
                ],
                plane_z=job["site"]["observation_level_asl_m"],
            )
            prim["starting_x_m"] = -1.0 * obs_lvl_intersection[0]
            prim["starting_y_m"] = -1.0 * obs_lvl_intersection[1]
            prim["magnet_azimuth_rad"] = primary["magnet_azimuth_rad"]
            prim["magnet_zenith_rad"] = primary["magnet_zenith_rad"]
            prim["magnet_cherenkov_pool_x_m"] = primary[
                "magnet_cherenkov_pool_x_m"
            ]
            prim["magnet_cherenkov_pool_y_m"] = primary[
                "magnet_cherenkov_pool_y_m"
            ]
            tabrec["primary"].append(prim)

            # cherenkov size
            # --------------
            crsz = ide.copy()
            crsz = _append_bunch_ssize(crsz, cherenkov_bunches)
            tabrec["cherenkovsize"].append(crsz)

            # assign grid
            # -----------
            grid_random_shift_x, grid_random_shift_y = prng.uniform(
                low=-0.5 * grid_geometry["bin_width"],
                high=0.5 * grid_geometry["bin_width"],
                size=2,
            )

            grhi = ide.copy()
            if job["artificial_core_limitation"]:
                _max_core_scatter_radius = np.interp(
                    x=primary["energy_GeV"],
                    xp=job["artificial_core_limitation"]["energy_GeV"],
                    fp=job["artificial_core_limitation"][
                        "max_scatter_radius_m"
                    ],
                )
                grid_bin_idxs_limitation = grid.where_grid_idxs_within_radius(
                    grid_geometry=grid_geometry,
                    radius=_max_core_scatter_radius,
                    center_x=-1.0 * grid_random_shift_x,
                    center_y=-1.0 * grid_random_shift_y,
                )
                grhi["artificial_core_limitation"] = 1
                grhi[
                    "artificial_core_limitation_radius_m"
                ] = _max_core_scatter_radius
                grhi["num_bins_thrown"] = len(grid_bin_idxs_limitation[0])
                grhi["area_thrown_m2"] = (
                    grhi["num_bins_thrown"] * grid_geometry["bin_area"]
                )
                logger.log("artificial core limitation is ON")
            else:
                grid_bin_idxs_limitation = None
                grhi["artificial_core_limitation"] = 0
                grhi["artificial_core_limitation_radius_m"] = -1.0
                grhi["num_bins_thrown"] = grid_geometry["total_num_bins"]
                grhi["area_thrown_m2"] = grid_geometry["total_area"]

            grhi["bin_width_m"] = grid_geometry["bin_width"]
            grhi["field_of_view_radius_deg"] = grid_geometry[
                "field_of_view_radius_deg"
            ]
            grhi["pointing_direction_x"] = grid_geometry["pointing_direction"][
                0
            ]
            grhi["pointing_direction_y"] = grid_geometry["pointing_direction"][
                1
            ]
            grhi["pointing_direction_z"] = grid_geometry["pointing_direction"][
                2
            ]
            grhi["random_shift_x_m"] = grid_random_shift_x
            grhi["random_shift_y_m"] = grid_random_shift_y
            grhi["magnet_shift_x_m"] = (
                -1.0 * primary["magnet_cherenkov_pool_x_m"]
            )
            grhi["magnet_shift_y_m"] = (
                -1.0 * primary["magnet_cherenkov_pool_x_m"]
            )
            grhi["total_shift_x_m"] = (
                grhi["random_shift_x_m"] + grhi["magnet_shift_x_m"]
            )
            grhi["total_shift_y_m"] = (
                grhi["random_shift_y_m"] + grhi["magnet_shift_y_m"]
            )

            grid_result = grid.assign(
                cherenkov_bunches=cherenkov_bunches,
                grid_geometry=grid_geometry,
                shift_x=grhi["total_shift_x_m"],
                shift_y=grhi["total_shift_y_m"],
                threshold_num_photons=job["grid"]["threshold_num_photons"],
                bin_idxs_limitation=grid_bin_idxs_limitation,
            )
            utils.tar_append(
                tarout=imgtar,
                file_name=random_seed.STRUCTURE.SEED_TEMPLATE_STR.format(
                    seed=event_seed
                )
                + ".f4.gz",
                file_bytes=grid.histogram_to_bytes(grid_result["histogram"]),
            )

            # grid statistics
            # ---------------
            grhi["num_bins_above_threshold"] = grid_result[
                "num_bins_above_threshold"
            ]
            grhi["overflow_x"] = grid_result["overflow_x"]
            grhi["underflow_x"] = grid_result["underflow_x"]
            grhi["overflow_y"] = grid_result["overflow_y"]
            grhi["underflow_y"] = grid_result["underflow_y"]
            tabrec["grid"].append(grhi)

            # cherenkov statistics
            # --------------------
            if cherenkov_bunches.shape[0] > 0:
                fase = ide.copy()
                fase = _append_bunch_statistics(
                    airshower_dict=fase, cherenkov_bunches=cherenkov_bunches
                )
                tabrec["cherenkovpool"].append(fase)

            reuse_event = grid_result["random_choice"]
            if reuse_event is not None:
                reuse_evth = event_header.copy()
                reuse_evth[cpw.I_EVTH_NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
                reuse_evth[cpw.I_EVTH_X_CORE_CM(reuse=1)] = (
                    cpw.M2CM * reuse_event["core_x_m"]
                )
                reuse_evth[cpw.I_EVTH_Y_CORE_CM(reuse=1)] = (
                    cpw.M2CM * reuse_event["core_y_m"]
                )
                utils.tar_append(
                    tarout=tarout,
                    file_name=cpw.TARIO_EVTH_FILENAME.format(event_id),
                    file_bytes=reuse_evth.tobytes(),
                )
                utils.tar_append(
                    tarout=tarout,
                    file_name=cpw.TARIO_BUNCHES_FILENAME.format(event_id),
                    file_bytes=reuse_event["cherenkov_bunches"].tobytes(),
                )
                crszp = ide.copy()
                crszp = _append_bunch_ssize(crszp, cherenkov_bunches)
                tabrec["cherenkovsizepart"].append(crszp)
                rase = ide.copy()
                rase = _append_bunch_statistics(
                    airshower_dict=rase,
                    cherenkov_bunches=reuse_event["cherenkov_bunches"],
                )
                tabrec["cherenkovpoolpart"].append(rase)
                rcor = ide.copy()
                rcor["bin_idx_x"] = reuse_event["bin_idx_x"]
                rcor["bin_idx_y"] = reuse_event["bin_idx_y"]
                rcor["core_x_m"] = reuse_event["core_x_m"]
                rcor["core_y_m"] = reuse_event["core_y_m"]
                tabrec["core"].append(rcor)
    logger.log("grid")

    nfs.copy(
        op.join(tmp_dir, "corsika.stdout"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stdout"),
    )
    nfs.copy(
        op.join(tmp_dir, "corsika.stderr"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stderr"),
    )

    # export grid histograms
    # ----------------------
    nfs.copy(
        src=tmp_grid_histogram_path,
        dst=op.join(job["feature_dir"], grid_histogram_filename),
    )
    logger.log("export_grid_histograms")

    return reuse_run_path, tabrec


def _run_merlict(job, reuse_run_path, tmp_dir):
    merlict_run_path = op.join(tmp_dir, _run_id_str(job) + "_merlict.cp")
    if not op.exists(merlict_run_path):
        merlict_rc = merlict.plenoscope_propagator(
            corsika_run_path=reuse_run_path,
            output_path=merlict_run_path,
            light_field_geometry_path=job["light_field_geometry_path"],
            merlict_plenoscope_propagator_path=job[
                "merlict_plenoscope_propagator_path"
            ],
            merlict_plenoscope_propagator_config_path=job[
                "merlict_plenoscope_propagator_config_path"
            ],
            random_seed=job["run_id"],
            stdout_postfix=".stdout",
            stderr_postfix=".stderr",
        )
        nfs.copy(
            merlict_run_path + ".stdout",
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stdout"),
        )
        nfs.copy(
            merlict_run_path + ".stderr",
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stderr"),
        )
        assert merlict_rc == 0

    return merlict_run_path


def _run_loose_trigger(
    job,
    tabrec,
    merlict_run_path,
    light_field_geometry,
    trigger_geometry,
    tmp_dir
):
    # loop over sensor responses
    # --------------------------
    merlict_run = pl.Run(merlict_run_path)
    table_past_trigger = []
    tmp_past_trigger_dir = op.join(tmp_dir, "past_trigger")
    os.makedirs(tmp_past_trigger_dir, exist_ok=True)

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_event_header.raw
        run_id = int(cevth[cpw.I_EVTH_RUN_NUMBER])
        airshower_id = int(cevth[cpw.I_EVTH_EVENT_NUMBER])
        ide = {
            spt.IDX: random_seed.STRUCTURE.random_seed_based_on(
                run_id=run_id, airshower_id=airshower_id
            )
        }

        # apply loose trigger
        # -------------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = pl.simple_trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            integration_time_slices=(
                job["sum_trigger"]["integration_time_slices"]
            ),
        )

        trg_resp_path = op.join(event._path, "refocus_sum_trigger.json")
        with open(trg_resp_path, "wt") as f:
            f.write(json.dumps(trigger_responses, indent=4))

        trg_maxr_path = op.join(
            event._path, "refocus_sum_trigger.focii_x_time_slices.uint32"
        )
        with open(trg_maxr_path, "wb") as f:
            f.write(max_response_in_focus_vs_timeslices.tobytes())

        # export trigger-truth
        # --------------------
        trgtru = ide.copy()
        trgtru["num_cherenkov_pe"] = int(
            event.simulation_truth.detector.number_air_shower_pulses()
        )
        trgtru["response_pe"] = int(
            np.max([focus["response_pe"] for focus in trigger_responses])
        )
        for o in range(len(trigger_responses)):
            trgtru["focus_{:02d}_response_pe".format(o)] = int(
                trigger_responses[o]["response_pe"]
            )
        tabrec["trigger"].append(trgtru)

        # passing loose trigger
        # ---------------------
        if trgtru["response_pe"] >= job["sum_trigger"]["threshold_pe"]:
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp[
                "unique_id_str"
            ] = random_seed.STRUCTURE.SEED_TEMPLATE_STR.format(
                seed=ptp[spt.IDX]
            )
            table_past_trigger.append(ptp)

            # export past loose trigger
            # -------------------------
            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)

            pl.tools.acp_format.compress_event_in_place(ptp["tmp_path"])
            final_tarname = ptp["unique_id_str"] + ".tar"
            plenoscope_event_dir_to_tar(
                event_dir=ptp["tmp_path"],
                output_tar_path=op.join(tmp_past_trigger_dir, final_tarname),
            )
            nfs.copy(
                src=op.join(tmp_past_trigger_dir, final_tarname),
                dst=op.join(job["past_trigger_dir"], final_tarname),
            )

    return tabrec, table_past_trigger, tmp_past_trigger_dir


def _assert_resources_exist(job):
    assert op.exists(job["corsika_primary_path"])
    assert op.exists(job["merlict_plenoscope_propagator_path"])
    assert op.exists(job["merlict_plenoscope_propagator_config_path"])
    assert op.exists(job["plenoscope_scenery_path"])
    assert op.exists(job["light_field_geometry_path"])
    assert op.exists(job["trigger_geometry_path"])


def _make_output_dirs(job):
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_reconstructed_cherenkov_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)


def run_job(job):
    _assert_resources_exist(job=job)
    _make_output_dirs(job=job)

    prng = np.random.Generator(np.random.MT19937(seed=job["run_id"]))

    time_log_path = op.join(job["log_dir"], _run_id_str(job) + "_runtime.jsonl")
    logger = logging.JsonlLog(time_log_path + ".tmp")
    job_path = op.join(job["log_dir"], _run_id_str(job) + "_job.json")
    with open(job_path + ".tmp", "wt") as f:
        f.write(json.dumps(job, indent=4))
    nfs.move(job_path + ".tmp", job_path)
    print('{{"run_id": {:d}"}}\n'.format(job["run_id"]))

    # draw primaries
    # --------------
    corsika_primary_steering = production.corsika_primary.draw_corsika_primary_steering(
        run_id=job["run_id"],
        site=job["site"],
        particle=job["particle"],
        site_particle_deflection=job["site_particle_deflection"],
        num_events=job["num_air_showers"],
        prng=prng,
    )
    logger.log("draw_primaries")

    # tmp dir
    # -------
    if job["tmp_dir"] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job["tmp_dir"], _run_id_str(job))
        os.makedirs(tmp_dir, exist_ok=True)
    logger.log("make_temp_dir:'{:s}'".format(tmp_dir))

    tabrec = {}

    reuse_run_path, tabrec = _run_corsika_and_grid_and_output_to_tmp_dir(
        job=job,
        prng=prng,
        logger=logger,
        tmp_dir=tmp_dir,
        corsika_primary_steering=corsika_primary_steering,
        tabrec=tabrec
    )

    merlict_run_path = _run_merlict(
        job=job,
        reuse_run_path=reuse_run_path,
        tmp_dir=tmp_dir,
    )

    logger.log("merlict")

    if not job["keep_tmp"]:
        os.remove(reuse_run_path)

    # prepare loose trigger
    # ----------------------
    light_field_geometry = pl.LightFieldGeometry(
        path=job["light_field_geometry_path"]
    )
    trigger_geometry = pl.simple_trigger.io.read_trigger_geometry_from_path(
        path=job["trigger_geometry_path"]
    )
    logger.log("prepare_trigger")

    tabrec, table_past_trigger, tmp_past_trigger_dir = _run_loose_trigger(
        job=job,
        tabrec=tabrec,
        merlict_run_path=merlict_run_path,
        light_field_geometry=light_field_geometry,
        trigger_geometry=trigger_geometry,
        tmp_dir=tmp_dir
    )

    logger.log("trigger")

    # Cherenkov classification
    # ------------------------
    roi_cfg = job["cherenkov_classification"]["region_of_interest"]
    dbscan_cfg = job["cherenkov_classification"]

    cer_phs_basename = _run_id_str(job) + "_reconstructed_cherenkov.tar"
    with pl.photon_stream.loph.LopfTarWriter(
        path=os.path.join(tmp_dir, cer_phs_basename),
        id_num_digits=random_seed.STRUCTURE.NUM_DIGITS_SEED,
    ) as cer_phs_run:
        for ptp in table_past_trigger:
            event = pl.Event(
                path=ptp["tmp_path"], light_field_geometry=light_field_geometry
            )
            trigger_responses = pl.simple_trigger.io.read_trigger_response_from_path(
                path=os.path.join(event._path, "refocus_sum_trigger.json")
            )
            roi = pl.simple_trigger.region_of_interest.from_trigger_response(
                trigger_response=trigger_responses,
                trigger_geometry=trigger_geometry,
                time_slice_duration=event.raw_sensor_response.time_slice_duration,
            )
            photons = pl.classify.RawPhotons.from_event(event)
            (
                cherenkov_photons,
                roi_settings,
            ) = pl.classify.cherenkov_photons_in_roi_in_image(
                roi=roi,
                photons=photons,
                roi_time_offset_start=roi_cfg["time_offset_start_s"],
                roi_time_offset_stop=roi_cfg["time_offset_stop_s"],
                roi_cx_cy_radius=np.deg2rad(roi_cfg["direction_radius_deg"]),
                roi_object_distance_offsets=roi_cfg[
                    "object_distance_offsets_m"
                ],
                dbscan_epsilon_cx_cy_radius=np.deg2rad(
                    dbscan_cfg["neighborhood_radius_deg"]
                ),
                dbscan_min_number_photons=dbscan_cfg["min_num_photons"],
                dbscan_deg_over_s=dbscan_cfg[
                    "direction_to_time_mixing_deg_per_s"
                ],
            )
            pl.classify.write_dense_photon_ids_to_event(
                event_path=op.abspath(event._path),
                photon_ids=cherenkov_photons.photon_ids,
                settings=roi_settings,
            )
            crcl = pl.classify.benchmark(
                pulse_origins=event.simulation_truth.detector.pulse_origins,
                photon_ids_cherenkov=cherenkov_photons.photon_ids,
            )
            crcl[spt.IDX] = ptp[spt.IDX]
            tabrec["cherenkovclassification"].append(crcl)

            # export reconstructed Cherenkov photons
            # --------------------------------------
            cer_phs = pl.photon_stream.loph.raw_sensor_response_to_photon_stream_in_loph_repr(
                raw_sensor_response=event.raw_sensor_response,
                cherenkov_photon_ids=cherenkov_photons.photon_ids,
            )
            cer_phs_run.add(identity=ptp[spt.IDX], phs=cer_phs)

    logger.log("cherenkov_classification")

    # extracting features
    # -------------------
    light_field_geometry_addon = pl.features.make_light_field_geometry_addon(
        light_field_geometry=light_field_geometry)

    logger.log("light_field_geometry_addons")

    for pt in table_past_trigger:
        event = pl.Event(
            path=pt["tmp_path"],
            light_field_geometry=light_field_geometry)
        try:
            lfft = pl.features.extract_features(
                cherenkov_photons=event.cherenkov_photons,
                light_field_geometry=light_field_geometry,
                light_field_geometry_addon=light_field_geometry_addon,
            )
            lfft[spt.IDX] = pt[spt.IDX]
            tabrec["features"].append(lfft)
        except Exception as excep:
            print("idx:", pt[spt.IDX], excep)
    logger.log("feature_extraction")

    # export event-table
    # ------------------
    table_filename = _run_id_str(job) + "_event_table.tar"
    event_table = spt.table_of_records_to_sparse_numeric_table(
        table_records=tabrec, structure=table.STRUCTURE
    )
    spt.write(
        path=op.join(tmp_dir, table_filename),
        table=event_table,
        structure=table.STRUCTURE,
    )
    nfs.copy(
        src=op.join(tmp_dir, table_filename),
        dst=op.join(job["feature_dir"], table_filename),
    )
    logger.log("export_event_table")

    # export past_trigger reconstructed cherenkov
    # -------------------------------------------
    nfs.copy(
        src=op.join(tmp_dir, cer_phs_basename),
        dst=op.join(
            job["past_trigger_reconstructed_cherenkov_dir"], cer_phs_basename
        ),
    )

    # end
    # ---
    logger.log("end")
    nfs.move(time_log_path + ".tmp", time_log_path)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)


def run_bundle(bundle):
    results = []
    for j, job in enumerate(bundle):
        msg = "\n#bundle {:d} of {:d}\n".format((j + 1), len(bundle))
        print(msg, file=sys.stdout)
        print(msg, file=sys.stderr)
        try:
            result = run_job(job=job)
        except Exception as exception_msg:
            print(exception_msg, file=sys.stderr)
            result = 0
        results.append(result)
    return results
