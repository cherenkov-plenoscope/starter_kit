from . import table
from . import unique
from . import grid
from . import merlict
from . import logging
from . import network_file_system as nfs
from . import utils
from . import production
from . import reconstruction

import sys
import numpy as np
import os
from os import path as op
import shutil
import time

import tempfile
import pandas
import json
import tarfile
import corsika_primary as cpw
import plenopy as pl
import sparse_numeric_table as spt
import gamma_ray_reconstruction as gamrec


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
    ase["num_photons"] = np.sum(cb[:, cpw.I.BUNCH.BSIZE])
    return ase


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    assert cb.shape[0] > 0
    CM2M = 1e2
    ase["maximum_asl_m"] = CM2M * np.median(cb[:, cpw.I.BUNCH.ZEM])
    ase["wavelength_median_nm"] = np.abs(np.median(cb[:, cpw.I.BUNCH.WVL]))
    ase["cx_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CX])
    ase["cy_median_rad"] = np.median(cb[:, cpw.I.BUNCH.CY])
    ase["x_median_m"] = CM2M * np.median(cb[:, cpw.I.BUNCH.X])
    ase["y_median_m"] = CM2M * np.median(cb[:, cpw.I.BUNCH.Y])
    ase["bunch_size_median"] = np.median(cb[:, cpw.I.BUNCH.BSIZE])
    return ase


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir + ".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


def _run_id_str(job):
    form = "{:0" + unique.RUN_ID_NUM_DIGITS + "d}"
    return form.format(job["run_id"])


def _init_grid_geometry_from_job(job):
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
    return grid_geometry


def _run_corsika_and_grid_and_output_to_tmp_dir(
    job, prng, tmp_dir, corsika_primary_steering, tabrec,
):
    grid_geometry = _init_grid_geometry_from_job(job=job)
    GRID_SKIP = int(job["grid"]["output_after_num_events"])
    assert GRID_SKIP > 0

    # loop over air-showers
    # ---------------------
    cherenkov_pools_path = op.join(tmp_dir, "cherenkov_pools.tar")
    tmp_grid_histogram_path = op.join(tmp_dir, "grid.tar")

    with cpw.event_tape.EventTapeWriter(
        path=cherenkov_pools_path
    ) as evttar, tarfile.open(tmp_grid_histogram_path, "w") as imgtar:

        corsika_run = cpw.CorsikaPrimary(
            corsika_path=job["corsika_primary_path"],
            steering_dict=corsika_primary_steering,
            stdout_path=op.join(tmp_dir, "corsika.stdout"),
            stderr_path=op.join(tmp_dir, "corsika.stderr"),
        )
        evttar.write_runh(runh=corsika_run.runh)

        for event_idx, corsika_event in enumerate(corsika_run):
            corsika_evth, cherenkov_bunches = corsika_event

            # assert match
            run_id = int(corsika_evth[cpw.I.EVTH.RUN_NUMBER])
            assert run_id == corsika_primary_steering["run"]["run_id"]
            event_id = event_idx + 1
            assert event_id == corsika_evth[cpw.I.EVTH.EVENT_NUMBER]
            uid = unique.make_uid(run_id=run_id, event_id=event_id)
            uid_str = unique.make_uid_str(run_id=run_id, event_id=event_id)

            ide = {spt.IDX: shower_id}

            primary = corsika_primary_steering["primaries"][event_idx]

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
            prim["momentum_x_GeV_per_c"] = corsika_evth[
                cpw.I.EVTH.PX_MOMENTUM_GEV_PER_C
            ]
            prim["momentum_y_GeV_per_c"] = corsika_evth[
                cpw.I.EVTH.PY_MOMENTUM_GEV_PER_C
            ]
            prim["momentum_z_GeV_per_c"] = (
                -1.0 * corsika_evth[cpw.I.EVTH.PZ_MOMENTUM_GEV_PER_C]
            )
            prim["first_interaction_height_asl_m"] = (
                -1.0
                * cpw.CM2M
                * corsika_evth[cpw.I.EVTH.Z_FIRST_INTERACTION_CM]
            )
            prim["starting_height_asl_m"] = (
                cpw.CM2M * corsika_evth[cpw.I.EVTH.STARTING_HEIGHT_CM]
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
                prng=prng,
                bin_idxs_limitation=grid_bin_idxs_limitation,
            )
            if event_idx % GRID_SKIP == 0:
                utils.tar_append(
                    evttar=imgtar,
                    file_name=shower_id_str + ".f4.gz",
                    file_bytes=grid.histogram_to_bytes(
                        grid_result["histogram"]
                    ),
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
                reuse_evth = corsika_evth.copy()
                reuse_evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
                reuse_evth[cpw.I.EVTH.X_CORE_CM(reuse=1)] = (
                    cpw.M2CM * reuse_event["core_x_m"]
                )
                reuse_evth[cpw.I.EVTH.Y_CORE_CM(reuse=1)] = (
                    cpw.M2CM * reuse_event["core_y_m"]
                )

                evttar.write_evth(evth=reuse_evth)
                evttar.write_bunches(bunches=reuse_event["cherenkov_bunches"])

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

    nfs.copy(
        op.join(tmp_dir, "corsika.stdout"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stdout"),
    )
    nfs.copy(
        op.join(tmp_dir, "corsika.stderr"),
        op.join(job["log_dir"], _run_id_str(job) + "_corsika.stderr"),
    )

    nfs.copy(
        src=op.join(tmp_dir, "grid.tar"),
        dst=op.join(job["feature_dir"], _run_id_str(job) + "_grid.tar"),
    )

    return cherenkov_pools_path, tabrec


def _run_merlict(job, cherenkov_pools_path, tmp_dir):
    detector_responses_path = op.join(tmp_dir, "detector_responses")
    if not op.exists(detector_responses_path):
        merlict_rc = merlict.plenoscope_propagator(
            corsika_run_path=cherenkov_pools_path,
            output_path=detector_responses_path,
            light_field_geometry_path=job["light_field_geometry_path"],
            merlict_plenoscope_propagator_path=job[
                "merlict_plenoscope_propagator_path"
            ],
            merlict_plenoscope_propagator_config_path=job[
                "merlict_plenoscope_propagator_config_path"
            ],
            random_seed=job["run_id"],
            photon_origins=True,
            stdout_path=op.join(tmp_dir, "merlict.stdout"),
            stderr_path=op.join(tmp_dir, "merlict.stderr"),
        )
        nfs.copy(
            op.join(tmp_dir, "merlict.stdout"),
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stdout"),
        )
        nfs.copy(
            op.join(tmp_dir, "merlict.stderr"),
            op.join(job["log_dir"], _run_id_str(job) + "_merlict.stderr"),
        )
        assert merlict_rc == 0

    return detector_responses_path


def _run_loose_trigger(
    job,
    tabrec,
    detector_responses_path,
    light_field_geometry,
    trigger_geometry,
    tmp_dir,
):
    # loop over sensor responses
    # --------------------------
    merlict_run = pl.Run(detector_responses_path)
    table_past_trigger = []
    tmp_past_trigger_dir = op.join(tmp_dir, "past_trigger")
    os.makedirs(tmp_past_trigger_dir, exist_ok=True)
    RAW_SKIP = int(job["raw_sensor_response"]["skip_num_events"])
    assert RAW_SKIP > 0

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_corsika_evth.raw
        run_id = int(cevth[cpw.I.EVTH.RUN_NUMBER])
        event_id = int(cevth[cpw.I.EVTH.EVENT_NUMBER])
        ide = {spt.IDX: unique.make_uid(run_id=run_id, event_id=event_id)}

        # apply loose trigger
        # -------------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices,
        ) = pl.trigger.estimate.first_stage(
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
            ptp["unique_id_str"] = unique.UID_FOTMAT_STR.format(ptp[spt.IDX])
            table_past_trigger.append(ptp)

            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)

            # export past loose trigger
            # -------------------------
            if ide[spt.IDX] % RAW_SKIP == 0:
                pl.tools.acp_format.compress_event_in_place(ptp["tmp_path"])
                final_tarname = ptp["unique_id_str"] + ".tar"
                plenoscope_event_dir_to_tar(
                    event_dir=ptp["tmp_path"],
                    output_tar_path=op.join(
                        tmp_past_trigger_dir, final_tarname
                    ),
                )
                nfs.copy(
                    src=op.join(tmp_past_trigger_dir, final_tarname),
                    dst=op.join(job["past_trigger_dir"], final_tarname),
                )

    return tabrec, table_past_trigger, tmp_past_trigger_dir


def _classify_cherenkov_photons(
    job,
    tabrec,
    tmp_dir,
    table_past_trigger,
    light_field_geometry,
    trigger_geometry,
):
    roi_cfg = job["cherenkov_classification"]["region_of_interest"]
    dbscan_cfg = job["cherenkov_classification"]

    with pl.photon_stream.loph.LopfTarWriter(
        path=os.path.join(tmp_dir, "reconstructed_cherenkov.tar"),
        id_num_digits=6+6,
    ) as cer_phs_run:
        for ptp in table_past_trigger:
            event = pl.Event(
                path=ptp["tmp_path"], light_field_geometry=light_field_geometry
            )
            trigger_responses = pl.trigger.io.read_trigger_response_from_path(
                path=os.path.join(event._path, "refocus_sum_trigger.json")
            )
            roi = pl.trigger.region_of_interest.from_trigger_response(
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
            cer_phs_run.add(uid=ptp[spt.IDX], phs=cer_phs)

    nfs.copy(
        src=op.join(tmp_dir, "reconstructed_cherenkov.tar"),
        dst=op.join(
            job["past_trigger_reconstructed_cherenkov_dir"],
            _run_id_str(job=job) + "_reconstructed_cherenkov.tar",
        ),
    )
    return tabrec


def _extract_features(
    tabrec, light_field_geometry, table_past_trigger, prng,
):
    light_field_geometry_addon = pl.features.make_light_field_geometry_addon(
        light_field_geometry=light_field_geometry
    )

    for pt in table_past_trigger:
        event = pl.Event(
            path=pt["tmp_path"], light_field_geometry=light_field_geometry
        )
        try:
            lfft = pl.features.extract_features(
                cherenkov_photons=event.cherenkov_photons,
                light_field_geometry=light_field_geometry,
                light_field_geometry_addon=light_field_geometry_addon,
                prng=prng,
            )
            lfft[spt.IDX] = pt[spt.IDX]
            tabrec["features"].append(lfft)
        except Exception as excep:
            print("idx:", pt[spt.IDX], excep)

    return tabrec


def _estimate_primary_trajectory(job, tmp_dir, light_field_geometry, tabrec):

    FUZZY_CONFIG = gamrec.trajectory.v2020nov12fuzzy0.config.compile_user_config(
        user_config=job["reconstruction"]["trajectory"]["fuzzy_method"]
    )
    MODEL_FIT_CONFIG = gamrec.trajectory.v2020dec04iron0b.config.compile_user_config(
        user_config=job["reconstruction"]["trajectory"]["core_axis_fit"]
    )

    _feature_table = spt.table_of_records_to_sparse_numeric_table(
        table_records={"features": tabrec["features"]},
        structure={"features": table.STRUCTURE["features"]},
    )
    shower_maximum_object_distance = spt.get_column_as_dict_by_index(
        table=_feature_table,
        level_key="features",
        column_key="image_smallest_ellipse_object_distance",
    )

    run = pl.photon_stream.loph.LopfTarReader(
        op.join(tmp_dir, "reconstructed_cherenkov.tar")
    )
    for event in run:
        airshower_id, loph_record = event

        if airshower_id in shower_maximum_object_distance:
            estimate, debug = gamrec.trajectory.v2020dec04iron0b.estimate(
                loph_record=loph_record,
                light_field_geometry=light_field_geometry,
                shower_maximum_object_distance=shower_maximum_object_distance[
                    airshower_id
                ],
                fuzzy_config=FUZZY_CONFIG,
                model_fit_config=MODEL_FIT_CONFIG,
            )

            if gamrec.trajectory.v2020dec04iron0b.is_valid_estimate(
                estimate=estimate
            ):
                rec = {}
                rec[spt.IDX] = airshower_id

                rec["cx_rad"] = estimate["primary_particle_cx"]
                rec["cy_rad"] = estimate["primary_particle_cy"]
                rec["x_m"] = estimate["primary_particle_x"]
                rec["y_m"] = estimate["primary_particle_y"]

                rec["fuzzy_cx_rad"] = debug["fuzzy_result"]["reco_cx"]
                rec["fuzzy_cy_rad"] = debug["fuzzy_result"]["reco_cy"]
                rec["fuzzy_main_axis_support_cx_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cx"
                ]
                rec["fuzzy_main_axis_support_cy_rad"] = debug["fuzzy_result"][
                    "main_axis_support_cy"
                ]
                rec["fuzzy_main_axis_support_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_support_uncertainty"]
                rec["fuzzy_main_axis_azimuth_rad"] = debug["fuzzy_result"][
                    "main_axis_azimuth"
                ]
                rec["fuzzy_main_axis_azimuth_uncertainty_rad"] = debug[
                    "fuzzy_result"
                ]["main_axis_azimuth_uncertainty"]

                tabrec["reconstructed_trajectory"].append(rec)

    return tabrec


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


def _export_event_table(job, tmp_dir, tabrec):
    event_table = spt.table_of_records_to_sparse_numeric_table(
        table_records=tabrec, structure=table.STRUCTURE
    )
    spt.write(
        path=op.join(tmp_dir, "event_table.tar"),
        table=event_table,
        structure=table.STRUCTURE,
    )
    nfs.copy(
        src=op.join(tmp_dir, "event_table.tar"),
        dst=op.join(job["feature_dir"], _run_id_str(job) + "_event_table.tar"),
    )


def _init_table_records():
    tabrec = {}
    for level_key in table.STRUCTURE:
        tabrec[level_key] = []
    return tabrec


def _export_job_to_log_dir(job):
    job_path = op.join(job["log_dir"], _run_id_str(job) + "_job.json")
    with open(job_path + ".tmp", "wt") as f:
        f.write(json.dumps(job, indent=4))
    nfs.move(job_path + ".tmp", job_path)


def run_job(job):
    _assert_resources_exist(job=job)
    _make_output_dirs(job=job)
    _export_job_to_log_dir(job=job)

    log_path = op.join(job["log_dir"], _run_id_str(job) + "_runtime.jsonl")
    jl = logging.JsonlLog(path=log_path + ".tmp")
    jl.log("starting run")

    jl.log("init prng")
    prng = np.random.Generator(np.random.MT19937(seed=job["run_id"]))

    with logging.TimeDelta(jl, "draw_primary"):
        corsika_primary_steering = production.corsika_primary.draw_corsika_primary_steering(
            run_id=job["run_id"],
            site=job["site"],
            particle=job["particle"],
            site_particle_deflection=job["site_particle_deflection"],
            num_events=job["num_air_showers"],
            prng=prng,
        )

    if job["tmp_dir"] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job["tmp_dir"], _run_id_str(job))
        os.makedirs(tmp_dir, exist_ok=True)
    jl.log("make tmp_dir: {:s}".format(tmp_dir))

    tabrec = _init_table_records()

    with logging.TimeDelta(jl, "corsika_and_grid"):
        (
            cherenkov_pools_path,
            tabrec,
        ) = _run_corsika_and_grid_and_output_to_tmp_dir(
            job=job,
            prng=prng,
            tmp_dir=tmp_dir,
            corsika_primary_steering=corsika_primary_steering,
            tabrec=tabrec,
        )
    with logging.TimeDelta(jl, "merlict"):
        detector_responses_path = _run_merlict(
            job=job,
            cherenkov_pools_path=cherenkov_pools_path,
            tmp_dir=tmp_dir,
        )

    if not job["keep_tmp"]:
        os.remove(cherenkov_pools_path)

    with logging.TimeDelta(jl, "read_geometry"):
        light_field_geometry = pl.LightFieldGeometry(
            path=job["light_field_geometry_path"]
        )
        trigger_geometry = pl.trigger.geometry.read(
            path=job["trigger_geometry_path"]
        )

    with logging.TimeDelta(jl, "pass_loose_trigger"):
        tabrec, table_past_trigger, tmp_past_trigger_dir = _run_loose_trigger(
            job=job,
            tabrec=tabrec,
            detector_responses_path=detector_responses_path,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            tmp_dir=tmp_dir,
        )

    with logging.TimeDelta(jl, "classify_cherenkov"):
        tabrec = _classify_cherenkov_photons(
            job=job,
            tabrec=tabrec,
            tmp_dir=tmp_dir,
            table_past_trigger=table_past_trigger,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
        )

    with logging.TimeDelta(jl, "extract_features"):
        tabrec = _extract_features(
            tabrec=tabrec,
            light_field_geometry=light_field_geometry,
            table_past_trigger=table_past_trigger,
            prng=prng,
        )

    with logging.TimeDelta(jl, "estimate_primary_trajectory"):
        tabrec = _estimate_primary_trajectory(
            job=job,
            tmp_dir=tmp_dir,
            light_field_geometry=light_field_geometry,
            tabrec=tabrec,
        )

    with logging.TimeDelta(jl, "export_event_table"):
        _export_event_table(job=job, tmp_dir=tmp_dir, tabrec=tabrec)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)
    jl.log("ending run")
    nfs.move(log_path + ".tmp", log_path)


def run_bundle(bundle):
    results = []
    for j, job in enumerate(bundle):
        msg = "\n#bundle {:d} of {:d}\n".format((j + 1), len(bundle))
        print(msg, file=sys.stdout)
        print(msg, file=sys.stderr)
        try:
            print('{{"run_id": {:d}"}}\n'.format(job["run_id"]))
            result = run_job(job=job)
        except Exception as exception_msg:
            print(exception_msg, file=sys.stderr)
            result = 0
        results.append(result)
    return results
