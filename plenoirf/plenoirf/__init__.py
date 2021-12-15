from . import summary
from . import analysis
from . import features
from . import table
from . import grid
from . import merlict
from . import logging
from . import map_and_reduce
from . import map_and_reduce_light_field_geometry
from . import network_file_system
from . import bundle
from . import provenance
from . import create_test_tables
from . import reconstruction
from . import utils
from . import production
from . import other_instruments
from . import single_thread_map_and_reduce
from . import unique

import os
import numpy as np
from os import path as op
from os.path import join as opj
import shutil
import subprocess
import random
import json
import glob
import tempfile
import pandas as pd
import tarfile
import io

import json_numpy
import plenopy as pl
import sparse_numeric_table as spt
import queue_map_reduce
from queue_map_reduce.tools import _log as qmrlog
import magnetic_deflection as mdfl
import gamma_ray_reconstruction as gamrec


MIN_PROTON_ENERGY_GEV = 5.0
MIN_HELIUM_ENERGY_GEV = 10.0

EXAMPLE_EXECUTABLES = {
    "corsika_primary_path": opj(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd",
    ),
    "merlict_plenoscope_propagator_path": opj(
        "build", "merlict", "merlict-plenoscope-propagation"
    ),
    "merlict_plenoscope_calibration_map_path": opj(
        "build", "merlict", "merlict-plenoscope-calibration-map"
    ),
    "merlict_plenoscope_calibration_reduce_path": opj(
        "build", "merlict", "merlict-plenoscope-calibration-reduce"
    ),
}

EXAMPLE_CONFIG_FILES = {
    "merlict_plenoscope_propagator_config_path": opj(
        "resources", "acp", "merlict_propagation_config.json"
    ),
    "plenoscope_scenery_path": opj("resources", "acp", "71m", "scenery"),
}

EXAMPLE_CONFIG = {
    "light_field_geometry": {
        "num_photons_per_block": 4 * 1000 * 1000,
        "num_blocks": 360,
    },
    "plenoscope_pointing": {"azimuth_deg": 0.0, "zenith_deg": 0.0},
    "sites": {
        "namibia": {
            "observation_level_asl_m": 2300,
            "earth_magnetic_field_x_muT": 12.5,
            "earth_magnetic_field_z_muT": -25.9,
            "atmosphere_id": 10,
            "geomagnetic_cutoff_rigidity_GV": 12.5,
            "coordinates_wgs1984": [-23.3425, 16.225556],
            "comment": "The Gamsberg-mesa in Khoma, Namibia, southern Africa.",
        },
        "chile": {
            "observation_level_asl_m": 5000,
            "earth_magnetic_field_x_muT": 20.815,
            "earth_magnetic_field_z_muT": -11.366,
            "atmosphere_id": 26,
            "geomagnetic_cutoff_rigidity_GV": 10.0,
            "coordinates_wgs1984": [-23.0193, -67.7532],
            "comment": "Llano de Chajnantor in Chile, southern America.",
        },
    },
    "particles": {
        "gamma": {
            "particle_id": 1,
            "energy_bin_edges_GeV": [
                utils.power10bin(decade=-1, bin=2, num_bins_per_decade=5),
                utils.power10bin(decade=3, bin=1, num_bins_per_decade=5),
            ],
            "max_scatter_angle_deg": 3.25,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": 0.0,
            "magnetic_deflection_max_off_axis_deg": 0.25,
        },
        "electron": {
            "particle_id": 3,
            "energy_bin_edges_GeV": [
                utils.power10bin(decade=-1, bin=3, num_bins_per_decade=5),
                utils.power10bin(decade=3, bin=1, num_bins_per_decade=5),
            ],
            "max_scatter_angle_deg": 6.5,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": -1.0,
            "magnetic_deflection_max_off_axis_deg": 0.5,
        },
        "proton": {
            "particle_id": 14,
            "energy_bin_edges_GeV": [
                max(
                    MIN_PROTON_ENERGY_GEV,
                    utils.power10bin(decade=0, bin=3, num_bins_per_decade=5),
                ),
                utils.power10bin(decade=3, bin=1, num_bins_per_decade=5),
            ],
            "max_scatter_angle_deg": 13,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": +1.0,
            "magnetic_deflection_max_off_axis_deg": 1.5,
        },
        "helium": {
            "particle_id": 402,
            "energy_bin_edges_GeV": [
                max(
                    MIN_HELIUM_ENERGY_GEV,
                    utils.power10bin(decade=1, bin=0, num_bins_per_decade=5),
                ),
                utils.power10bin(decade=3, bin=1, num_bins_per_decade=5),
            ],
            "max_scatter_angle_deg": 13,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": +2.0,
            "magnetic_deflection_max_off_axis_deg": 1.5,
        },
    },
    "grid": production.example.EXAMPLE_GRID,
    "sum_trigger": {
        "object_distances_m": [
            5000.0,
            6164.0,
            7600.0,
            9369.0,
            11551.0,
            14240.0,
            17556.0,
            21644.0,
            26683.0,
            32897.0,
            40557.0,
            50000.0,
        ],
        "threshold_pe": 115,
        "integration_time_slices": 10,
        "image": {
            "image_outer_radius_deg": 3.25 - 0.033335,
            "pixel_spacing_deg": 0.06667,
            "pixel_radius_deg": 0.146674,
            "max_number_nearest_lixel_in_pixel": 7,
        },
    },
    "cherenkov_classification": {
        "region_of_interest": {
            "time_offset_start_s": -10e-9,
            "time_offset_stop_s": 10e-9,
            "direction_radius_deg": 2.0,
            "object_distance_offsets_m": [4000.0, 2000.0, 0.0, -2000.0,],
        },
        "min_num_photons": 17,
        "neighborhood_radius_deg": 0.075,
        "direction_to_time_mixing_deg_per_s": 0.375e9,
    },
    "reconstruction": {
        "trajectory": gamrec.trajectory.v2020dec04iron0b.config.make_example_config_for_71m_plenoscope(
            fov_radius_deg=3.25
        ),
    },
    "raw_sensor_response": {"skip_num_events": 50,},
    "runs": {
        "gamma": {"num": 64, "first_run_id": 1},
        "electron": {"num": 64, "first_run_id": 1},
        "proton": {"num": 64, "first_run_id": 1},
        "helium": {"num": 64, "first_run_id": 1},
    },
    "magnetic_deflection": {"num_energy_supports": 512, "max_energy_GeV": 64},
    "num_airshowers_per_run": 100,
    "artificial_core_limitation": {
        "gamma": None,
        "electron": None,
        "proton": None,
        "helium": None,
    },
}


def init(out_dir, config=EXAMPLE_CONFIG, cfg_files=EXAMPLE_CONFIG_FILES):
    out_absdir = op.abspath(out_dir)
    os.makedirs(out_absdir)
    os.makedirs(opj(out_absdir, "input"))

    json_numpy.write(
        path=opj(out_absdir, "input", "config.json" + "tmp"), out_dict=config,
    )
    network_file_system.move(
        opj(out_absdir, "input", "config.json" + "tmp"),
        opj(out_absdir, "input", "config.json"),
    )

    network_file_system.copy(
        src=cfg_files["plenoscope_scenery_path"],
        dst=opj(out_absdir, "input", "scenery"),
    )
    network_file_system.copy(
        src=cfg_files["merlict_plenoscope_propagator_config_path"],
        dst=opj(out_absdir, "input", "merlict_propagation_config.json"),
    )


def _estimate_magnetic_deflection_of_air_showers(
    cfg, out_absdir, map_and_reduce_pool
):
    qmrlog("Estimating magnetic deflection.")
    mdfl_dir = opj(out_absdir, "magnetic_deflection")

    if op.exists(mdfl_dir):
        mdflcfg = mdfl.read_config(work_dir=mdfl_dir)

        for particle in cfg["particles"]:
            assert particle in mdflcfg["particles"]
        for site in cfg["sites"]:
            assert site in mdflcfg["sites"]
        np.testing.assert_almost_equal(
            cfg["plenoscope_pointing"]["azimuth_deg"],
            mdflcfg["pointing"]["azimuth_deg"],
            decimal=2,
        )
        np.testing.assert_almost_equal(
            cfg["plenoscope_pointing"]["zenith_deg"],
            mdflcfg["pointing"]["zenith_deg"],
            decimal=2,
        )
    else:
        mdfl.init(
            work_dir=mdfl_dir,
            particles=cfg["particles"],
            sites=cfg["sites"],
            pointing=cfg["plenoscope_pointing"],
            max_energy=cfg["magnetic_deflection"]["max_energy_GeV"],
            num_energy_supports=cfg["magnetic_deflection"][
                "num_energy_supports"
            ],
        )

        jobs = mdfl.make_jobs(work_dir=mdfl_dir)
        _ = map_and_reduce_pool.map(mdfl.map_and_reduce.run_job, jobs)
        mdfl.reduce(work_dir=mdfl_dir)


def _estimate_light_field_geometry_of_plenoscope(
    cfg, out_absdir, map_and_reduce_pool, executables
):
    qmrlog("Estimating light-field-geometry.")

    if op.exists(opj(out_absdir, "light_field_geometry")):
        assert utils.contains_same_bytes(
            opj(out_absdir, "input", "scenery", "scenery.json"),
            opj(
                out_absdir,
                "light_field_geometry",
                "input",
                "scenery",
                "scenery.json",
            ),
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="light_field_geometry_", dir=out_absdir
        ) as tmp_dir:
            lfg_jobs = map_and_reduce_light_field_geometry.make_jobs(
                merlict_map_path=executables[
                    "merlict_plenoscope_calibration_map_path"
                ],
                scenery_path=opj(out_absdir, "input", "scenery"),
                out_dir=tmp_dir,
                num_photons_per_block=cfg["light_field_geometry"][
                    "num_photons_per_block"
                ],
                num_blocks=cfg["light_field_geometry"]["num_blocks"],
                random_seed=0,
            )
            _ = map_and_reduce_pool.map(
                map_and_reduce_light_field_geometry.run_job, lfg_jobs
            )
            subprocess.call(
                [
                    executables["merlict_plenoscope_calibration_reduce_path"],
                    "--input",
                    tmp_dir,
                    "--output",
                    opj(out_absdir, "light_field_geometry"),
                ]
            )

    if not op.exists(opj(out_absdir, "light_field_geometry", "plot")):
        qmrlog("Plotting light-field-geometry.")
        lfg = pl.LightFieldGeometry(opj(out_absdir, "light_field_geometry"))
        pl.plot.light_field_geometry.save_all(
            light_field_geometry=lfg,
            out_dir=opj(out_absdir, "light_field_geometry", "plot"),
        )


def _estimate_trigger_geometry_of_plenoscope(
    cfg, out_absdir,
):
    qmrlog("Estimating trigger-geometry.")

    if not op.exists(opj(out_absdir, "trigger_geometry")):
        light_field_geometry = pl.LightFieldGeometry(
            path=opj(out_absdir, "light_field_geometry")
        )
        img = cfg["sum_trigger"]["image"]
        trigger_image_geometry = pl.trigger.geometry.init_trigger_image_geometry(
            image_outer_radius_rad=np.deg2rad(img["image_outer_radius_deg"]),
            pixel_spacing_rad=np.deg2rad(img["pixel_spacing_deg"]),
            pixel_radius_rad=np.deg2rad(img["pixel_radius_deg"]),
            max_number_nearest_lixel_in_pixel=img[
                "max_number_nearest_lixel_in_pixel"
            ],
        )
        trigger_geometry = pl.trigger.geometry.init_trigger_geometry(
            light_field_geometry=light_field_geometry,
            trigger_image_geometry=trigger_image_geometry,
            object_distances=cfg["sum_trigger"]["object_distances_m"],
        )
        pl.trigger.geometry.write(
            trigger_geometry=trigger_geometry,
            path=opj(out_absdir, "trigger_geometry"),
        )
        tss = pl.trigger.geometry.init_summation_statistics(
            trigger_geometry=trigger_geometry
        )
        pl.trigger.plot.write_figures_to_directory(
            trigger_geometry=trigger_geometry,
            trigger_summation_statistics=tss,
            out_dir=opj(out_absdir, "trigger_geometry", "plot"),
        )


def _populate_table_of_thrown_air_showers(
    cfg,
    out_absdir,
    map_and_reduce_pool,
    executables,
    tmp_absdir,
    date_dict_now,
    KEEP_TMP,
    LAZY_REDUCTION=False,
    num_parallel_jobs=2000,
):
    qmrlog("Estimating instrument-response.")
    table_absdir = opj(out_absdir, "event_table")
    os.makedirs(table_absdir, exist_ok=True)

    prov = provenance.make_provenance()
    prov = provenance.add_corsika(
        prov=prov, corsika_primary_path=corsika_primary_path
    )

    qmrlog("Write provenance.")
    json_numpy.write(
        path=opj(table_absdir, "provenance.json"), out_dict=prov,
    )

    deflection = mdfl.read_deflection(
        work_dir=opj(out_absdir, "magnetic_deflection"), style="dict",
    )

    irf_jobs = []
    for site_key in cfg["sites"]:
        site_absdir = opj(table_absdir, site_key)
        if op.exists(site_absdir):
            continue
        os.makedirs(site_absdir, exist_ok=True)

        for particle_key in cfg["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            if op.exists(site_particle_absdir):
                continue
            os.makedirs(site_particle_absdir, exist_ok=True)

            run_id = cfg["runs"][particle_key]["first_run_id"]
            for job_idx in np.arange(cfg["runs"][particle_key]["num"]):
                assert run_id > 0

                irf_job = {
                    "run_id": run_id,
                    "num_air_showers": cfg["num_airshowers_per_run"],
                    "plenoscope_pointing": cfg["plenoscope_pointing"],
                    "particle": cfg["particles"][particle_key],
                    "site": cfg["sites"][site_key],
                    "site_particle_deflection": deflection[site_key][
                        particle_key
                    ],
                    "grid": cfg["grid"],
                    "raw_sensor_response": cfg["raw_sensor_response"],
                    "sum_trigger": cfg["sum_trigger"],
                    "cherenkov_classification": cfg[
                        "cherenkov_classification"
                    ],
                    "corsika_primary_path": executables[
                        "corsika_primary_path"
                    ],
                    "plenoscope_scenery_path": opj(
                        out_absdir, "input", "scenery"
                    ),
                    "merlict_plenoscope_propagator_path": executables[
                        "merlict_plenoscope_propagator_path"
                    ],
                    "light_field_geometry_path": opj(
                        out_absdir, "light_field_geometry"
                    ),
                    "trigger_geometry_path": opj(
                        out_absdir, "trigger_geometry"
                    ),
                    "merlict_plenoscope_propagator_config_path": opj(
                        out_absdir, "input", "merlict_propagation_config.json"
                    ),
                    "log_dir": opj(site_particle_absdir, "log.map"),
                    "past_trigger_dir": opj(
                        site_particle_absdir, "past_trigger.map"
                    ),
                    "past_trigger_reconstructed_cherenkov_dir": opj(
                        site_particle_absdir,
                        "past_trigger_reconstructed_cherenkov_dir.map",
                    ),
                    "feature_dir": opj(site_particle_absdir, "features.map"),
                    "keep_tmp": KEEP_TMP,
                    "tmp_dir": tmp_absdir,
                    "date": date_dict_now,
                    "artificial_core_limitation": cfg[
                        "artificial_core_limitation"
                    ][particle_key],
                    "reconstruction": cfg["reconstruction"],
                }
                run_id += 1
                irf_jobs.append(irf_job)

    random.shuffle(irf_jobs)

    irf_bundles = bundle.bundle_jobs(
        jobs=irf_jobs, desired_num_bunbles=num_parallel_jobs
    )

    _ = map_and_reduce_pool.map(map_and_reduce.run_bundle, irf_bundles)

    qmrlog("Reduce instrument-response.")

    for site_key in cfg["sites"]:
        site_absdir = opj(table_absdir, site_key)
        for particle_key in cfg["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            log_absdir = opj(site_particle_absdir, "log.map")
            feature_absdir = opj(site_particle_absdir, "features.map")

            # run-time
            # ========
            log_abspath = opj(site_particle_absdir, "runtime.csv")
            if not op.exists(log_abspath) or not LAZY_REDUCTION:
                _lop_paths = glob.glob(opj(log_absdir, "*_runtime.jsonl"))
                logging.reduce(
                    list_of_log_paths=_lop_paths, out_path=log_abspath
                )
            qmrlog("Reduce {:s} {:s} run-time.".format(site_key, particle_key))

            # event table
            # ===========
            event_table_abspath = opj(site_particle_absdir, "event_table.tar")
            if not op.exists(event_table_abspath) or not LAZY_REDUCTION:
                _feature_paths = glob.glob(
                    opj(feature_absdir, "*_event_table.tar")
                )
                event_table = spt.concatenate_files(
                    list_of_table_paths=_feature_paths,
                    structure=table.STRUCTURE,
                )
                spt.write(
                    path=event_table_abspath,
                    table=event_table,
                    structure=table.STRUCTURE,
                )
            qmrlog(
                "Reduce {:s} {:s} event_table.".format(site_key, particle_key)
            )

            # grid images
            # ===========
            grid_abspath = opj(site_particle_absdir, "grid.tar")
            if not op.exists(grid_abspath) or not LAZY_REDUCTION:
                _grid_paths = glob.glob(opj(feature_absdir, "*_grid.tar"))
                grid.reduce(
                    list_of_grid_paths=_grid_paths, out_path=grid_abspath
                )
            qmrlog("Reduce {:s} {:s} grid.".format(site_key, particle_key))

            # cherenkov-photon-stream
            # =======================
            loph_abspath = opj(site_particle_absdir, "cherenkov.phs.loph.tar")
            tmp_loph_abspath = loph_abspath + ".tmp"
            qmrlog(
                "Reduce {:s} {:s} cherenkov phs.".format(
                    site_key, particle_key
                )
            )
            if not op.exists(loph_abspath) or not LAZY_REDUCTION:
                qmrlog("compile ", loph_abspath)
                _cer_run_paths = glob.glob(
                    opj(
                        site_particle_absdir,
                        "past_trigger_reconstructed_cherenkov_dir.map",
                        "*_reconstructed_cherenkov.tar",
                    )
                )
                _cer_run_paths.sort()
                pl.photon_stream.loph.concatenate_tars(
                    in_paths=_cer_run_paths, out_path=tmp_loph_abspath
                )
                network_file_system.move(tmp_loph_abspath, loph_abspath)


def run(
    path,
    map_and_reduce_pool=single_thread_map_and_reduce,
    num_parallel_jobs=2000,
    executables=EXAMPLE_EXECUTABLES,
    TMP_DIR_ON_WORKERNODE=True,
    KEEP_TMP=False,
    LAZY_REDUCTION=False,
):
    date_dict_now = utils.date_dict_now()
    qmrlog("Start run()")

    out_absdir = op.abspath(path)
    for exe_path in executables:
        executables[exe_path] = op.abspath(executables[exe_path])

    if TMP_DIR_ON_WORKERNODE:
        tmp_absdir = None
        qmrlog("Use tmp_dir on workernodes.")
    else:
        tmp_absdir = opj(out_absdir, "tmp")
        os.makedirs(tmp_absdir, exist_ok=True)
        qmrlog("Use tmp_dir in out_dir {:s}.".format(tmp_absdir))

    qmrlog("Read config")
    cfg = json_numpy.read(opj(out_absdir, "input", "config.json"))

    _estimate_magnetic_deflection_of_air_showers(
        cfg=cfg, out_absdir=out_absdir, map_and_reduce_pool=map_and_reduce_pool
    )

    _estimate_light_field_geometry_of_plenoscope(
        cfg=cfg,
        out_absdir=out_absdir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
    )

    _estimate_trigger_geometry_of_plenoscope(cfg=cfg, out_absdir=out_absdir)

    _populate_table_of_thrown_air_showers(
        cfg=cfg,
        out_absdir=out_absdir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
        tmp_absdir=tmp_absdir,
        KEEP_TMP=KEEP_TMP,
        date_dict_now=date_dict_now,
        LAZY_REDUCTION=LAZY_REDUCTION,
        num_parallel_jobs=num_parallel_jobs,
    )

    qmrlog("End run().")
