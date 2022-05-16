from . import summary
from . import analysis
from . import features
from . import table
from . import grid
from . import logging
from . import instrument_response
from . import network_file_system as nfs
from . import bundle
from . import provenance
from . import create_test_tables
from . import reconstruction
from . import utils
from . import production
from . import other_instruments
from . import unique

import os
import numpy as np
from os import path as op
from os.path import join as opj
import shutil
import subprocess
import multiprocessing
import random
import glob
import tempfile
import pandas as pd
import tarfile
import io

import json_numpy
import binning_utils
import plenopy as pl
import sparse_numeric_table as spt
import queue_map_reduce
from queue_map_reduce.tools import _log as qmrlog
import magnetic_deflection as mdfl
import gamma_ray_reconstruction as gamrec


MIN_PROTON_ENERGY_GEV = 5.0
MIN_HELIUM_ENERGY_GEV = 10.0

EXAMPLE_EXECUTABLE_PATHS = {
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

EXAMPLE_CONFIG_FILE_PATHS = {
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
                binning_utils.power10.lower_bin_edge(decade=-1, bin=2, num_bins_per_decade=5),
                binning_utils.power10.lower_bin_edge(decade=3, bin=1, num_bins_per_decade=5),
            ],
            "max_scatter_angle_deg": 3.25,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": 0.0,
            "magnetic_deflection_max_off_axis_deg": 0.25,
        },
        "electron": {
            "particle_id": 3,
            "energy_bin_edges_GeV": [
                binning_utils.power10.lower_bin_edge(decade=-1, bin=3, num_bins_per_decade=5),
                binning_utils.power10.lower_bin_edge(decade=3, bin=1, num_bins_per_decade=5),
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
                    binning_utils.power10.lower_bin_edge(decade=0, bin=3, num_bins_per_decade=5),
                ),
                binning_utils.power10.lower_bin_edge(decade=3, bin=1, num_bins_per_decade=5),
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
                    binning_utils.power10.lower_bin_edge(decade=1, bin=0, num_bins_per_decade=5),
                ),
                binning_utils.power10.lower_bin_edge(decade=3, bin=1, num_bins_per_decade=5),
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


def init(
    run_dir, config=EXAMPLE_CONFIG, config_file_paths=EXAMPLE_CONFIG_FILE_PATHS
):
    run_dir = op.abspath(run_dir)
    os.makedirs(run_dir)
    os.makedirs(opj(run_dir, "input"))

    json_numpy.write(
        path=opj(run_dir, "input", "config.json" + "tmp"), out_dict=config,
    )
    nfs.move(
        opj(run_dir, "input", "config.json" + "tmp"),
        opj(run_dir, "input", "config.json"),
    )

    nfs.copy(
        src=config_file_paths["plenoscope_scenery_path"],
        dst=opj(run_dir, "input", "scenery"),
    )
    nfs.copy(
        src=config_file_paths["merlict_plenoscope_propagator_config_path"],
        dst=opj(run_dir, "input", "merlict_propagation_config.json"),
    )


def _estimate_magnetic_deflection_of_air_showers(
    config, run_dir, map_and_reduce_pool
):
    qmrlog("Estimating magnetic deflection.")
    mdfl_dir = opj(run_dir, "magnetic_deflection")

    if op.exists(mdfl_dir):
        mdflconfig = mdfl.read_config(work_dir=mdfl_dir)

        for particle in config["particles"]:
            assert particle in mdflconfig["particles"]
        for site in config["sites"]:
            assert site in mdflconfig["sites"]
        np.testing.assert_almost_equal(
            config["plenoscope_pointing"]["azimuth_deg"],
            mdflconfig["pointing"]["azimuth_deg"],
            decimal=2,
        )
        np.testing.assert_almost_equal(
            config["plenoscope_pointing"]["zenith_deg"],
            mdflconfig["pointing"]["zenith_deg"],
            decimal=2,
        )
    else:
        mdfl.init(
            work_dir=mdfl_dir,
            particles=config["particles"],
            sites=config["sites"],
            pointing=config["plenoscope_pointing"],
            max_energy=config["magnetic_deflection"]["max_energy_GeV"],
            num_energy_supports=config["magnetic_deflection"][
                "num_energy_supports"
            ],
        )

        jobs = mdfl.make_jobs(work_dir=mdfl_dir)
        _ = map_and_reduce_pool.map(mdfl.map_and_reduce.run_job, jobs)
        mdfl.reduce(work_dir=mdfl_dir)


def _estimate_light_field_geometry_of_plenoscope(
    config, run_dir, map_and_reduce_pool, executables
):
    qmrlog("Estimating light-field-geometry.")

    if op.exists(opj(run_dir, "light_field_geometry")):
        assert utils.contains_same_bytes(
            opj(run_dir, "input", "scenery", "scenery.json"),
            opj(
                run_dir,
                "light_field_geometry",
                "input",
                "scenery",
                "scenery.json",
            ),
        )
    else:
        with tempfile.TemporaryDirectory(
            prefix="light_field_geometry_", dir=run_dir
        ) as map_dir:
            lfg_jobs = production.light_field_geometry.make_jobs(
                merlict_map_path=executables[
                    "merlict_plenoscope_calibration_map_path"
                ],
                scenery_path=opj(run_dir, "input", "scenery"),
                map_dir=map_dir,
                num_photons_per_block=config["light_field_geometry"][
                    "num_photons_per_block"
                ],
                num_blocks=config["light_field_geometry"]["num_blocks"],
                random_seed=0,
            )
            _ = map_and_reduce_pool.map(
                production.light_field_geometry.run_job, lfg_jobs
            )
            production.light_field_geometry.reduce(
                merlict_reduce_path=executables[
                    "merlict_plenoscope_calibration_reduce_path"
                ],
                map_dir=map_dir,
                out_dir=opj(run_dir, "light_field_geometry"),
            )

    if not op.exists(opj(run_dir, "light_field_geometry", "plot")):
        qmrlog("Plotting light-field-geometry.")
        lfg = pl.LightFieldGeometry(opj(run_dir, "light_field_geometry"))
        pl.plot.light_field_geometry.save_all(
            light_field_geometry=lfg,
            out_dir=opj(run_dir, "light_field_geometry", "plot"),
        )


def _estimate_trigger_geometry_of_plenoscope(
    config, run_dir,
):
    qmrlog("Estimating trigger-geometry.")
    if not op.exists(opj(run_dir, "trigger_geometry")):
        light_field_geometry = pl.LightFieldGeometry(
            path=opj(run_dir, "light_field_geometry")
        )
        img = config["sum_trigger"]["image"]
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
            object_distances=config["sum_trigger"]["object_distances_m"],
        )
        pl.trigger.geometry.write(
            trigger_geometry=trigger_geometry,
            path=opj(run_dir, "trigger_geometry"),
        )
        tss = pl.trigger.geometry.init_summation_statistics(
            trigger_geometry=trigger_geometry
        )
        pl.trigger.plot.write_figures_to_directory(
            trigger_geometry=trigger_geometry,
            trigger_summation_statistics=tss,
            out_dir=opj(run_dir, "trigger_geometry", "plot"),
        )


def _populate_table_of_thrown_air_showers(
    config,
    run_dir,
    map_and_reduce_pool,
    executables,
    tmp_absdir,
    date_dict,
    KEEP_TMP,
    LAZY_REDUCTION=False,
    num_parallel_jobs=2000,
):
    qmrlog("Estimating instrument-response.")
    table_absdir = opj(run_dir, "event_table")
    os.makedirs(table_absdir, exist_ok=True)

    prov = provenance.make_provenance()
    prov = provenance.add_corsika(
        prov=prov, corsika_primary_path=executables["corsika_primary_path"]
    )

    qmrlog("Write provenance.")
    json_numpy.write(
        path=opj(table_absdir, "provenance.json"), out_dict=prov,
    )

    deflection = mdfl.read_deflection(
        work_dir=opj(run_dir, "magnetic_deflection"), style="dict",
    )

    irf_jobs = []
    for site_key in config["sites"]:
        site_absdir = opj(table_absdir, site_key)
        if op.exists(site_absdir):
            continue
        os.makedirs(site_absdir, exist_ok=True)

        for particle_key in config["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            if op.exists(site_particle_absdir):
                continue
            os.makedirs(site_particle_absdir, exist_ok=True)

            run_id = config["runs"][particle_key]["first_run_id"]
            for job_idx in np.arange(config["runs"][particle_key]["num"]):
                assert run_id > 0

                irf_job = instrument_response.make_job_dict(
                    run_dir=run_dir,
                    production_key="event_table",
                    run_id=run_id,
                    site_key=site_key,
                    particle_key=particle_key,
                    config=config,
                    deflection_table=deflection,
                    num_air_showers=config["num_airshowers_per_run"],
                    corsika_primary_path=executables["corsika_primary_path"],
                    merlict_plenoscope_propagator_path=executables[
                        "merlict_plenoscope_propagator_path"
                    ],
                    tmp_dir=tmp_absdir,
                    keep_tmp_dir=KEEP_TMP,
                    date_dict=date_dict,
                )

                run_id += 1
                irf_jobs.append(irf_job)

    random.shuffle(irf_jobs)

    irf_jobs_in_bundles = bundle.make_jobs_in_bundles(
        jobs=irf_jobs, desired_num_bunbles=num_parallel_jobs
    )

    _ = map_and_reduce_pool.map(
        instrument_response.run_jobs_in_bundles, irf_jobs_in_bundles
    )

    qmrlog("Reduce instrument-response.")

    for site_key in config["sites"]:
        for particle_key in config["particles"]:
            instrument_response.reduce(
                run_dir=run_dir,
                production_key="event_table",
                site_key=site_key,
                particle_key=particle_key,
                LAZY=LAZY_REDUCTION,
            )


def run(
    run_dir,
    map_and_reduce_pool=multiprocessing.Pool(1),
    num_parallel_jobs=2000,
    executables=EXAMPLE_EXECUTABLE_PATHS,
    TMP_DIR_ON_WORKERNODE=True,
    KEEP_TMP=False,
    LAZY_REDUCTION=False,
):
    date_dict = provenance.get_time_dict_now()
    qmrlog("Start run()")

    run_dir = op.abspath(run_dir)
    for exe_path in executables:
        executables[exe_path] = op.abspath(executables[exe_path])

    if TMP_DIR_ON_WORKERNODE:
        tmp_absdir = None
        qmrlog("Use tmp_dir on workernodes.")
    else:
        tmp_absdir = opj(run_dir, "tmp")
        os.makedirs(tmp_absdir, exist_ok=True)
        qmrlog("Use tmp_dir in out_dir {:s}.".format(tmp_absdir))

    qmrlog("Read config")
    config = json_numpy.read(opj(run_dir, "input", "config.json"))

    _estimate_magnetic_deflection_of_air_showers(
        config=config, run_dir=run_dir, map_and_reduce_pool=map_and_reduce_pool
    )

    _estimate_light_field_geometry_of_plenoscope(
        config=config,
        run_dir=run_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
    )

    _estimate_trigger_geometry_of_plenoscope(config=config, run_dir=run_dir)

    _populate_table_of_thrown_air_showers(
        config=config,
        run_dir=run_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        executables=executables,
        tmp_absdir=tmp_absdir,
        KEEP_TMP=KEEP_TMP,
        date_dict=date_dict,
        LAZY_REDUCTION=LAZY_REDUCTION,
        num_parallel_jobs=num_parallel_jobs,
    )

    qmrlog("End run().")
