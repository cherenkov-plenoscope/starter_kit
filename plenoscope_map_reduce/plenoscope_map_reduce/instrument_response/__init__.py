from . import random
from . import table
from . import grid
from . import merlict
from . import logging
from . import query
from . import map_and_reduce
from . import network_file_system
from .. import light_field_geometry as plmr_light_field_geometry

import os
import numpy as np
from os import path as op
from os.path import join as opj
import shutil
import subprocess
import random
import json
import multiprocessing
import glob
import tempfile

import sun_grid_engine_map as sge


EXAMPLE_EXECUTABLES = {
    "corsika_primary_path": opj(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd"),

    "merlict_plenoscope_propagator_path": opj(
        "build",
        "merlict",
        "merlict-plenoscope-propagation"),

    "merlict_plenoscope_calibration_map_path": opj(
        "build",
        "merlict",
        "merlict-plenoscope-calibration-map"),

    "merlict_plenoscope_calibration_reduce_path": opj(
        "build",
        "merlict",
        "merlict-plenoscope-calibration-reduce"),
}

EXAMPLE_CONFIG_FILES = {
    "merlict_plenoscope_propagator_config_path": opj(
        "resources",
        "acp",
        "merlict_propagation_config.json"),

    "plenoscope_scenery_path": opj(
        "resources",
        "acp",
        "71m",
        "scenery"),
}

EXAMPLE_CONFIG = {
    "light_field_geometry": {
        "num_photons_per_block": 1000*1000,
        "num_blocks": 1337,
    },

    "plenoscope_pointing": {
        "azimuth_deg": 0.,
        "zenith_deg": 0.
    },

    "sites": {
        "namibia": {
            "observation_level_asl_m": 2300,
            "earth_magnetic_field_x_muT": 12.5,
            "earth_magnetic_field_z_muT": -25.9,
            "atmosphere_id": 10,
        },
        "chile": {
            "observation_level_asl_m": 5000,
            "earth_magnetic_field_x_muT": 20.815,
            "earth_magnetic_field_z_muT": -11.366,
            "atmosphere_id": 26,
        },
    },

    "particles": {
        "gamma": {
            "particle_id": 1,
            "energy_bin_edges_GeV": [0.5, 100],
            "max_scatter_angle_deg": 5,
            "energy_power_law_slope": -1.7,
        },
        "electron": {
            "particle_id": 3,
            "energy_bin_edges_GeV": [0.5, 100],
            "max_scatter_angle_deg": 30,
            "energy_power_law_slope": -1.7,
        },
        "proton": {
            "particle_id": 14,
            "energy_bin_edges_GeV": [5, 100],
            "max_scatter_angle_deg": 30,
            "energy_power_law_slope": -1.7,
        },
    },

    "grid": {
        "num_bins_radius": 512,
        "threshold_num_photons": 50
    },

    "sum_trigger": {
        "patch_threshold": 103,
        "integration_time_in_slices": 10,
        "min_num_neighbors": 3,
        "object_distances": [10e3, 15e3, 20e3],
    },

    "num_runs": {
        "gamma": 1,
        "electron": 4,
        "proton": 5
    },

    "num_airshowers_per_run": 100,
}


def init(out_dir, config=EXAMPLE_CONFIG, cfg_files=EXAMPLE_CONFIG_FILES):
    out_absdir = op.abspath(out_dir)
    os.makedirs(out_absdir)
    os.makedirs(opj(out_absdir, 'input'))
    with open(opj(out_absdir, 'input', 'config.json'+'tmp'), "wt") as fout:
        fout.write(json.dumps(config, indent=4))
    shutil.move(
        opj(out_absdir, 'input', 'config.json'+'tmp'),
        opj(out_absdir, 'input', 'config.json'))

    network_file_system.copy(
        src=cfg_files['plenoscope_scenery_path'],
        dst=opj(out_absdir, 'input', 'scenery'))

    network_file_system.copy(
        src=cfg_files['merlict_plenoscope_propagator_config_path'],
        dst=opj(out_absdir, 'input', 'merlict_propagation_config.json'))


def run(
    path,
    MULTIPROCESSING_POOL="sun_grid_engine",
    executables=EXAMPLE_EXECUTABLES,
    TMP_DIR_ON_WORKERNODE=True,
    KEEP_TMP=False,
    LAZY_REDUCTION=False,
):
    date_dict_now = map_and_reduce.date_dict_now()
    sge._print("Start run()")

    out_absdir = op.abspath(path)
    for exe_path in executables:
        executables[exe_path] = op.abspath(executables[exe_path])

    if TMP_DIR_ON_WORKERNODE:
        tmp_absdir = None
        sge._print("Use tmp_dir on workernodes.")
    else:
        tmp_absdir = opj(out_absdir, "tmp")
        os.makedirs(tmp_absdir, exist_ok=True)
        sge._print("Use tmp_dir in out_dir {:s}.".format(tmp_absdir))

    if MULTIPROCESSING_POOL == "sun_grid_engine":
        pool = sge
        sge._print("Use sun-grid-engine multiprocessing-pool.")
    elif MULTIPROCESSING_POOL == "local":
        pool = multiprocessing.Pool(8)
        sge._print("Use local multiprocessing-pool.")
    else:
        raise KeyError(
            "Unknown MULTIPROCESSING_POOL: {:s}".format(MULTIPROCESSING_POOL))

    sge._print("Read config")
    with open(opj(out_absdir, 'input', 'config.json'), "rt") as fin:
        cfg = json.loads(fin.read())

    sge._print("Estimating light-field-geometry.")
    # ============================================
    if op.exists(opj(out_absdir, 'light_field_geometry')):
        assert map_and_reduce.contains_same_bytes(
            opj(
                out_absdir,
                'input',
                'scenery',
                'scenery.json'),
            opj(
                out_absdir,
                'light_field_geometry',
                'input',
                'scenery',
                'scenery.json'))
    else:
        with tempfile.TemporaryDirectory(
            prefix='light_field_geometry_',
            dir=out_absdir
        ) as tmp_dir:
            lfg_jobs = plmr_light_field_geometry.make_jobs(
                merlict_map_path=executables[
                    "merlict_plenoscope_calibration_map_path"],
                scenery_path=opj(out_absdir, 'input', 'scenery'),
                out_dir=tmp_dir,
                num_photons_per_block=cfg[
                    'light_field_geometry']['num_photons_per_block'],
                num_blocks=cfg[
                    'light_field_geometry']['num_blocks'],
                random_seed=0)
            rc = pool.map(plmr_light_field_geometry.run_job, lfg_jobs)
            subprocess.call([
                executables["merlict_plenoscope_calibration_reduce_path"],
                '--input', tmp_dir,
                '--output', opj(out_absdir, 'light_field_geometry')])

    sge._print("Estimating instrument-response.")
    # ===========================================
    irf_jobs = []
    run_id = 1
    for site_key in cfg["sites"]:
        site_absdir = opj(out_absdir, site_key)
        if op.exists(site_absdir):
            continue
        os.makedirs(site_absdir, exist_ok=True)

        for particle_key in cfg["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            if op.exists(site_particle_absdir):
                continue
            os.makedirs(site_particle_absdir, exist_ok=True)
            for job_idx in np.arange(cfg["num_runs"][particle_key]):

                irf_job = {
                    "run_id": run_id,
                    "num_air_showers": cfg["num_airshowers_per_run"],
                    "plenoscope_pointing": cfg["plenoscope_pointing"],
                    "particle": cfg["particles"][particle_key],
                    "site": cfg["sites"][site_key],
                    "grid": cfg["grid"],
                    "sum_trigger": cfg["sum_trigger"],
                    "corsika_primary_path": executables[
                        "corsika_primary_path"],
                    "plenoscope_scenery_path": opj(
                        out_absdir,
                        'input',
                        'scenery'),
                    "merlict_plenoscope_propagator_path": executables[
                        "merlict_plenoscope_propagator_path"],
                    "light_field_geometry_path":
                        opj(out_absdir, 'light_field_geometry'),
                    "merlict_plenoscope_propagator_config_path": opj(
                        out_absdir,
                        'input',
                        'merlict_propagation_config.json'),
                    "log_dir":
                        opj(site_particle_absdir, "log.map"),
                    "past_trigger_dir":
                        opj(site_particle_absdir, "past_trigger.map"),
                    "feature_dir":
                        opj(site_particle_absdir, "features.map"),
                    "keep_tmp": KEEP_TMP,
                    "tmp_dir": tmp_absdir,
                    "date": date_dict_now,
                }
                run_id += 1
                irf_jobs.append(irf_job)

    random.shuffle(irf_jobs)
    rc = pool.map(map_and_reduce.run_job, irf_jobs)
    sge._print("Reduce instrument-response.")

    for site_key in cfg["sites"]:
        site_absdir = opj(out_absdir, site_key)
        for particle_key in cfg["particles"]:
            site_particle_absdir = opj(site_absdir, particle_key)
            log_absdir = opj(site_particle_absdir, "log.map")
            feature_absdir = opj(site_particle_absdir, "features.map")

            # run-time
            # ========
            log_abspath = opj(site_particle_absdir, 'runtime.csv')
            if not op.exists(log_abspath) or not LAZY_REDUCTION:
                _lop_paths = glob.glob(opj(log_absdir, "*_runtime.jsonl"))
                logging.reduce(
                    list_of_log_paths=_lop_paths,
                    out_path=log_abspath)
            sge._print(
                "Reduce {:s} {:s} run-time.".format(site_key, particle_key))

            # event table
            # ===========
            event_table_abspath = opj(
                site_particle_absdir,
                'event_table.tar')
            if not op.exists(event_table_abspath) or not LAZY_REDUCTION:
                _feature_paths = glob.glob(
                    opj(feature_absdir, "*_event_table.tar"))
                table.reduce(
                    list_of_feature_paths=_feature_paths,
                    out_path=event_table_abspath)
            sge._print(
                "Reduce {:s} {:s} event_table.".format(site_key, particle_key))

            # grid images
            # ===========
            grid_abspath = opj(site_particle_absdir, 'grid.tar')
            if not op.exists(grid_abspath) or not LAZY_REDUCTION:
                _grid_paths = glob.glob(opj(feature_absdir, "*_grid.tar"))
                grid.reduce(
                    list_of_grid_paths=_grid_paths,
                    out_path=grid_abspath)
            sge._print(
                "Reduce {:s} {:s} grid.".format(site_key, particle_key))

    sge._print("End main().")
