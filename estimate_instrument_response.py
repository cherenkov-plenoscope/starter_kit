#! /usr/bin/env python
"""
Estimate the instrument-response for the Cherenkov-plenoscope.

Usage: trigger_sensitivity [--out_dir=DIR]

Options:
    -h --help               Prints this help message.
    -o --out_dir=DIR        Output directory [default: ./run]
"""
import docopt
import sun_grid_engine_map as sge
import plenoscope_map_reduce as plmr
import os
import numpy as np
from os import path as op
import shutil
import subprocess
import random
import plenopy as pl
import json
import multiprocessing


def absjoin(*args):
    return op.abspath(op.join(*args))


MULTIPROCESSING_POOL = ""
if MULTIPROCESSING_POOL == "sun_grid_engine":
    pool = sge
else:
    pool = multiprocessing.Pool(8)


CORSIKA_PRIMARY_PATH = absjoin(
    "build",
    "corsika",
    "modified",
    "corsika-75600",
    "run",
    "corsika75600Linux_QGSII_urqmd")

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = absjoin(
    "build",
    "merlict",
    "merlict-plenoscope-propagation")

MERLICT_PLENOSCOPE_CALIBRATION_MAP_PATH = absjoin(
    "build",
    "merlict",
    "merlict-plenoscope-calibration-map")

MERLICT_PLENOSCOPE_CALIBRATION_REDUCE_PATH = absjoin(
    "build",
    "merlict",
    "merlict-plenoscope-calibration-reduce")

PLENOSCOPE_SCENERY_PATH = absjoin(
    "resources",
    "acp",
    "71m",
    "scenery")

MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH = absjoin(
    "resources",
    "acp",
    "merlict_propagation_config.json")

SITES = {
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
}

PARTICLES = {
    "gamma": {
        "particle_id": 1,
        "energy_bin_edges_GeV": [0.5, 10, 100],
        "max_zenith_deg_vs_energy": [5, 5, 5],
        "max_depth_g_per_cm2_vs_energy": [0, 0, 0],
        "energy_power_law_slope": -1.5,
    },
    "electron": {
        "particle_id": 3,
        "energy_bin_edges_GeV": [1, 10, 100],
        "max_zenith_deg_vs_energy": [30, 30, 30],
        "max_depth_g_per_cm2_vs_energy": [0, 0, 0],
        "energy_power_law_slope": -1.5,
    },
    "proton": {
        "particle_id": 14,
        "energy_bin_edges_GeV": [5, 10, 100],
        "max_zenith_deg_vs_energy": [30, 30, 30],
        "max_depth_g_per_cm2_vs_energy": [0, 0, 0],
        "energy_power_law_slope": -1.5,
    },
}

GRID = {
    "num_bins_radius": 512,
    "threshold_num_photons": 50
}

SUM_TRIGGER = {
    "patch_threshold": 103,
    "integration_time_in_slices": 10,
    "min_num_neighbors": 3,
    "object_distances": [10e3, 15e3, 20e3],
}

NUM_RUNS = {
    "gamma": 3,
    "electron": 3,
    "proton": 3
}
NUM_AIRSHOWERS_PER_RUN = 100

if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        out_dir = op.abspath(arguments['--out_dir'])
    except docopt.DocoptExit as e:
        print(e)

    date_dict_now = plmr.instrument_response.date_dict_now()
    print("-------start---------")

    print("-------light-field-geometry---------")
    os.makedirs(out_dir, exist_ok=True)
    lfg_path = absjoin(out_dir, 'light_field_geometry')
    if not op.exists(lfg_path):
        lfg_tmp_dir = lfg_path+".tmp"
        os.makedirs(lfg_tmp_dir)
        lfg_jobs = plmr.make_jobs_light_field_geometry(
            merlict_map_path=MERLICT_PLENOSCOPE_CALIBRATION_MAP_PATH,
            scenery_path=PLENOSCOPE_SCENERY_PATH,
            out_dir=lfg_tmp_dir,
            num_photons_per_block=1000*1000,
            num_blocks=1337,
            random_seed=0)
        rc = pool.map(plmr.run_job_light_field_geometry, lfg_jobs)
        subprocess.call([
            MERLICT_PLENOSCOPE_CALIBRATION_REDUCE_PATH,
            '--input', lfg_tmp_dir,
            '--output', lfg_path])
        shutil.rmtree(lfg_tmp_dir)

    print("-------instrument-response---------")
    irf_jobs = []
    run_id = 1
    for site_key in SITES:
        site_dir = op.join(out_dir, site_key)
        if op.exists(site_dir):
            continue
        os.makedirs(site_dir, exist_ok=True)

        for particle_key in PARTICLES:
            site_particle_dir = op.join(site_dir, particle_key)
            if op.exists(site_particle_dir):
                continue
            os.makedirs(site_particle_dir, exist_ok=True)
            for job_idx in np.arange(NUM_RUNS[particle_key]):

                irf_job = {
                    "run_id": run_id,
                    "num_air_showers": NUM_AIRSHOWERS_PER_RUN,
                    "particle": PARTICLES[particle_key],
                    "site": SITES[site_key],
                    "grid": GRID,
                    "sum_trigger": SUM_TRIGGER,
                    "corsika_primary_path": CORSIKA_PRIMARY_PATH,
                    "plenoscope_scenery_path": PLENOSCOPE_SCENERY_PATH,
                    "merlict_plenoscope_propagator_path":
                        MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
                    "light_field_geometry_path": lfg_path,
                    "merlict_plenoscope_propagator_config_path":
                        MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH,
                    "log_dir": op.join(site_particle_dir, "log"),
                    "past_trigger_dir":
                        op.join(site_particle_dir, "past_trigger"),
                    "feature_dir": op.join(site_particle_dir, "features"),
                    "non_temp_work_dir": None,
                    "date": date_dict_now,
                }
                run_id += 1
                irf_jobs.append(irf_job)

    random.shuffle(irf_jobs)
    rc = pool.map(plmr.instrument_response.run_job, irf_jobs)
    print("-------instrument-response---------")
