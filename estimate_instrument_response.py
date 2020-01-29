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

executables = {
    "corsika_primary_abspath": absjoin(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd"),

    "merlict_plenoscope_propagator_abspath": absjoin(
        "build",
        "merlict",
        "merlict-plenoscope-propagation"),

    "merlict_plenoscope_calibration_map_abspath": absjoin(
        "build",
        "merlict",
        "merlict-plenoscope-calibration-map"),

    "merlict_plenoscope_calibration_reduce_abspath": absjoin(
        "build",
        "merlict",
        "merlict-plenoscope-calibration-reduce"),
}

cfg_files = {
    "merlict_plenoscope_propagator_config_abspath": absjoin(
        "resources",
        "acp",
        "merlict_propagation_config.json"),

    "plenoscope_scenery_abspath": absjoin(
        "resources",
        "acp",
        "71m",
        "scenery"),
}

cfg = {
    "plenoscope_scenery_relpath": op.join(
        "input",
        "scenery"),

    "merlict_plenoscope_propagator_config_relpath": op.join(
        "input",
        "merlict_propagation_config.json"),

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
            "energy_power_law_slope": -1.5,
        },
        "electron": {
            "particle_id": 3,
            "energy_bin_edges_GeV": [1, 100],
            "max_scatter_angle_deg": 30,
            "energy_power_law_slope": -1.5,
        },
        "proton": {
            "particle_id": 14,
            "energy_bin_edges_GeV": [5, 100],
            "max_scatter_angle_deg": 30,
            "energy_power_law_slope": -1.5,
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
        "gamma": 3,
        "electron": 3,
        "proton": 3
    },

    "num_airshowers_per_run": 100,
}


if __name__ == '__main__':
    try:
        arguments = docopt.docopt(__doc__)
        out_absdir = op.abspath(arguments['--out_dir'])
    except docopt.DocoptExit as e:
        print(e)

    date_dict_now = plmr.instrument_response.date_dict_now()
    sge._print("Start main()")
    os.makedirs(out_absdir, exist_ok=True)

    sge._print("Copy resources")
    # ==========================
    input_absdir = op.join(out_absdir, 'input')
    os.makedirs(input_absdir, exist_ok=True)

    for cfg_file in cfg_files:
        cfg_file_abspath = cfg_files[cfg_file]
        local_cfg_file_abspath = op.join(
            input_absdir,
            op.basename(cfg_file_abspath))
        local_cfg_file_relpath = op.join(
            'input',
            op.basename(cfg_file_abspath))
        if not op.exists(local_cfg_file_abspath):
            plmr.instrument_response.safe_copy(
                src=cfg_file_abspath,
                dst=local_cfg_file_abspath)
        cfg_file_rel = cfg_file.replace('abspath', 'relpath')
        cfg[cfg_file_rel] = local_cfg_file_relpath

    cfg['executables'] = executables

    with open(op.join(input_absdir, "config.json"), "wt") as fout:
        fout.write(json.dumps(cfg, indent=4))

    sge._print("Estimating light-field-geometry.")
    # ============================================
    lfg_abspath = op.join(out_absdir, 'light_field_geometry')
    if not op.exists(lfg_abspath):
        lfg_tmp_absdir = lfg_abspath+".tmp"
        os.makedirs(lfg_tmp_absdir)
        lfg_jobs = plmr.make_jobs_light_field_geometry(
            merlict_map_path=executables[
                "merlict_plenoscope_calibration_map_abspath"],
            scenery_path=op.join(
                out_absdir,
                cfg["plenoscope_scenery_relpath"]),
            out_dir=lfg_tmp_absdir,
            num_photons_per_block=cfg[
                'light_field_geometry']['num_photons_per_block'],
            num_blocks=cfg[
                'light_field_geometry']['num_blocks'],
            random_seed=0)
        rc = pool.map(plmr.run_job_light_field_geometry, lfg_jobs)
        subprocess.call([
            executables["merlict_plenoscope_calibration_reduce_abspath"],
            '--input', lfg_tmp_absdir,
            '--output', lfg_abspath])
        shutil.rmtree(lfg_tmp_absdir)

    sge._print("Estimating instrument-response.")
    # ===========================================
    irf_jobs = []
    run_id = 1
    for site_key in cfg["sites"]:
        site_absdir = op.join(out_absdir, site_key)
        if op.exists(site_absdir):
            continue
        os.makedirs(site_absdir, exist_ok=True)

        for particle_key in cfg["particles"]:
            site_particle_absdir = op.join(site_absdir, particle_key)
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
                        "corsika_primary_abspath"],
                    "plenoscope_scenery_path": op.join(
                        out_absdir,
                        cfg["plenoscope_scenery_relpath"]),
                    "merlict_plenoscope_propagator_path": executables[
                        "merlict_plenoscope_propagator_abspath"],
                    "light_field_geometry_path": lfg_abspath,
                    "merlict_plenoscope_propagator_config_path": op.join(
                        out_absdir,
                        cfg["merlict_plenoscope_propagator_config_relpath"]),
                    "log_dir": op.join(site_particle_absdir, "log"),
                    "past_trigger_dir":
                        op.join(site_particle_absdir, "past_trigger"),
                    "feature_dir": op.join(site_particle_absdir, "features"),
                    "non_temp_work_dir": None,
                    "date": date_dict_now,
                }
                run_id += 1
                irf_jobs.append(irf_job)

    rc = pool.map(plmr.instrument_response.run_job, irf_jobs)
    sge._print("End main().")
