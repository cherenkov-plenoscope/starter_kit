#!/usr/bin/python
import sys
import json
import os
import shutil
import multiprocessing
import magnetic_deflection as mdfl
import sun_grid_engine_map
import plenoirf

argv = plenoirf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
work_dir = argv[1]

if shutil.which('qsub'):
    pool = sun_grid_engine_map
    print("use sun-grid-engine")
else:
    pool = multiprocessing.Pool(8)
    print("use multiprocessing on localhost")

particles = {
    "gamma": {
        "particle_id": 1,
        "energy_bin_edges_GeV": [0.5, 100],
        "max_scatter_angle_deg": 5,
        "energy_power_law_slope": -1.7,
        "electric_charge_qe": 0.,
        "magnetic_deflection_max_off_axis_deg": 0.25,
    },
    "electron": {
        "particle_id": 3,
        "energy_bin_edges_GeV": [0.5, 100],
        "max_scatter_angle_deg": 10,
        "energy_power_law_slope": -1.7,
        "electric_charge_qe": -1.,
        "magnetic_deflection_max_off_axis_deg": 0.5,
    },
    "proton": {
        "particle_id": 14,
        "energy_bin_edges_GeV": [5, 100],
        "max_scatter_angle_deg": 30,
        "energy_power_law_slope": -1.7,
        "electric_charge_qe": +1.,
        "magnetic_deflection_max_off_axis_deg": 1.5,
    },
    "helium": {
        "particle_id": 402,
        "energy_bin_edges_GeV": [10, 100],
        "max_scatter_angle_deg": 30,
        "energy_power_law_slope": -1.7,
        "electric_charge_qe": +2.,
        "magnetic_deflection_max_off_axis_deg": 1.5,
    },
}

plenoscope_pointing = {
    "azimuth_deg": 0.,
    "zenith_deg": 0.
}

sites = {
    "namibia": {
        "observation_level_asl_m": 2300,
        "earth_magnetic_field_x_muT": 12.5,
        "earth_magnetic_field_z_muT": -25.9,
        "atmosphere_id": 10,
        "geomagnetic_cutoff_rigidity_GV": 12.5,
    },
    "namibiaOff": {
        "observation_level_asl_m": 2300,
        "earth_magnetic_field_x_muT": 1e-6,
        "earth_magnetic_field_z_muT": 1e-6,
        "atmosphere_id": 10,
        "geomagnetic_cutoff_rigidity_GV": 0.,
    },
    "chile": {
        "observation_level_asl_m": 5000,
        "earth_magnetic_field_x_muT": 20.815,
        "earth_magnetic_field_z_muT": -11.366,
        "atmosphere_id": 26,
        "geomagnetic_cutoff_rigidity_GV": 10.0,
    },
    "lapalma": {
        "observation_level_asl_m": 2200,
        "earth_magnetic_field_x_muT": 30.419,
        "earth_magnetic_field_z_muT": 23.856,
        "atmosphere_id": 8,
        "geomagnetic_cutoff_rigidity_GV": 12.5,
    },
}

mdfl.A_init_work_dir(
    particles=particles,
    sites=sites,
    plenoscope_pointing=plenoscope_pointing,
    max_energy=64.0,
    num_energy_supports=512,
    work_dir=work_dir)

jobs = mdfl.B_make_jobs_from_work_dir(work_dir=work_dir)

job_results = pool.map(mdfl.map_and_reduce.run_job, jobs)

mdfl.C_reduce_job_results_in_work_dir(
    job_results=job_results,
    work_dir=work_dir)

mdfl.D_summarize_raw_deflection(out_dir=work_dir)

