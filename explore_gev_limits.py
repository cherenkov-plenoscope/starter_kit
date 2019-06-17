import numpy as np
import tempfile
import os
import gev_limits as gli
import sun_grid_engine_map as sge



MERLICT_PATH = os.path.abspath('./build/merlict/merlict-eventio-converter')
assert os.path.exists(MERLICT_PATH)

out_dir = os.path.join('./__gev_limits_gamma')
out_dir = os.path.abspath(out_dir)
map_and_reduce_dir = os.path.join(out_dir, 'map_and_reduce')
os.makedirs(map_and_reduce_dir)

portal_instrument = {
    'aperture_radius': 35.5,
    'num_paxel_on_diagonal': 8,
    'field_of_view_radius_deg': 3.25,
    'num_pixel_on_diagonal': int(np.round(6.5/0.0667)),
    'time_radius': 25e-9,
    'num_time_slices': 100,
    'mirror_reflectivity': 0.8,
    'photo_detection_efficiency': 0.25,
}

gamma_particle = {
    'prmpar': 1,
    'E_start': 0.8,
    'E_stop': 1.6,
    'max_theta_deg': 4.25,
}

gamsberg_site = {
    'atmosphere': 10,
    'observation_level_altitude_asl': 2347.0,
    'earth_magnetic_field_x_muT': 12.5,
    'earth_magnetic_field_z_muT': -25.9,
}

jobs = gli.map_and_reduce.make_jobs(
    map_and_reduce_dir=map_and_reduce_dir,
    random_seed=1,
    num_runs=2232,
    num_events_in_run=2560,
    eventio_converter_path=MERLICT_PATH,
    instrument=portal_instrument,
    particle=gamma_particle,
    site=gamsberg_site,
    trigger_threshold=50)

for job in jobs:
    gli.map_and_reduce.run_job(job)

all_runs_path = os.path.join(out_dir, 'run.lut')
gli.lookup.concatenate([
    jobs[0]['out_path'],
    jobs[1]['out_path']
    ], all_runs_path)

lut = gli.lookup.LookUpTable(all_runs_path)

thrown = gli.thrown_structure.read_events_thrown(
    os.path.join(all_runs_path, 'thrown.float32'))
