import numpy as np
import tempfile
import os
import gev_limits as gli
import sun_grid_engine_map as sge
import shutil

out_dir = os.path.abspath('.')

gamma_particle = {
    'prmpar': 1,
    'E_start': 0.8,
    'E_stop': 2.4,
    'E_slope': 0.0,
    'max_theta_deg': 4.25,
    'min_theta_deg': 0.,
    'min_phi_deg': 0.,
    'max_phi_deg': 360.,
    'XSCAT_m': 300,
    'YSCAT_m': 0,
}

electron_particle = {
    'prmpar': 3,
    'E_start': 0.8,
    'E_stop': 100,
    'E_slope': -1.0,
    'max_theta_deg': 4.25,
    'min_theta_deg': 0.,
    'min_phi_deg': 0.,
    'max_phi_deg': 360.,
    'XSCAT_m': 1100,
    'YSCAT_m': 0,
}

proton_particle = {
    'prmpar': 14,
    'E_start': 5,
    'E_stop': 100,
    'E_slope': -1.0,
    'max_theta_deg': 6.5,
    'min_theta_deg': 0.,
    'min_phi_deg': 0.,
    'max_phi_deg': 360.,
    'XSCAT_m': 600,
    'YSCAT_m': 0,
}

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

TRIGGER_THRESHOLD = 50

gamsberg_site = {
    'atmosphere': 10,
    'observation_level_altitude_asl': 2347.0,
    'earth_magnetic_field_x_muT': 12.5,
    'earth_magnetic_field_z_muT': -25.9,
}

paranal_site = {
    'atmosphere': 26,
    'observation_level_altitude_asl': 5000.0,
    'earth_magnetic_field_x_muT': 20.815,
    'earth_magnetic_field_z_muT': -11.366,
}

NUM_RUNS = 2
NUM_EVENTS_IN_RUN = 2560

MERLICT_PATH = os.path.abspath('./build/merlict/merlict-eventio-converter')
assert os.path.exists(MERLICT_PATH)

particles = {
    "gamma": gamma_particle,
    "electron": electron_particle,
    "proton": proton_particle
}

sites = {
    "paranal": paranal_site,
    "gamsberg": gamsberg_site
}

for site in sites:
    for particle in particles:
        site_dir = os.path.join(out_dir ,'__gev_limits_{:s}'.format(site))
        map_and_reduce_dir = os.path.join(
            site_dir, 'map_and_reduce_{:s}'.format(particle))
        os.makedirs(map_and_reduce_dir)

        __site = sites[site].copy()
        if particle in ['electron', 'proton']:
            __site['earth_magnetic_field_x_muT'] = 1e-9
            __site['earth_magnetic_field_z_muT'] = 1e-9

        jobs = gli.map_and_reduce.make_jobs(
            map_and_reduce_dir=map_and_reduce_dir,
            random_seed=1,
            num_runs=NUM_RUNS,
            num_events_in_run=NUM_EVENTS_IN_RUN,
            eventio_converter_path=MERLICT_PATH,
            instrument=portal_instrument,
            particle=particles[particle],
            site=__site,
            trigger_threshold=TRIGGER_THRESHOLD)

        for job in jobs:
            gli.map_and_reduce.run_job(job)
        #sge.map(gli.map_and_reduce.run_job, jobs)

        lut_path = os.path.join(site_dir, '{:s}.lut'.format(particle))

        lut_paths = []
        for job in jobs:
            lut_paths.append(job['out_path'])
        gli.lookup.concatenate(lut_paths, lut_path)

        lut = gli.lookup.LookUpTable(lut_path)
        shutil.rmtree(map_and_reduce_dir)

        thrown = gli.thrown_structure.read_events_thrown(
            os.path.join(lut_path, 'thrown.float32'))
