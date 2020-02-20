import plenoirf
import magnetic_deflection as mdfl
import json
import os
import sun_grid_engine_map
import multiprocessing
import random
import pandas as pd
import shutil
import pickle
import corsika_primary_wrapper as cpw


work_dir = os.path.join('.', 'explore_deflection')

particles = plenoirf.EXAMPLE_CONFIG['particles']
plenoscope_pointing = plenoirf.EXAMPLE_CONFIG['plenoscope_pointing']
sites = plenoirf.EXAMPLE_CONFIG['sites']

CORSIKA_PRIMARY_PATH = os.path.abspath(os.path.join(
    "build",
    "corsika",
    "modified",
    "corsika-75600",
    "run",
    "corsika75600Linux_QGSII_urqmd"))

max_energy_GeV = 24.0

# num_parallel = 12
# pool = multiprocessing.Pool(num_parallel)
pool = sun_grid_engine_map


def power_space(start, stop, power_index, num, iterations=10000):
    assert num >= 2
    num_points_without_start_and_end = num - 2
    if num_points_without_start_and_end > 1:
        full = []
        for iti in range(iterations):
            points = np.sort(cpw.random_distributions.draw_power_law(
                lower_limit=start,
                upper_limit=stop,
                power_slope=power_index,
                num_samples=num_points_without_start_and_end))
            points = [start] + points.tolist() + [stop]
            full.append(points)
        full = np.array(full)
        return np.mean(full, axis=0)
    else:
        return np.array([start, stop])


def sort_combined_results(
    combined_results,
    particles,
    sites,
):
    df = pd.DataFrame(combined_results)

    KEEP_KEYS = [
        "particle_id",
        "energy_GeV",
        "primary_azimuth_deg",
        "primary_zenith_deg",
        "cherenkov_pool_x_m",
        "cherenkov_pool_y_m",
        "off_axis_deg",
        "num_valid_Cherenkov_pools",
        "num_thrown_Cherenkov_pools",
        "total_num_events",
    ]

    res = {}
    for site_key in sites:
        res[site_key] = {}
        for particle_key in particles:
            site_mask = (df['site_key'] == site_key).values
            particle_mask = (df['particle_key'] == particle_key).values
            mask = np.logical_and(site_mask, particle_mask)
            site_particle_df = df[mask]
            site_particle_df = site_particle_df[site_particle_df['valid']]
            site_particle_keep_df = site_particle_df[KEEP_KEYS]
            site_particle_rec = site_particle_keep_df.to_records(index=False)
            argsort = np.argsort(site_particle_rec['energy_GeV'])
            site_particle_rec = site_particle_rec[argsort]
            res[site_key][particle_key] = site_particle_rec
    return res


def write_recarray_to_csv(recarray, path):
    df = pd.DataFrame(recarray)
    csv = df.to_csv(index=False)
    with open(path+".tmp", 'wt') as f:
        f.write(csv)
    shutil.move(path+".tmp", path)


def make_jobs(
    sites,
    particles,
    plenoscope_pointing,
    max_energy,
    num_energy_supports,
    energy_supports_power_law_slope=-1.7,
    iteration_speed=0.9,
    initial_num_events_per_iteration=2**5,
    max_total_num_events=2**12,
    corsika_primary_path=CORSIKA_PRIMARY_PATH,
):
    jobs = []
    for site_key in sites:
        for particle_key in particles:
            site = sites[site_key]
            particle_id = particles[particle_key]["particle_id"]
            max_off_axis_deg = .1*particles[particle_key]["max_scatter_angle_deg"]
            min_energy = np.min(particles[particle_key]["energy_bin_edges_GeV"])
            energy_supports = power_space(
                start=min_energy,
                stop=max_energy,
                power_index=energy_supports_power_law_slope,
                num=num_energy_supports)
            for energy_idx in range(len(energy_supports)):
                job = {}
                job['site'] = site
                job['primary_energy'] = energy_supports[energy_idx]
                job['primary_particle_id'] = particle_id
                job['instrument_azimuth_deg'] = plenoscope_pointing['azimuth_deg']
                job['instrument_zenith_deg'] = plenoscope_pointing['zenith_deg']
                job['max_off_axis_deg'] = max_off_axis_deg
                job['corsika_primary_path'] = CORSIKA_PRIMARY_PATH
                job['site_key'] = site_key
                job['particle_key'] = particle_key
                job['iteration_speed'] = iteration_speed
                job['initial_num_events_per_iteration'] = (
                    initial_num_events_per_iteration)
                job['max_total_num_events'] = max_total_num_events
                jobs.append(job)
    return jobs


def sort_jobs_by_key(jobs, key):
    _values = [job[key] for job in jobs]
    _values_argsort = np.argsort(_values)
    jobs_sorted = [jobs[_values_argsort[i]] for i in range(len(jobs))]
    return jobs_sorted


os.makedirs(work_dir, exist_ok=True)

jobs = make_jobs(
    sites=sites,
    particles=particles,
    plenoscope_pointing=plenoscope_pointing,
    max_energy=24,
    num_energy_supports=256)

print(len(jobs))

jobs_sorted_energy = sort_jobs_by_key(jobs=jobs, key='primary_energy')

"""
# random.shuffle(jobs)
combined_results = pool.map(mdfl.run_job, jobs)

res = sort_combined_results(
    combined_results=combined_results,
    sites=sites,
    particles=particles)

for site_key in sites:
    for particle_key in particles:
        out_path = os.path.join(work_dir, '{:s}_{:s}.csv'.format(
            site_key, particle_key))
        write_recarray_to_csv(
            recarray=res[site_key][particle_key],
            path=out_path)
"""

