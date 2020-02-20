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
        "primary_cx",
        "primary_cy",
        "cherenkov_pool_x_m",
        "cherenkov_pool_y_m",
        "rel_uncertainty",
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
    max_energy,
    num_energy_supports,
    iteration_speed=0.9,
    max_iterations=25,
    power_slope=-1.7,
    corsika_primary_path=CORSIKA_PRIMARY_PATH,
):
    jobs = []
    for site_key in sites:
        for particle_key in particles:
            site = sites[site_key]
            particle_id = particles[particle_key]["particle_id"]
            max_off_axis_deg = .1*particles[particle_key]["max_scatter_angle_deg"]
            min_energy = np.min(particles[particle_key]["energy_bin_edges_GeV"])
            energy_supports = np.sort(cpw.random_distributions.draw_power_law(
                lower_limit=min_energy,
                upper_limit=max_energy,
                power_slope=power_slope,
                num_samples=num_energy_supports))
            for energy_idx in range(len(energy_supports)):
                job = {}
                job['site'] = site
                job['primary_energy'] = energy_supports[energy_idx]
                job['primary_particle_id'] = particle_id
                job['instrument_azimuth_deg'] = plenoscope_pointing['azimuth_deg']
                job['instrument_zenith_deg'] = plenoscope_pointing['zenith_deg']
                job['max_off_axis_deg'] = max_off_axis_deg
                job['primary_particle_id'] = particle_id
                job['corsika_primary_path'] = CORSIKA_PRIMARY_PATH
                job['site_key'] = site_key
                job['particle_key'] = particle_key
                job['iteration_speed'] = iteration_speed
                job['max_iterations'] = max_iterations
                jobs.append(job)
    return jobs


os.makedirs(work_dir, exist_ok=True)

jobs = make_jobs(
    sites=sites,
    particles=particles,
    max_energy=24,
    num_energy_supports=64)

print(len(jobs))
if os.path.exists('combined_results.pkl'):
    with open('combined_results.pkl', 'rb') as f:
        combined_results = pickle.loads(f.read())
else:
    random.shuffle(jobs)
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
def _estimate_num_additional_steps(
    primary_zenith_deg,
    max_off_axis_deg
):
    return (
        np.abs(np.gradient(primary_zenith_deg)) // max_off_axis_deg
    ).astype(np.int)


def _estimate_additional_energies(
    num_additional_steps,
    energy_bin_edges,
):
    additional_energies = []
    for idx in range(len(energy_bin_edges) - 1):
        energy_start = energy_bin_edges[idx]
        energy_stop = energy_bin_edges[idx + 1]

        additional_including_existing_bin_edges = np.geomspace(
            energy_start,
            energy_stop,
            2+num_additional_steps[idx])
        additional = additional_including_existing_bin_edges[1:-1]
        additional = additional.tolist()

        additional_energies += additional
    return additional_energies


def estimate_additional_energies(
    primary_zenith_deg,
    max_off_axis_deg,
    energy_bin_edges,
):
    num_additional_steps = _estimate_num_additional_steps(
        primary_zenith_deg=primary_zenith_deg,
        max_off_axis_deg=max_off_axis_deg)

    return _estimate_additional_energies(
        num_additional_steps=num_additional_steps,
        energy_bin_edges=energy_bin_edges)
"""
