import plenoirf
import magnetic_deflection as mdfl
import os
import sun_grid_engine_map
import multiprocessing


work_dir = os.path.join('.', 'explore_deflection')

particles = plenoirf.EXAMPLE_CONFIG['particles']
plenoscope_pointing = plenoirf.EXAMPLE_CONFIG['plenoscope_pointing']
sites = plenoirf.EXAMPLE_CONFIG['sites']

max_energy_GeV = 32.0

# pool = multiprocessing.Pool(8)
pool = sun_grid_engine_map

if not os.path.exists(work_dir):
    jobs = mdfl.map_and_reduce.make_jobs(
        sites=sites,
        particles=particles,
        plenoscope_pointing=plenoscope_pointing,
        max_energy=max_energy_GeV,
        num_energy_supports=256)
    print(len(jobs))
    jobs_sorted_energy = mdfl.map_and_reduce.sort_jobs_by_key(
        jobs=jobs,
        key='primary_energy')
    combined_results = pool.map(
        mdfl.map_and_reduce.run_job,
        jobs_sorted_energy)
    deflection_table = mdfl.map_and_reduce.structure_combined_results(
        combined_results=combined_results,
        sites=sites,
        particles=particles)
    mdfl.map_and_reduce.write_deflection_table(deflection_table, work_dir)

deflection_table = mdfl.map_and_reduce.read_deflection_table(work_dir)
