import os
import random
import magnetic_deflection as md
import sun_grid_engine_map as sge
import acp_instrument_response_function as irf

particle_types = ['gamma', 'electron', 'proton']

location_steerong_card = irf.utils.read_json(
    'resources/acp/71m/chile_paranal.json')

reduce_dir = os.path.abspath('run/magnetic_deflection/reduce')
merlict_path=os.path.abspath('build/merlict/merlict-magnetic-field-explorer')

jobs = []
for particle_type in particle_types:
    jobs += md.map_and_reduce.make_jobs(
        particle_type=particle_type,
        max_zenith_scatter_angle_deg=15.,
        observation_level_altitude_asl=
            location_steerong_card['observation_level_altitude_asl'],
        earth_magnetic_field_x_muT=
            location_steerong_card['earth_magnetic_field_x_muT'],
        earth_magnetic_field_z_muT=
            location_steerong_card['earth_magnetic_field_z_muT'],
        atmosphere_model=location_steerong_card['atmosphere_model'],
        E_start=0.25,
        E_stop=10.,
        out_dir=os.path.join(reduce_dir, particle_type),
        merlict_path=merlict_path,
        num_runs=100,
        num_events_in_run=100)

random.shuffle(jobs)
rc = sge.map(irf.trigger_simulation.run_job, jobs)

for particle_type in particle_types:
    md.map_and_reduce.reduce_output(
        in_dir=os.path.join(reduce_dir, particle_type),
        out_path=os.path.join(reduce_dir, particle_type+".float32"))

