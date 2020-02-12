import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json
from .. import table
from .. import merlict
from .. import grid
from . import figure
from . import effective


def argv_since_py(sys_argv):
    argv = []
    for arg in sys_argv:
        if len(argv) > 0:
            argv.append(arg)
        if '.py' in arg:
            argv.append(arg)
    return argv


def read_event_table_cache(summary_dir, run_dir, site_key, particle_key):
    cache_path = opj(summary_dir, 'cache', '{:s}_{:s}_event_table.tar'.format(
        site_key, particle_key))
    if not os.path.exists(cache_path):
        event_table = table.read(opj(
            run_dir,
            site_key,
            particle_key,
            'event_table.tar'))
        os.makedirs(opj(summary_dir, 'cache'), exist_ok=True)
        table.write_bin(path=cache_path, event_table=event_table)
    else:
        event_table = table.read_bin(cache_path)
    return event_table


def read_summary_config(summary_dir):
    with open(opj(summary_dir, 'summary_config.json'), 'rt') as fin:
        config = json.loads(fin.read())
    return config


def init(
    run_dir,
    summary_dir,
    figure_config_16by9=figure.CONFIG_16_9,
):
    os.makedirs(summary_dir, exist_ok=True)
    irf_cfg = read_instrument_response_config(run_dir=run_dir)

    # guess energy_bin_edges
    particles = irf_cfg['config']['particles']
    min_energies = []
    max_energies = []
    for particle_key in particles:
        e_bins = particles[particle_key]['energy_bin_edges_GeV']
        min_energies.append(np.min(e_bins))
        max_energies.append(np.max(e_bins))
    min_energy = np.min(min_energies)
    max_energy = np.max(max_energies)

    num_events_past_trigger = 10*1000
    sites = irf_cfg['config']['sites']
    for site_key in sites:
        for particle_key in particles:
            event_table = read_event_table_cache(
                summary_dir=summary_dir,
                run_dir=run_dir,
                site_key=site_key,
                particle_key=particle_key)
            if event_table['pasttrigger'].shape[0] < num_events_past_trigger:
                num_events_past_trigger = event_table['pasttrigger'].shape[0]

    num_energy_bins = int(np.sqrt(num_events_past_trigger)//2)
    num_energy_bins = 2*(num_energy_bins//2)
    num_energy_bins = np.max([np.min([num_energy_bins, 2**6]), 2**2])
    energy_bin_edges = np.geomspace(min_energy, max_energy, num_energy_bins+1)

    cfg = {}
    cfg['energy_bin_edges_GeV'] = list(energy_bin_edges)
    cfg['energy_bin_edges_GeV_coarse'] = list(energy_bin_edges[::2])
    cfg['c_bin_edges_deg'] = list(grid_direction.guess_c_bin_edges(
        num_events=num_events_past_trigger))
    cfg['figure_16_9'] = figure_config_16by9
    with open(opj(summary_dir, 'summary_config.json'), 'wt') as fout:
        fout.write(json.dumps(cfg, indent=4))


def read_instrument_response_config(run_dir):
    with open(opj(run_dir, 'input', 'config.json'), 'rt') as f:
        config = json.loads(f.read())
    light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        opj(run_dir, 'input', 'scenery', 'scenery.json'))
    grid_geometry = grid.init(
        plenoscope_diameter=2*light_field_sensor_geometry[
            'expected_imaging_system_aperture_radius'],
        num_bins_radius=config['grid']['num_bins_radius'])
    with open(opj(run_dir, 'input', 'scenery', 'scenery.json'), 'rt') as f:
        plenoscope_scenery = json.loads(f.read())
    _prop_cfg_path = opj(run_dir, 'input','merlict_propagation_config.json')
    with open(_prop_cfg_path, 'rt') as f:
        merlict_propagation_config = json.loads(f.read())
    bundle = {
        'config': config,
        'light_field_sensor_geometry': light_field_sensor_geometry,
        'plenoscope_scenery': plenoscope_scenery,
        'grid_geometry': grid_geometry,
        'merlict_propagation_config': merlict_propagation_config,
    }
    return bundle
