import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json
from . import runtime
from .. import table
from .. import merlict
from .. import grid
from . import grid_direction
from . import grid_area
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


num_energy_bins = 10
EXAMPLE_ENERGY_BIN_EDGES = np.geomspace(0.5, 1000, num_energy_bins + 1)


def summarize(
    run_dir,
    out_dir,
    energy_bin_edges=EXAMPLE_ENERGY_BIN_EDGES,
    figure_config_16by9=figure.CONFIG_16_9,
    num_c_bins=None,
):
    fc16by9 = figure_config_16by9
    fc5by4 = fc16by9.copy()
    fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(opj(out_dir, 'cache'), exist_ok=True)

    inp = read_instrument_response_config(run_dir)
    config = inp['config']
    for site_key in config['sites']:
        for particle_key in config['particles']:
            prefix = '{:s}_{:s}'.format(site_key, particle_key)

            event_table = table.read(opj(
                run_dir,
                site_key,
                particle_key,
                'event_table.tar'))

            # runtime
            # -------
            runtime_table = runtime.read(opj(
                run_dir,
                site_key,
                particle_key,
                'runtime.csv'))
            '''
            extended_runtime_table = runtime.merge_event_table(
                runtime_table=runtime_table,
                event_table=event_table)

            runtime.write(
                path=opj(out_dir, prefix+'_runtime.csv'),
                table=extended_runtime_table,)

            runtime.write_relative_runtime(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_relative_runtime'),
                figure_config=fc16by9)

            runtime.write_speed(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_speed_runtime'),
                figure_config=fc16by9)

            # grid directions
            # ---------------
            com_pri_grd = table.merge(
                event_table=event_table,
                level_keys=['primary', 'grid'])

            c_bin_edges = grid_direction.guess_c_bin_edges(
                num_events=com_pri_grd['primary'].shape[0],
                num_c_bins=num_c_bins)
            icu, ecu, nev = grid_direction.histogram_grid_trigger(
                event_table_common_primary_grid=com_pri_grd,
                energy_bin_edges=energy_bin_edges,
                c_bin_edges=c_bin_edges)
            grid_direction.write_qube_of_figures(
                out_path=opj(out_dir, prefix+'_grid_direction'),
                intensity_cube=icu,
                exposure_cube=ecu,
                num_events_stack=nev,
                c_bin_edges=c_bin_edges,
                energy_bin_edges=energy_bin_edges,
                figure_config=fc5by4)

            c_bin_edges = grid_direction.guess_c_bin_edges(
                num_events=event_table['pasttrigger'].shape[0],
                num_c_bins=num_c_bins)
            icu, ecu, nev = grid_direction.histogram_plenoscope_trigger(
                event_table=event_table,
                energy_bin_edges=energy_bin_edges,
                c_bin_edges=c_bin_edges)
            grid_direction.write_qube_of_figures(
                out_path=opj(out_dir, prefix+'_pasttrigger_direction'),
                intensity_cube=icu,
                exposure_cube=ecu,
                num_events_stack=nev,
                c_bin_edges=c_bin_edges,
                energy_bin_edges=energy_bin_edges,
                figure_config=fc5by4)

            '''
            # grid areas pasttrigger
            # ----------------------

            mrg_table = table.merge(
                event_table=event_table,
                level_keys=[
                    'primary',
                    'grid',
                    'core',
                    'cherenkovsize',
                    'cherenkovpool',
                    'cherenkovsizepart',
                    'cherenkovpoolpart',
                    'trigger',
                    'pasttrigger'])

            _grid_pasttrigger_path = opj(
                out_dir,
                'cache',
                '{:s}_{:s}_grid_pasttrigger.tar'.format(
                    site_key,
                    particle_key))
            if os.path.exists(_grid_pasttrigger_path):
                grid_histograms_pasttrigger = grid.read_histograms(
                    path=_grid_pasttrigger_path,
                    indices=mrg_table['pasttrigger'])
            else:
                grid_histograms_pasttrigger = grid.read_histograms(
                    path=opj(run_dir, site_key, particle_key, 'grid.tar'),
                    indices=mrg_table['pasttrigger'])
                grid.write_histograms(
                    path=_grid_pasttrigger_path,
                    grid_histograms=grid_histograms_pasttrigger)


            c_bin_edges = grid_direction.guess_c_bin_edges(
                num_events=mrg_table['pasttrigger'].shape[0],
                num_c_bins=num_c_bins)

            adh = grid_area.histogram_area_direction(
                energy_bin_edges=energy_bin_edges,
                primary_table=mrg_table['primary'],
                grid_histograms=grid_histograms_pasttrigger,
                grid_geometry=inp['grid_geometry'],
                c_bin_edges_deg=c_bin_edges)

            print(adh)

            for energy_idx in range(len(energy_bin_edges) - 1):
                grid_area.write_area_direction(
                    path=opj(
                        out_dir,
                        '{:s}_{:s}_grid_pasttrigger_{:06d}'.format(
                            site_key,
                            particle_key,
                            energy_idx)),
                    grid_intensity=adh['grid_intensities'][energy_idx],
                    grid_xy_bin_edges=inp['grid_geometry']['xy_bin_edges'],
                    direction_intensity=adh['direction_intensities'][energy_idx],
                    c_bin_edges_deg=c_bin_edges,
                    num_airshower=adh['num_airshowers'][energy_idx],
                    energy_GeV_start=energy_bin_edges[energy_idx],
                    energy_GeV_stop=energy_bin_edges[energy_idx + 1],
                    figure_config=figure.CONFIG_16_9)


            # effective quantity
            # ------------------
            source_type = {
                'point': {'max_between_reconstruction_and_pointing_deg': 2.0},
                'diffuse': {'max_between_reconstruction_and_pointing_deg': 180.0},
            }


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
