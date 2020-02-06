import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json
from . import runtime
from .. import table
from .. import merlict
from .. import grid
from . import grid_trigger_vs_direction
from . import figure


num_energy_bins = 10
EXAMPLE_ENERGY_BIN_EDGES = np.geomspace(0.5, 1000, num_energy_bins + 1)


def summarize(
    run_dir,
    out_dir,
    energy_bin_edges=EXAMPLE_ENERGY_BIN_EDGES,
    figure_config=figure.CONFIG,
    num_c_bins=None,
):
    with open(opj(run_dir, 'input', 'config.json'), 'rt') as f:
        config = json.loads(f.read())
    os.makedirs(out_dir, exist_ok=True)

    light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        opj(run_dir, 'input', 'scenery', 'scenery.json'))

    grid_geometry = grid.init(
        plenoscope_diameter=2*light_field_sensor_geometry[
            'expected_imaging_system_aperture_radius'],
        num_bins_radius=config['grid']['num_bins_radius'])

    for site_key in config['sites']:
        for particle_key in config['particles']:
            prefix = '{:s}_{:s}'.format(site_key, particle_key)

            event_table = table.read(opj(
                run_dir,
                site_key,
                particle_key,
                'event_table.tar'))
            runtime_table = runtime.read(opj(
                run_dir,
                site_key,
                particle_key,
                'runtime.csv'))

            extended_runtime_table = runtime.merge_event_table(
                runtime_table=runtime_table,
                event_table=event_table)

            runtime.write(
                path=opj(out_dir, prefix+'_runtime.csv'),
                table=extended_runtime_table,)

            runtime.write_relative_runtime(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_relative_runtime'),
                figure_config=figure_config)

            runtime.write_speed(
                table=extended_runtime_table,
                out_path=opj(out_dir, prefix+'_speed_runtime'),
                figure_config=figure_config)

            com_pri_grd = table.merge(
                event_table=event_table,
                level_keys=['primary', 'grid'])

            if num_c_bins is None:
                num_c_bins = int(0.05*np.sqrt(com_pri_grd['primary'].shape[0]))
                num_c_bins = np.max([np.min([num_c_bins, 129]), 17])

            grid_trigger_vs_direction.write(
                event_table_common_primary_grid=com_pri_grd,
                grid_geometry=grid_geometry,
                energy_bin_edges=energy_bin_edges,
                max_zenith_deg=40,
                out_path=opj(out_dir, prefix+'_grid_direction'),
                figure_config=figure_config,
                num_c_bins=num_c_bins)

            com_pri_grd_pat = table.merge(
                event_table=event_table,
                level_keys=['primary', 'grid', 'pasttrigger'])

            if num_c_bins is None:
                num_c_bins = int(
                    0.05*np.sqrt(com_pri_grd_pat['primary'].shape[0]))
                num_c_bins = np.max([np.min([num_c_bins, 129]), 17])

            grid_trigger_vs_direction.write(
                event_table_common_primary_grid=com_pri_grd_pat,
                grid_geometry=grid_geometry,
                energy_bin_edges=energy_bin_edges,
                max_zenith_deg=40,
                out_path=opj(out_dir, prefix+'_pasttrigger_direction'),
                figure_config=figure_config,
                num_c_bins=num_c_bins)


    return config, event_table, grid_geometry
