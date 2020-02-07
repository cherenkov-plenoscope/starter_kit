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
from . import figure


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
    fc4by3 = fc16by9.copy()
    fc4by3['cols'] = fc16by9['cols']*(9/16)*(4/3)
    fc5by4 = fc16by9.copy()
    fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

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

            # runtime
            # -------
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

            # effective quantity
            # ------------------
            source_type = {
                'point': {'max_between_reconstruction_and_pointing_deg': 2.0},
                'diffuse': {'max_between_reconstruction_and_pointing_deg': 180.0},
            }


    return config, event_table, grid_geometry
