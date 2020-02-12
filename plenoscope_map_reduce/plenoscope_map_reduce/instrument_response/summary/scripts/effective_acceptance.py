#!/usr/bin/python
import sys
from plenoscope_map_reduce import instrument_response as irf
import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
run_dir = argv[1]
summary_dir = argv[2]

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)


def reconstruct_direction_cx_cy(event_features):
    return (
        event_features['image_infinity_cx_mean'],
        event_features['image_infinity_cy_mean'])


for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        energy_bin_edges = np.array(sum_config['energy_bin_edges_GeV'])

        event_table = irf.summary.read_event_table_cache(
            summary_dir=summary_dir,
            run_dir=run_dir,
            site_key=site_key,
            particle_key=particle_key)

        num_trials = irf_config['grid_geometry']['num_bins_diameter']**2

        scenarios = {
            'point': {'max_between_pointing_primary_deg': 2.5},
            'diffuse': {'max_between_pointing_primary_deg': 180.0},
        }

        mrg_pri_grd = irf.table.merge(
            event_table=event_table,
            level_keys=['primary', 'grid'])

        rec_cx, rec_cy = reconstruct_direction_cx_cy(
            event_features=event_table['features'])
        c_radial = np.hypot(rec_cx, rec_cy)
        pasttrigger_onregion_mask = c_radial <= np.deg2rad(
            scenarios['point']['max_between_pointing_primary_deg'])
        pasttrigger_onregion_indices = irf.table.mask_to_indices(
                level=event_table['features'],
                mask=pasttrigger_onregion_mask)

        for scenario in scenarios:
            primary_inside_cone_mask = irf.query.query_in_viewcone_cx_xy(
                cx_deg=0.0,
                cy_deg=0.0,
                cone_opening_angle_deg=scenarios[scenario][
                    'max_between_pointing_primary_deg'],
                primary_table=mrg_pri_grd['primary'],)
            primary_inside_cone_indices = irf.table.mask_to_indices(
                level=mrg_pri_grd['primary'],
                mask=primary_inside_cone_mask)
            mrg_pri_grd_con = {
                'primary': irf.table.by_indices(
                    event_table=mrg_pri_grd,
                    level_key='primary',
                    indices=primary_inside_cone_indices),
                'grid': irf.table.by_indices(
                    event_table=mrg_pri_grd,
                    level_key='grid',
                    indices=primary_inside_cone_indices),}

            if scenario == 'point':
                max_scatter = mrg_pri_grd_con['grid']['area_thrown_m2']
                quantity_label = 'area / (m$^{2}$)'
                y_start=1e2
                y_stop=1e7
            else:
                y_start=1e0
                y_stop=1e5
                max_scatter = (
                    mrg_pri_grd_con['grid']['area_thrown_m2']*
                    mrg_pri_grd_con['primary']['solid_angle_thrown_sr'])
                quantity_label = 'acceptance / (m$^{2}$ sr)'

            qa_grid = irf.summary.effective.estimate_effective_quantity(
                energy_bin_edges = energy_bin_edges,
                energies = mrg_pri_grd_con['primary']['energy_GeV'],
                max_scatter_quantities = max_scatter,
                thrown_mask=np.ones(mrg_pri_grd_con['primary'].shape[0]),
                thrown_weights=num_trials*np.ones(mrg_pri_grd_con['primary'].shape[0]),
                detection_mask = np.ones(mrg_pri_grd_con['primary'].shape[0]),
                detection_weights = mrg_pri_grd_con['grid']['num_bins_above_threshold'])

            irf.summary.effective.write_effective_quantity_figure(
                y_start=y_start,
                y_stop=y_stop,
                effective_quantity=qa_grid,
                quantity_label='grid-'+quantity_label,
                path=opj(summary_dir, '{:s}_{:s}_grid'.format(
                    prefix_str, scenario)))

            pasttrigger_mask = irf.table.make_mask_of_right_in_left(
                left_level=mrg_pri_grd_con['primary'],
                right_level=pasttrigger_onregion_indices)

            qa_trigger = irf.summary.effective.estimate_effective_quantity(
                energy_bin_edges = energy_bin_edges,
                energies = mrg_pri_grd_con['primary']['energy_GeV'],
                max_scatter_quantities = max_scatter,
                thrown_mask=np.ones(mrg_pri_grd_con['primary'].shape[0]),
                thrown_weights=num_trials*np.ones(mrg_pri_grd_con['primary'].shape[0]),
                detection_mask = pasttrigger_mask,
                detection_weights = mrg_pri_grd_con['grid']['num_bins_above_threshold'])

            irf.summary.effective.write_effective_quantity_figure(
                y_start=y_start,
                y_stop=y_stop,
                effective_quantity=qa_trigger,
                quantity_label='trigger-'+quantity_label,
                path=opj(summary_dir, '{:s}_{:s}_trigger'.format(
                    prefix_str, scenario)))
