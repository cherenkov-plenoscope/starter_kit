#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os
import json
import magnetic_deflection as mdfl

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 2
run_dir = argv[1]
summary_dir = os.path.join(run_dir, 'summary')

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)

trigger_threshold = irf_config['config']['sum_trigger']['patch_threshold']

xy_bin_edges = np.linspace(
    np.min(irf_config['grid_geometry']['xy_bin_edges']),
    np.max(irf_config['grid_geometry']['xy_bin_edges']),
    64)

for site_key in irf_config['config']['sites']:

    tables = {}
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)
        tables[particle_key] = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray(
                {spt.IDX: event_table['trigger'][spt.IDX]}),
            level_keys=['primary', 'grid', 'core', 'trigger'])

    for particle_key in irf_config['config']['particles']:

        passed_trigger = (
            tables[particle_key]['trigger']['response_pe'] >=
            trigger_threshold
        )

        hist_thrown = np.histogram2d(
            tables[particle_key]['core']['core_x_m'],
            tables[particle_key]['core']['core_y_m'],
            bins=(
                xy_bin_edges,
                xy_bin_edges)
        )[0]

        hist_detected = np.histogram2d(
            tables[particle_key]['core']['core_x_m'][passed_trigger],
            tables[particle_key]['core']['core_y_m'][passed_trigger],
            bins=(
                xy_bin_edges,
                xy_bin_edges)
        )[0]

        hist = np.zeros(shape=hist_thrown.shape)
        exposure_mask = hist_thrown > 100
        hist[exposure_mask] = (
            hist_detected[exposure_mask]/
            hist_thrown[exposure_mask]
        )

        fc16by9 = sum_config['figure_16_9']
        fc5by4 = fc16by9.copy()
        fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)
        fig = irf.summary.figure.figure(fc5by4)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax_cb = fig.add_axes([0.85, 0.1, 0.02, 0.8])
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_aspect('equal')
        _pcm_grid = ax.pcolormesh(
            xy_bin_edges*1e-3,
            xy_bin_edges*1e-3,
            np.transpose(hist),
            norm=plt_colors.PowerNorm(gamma=0.5),
            cmap='Blues')
        plt.colorbar(_pcm_grid, cax=ax_cb, extend='max')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('x/m')
        ax.set_ylabel('y/m')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(summary_dir, '{:s}_{:s}_core_positions.png'.format(
                site_key,
                particle_key))
        )
        plt.close(fig)
