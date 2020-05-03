#!/usr/bin/python
import sys

import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json

import sparse_table as spt
import plenoirf as irf

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

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        # read
        # ----
        event_table = spt.read(
            path=os.path.join(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)

        mrg_table = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray({
                spt.IDX: event_table['pasttrigger'][spt.IDX]}),
            level_keys=[
                'primary',
                'grid',
                'core',
                'cherenkovsize',
                'cherenkovpool',
                'cherenkovsizepart',
                'cherenkovpoolpart',
                'trigger',
                'pasttrigger'
            ]
        )

        grid_histograms_pasttrigger = irf.grid.read_histograms(
            path=opj(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'grid.tar'),
            indices=mrg_table['pasttrigger'][spt.IDX])

        # summarize
        # ---------
        energy_bin_edges_GeV = sum_config['energy_bin_edges_GeV_coarse']
        c_bin_edges_deg = sum_config['c_bin_edges_deg']

        _primary_incidents = irf.query.primary_incident_vector(
            mrg_table['primary'])
        primary_cx = _primary_incidents[:, 0]
        primary_cy = _primary_incidents[:, 1]

        grid_intensities = []
        num_airshowers = []
        for energy_idx in range(len(energy_bin_edges_GeV) - 1):
            energy_GeV_start = energy_bin_edges_GeV[energy_idx]
            energy_GeV_stop = energy_bin_edges_GeV[energy_idx + 1]
            mask = np.logical_and(
                mrg_table['primary']['energy_GeV'] > energy_GeV_start,
                mrg_table['primary']['energy_GeV'] <= energy_GeV_stop)
            primary_matches = mrg_table['primary'][mask]
            grid_intensity = np.zeros((
                irf_config['grid_geometry']['num_bins_diameter'],
                irf_config['grid_geometry']['num_bins_diameter']))
            num_airshower = 0
            for mat in primary_matches:
                grid_intensity += irf.grid.bytes_to_histogram(
                    grid_histograms_pasttrigger[mat[spt.IDX]])
                num_airshower += 1
            grid_intensities.append(grid_intensity)
            num_airshowers.append(num_airshower)

        grid_intensities = np.array(grid_intensities)
        num_airshowers = np.array(num_airshowers)

        # write
        # -----
        fc16by9 = sum_config['figure_16_9']
        fc5by4 = fc16by9.copy()
        fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

        for energy_idx in range(len(energy_bin_edges_GeV) - 1):
            grid_intensity = grid_intensities[energy_idx]
            num_airshower = num_airshowers[energy_idx]

            fig = irf.summary.figure.figure(fc5by4)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax_cb = fig.add_axes([0.85, 0.1, 0.02, 0.8])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_aspect('equal')
            _pcm_grid = ax.pcolormesh(
                np.array(irf_config['grid_geometry']['xy_bin_edges'])*1e-3,
                np.array(irf_config['grid_geometry']['xy_bin_edges'])*1e-3,
                np.transpose(grid_intensity/num_airshower),
                norm=plt_colors.PowerNorm(gamma=0.5),
                cmap='Blues')
            plt.colorbar(_pcm_grid, cax=ax_cb, extend='max')
            ax.set_title(
                'num. airshower {:d}, energy {:.1f} - {:.1f}GeV'.format(
                    num_airshower,
                    energy_bin_edges_GeV[energy_idx],
                    energy_bin_edges_GeV[energy_idx + 1]))
            ax.set_xlabel('x/km')
            ax.set_ylabel('y/km')
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)

            plt.savefig(
                opj(
                    summary_dir,
                    '{:s}_{:s}_{:06d}.{:s}'.format(
                        prefix_str,
                        'grid_area_pasttrigger',
                        energy_idx,
                        fc5by4['format'])))
            plt.close(fig)
