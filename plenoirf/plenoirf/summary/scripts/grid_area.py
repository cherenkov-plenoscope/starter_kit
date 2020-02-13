#!/usr/bin/python
import sys

import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json

import plenoirf as irf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
run_dir = argv[1]
summary_dir = argv[2]

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        # read
        # ----
        event_table = irf.summary.read_event_table_cache(
            summary_dir=summary_dir,
            run_dir=run_dir,
            site_key=site_key,
            particle_key=particle_key)

        mrg_table = irf.table.merge(
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
            summary_dir,
            'cache',
            '{:s}_{:s}_grid_pasttrigger.tar'.format(
                site_key,
                particle_key))
        if os.path.exists(_grid_pasttrigger_path):
            grid_histograms_pasttrigger = irf.grid.read_histograms(
                path=_grid_pasttrigger_path)
        else:
            grid_histograms_pasttrigger = irf.grid.read_histograms(
                path=opj(run_dir, site_key, particle_key, 'grid.tar'),
                indices=mrg_table['pasttrigger'])
            irf.grid.write_histograms(
                path=_grid_pasttrigger_path,
                grid_histograms=grid_histograms_pasttrigger)

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
                    grid_histograms_pasttrigger[(
                        mat['run_id'],
                        mat['airshower_id'])])
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
