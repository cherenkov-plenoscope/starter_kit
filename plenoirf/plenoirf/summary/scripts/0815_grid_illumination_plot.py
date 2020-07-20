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
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

trigger_threshold = sum_config['trigger']['threshold_pe']
trigger_modus = sum_config["trigger"]["modus"]
energy_bin_edges_GeV = energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins_coarse'] + 1
)
num_grid_bins_on_edge = irf_config['grid_geometry']['num_bins_diameter']

fc16by9 = sum_config['plot']['16_by_9']
fc5by4 = fc16by9.copy()
fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

os.makedirs(pa['out_dir'], exist_ok=True)

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        # read
        # ----
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'
            ),
            structure=irf.table.STRUCTURE
        )

        idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=event_table['trigger'],
            threshold=trigger_threshold,
            modus=trigger_modus,
        )

        detected_events = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_detected,
            level_keys=[
                'primary',
                'grid',
                'core',
                'cherenkovsize',
                'cherenkovpool',
                'cherenkovsizepart',
                'cherenkovpoolpart',
                'trigger',
            ]
        )

        detected_grid_histograms = irf.grid.read_histograms(
            path=opj(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'grid.tar'
            ),
            indices=idx_detected
        )

        # summarize
        # ---------
        grid_intensities = []
        num_airshowers = []
        for energy_idx in range(len(energy_bin_edges_GeV) - 1):
            energy_GeV_start = energy_bin_edges_GeV[energy_idx]
            energy_GeV_stop = energy_bin_edges_GeV[energy_idx + 1]
            energy_mask = np.logical_and(
                detected_events['primary']['energy_GeV'] > energy_GeV_start,
                detected_events['primary']['energy_GeV'] <= energy_GeV_stop
            )
            idx_energy_range = detected_events['primary'][energy_mask][spt.IDX]
            grid_intensity = np.zeros((
                num_grid_bins_on_edge,
                num_grid_bins_on_edge)
            )
            num_airshower = 0
            for idx in idx_energy_range:
                grid_intensity += irf.grid.bytes_to_histogram(
                    detected_grid_histograms[idx]
                )
                num_airshower += 1
            grid_intensities.append(grid_intensity)
            num_airshowers.append(num_airshower)

        grid_intensities = np.array(grid_intensities)
        num_airshowers = np.array(num_airshowers)

        # write
        # -----
        for energy_idx in range(len(energy_bin_edges_GeV) - 1):
            grid_intensity = grid_intensities[energy_idx]
            num_airshower = num_airshowers[energy_idx]

            vmin = None
            vmax = None
            normalized_grid_intensity = grid_intensity
            if num_airshower > 0:
                normalized_grid_intensity /= num_airshower
            else:
                vmin = 0
                vmax = 1
            fig = irf.summary.figure.figure(fc5by4)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax_cb = fig.add_axes([0.85, 0.1, 0.02, 0.8])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_aspect('equal')
            _pcm_grid = ax.pcolormesh(
                np.array(irf_config['grid_geometry']['xy_bin_edges'])*1e-3,
                np.array(irf_config['grid_geometry']['xy_bin_edges'])*1e-3,
                np.transpose(normalized_grid_intensity),
                norm=plt_colors.PowerNorm(gamma=0.5),
                cmap='Blues',
                vmin=vmin,
                vmax=vmax
            )
            plt.colorbar(_pcm_grid, cax=ax_cb, extend='max')
            ax.set_title(
                'num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f}GeV'.format(
                    num_airshower,
                    energy_bin_edges_GeV[energy_idx],
                    energy_bin_edges_GeV[energy_idx + 1]
                ),
                family='monospace'
            )
            ax.set_xlabel('x/km')
            ax.set_ylabel('y/km')
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)

            plt.savefig(
                opj(
                    pa['out_dir'],
                    '{:s}_{:s}_{:06d}.{:s}'.format(
                        prefix_str,
                        'grid_area_pasttrigger',
                        energy_idx,
                        fc5by4['format'])))
            plt.close(fig)
