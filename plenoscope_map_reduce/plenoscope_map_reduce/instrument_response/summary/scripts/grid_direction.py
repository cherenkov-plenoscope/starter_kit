#!/usr/bin/python
import sys

import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json

from plenoscope_map_reduce import instrument_response as irf

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

        # summarize
        # ---------
        energy_bin_edges = sum_config['energy_bin_edges_GeV_coarse']
        c_bin_edges_deg = sum_config['c_bin_edges_deg']

        pasttrigger_mask = irf.table.make_mask_of_right_in_left(
                left_level=event_table['primary'],
                right_level=event_table['pasttrigger'])
        _in_vec = irf.query.primary_incident_vector(event_table['primary'])
        primary_cx = _in_vec[:, 0]
        primary_cy = _in_vec[:, 1]

        intensity_cube = []
        exposure_cube = []
        num_events_stack = []
        for eidx in range(len(energy_bin_edges) - 1):
            e_mask = np.logical_and(
                event_table['primary']['energy_GeV'] >= energy_bin_edges[eidx],
                event_table['primary']['energy_GeV'] < energy_bin_edges[eidx + 1])

            num_events = np.sum(pasttrigger_mask[e_mask])
            num_events_stack.append(num_events)

            his = np.histogram2d(
                np.rad2deg(primary_cx[e_mask]),
                np.rad2deg(primary_cy[e_mask]),
                weights=pasttrigger_mask[e_mask],
                bins=[c_bin_edges_deg, c_bin_edges_deg])[0]

            exposure = np.histogram2d(
                np.rad2deg(primary_cx[e_mask]),
                np.rad2deg(primary_cy[e_mask]),
                bins=[c_bin_edges_deg, c_bin_edges_deg])[0]

            his[exposure > 0] = his[exposure > 0]/exposure[exposure > 0]
            exposure_cube.append(exposure > 0)
            intensity_cube.append(his)
        intensity_cube = np.array(intensity_cube)
        exposure_cube = np.array(exposure_cube)
        num_events_stack = np.array(num_events_stack)

        # write
        # -----
        fc16by9 = sum_config['figure_16_9']
        fc5by4 = fc16by9.copy()
        fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)
        vmax = np.max(intensity_cube)

        for energy_idx in range(len(energy_bin_edges) - 1):

            fig = irf.summary.figure.figure(fc5by4)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax_cb = fig.add_axes([0.85, 0.1, 0.02, 0.8])
            ax.spines['top'].set_color('none')
            ax.spines['right'].set_color('none')
            ax.set_aspect('equal')
            _pcm_grid = ax.pcolormesh(
                c_bin_edges_deg,
                c_bin_edges_deg,
                np.transpose(intensity_cube[energy_idx, :, :]),
                norm=plt_colors.PowerNorm(gamma=0.5),
                cmap='Blues',
                vmin=0.,
                vmax=vmax)
            ax.set_xlim([np.min(c_bin_edges_deg), np.max(c_bin_edges_deg)])
            ax.set_ylim([np.min(c_bin_edges_deg), np.max(c_bin_edges_deg)])
            plt.colorbar(_pcm_grid, cax=ax_cb, extend='max')
            ax.set_xlabel('primary cx/deg')
            ax.set_ylabel('primary cy/deg')
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.set_title(
                'num. airshower {:d}, energy {:.1f} - {:.1f}GeV'.format(
                num_events_stack[energy_idx],
                energy_bin_edges[energy_idx],
                energy_bin_edges[energy_idx + 1]))
            for rr in [10, 20, 30, 40, 50]:
                irf.summary.figure.ax_add_circle(
                    ax=ax,
                    x=0,
                    y=0,
                    r=rr,
                    color='k',
                    linewidth=0.66,
                    linestyle='-',
                    alpha=0.1)

            num_c_bins = len(c_bin_edges_deg) - 1
            for ix in range(num_c_bins):
                for iy in range(num_c_bins):
                    if not exposure_cube[energy_idx][ix][iy]:
                        irf.summary.figure.ax_add_hatches(
                            ax=ax,
                            ix=ix,
                            iy=iy,
                            x_bin_edges=c_bin_edges_deg,
                            y_bin_edges=c_bin_edges_deg)
            plt.savefig(
                opj(
                    summary_dir,
                    '{:s}_{:s}_{:06d}.{:s}'.format(
                        prefix_str,
                        'grid_direction_pasttrigger',
                        energy_idx,
                        fc5by4['format'])))
            plt.close(fig)
