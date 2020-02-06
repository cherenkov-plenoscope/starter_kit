import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pandas as pd
import numpy as np
import os
import json

from .. import grid
from . import figure


def add_hatches(ax, ix, iy, x_bin_edges, y_bin_edges, alpha=0.1):
    x0 = x_bin_edges[ix]
    x1 = x_bin_edges[ix + 1]
    y0 = y_bin_edges[iy]
    y1 = y_bin_edges[iy + 1]
    ax.plot([x0, x1], [y0, y1], '-k', alpha=alpha)


def write(
    event_table_common_primary_grid,
    grid_geometry,
    energy_bin_edges,
    max_zenith_deg,
    out_path,
    figure_config,
    num_c_bins=26,
):
    cpg = event_table_common_primary_grid

    NUM_TRIALS_ON_GRID = 1024*1024
    AREA_GRID_M2 = (72**2)*NUM_TRIALS_ON_GRID

    cxs = np.cos(cpg['primary']['azimuth_rad'])*cpg['primary']['zenith_rad']
    cys = np.sin(cpg['primary']['azimuth_rad'])*cpg['primary']['zenith_rad']
    cpg_incident_vectors = grid._make_bunch_direction(cxs, cys)

    ghis = []
    exposure_masks = []
    nums_events = []
    for eidx in range(len(energy_bin_edges) - 1):
        e_mask = np.logical_and(
            cpg['primary']['energy_GeV'] >= energy_bin_edges[eidx],
            cpg['primary']['energy_GeV'] < energy_bin_edges[eidx + 1])
        num_events = np.sum(e_mask)
        nums_events.append(num_events)

        c_bins = np.linspace(-max_zenith_deg, max_zenith_deg, num_c_bins+1)
        his = np.histogram2d(
            np.rad2deg(cpg_incident_vectors[e_mask, 0]),
            np.rad2deg(cpg_incident_vectors[e_mask, 1]),
            bins=[c_bins, c_bins],
            weights=cpg['grid']['num_bins_above_threshold'][e_mask])[0]

        exposure = np.histogram2d(
            np.rad2deg(cpg_incident_vectors[e_mask, 0]),
            np.rad2deg(cpg_incident_vectors[e_mask, 1]),
            bins=[c_bins, c_bins])[0]
        his[exposure > 0] = his[exposure > 0]/exposure[exposure > 0]
        exposure_masks.append(exposure > 0)
        ghis.append(his)
    ghis = np.array(ghis)
    exposure_masks = np.array(exposure_masks)

    max_num_bins_above_threshold = np.max(ghis)
    for eidx in range(len(energy_bin_edges) - 1):
        fig = figure.figure(figure_config)
        ax_size = [0.1, 0.15, 0.8, 0.75]
        ax = fig.add_axes(ax_size)
        ax_cb = fig.add_axes([0.8, 0.15, 0.02, 0.75])
        ax.set_title(
            '{: 1.1f} to {: 1.1f} GeV, {:1.1e} events'.format(
                energy_bin_edges[eidx],
                energy_bin_edges[eidx + 1],
                float(nums_events[eidx])))
        _pcm = ax.pcolormesh(
            c_bins,
            c_bins,
            np.transpose(ghis[eidx, :, :]),
            norm=colors.PowerNorm(gamma=0.5),
            cmap='Blues',
            vmax=max_num_bins_above_threshold)
        plt.colorbar(_pcm, cax=ax_cb, extend='max')
        ax.set_xlabel('primary cx/deg')
        ax.set_ylabel('primary cy/deg')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.33)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        for ix in range(num_c_bins):
            for iy in range(num_c_bins):
                if not exposure_masks[eidx][ix][iy]:
                    add_hatches(
                        ax=ax,
                        ix=ix,
                        iy=iy,
                        x_bin_edges=c_bins,
                        y_bin_edges=c_bins)

        ax.set_aspect('equal')
        fig.savefig(
            out_path+"_{:06d}.{:s}".format(eidx, figure_config['format']))
        plt.close(fig)
