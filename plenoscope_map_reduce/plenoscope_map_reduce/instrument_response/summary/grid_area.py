import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

import numpy as np

from .. import grid
from .. import query
from .. import table
from . import figure


def histogram_area_direction(
    energy_bin_edges,
    primary_table,
    grid_histograms,
    grid_geometry,
    c_bin_edges_deg,
):
    primary_incidents = query.primary_incident_vector(primary_table)
    primary_cx = primary_incidents[:, 0]
    primary_cy = primary_incidents[:, 1]
    num_bins_diameter = grid_geometry['num_bins_diameter']
    grid_intensities = []
    direction_intensities = []
    num_airshowers = []
    for energy_idx in range(len(energy_bin_edges) - 1):
        energy_GeV_start = energy_bin_edges[energy_idx]
        energy_GeV_stop = energy_bin_edges[energy_idx + 1]
        mask = np.logical_and(
            primary_table['energy_GeV'] > energy_GeV_start,
            primary_table['energy_GeV'] <= energy_GeV_stop)
        primary_matches = primary_table[mask]
        grid_intensity = np.zeros((num_bins_diameter, num_bins_diameter))
        num_airshower = 0
        for mat in primary_matches:
            grid_intensity += grid.bytes_to_histogram(
                grid_histograms[(mat['run_id'], mat['airshower_id'])])
            num_airshower += 1
        direction_intensity = np.histogram2d(
            np.rad2deg(primary_cx[mask]),
            np.rad2deg(primary_cy[mask]),
            bins=[c_bin_edges_deg, c_bin_edges_deg])[0]
        grid_intensities.append(grid_intensity)
        direction_intensities.append(direction_intensity)
        num_airshowers.append(num_airshower)
    return {
        'grid_intensities': np.array(grid_intensities),
        'direction_intensities': np.array(direction_intensities),
        'num_airshowers': np.array(num_airshowers)}


def write_area_direction(
    path,
    grid_intensity,
    grid_xy_bin_edges,
    direction_intensity,
    c_bin_edges_deg,
    num_airshower,
    energy_GeV_start,
    energy_GeV_stop,
    figure_config=figure.CONFIG_16_9
):
    fig = figure.figure(figure_config)

    ch = 0.6
    cw = 0.31
    ax_grid = fig.add_axes([0.08, 0.3, cw, ch])
    ax_grid_cb = fig.add_axes([0.4, 0.3, 0.02, ch])
    ax_grid.spines['top'].set_color('none')
    ax_grid.spines['right'].set_color('none')
    ax_grid.set_aspect('equal')
    _pcm_grid = ax_grid.pcolormesh(
        np.array(grid_xy_bin_edges)*1e-3,
        np.array(grid_xy_bin_edges)*1e-3,
        np.transpose(grid_intensity/num_airshower),
        norm=plt_colors.PowerNorm(gamma=0.5),
        cmap='Blues')
    plt.colorbar(_pcm_grid, cax=ax_grid_cb, extend='max')
    ax_grid.set_xlabel('x/km')
    ax_grid.set_ylabel('y/km')
    ax_grid.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax_grid.set_title('Cherenkov-photons/airshower$^{-1}$')

    ax_poin = fig.add_axes([0.58, 0.3, cw, ch])
    ax_poin_cb = fig.add_axes([0.9, 0.3, 0.02, ch])
    ax_poin.spines['top'].set_color('none')
    ax_poin.spines['right'].set_color('none')
    ax_poin.set_aspect('equal')
    _pcm_poin = ax_poin.pcolormesh(
        c_bin_edges_deg,
        c_bin_edges_deg,
        np.transpose(direction_intensity),
        norm=plt_colors.PowerNorm(gamma=0.5),
        cmap='Blues')
    plt.colorbar(_pcm_poin, cax=ax_poin_cb, extend='max')
    ax_poin.set_xlabel('cx/deg')
    ax_poin.set_ylabel('cy/deg')
    figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=10)
    figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=20)
    figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=30)
    ax_poin.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax_poin.set_title('primary-direction')

    ax_slid = fig.add_axes([0.4, 0.1, 0.5, 0.1])
    ax_slid.spines['top'].set_color('none')
    ax_slid.spines['right'].set_color('none')
    ax_slid.spines['left'].set_color('none')
    ax_slid.set_xlabel('energy/GeV')
    ax_slid.set_xlim([1e-1, 1e3])
    ax_slid.semilogx()
    ax_slid.set_yticklabels([])
    ax_slid.set_yticks([])
    ax_slid.plot([energy_GeV_start, energy_GeV_start], [0, 1], 'k-')
    ax_slid.plot([energy_GeV_stop, energy_GeV_stop], [0, 1], 'k-')
    ax_slid.plot([energy_GeV_start, energy_GeV_stop], [1, 1], 'k-')

    ax_text = fig.add_axes([0.1, 0.1, 0.1, 0.1])
    ax_text.set_axis_off()
    ax_text.text(0, 0, 'num. airshower: {:d}'.format(num_airshower))

    plt.savefig(path+'.'+figure_config['format'])
    plt.close(fig)
