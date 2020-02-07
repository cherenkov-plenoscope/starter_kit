import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

import pandas as pd
import numpy as np
import os
import json

from .. import grid
from .. import query
from .. import table
from . import figure


def guess_c_bin_edges(
    num_events,
    num_c_bins,
    min_max_c_deg=40,
):
    if num_c_bins is None:
        _num_c_bins = int(0.05*np.sqrt(num_events))
        _num_c_bins = np.max([np.min([_num_c_bins, 129]), 17])
    else:
        _num_c_bins = num_c_bins
    return np.linspace(-min_max_c_deg, min_max_c_deg, _num_c_bins + 1)


def add_hatches(ax, ix, iy, x_bin_edges, y_bin_edges, alpha=0.1):
    x0 = x_bin_edges[ix]
    x1 = x_bin_edges[ix + 1]
    y0 = y_bin_edges[iy]
    y1 = y_bin_edges[iy + 1]
    ax.plot([x0, x1], [y0, y1], '-k', alpha=alpha)


def write_qube_of_figures(
    out_path,
    intensity_cube,
    exposure_cube,
    num_events_stack,
    c_bin_edges,
    energy_bin_edges,
    figure_config,
    cmap='Blues'
):
    num_c_bins = len(c_bin_edges) - 1
    max_intensity = np.max(intensity_cube)
    for eidx in range(len(energy_bin_edges) - 1):
        fig = figure.figure(figure_config)
        ax_size = [0.1, 0.15, 0.8, 0.75]
        ax = fig.add_axes(ax_size)
        ax_cb = fig.add_axes([0.8, 0.15, 0.02, 0.75])
        ax.set_title(
            '{: 1.1f} to {: 1.1f} GeV, {:1.1e} events'.format(
                energy_bin_edges[eidx],
                energy_bin_edges[eidx + 1],
                float(num_events_stack[eidx])))
        _pcm = ax.pcolormesh(
            c_bin_edges,
            c_bin_edges,
            np.transpose(intensity_cube[eidx, :, :]),
            norm=plt_colors.PowerNorm(gamma=0.5),
            cmap=cmap,
            vmax=max_intensity)
        plt.colorbar(_pcm, cax=ax_cb, extend='max')
        ax.set_xlabel('primary cx/deg')
        ax.set_ylabel('primary cy/deg')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.33)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')

        for ix in range(num_c_bins):
            for iy in range(num_c_bins):
                if not exposure_cube[eidx][ix][iy]:
                    add_hatches(
                        ax=ax,
                        ix=ix,
                        iy=iy,
                        x_bin_edges=c_bin_edges,
                        y_bin_edges=c_bin_edges)

        ax.set_aspect('equal')
        fig.savefig(
            out_path+"_{:06d}.{:s}".format(eidx, figure_config['format']))
        plt.close(fig)


def histogram_grid_trigger(
    event_table_common_primary_grid,
    energy_bin_edges,
    c_bin_edges,
):
    cpg = event_table_common_primary_grid
    cpg_incident_vectors = query.primary_incident_vector(cpg['primary'])
    intensity_cube = []
    exposure_cube = []
    num_events_stack = []
    for eidx in range(len(energy_bin_edges) - 1):
        e_mask = np.logical_and(
            cpg['primary']['energy_GeV'] >= energy_bin_edges[eidx],
            cpg['primary']['energy_GeV'] < energy_bin_edges[eidx + 1])
        num_events = np.sum(e_mask)
        num_events_stack.append(num_events)

        his = np.histogram2d(
            np.rad2deg(cpg_incident_vectors[e_mask, 0]),
            np.rad2deg(cpg_incident_vectors[e_mask, 1]),
            bins=[c_bin_edges, c_bin_edges],
            weights=cpg['grid']['num_bins_above_threshold'][e_mask])[0]

        exposure = np.histogram2d(
            np.rad2deg(cpg_incident_vectors[e_mask, 0]),
            np.rad2deg(cpg_incident_vectors[e_mask, 1]),
            bins=[c_bin_edges, c_bin_edges])[0]
        his[exposure > 0] = his[exposure > 0]/exposure[exposure > 0]
        exposure_cube.append(exposure > 0)
        intensity_cube.append(his)
    intensity_cube = np.array(intensity_cube)
    exposure_cube = np.array(exposure_cube)
    num_events_stack = np.array(num_events_stack)
    return intensity_cube, exposure_cube, num_events_stack


def histogram_plenoscope_trigger(
    event_table,
    energy_bin_edges,
    c_bin_edges,
):
    pasttrigger_mask = table.make_mask_of_right_in_left(
            left_level=event_table['primary'],
            right_level=event_table['pasttrigger'])
    incident_vectors = query.primary_incident_vector(event_table['primary'])
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
            np.rad2deg(incident_vectors[e_mask, 0]),
            np.rad2deg(incident_vectors[e_mask, 1]),
            weights=pasttrigger_mask[e_mask],
            bins=[c_bin_edges, c_bin_edges])[0]

        exposure = np.histogram2d(
            np.rad2deg(incident_vectors[e_mask, 0]),
            np.rad2deg(incident_vectors[e_mask, 1]),
            bins=[c_bin_edges, c_bin_edges])[0]

        his[exposure > 0] = his[exposure > 0]/exposure[exposure > 0]
        exposure_cube.append(exposure > 0)
        intensity_cube.append(his)
    intensity_cube = np.array(intensity_cube)
    exposure_cube = np.array(exposure_cube)
    num_events_stack = np.array(num_events_stack)
    return intensity_cube, exposure_cube, num_events_stack
