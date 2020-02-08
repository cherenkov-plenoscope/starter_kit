import os
from os.path import join as opj
import pandas as pd
import numpy as np
import json
import plenoscope_map_reduce as plmr
from plenoscope_map_reduce import instrument_response as irf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

run_dir = 'run-2020-02-04_1924'
site_key = 'namibia'
particle_key = 'electron'

inp = irf.summary.read_input(run_dir)

cfg = inp['config']

event_table = irf.table.read(opj(
    run_dir,
    site_key,
    particle_key,
    'event_table.tar'))


pasttrigger_indices = event_table['pasttrigger']

grid_hists = irf.grid.read_histograms(
    path=opj(
        run_dir,
        site_key,
        particle_key,
        'grid.tar'),
    indices=pasttrigger_indices)


num_energy_bins = 9
energy_bin_edges = np.geomspace(0.5, 1000, num_energy_bins + 1)

c_bin_edges_deg = np.linspace(-30, 30, 26)


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
        'pasttrigger',
    ])

primary_incidents = irf.query.primary_incident_vector(mrg_table['primary'])
primary_cx = primary_incidents[:, 0]
primary_cy = primary_incidents[:, 1]

for energy_idx in range(num_energy_bins):
    energy_GeV_start = energy_bin_edges[energy_idx]
    energy_GeV_stop = energy_bin_edges[energy_idx + 1]

    mask = np.logical_and(
        mrg_table['primary']['energy_GeV'] > energy_GeV_start,
        mrg_table['primary']['energy_GeV'] <= energy_GeV_stop)

    primary_matches = mrg_table['primary'][mask]

    num_bins_diameter = inp['grid_geometry']['num_bins_diameter']
    grid_intensity = np.zeros((num_bins_diameter, num_bins_diameter))
    num_airshower = 0
    for mat in primary_matches:
        grid_intensity += irf.grid.bytes_to_histogram(
            grid_hists[(mat['run_id'], mat['airshower_id'])])
        num_airshower += 1

    direction_intensity = np.histogram2d(
        np.rad2deg(primary_cx[mask]),
        np.rad2deg(primary_cy[mask]),
        bins=[c_bin_edges_deg, c_bin_edges_deg])[0]

    grid_intensity = grid_intensity
    grid_xy_bin_edges = inp['grid_geometry']['xy_bin_edges']
    direction_intensity = direction_intensity
    num_airshower = num_airshower
    c_bin_edges_deg = c_bin_edges_deg
    path = "{:s}_{:s}_grid_{:06d}".format(site_key, particle_key, energy_idx)
    figure_config = irf.summary.figure.CONFIG_16_9
    # ========================================================================
    fig = irf.summary.figure.figure(figure_config)

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
    irf.summary.figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=10)
    irf.summary.figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=20)
    irf.summary.figure.ax_add_circle(ax=ax_poin, x=0, y=0, r=30)
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



'''
num_trials = inp['grid_geometry']['num_bins_diameter']**2

scenarios = {
    'point': {'max_between_pointing_primary_deg': 2.0},
    'diffuse': {'max_between_pointing_primary_deg': 180.0},
}

for scenario in scenarios:
    inside_cone = irf.query.query_in_viewcone_cx_xy(
        cx_deg=0.0,
        cy_deg=0.0,
        cone_opening_angle_deg=scenarios[scenario]['max_between_pointing_primary_deg'],
        primary_table=event_table['primary'],)

    _event_table = event_table.copy()
    _event_table['primary'] = _event_table['primary'][inside_cone]

    mrg_pri_grd = irf.table.merge(
        event_table=event_table,
        level_keys=['primary', 'grid'])

    if scenario == 'point':
        max_scatter = mrg_pri_grd['grid']['area_thrown_m2']
        quantity_label = 'area / (m$^{2}$)'
    else:
        max_scatter = (
            mrg_pri_grd['grid']['area_thrown_m2']*
            mrg_pri_grd['primary']['solid_angle_thrown_sr'])
        quantity_label = 'acceptance / (m$^{2}$ sr)'

    qa_grid = irf.summary.effective.estimate_effective_quantity(
        energy_bin_edges = energy_bin_edges,
        energies = mrg_pri_grd['primary']['energy_GeV'],
        max_scatter_quantities = max_scatter,
        thrown_mask=num_trials*np.ones(mrg_pri_grd['primary'].shape[0]),
        detection_mask = mrg_pri_grd['grid']['num_bins_above_threshold'])

    irf.summary.effective.write_effective_quantity_figure(
        effective_quantity=qa_grid,
        quantity_label='grid-'+quantity_label,
        path='{:s}_grid'.format(scenario))

    pasttrigger_mask = irf.table.make_mask_of_right_in_left(
        left_level=mrg_pri_grd['primary'],
        right_level=event_table['pasttrigger'])

    qa_trigger = irf.summary.effective.estimate_effective_quantity(
        energy_bin_edges = energy_bin_edges,
        energies = mrg_pri_grd['primary']['energy_GeV'],
        max_scatter_quantities = max_scatter,
        thrown_mask=num_trials*np.ones(mrg_pri_grd['primary'].shape[0]),
        detection_mask = pasttrigger_mask*mrg_pri_grd['grid']['num_bins_above_threshold'])

    irf.summary.effective.write_effective_quantity_figure(
        effective_quantity=qa_trigger,
        quantity_label='trigger-'+quantity_label,
        path='{:s}_trigger'.format(scenario))
'''
