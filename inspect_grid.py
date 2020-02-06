import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import pandas as pd
import numpy as np
import os
from os import path as op
import json
import multiprocessing
from plenoscope_map_reduce import instrument_response as irf



run_dir = 'run-2020-02-04_0955'
out_dir = 'walk'
os.makedirs(out_dir, exist_ok=True)

with open(op.join(run_dir, 'input', 'config.json')) as fin:
    cfg = json.loads(fin.read())

_scenery = irf.merlict.read_plenoscope_geometry(
    op.join(run_dir, 'input', 'scenery', "scenery.json"))

grid_geometry = irf.grid.init(
    plenoscope_diameter=2*_scenery['expected_imaging_system_aperture_radius'],
    num_bins_radius=cfg['grid']['num_bins_radius'])

site = 'namibia'
particle = 'electron'
sp_dir = op.join(run_dir, site, particle)

grid_histograms = irf.grid.read_histograms(op.join(sp_dir, 'grid.tar'))
evttab = irf.table.read(op.join(sp_dir, 'event_table.tar'))



def ax_add_circle(ax, x, y, r, linetyle='k-', num_steps=1000):
    phi = np.linspace(0, 2*np.pi, num_steps)
    xs = x + r*np.cos(phi)
    ys = y + r*np.sin(phi)
    ax.plot(xs, ys, linetyle)


def ax_add_slider(ax, start, stop, values, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    ax.set_xticklabels([])
    ax.set_xticks([])
    ax.spines['bottom'].set_visible(False)
    ax.set_xlabel(label)
    for value in values:
        ax.plot(
            [0, 1],
            [value, value],
            "k",
            linewidth=5)


def write_histogram_figure(
    out_path,
    grid_intensity,
    num_airshower,
    xy_bin_edges,
    view
):
    scale = 0.6
    fig = plt.figure(figsize=(16*scale, 9*scale), dpi=120/scale)

    ax_hist = fig.add_axes([0.05, 0.1, 0.5, 0.8])
    ax_hist_cb = fig.add_axes([0.55, 0.1, 0.02, 0.8])
    ax_hist.spines['top'].set_color('none')
    ax_hist.spines['right'].set_color('none')
    ax_hist.set_aspect('equal')
    _pcm = ax_hist.pcolormesh(
        np.array(xy_bin_edges)*1e-3,
        np.array(xy_bin_edges)*1e-3,
        grid_intensity/num_airshower,
        norm=colors.PowerNorm(gamma=0.5),
        cmap='Blues')
    plt.colorbar(_pcm, cax=ax_hist_cb, extend='max')
    ax_hist.set_xlabel('x/km')
    ax_hist.set_ylabel('y/km')
    ax_hist.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)

    ax_poiting = fig.add_axes([0.65, 0.6, 0.3, 0.3])
    ax_poiting.set_aspect('equal')
    ax_poiting.spines['top'].set_color('none')
    ax_poiting.spines['right'].set_color('none')
    ax_poiting.spines['bottom'].set_color('none')
    ax_poiting.spines['left'].set_color('none')
    ax_poiting.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax_add_circle(ax_poiting, 0, 0, 30, '-k')
    ax_add_circle(ax_poiting, 0, 0, 20, '-k')
    ax_add_circle(ax_poiting, 0, 0, 10, '-k')
    ax_add_circle(ax_poiting, view['cx'], view['cy'], view['oa'], '-r')
    ax_poiting.set_xlabel('incident x/deg')
    ax_poiting.set_ylabel('incident y/deg')

    ax_energy = fig.add_axes([0.7, 0.1, 0.03, 0.3])
    ax_energy.spines['right'].set_visible(False)
    ax_energy.spines['top'].set_visible(False)
    ax_add_slider(
        ax=ax_energy,
        start=1e-1,
        stop=1e3,
        values=[view['el'], view['eu']],
        label='energy/GeV',
        log=True)

    ax_size = fig.add_axes([0.8, 0.1, 0.03, 0.3])
    ax_size.spines['right'].set_visible(False)
    ax_size.spines['top'].set_visible(False)
    ax_add_slider(
        ax=ax_size,
        start=1e0,
        stop=1e6,
        values=[np.sum(grid_intensity)/num_airshower],
        label='size',
        log=True)

    ax_nshow = fig.add_axes([0.9, 0.1, 0.03, 0.3])
    ax_nshow.spines['right'].set_visible(False)
    ax_nshow.spines['top'].set_visible(False)
    ax_add_slider(
        ax=ax_nshow,
        start=1e0,
        stop=1e6,
        values=[num_airshower],
        label='shower',
        log=True)

    plt.savefig(out_path)
    plt.close(fig)


def move_linear(view_stations, num_steps_per_station=60):
    num_steps = num_steps_per_station
    views = []
    station = 0
    while station < (len(view_stations) - 1):
        start = view_stations[station]
        stop = view_stations[station + 1]
        di = {
            'el': np.linspace(start['el'], stop['el'], num_steps),
            'eu': np.linspace(start['eu'], stop['eu'], num_steps),
            'cx': np.linspace(start['cx'], stop['cx'], num_steps),
            'cy': np.linspace(start['cy'], stop['cy'], num_steps),
            'oa': np.linspace(start['oa'], stop['oa'], num_steps),}
        views += pd.DataFrame(di).to_dict(orient='records')
        station += 1
    return views


def run_job(job):
    view = job['view']
    grid_intensity, num_airshower = irf.query.query_grid_histograms(
        energy_GeV_start=view['el'],
        energy_GeV_stop=view['eu'],
        cx_deg=view['cx'],
        cy_deg=view['cy'],
        cone_opening_angle_deg=view['oa'],
        primary_table=evttab['primary'],
        grid_histograms=job['grid_histograms'],
        num_bins_radius=job['num_bins_radius'])
    print('q: ', num_airshower)
    out_path = op.join(job['out_dir'], "{:06d}.jpg".format(job['id']))
    print('o: ', out_path)
    write_histogram_figure(
        out_path=out_path,
        grid_intensity=grid_intensity,
        num_airshower=num_airshower,
        xy_bin_edges=job['grid_geometry']['xy_bin_edges'],
        view=job['view'])
    return 0



view_stations = [
    {"el": 5, "eu": 50, "cx": 10, "cy": 0, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": 7, "cy": 7, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": 0, "cy": 10, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": -7, "cy": 7, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": -10, "cy": 0, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": -7, "cy": -7, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": 0, "cy": -10, "oa": 2.5},
    {"el": 5, "eu": 50, "cx": 7, "cy": -7, "oa": 2.5},
]

views = move_linear(view_stations, 15)

jobs = []
for idx, view in enumerate(views):
    job = {}
    job['id'] = idx
    job['out_dir'] = out_dir
    job['view'] = view
    job['primary'] = evttab['primary']
    job['grid_geometry'] = grid_geometry
    job['grid_histograms'] = grid_histograms
    job['num_bins_radius'] = cfg['grid']['num_bins_radius']
    jobs.append(job)

"""
for job in jobs:
    run_job(job)
"""

np.histogram2d(
    )