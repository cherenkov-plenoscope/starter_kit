import pandas as pd
import numpy as np
import os
import io
import gzip
import json
from os import path as op
import glob
import plenoscope_map_reduce as plmr
from plenoscope_map_reduce import instrument_response as irf
import tarfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import multiprocessing


EXAMPLE_IRF_DIR = op.join("..", "run-2020-01-27_1011")

irf_dir = EXAMPLE_IRF_DIR

with open(op.join(irf_dir, "config.json"), "rt") as f:
    cfg = json.loads(f.read())


_scenery_path = op.join(
    "resources",
    "acp",
    "71m",
    "scenery",
    "scenery.json")
_light_field_sensor_geometry = irf.merlict.read_plenoscope_geometry(
    merlict_scenery_path=_scenery_path)
plenoscope_diameter = 2.0*_light_field_sensor_geometry[
        "expected_imaging_system_aperture_radius"]
plenoscope_grid_geometry = irf.grid.init(
    plenoscope_diameter=plenoscope_diameter,
    num_bins_radius=cfg["grid"]["num_bins_radius"])


particle_keys = list(cfg["particles"].keys())
particle_keys = ["electron"]
site_keys = list(cfg["sites"].keys())

# collect features
site_key = "namibia"

features = {}
grids = {}
for particle_key in particle_keys:
    site_particle_dir = op.join(irf_dir, site_key, particle_key)
    feature_dir = op.join(site_particle_dir, "features")

    pp = "{:s}_{:s}.tar".format(site_key, particle_key)
    if not op.exists(pp):
        features[particle_key] = irf.table.reduce_site_particle(
            site_particle_feature_dir=feature_dir,
            format_suffix=irf.table.FORMAT_SUFFIX,
            config=irf.table.CONFIG)

        irf.table.write_site_particle(
            path=pp,
            table=features[particle_key],
            config=irf.table.CONFIG)
    else:
        features[particle_key] = irf.table.read_site_particle(
            pp,
            config=irf.table.CONFIG)

    grids[particle_key] = irf.grid.reduce_histograms(feature_dir=feature_dir)


def add_circle(ax, x, y, r, linestyle='k-', num_points=1000):
    xs = np.zeros(num_points)
    ys = np.zeros(num_points)
    phis = np.linspace(0, 2*np.pi, num_points)
    xs = np.cos(phis)*r + x
    ys = np.sin(phis)*r + y
    ax.plot(xs, ys, linestyle)


def rm_splines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)


def add_slider_axes(ax, start, stop, values, label, log=False):
    ax.set_ylim([start, stop])
    if log:
        ax.semilogy()
    rm_splines(ax=ax)
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

def save_view(
    path = '',
    energy_GeV_start = 0.5,
    energy_GeV_stop = 5,
    cx_deg = 0.,
    cy_deg = -5.0,
    cone_opening_angle_deg = 1.5,
    primary_table = features[particle_key]['primary'],
    grid_histograms = grids[particle_key],
    plenoscope_grid_geometry = plenoscope_grid_geometry,
    ff=1.5
):
    histogram, num_airshower = irf.query.query_grid_histograms(
        energy_GeV_start=energy_GeV_start,
        energy_GeV_stop=energy_GeV_stop,
        cx_deg=cx_deg,
        cy_deg=cy_deg,
        cone_opening_angle_deg=cone_opening_angle_deg,
        primary_table=primary_table,
        grid_histograms=grid_histograms,
        num_bins_radius=plenoscope_grid_geometry['num_bins_radius'])
    grid_bin_edges = plenoscope_grid_geometry['xy_bin_edges']
    fig = plt.figure(figsize=(16/ff, 9/ff), dpi=120*ff)
    ax_hist = fig.add_axes((0.075, 0.05, 0.45, 0.9))
    rm_splines(ax_hist)
    hh = histogram
    hh_log = hh.copy()/num_airshower
    hh_log[hh>1] = np.log10(hh[hh>1])
    ax_hist.pcolormesh(
        grid_bin_edges*1e-3,
        grid_bin_edges*1e-3,
        hh_log,
        cmap='Greys')
    ax_hist.set_aspect('equal')
    ax_hist.set_xlabel('observation-level x/km')
    ax_hist.set_ylabel('observation-level y/km')
    ax_hist.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax_incident = fig.add_axes((0.675, 0.5, 0.4, 0.4))
    rm_splines(ax_incident)
    add_circle(ax_incident, 0, 0, 10)
    add_circle(ax_incident, 0, 0, 20)
    add_circle(ax_incident, 0, 0, 30)
    ax_incident.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    add_circle(ax_incident, cx_deg, cy_deg, cone_opening_angle_deg, 'r-')
    ax_incident.set_aspect('equal')
    ax_incident.set_xlabel('incident-x/deg')
    ax_incident.set_ylabel('incident-y/deg')
    ax_size = fig.add_axes((0.575, 0.15, 0.03, 0.7))
    add_slider_axes(
        ax=ax_size,
        start=1e0,
        stop=1e6,
        values=[np.sum(histogram)/num_airshower],
        label='size/1',
        log=True)
    ax_size = fig.add_axes((0.675, 0.15, 0.03, 0.7))
    add_slider_axes(
        ax=ax_size,
        start=1e-1,
        stop=1e4,
        values=[energy_GeV_start, energy_GeV_stop],
        label='energy/GeV',
        log=True)
    ax_text = fig.add_axes((0.725, 0.05, 0.3, 0.3))
    ax_text.set_axis_off()
    ax_text.set_xlim([0, 10])
    ax_text.set_ylim([0, 10])
    ax_text.text(1, 10, 'num. airshower {: 6d}'.format(num_airshower))
    ax_text.text(1, 9, 'incident-x/deg {:1.1f}'.format(cx_deg))
    ax_text.text(1, 8, 'incident-y/deg {:1.1f}'.format(cy_deg))
    ax_text.text(1, 7, 'opening angle/deg {:1.1f}'.format(
        cone_opening_angle_deg))
    fig.savefig(path)
    plt.close(fig)


def _move_linear(view_stations, num_steps_per_station=15):
    num_steps = num_steps_per_station
    views = []
    station = 0
    while station < (len(view_stations) - 1):
        start = view_stations[station]
        stop = view_stations[station + 1]
        block_views = np.array([
                np.linspace(start['el'], stop['el'], num_steps),
                np.linspace(start['eu'], stop['eu'], num_steps),
                np.linspace(start['cx'], stop['cx'], num_steps),
                np.linspace(start['cy'], stop['cy'], num_steps),
                np.linspace(start['op'], stop['op'], num_steps),
            ])
        views.append(block_views.T)
        station += 1
    return np.concatenate(views)


view_stations = [
    {"el": 5, "eu": 50, 'cx': 0.0, 'cy': 0.0, 'op': 8.0},
    {"el": 0.5, "eu": 5, 'cx': 15.0, 'cy': 0.0, 'op': 2.0},
    {"el": 0.5, "eu": 5, 'cx': 0.0, 'cy': 15.0, 'op': 2.0},
    {"el": 0.5, "eu": 5, 'cx': -15.0, 'cy': 0.0, 'op': 2.0},
    {"el": 0.5, "eu": 5, 'cx': 0.0, 'cy': -15.0, 'op': 2.0},
    {"el": 0.5, "eu": 5, 'cx': 15.0, 'cy': 0.0, 'op': 2.0},
    {"el": 5, "eu": 50, 'cx': 0.0, 'cy': 0.0, 'op': 8.0},
]

views = _move_linear(view_stations, num_steps_per_station=30)

jobs = []
for idx, v in enumerate(views):
    job = {}
    job['out_dir'] = os.path.curdir
    job['idx'] = idx
    job['el'] = v[0]
    job['eu'] = v[1]
    job['cx'] = v[2]
    job['cy'] = v[3]
    job['op'] = v[4]
    job['primary'] = features[particle_key]['primary']
    job['grid'] = grids[particle_key]
    job['grid_geometry'] = plenoscope_grid_geometry
    jobs.append(job)


def run_job(job):
    out_path = os.path.join(
        job['out_dir'],
        "{:06d}.{:s}".format(job['idx'], 'jpg'))
    save_view(
        path=out_path,
        energy_GeV_start = job['el'],
        energy_GeV_stop = job['eu'],
        cx_deg = job['cx'],
        cy_deg = job['cy'],
        cone_opening_angle_deg = job['op'],
        primary_table = job['primary'],
        grid_histograms = job['grid'],
        plenoscope_grid_geometry = job['grid_geometry'])
    return 0


pool = multiprocessing.Pool(8)
pool.map(run_job, jobs)
