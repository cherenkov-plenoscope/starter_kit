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



def resizs_hist(hist, shape):
    xf = hist.shape[0] // shape[0]
    assert hist.shape[0] % shape[0] == 0
    yf = hist.shape[1] // shape[1]
    assert hist.shape[1] % shape[1] == 0
    out = np.zeros(shape, dtype=hist.dtype)
    for x in range(shape[0]):
        for y in range(shape[1]):
            x_start = x*xf
            x_stop = (x+1)*xf - 1
            y_start = y*yf
            y_stop = (y+1)*yf - 1
            out[x, y] = np.sum(hist[x_start:x_stop, y_start:y_stop])
    return out




EXAMPLE_IRF_DIR = op.join("..", "run-2020-01-27_1011")

irf_dir = EXAMPLE_IRF_DIR

with open(op.join(irf_dir, "config.json"), "rt") as f:
    cfg = json.loads(f.read())

particle_keys = list(cfg["particles"].keys())
particle_keys = ["electron"]
site_keys = list(cfg["sites"].keys())

# collect features
site_key = "namibia"


query_str = '(run_id == {:d}) and (airshower_id == {:d})'

grid_shape = (128, 128)

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

    # reduce grid_histogram
    num_bins_grid_edge = 1024
    num_energy_bins = 10
    energy_start = 0.25
    energy_stop = 1000
    energy_bin_edges = np.geomspace(
        energy_start,
        energy_stop,
        num_energy_bins + 1)

    num_cxy_bins = 10
    cx_start_stop = np.deg2rad(30)
    cxy_bin_edges = np.linspace(
        -cx_start_stop,
        cx_start_stop,
        num_cxy_bins + 1)

    grids[particle_key] = irf.grid.reduce_histograms(feature_dir=feature_dir)

energy = 1
energy_abs_delta = 1
cx_deg = 0.0
cy_deg = 0.0
c_delta_deg = 1.5

prm = features['electron']['primary']


cxs = np.cos(prm['azimuth_rad'])*prm['zenith_rad']
cys = np.sin(prm['azimuth_rad'])*prm['zenith_rad']

directions = irf.grid._make_bunch_direction(cxs, cys)
target_direction = irf.grid._make_bunch_direction(
    np.array([np.deg2rad(cx_deg)]),
    np.array([np.deg2rad(cy_deg)]))

deltas = irf.grid._make_angle_between(
    directions=directions,
    direction=target_direction.T)[:, 0]
mask_cxcy = deltas <= np.deg2rad(c_delta_deg)
mask_energy = np.logical_and(
    prm['energy_GeV'] > (energy - energy_abs_delta),
    prm['energy_GeV'] <= (energy + energy_abs_delta))
mask_match = np.logical_and(mask_cxcy, mask_energy)

match = prm[mask_match]

cx_match = cxs[mask_match]
cy_match = cys[mask_match]

hist = np.zeros((1024, 1024))
num_airshower = 0
for asho in match:
    seed = irf.table.random_seed_based_on(
        run_id=asho['run_id'],
        airshower_id=asho['airshower_id'])
    print(seed)
    hist += irf.grid.bytes_to_histogram(grids[particle_key][seed])
    num_airshower += 1

hh, nn = irf.query.query_grid_histograms(
    energy_GeV_start=100,
    energy_GeV_stop=1000,
    cx_deg=0.,
    cy_deg=0.,
    cone_opening_angle_deg=5,
    primary_table=features[particle_key]['primary'],
    grid_histograms=grids[particle_key],
    num_bins_radius=512)