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


def run_ids_in_dir(feature_dir, wild_card):
    paths = glob.glob(op.join(feature_dir, wild_card))
    run_ids = []
    for path in paths:
        basename = op.basename(path)
        run_ids.append(int(basename[0:6]))
    return list(set(run_ids))


def reduce_site_particle_features(
    site_particle_feature_dir,
    format_suffix='csv',
    table_config=irf.TABLE,
):
    wild_card = '*.{:s}'.format(format_suffix)
    spf_dir = site_particle_feature_dir
    features = {}
    for level in table_config["level"]:
        features[level] = []
        for run_id in run_ids_in_dir(spf_dir, wild_card=wild_card):
            fpath = op.join(
                feature_dir,
                "{:06d}_{:s}.{:s}".format(run_id, level, format_suffix))
            _rec = irf.read_table_to_recarray(
                path=fpath,
                table_config=table_config,
                level=level)
            features[level].append(_rec)
    for level in table_config["level"]:
        print(particle_key, level)
        ll = features[level]
        cat = np.concatenate(ll)
        features[level] = cat
    return features


def write_site_particle_features(path, features, table_config=irf.TABLE):
    assert op.splitext(path)[1] == ".tar"
    with tarfile.open(path, "w") as tarout:
        for level in table_config['level']:
            level_df = pd.DataFrame(features[level])
            level_csv = level_df.to_csv(index=False)
            level_filename = "{:s}.csv".format(level)
            with io.BytesIO() as fbuff:
                fbuff.write(str.encode(level_csv))
                fbuff.seek(0)
                tarinfo = tarfile.TarInfo(name=level_filename)
                tarinfo.size = len(fbuff.getvalue())
                tarout.addfile(tarinfo=tarinfo, fileobj=fbuff)


def read_site_particle_features(path, table_config=irf.TABLE):
    features = {}
    with tarfile.open(path, "r") as tarin:
        for level in table_config['level']:
            level_filename = "{:s}.csv".format(level)
            tarinfo = tarin.getmember(level_filename)
            with io.BytesIO() as fbuff:
                fbuff.write(tarin.extractfile(tarinfo).read())
                fbuff.seek(0)
                features[level] = irf.read_table_to_recarray(
                    fbuff,
                    table_config=table_config,
                    level=level)
    return features


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


features = {}
for particle_key in particle_keys:
    site_particle_dir = op.join(irf_dir, site_key, particle_key)
    feature_dir = op.join(site_particle_dir, "features")

    pp = "{:s}_{:s}.tar".format(site_key, particle_key)
    if not op.exists(pp):
        features[particle_key] = reduce_site_particle_features(
            site_particle_feature_dir=feature_dir,
            format_suffix='csv',
            table_config=irf.TABLE)

        write_site_particle_features(
            path=pp,
            features=features[particle_key],
            table_config=irf.TABLE)
    else:
        features[particle_key] = read_site_particle_features(
            pp,
            table_config=irf.TABLE)

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

    his = []
    for i_energy in range(num_energy_bins):
        his.append([])
        for i_cx in range(num_cxy_bins):
            his[i_energy].append([])
            for i_cy in range(num_cxy_bins):
                his[i_energy][i_cx].append([])

    prm_df = pd.DataFrame(features[particle_key]["primary"])
    prm_df = prm_df.set_index(list(irf.TABLE['index'].keys()))

    run_ids = run_ids_in_dir(feature_dir, wild_card="*grid_images.tar")
    for run_id in run_ids:
        tarpath = op.join(
            feature_dir,
            "{:06d}_grid_images.tar".format(run_id))

        with tarfile.open(tarpath, "r") as tarin:
            for tarinfo in tarin:
                airshower_id = int(tarinfo.name[0:6])

                ss = prm_df.query(query_str.format(run_id, airshower_id))

                energy = ss['energy_GeV'].values[0]
                az = ss['azimuth_rad'].values[0]
                zd = ss['zenith_rad'].values[0]

                en_bin = np.digitize(energy, energy_bin_edges) - 1
                if en_bin == -1 or en_bin == len(energy_bin_edges)-1:
                    print('energy out of range', energy)
                    continue

                cx = np.cos(az)*zd
                cy = np.sin(az)*zd

                cx_bin = np.digitize(cx, cxy_bin_edges) - 1
                if cx_bin == -1 or cx_bin == len(cxy_bin_edges)-1:
                    print('energy out of cx range', cx)
                    continue

                cy_bin = np.digitize(cy, cxy_bin_edges) - 1
                if cy_bin == -1 or cy_bin == len(cxy_bin_edges)-1:
                    print('energy out of cy range', cy)
                    continue

                print(run_id, airshower_id, en_bin, cx_bin, cy_bin)
                his[en_bin][cx_bin][cy_bin].append(
                    tarin.extractfile(tarinfo).read())

                '''
                arr_bytes = gzip.decompress(tarin.extractfile(tarinfo).read())
                arr = np.frombuffer(arr_bytes, dtype='<f4')
                arr = arr.reshape((1024, 1024), order='c')
                '''