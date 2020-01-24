import pandas as pd
import numpy as np
import os
import json
from os import path as op
import glob
import plenoscope_map_reduce as plmr
from plenoscope_map_reduce import instrument_response as irf


def run_ids_in_dir(feature_dir, wild_card):
    paths = glob.glob(op.join(feature_dir, wild_card))
    run_ids = []
    for path in paths:
        basename = op.basename(path)
        run_ids.append(int(basename[0:6]))
    return list(set(run_ids))


EXAMPLE_IRF_DIR = op.join(".", "run2")

irf_dir = EXAMPLE_IRF_DIR

with open(op.join(irf_dir, "config.json"), "rt") as f:
    cfg = json.loads(f.read())

particle_keys = list(cfg["particles"].keys())
site_keys = list(cfg["sites"].keys())

# collect features
site_key = "namibia"


features = {}
for particle_key in particle_keys:
    site_particle_dir = op.join(irf_dir, site_key, particle_key)
    feature_dir = op.join(site_particle_dir, "features")

    features[particle_key] = {}
    for level in irf.TABLE["level"]:
        features[particle_key][level] = []
        for run_id in run_ids_in_dir(feature_dir, wild_card="*.csv"):
            fpath = op.join(
                feature_dir,
                "{:06d}_{:s}.csv".format(run_id, level))
            print(fpath)
            assert op.exists(fpath)
            rec = irf.read_table_to_recarray(
                path=fpath,
                table_config=irf.TABLE,
                level=level)
            features[particle_key][level].append(rec)

    for level in irf.TABLE["level"]:
        print(particle_key, level)
        ll = features[particle_key][level]
        cat = np.concatenate(ll)
        features[particle_key][level] = cat
