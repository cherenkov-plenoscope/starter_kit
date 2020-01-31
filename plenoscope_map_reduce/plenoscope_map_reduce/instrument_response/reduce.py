import os
from os import path as op
import shutil
import pandas as pd
import glob
import tarfile
from . import logging
from . import table
from . import grid


def reduce_site_particle(site_particle_dir, out_dir=None, lazy=True):
    if out_dir is None:
        out_dir = site_particle_dir
    log_dir = op.join(site_particle_dir, "log")
    features_dir = op.join(site_particle_dir, "features")
    past_trigger_dir = op.join(site_particle_dir, "past_trigger")

    # run-time
    # ========
    log_path = op.join(out_dir, 'runtime.csv')
    if not op.exists(log_path) or not lazy:
        lop_paths = glob.glob(op.join(log_dir, "*_runtime.jsonl"))
        log_records = logging.reduce(list_of_log_paths=lop_paths)
        log_df = pd.DataFrame(log_records)
        log_df.to_csv(log_path+".tmp", index=False)
        shutil.move(log_path+".tmp", log_path)

    # features
    # ========
    feature_path = op.join(out_dir, 'features.tar')
    if not op.exists(feature_path) or not lazy:
        evttab = table.reduce_feature_dir(
            feature_dir=op.join(site_particle_dir, "features"),
            format_suffix=table.FORMAT_SUFFIX,
            config=table.CONFIG)
        table.write_site_particle(
            path=feature_path,
            event_table=evttab,
            config=table.CONFIG)

    # grid images
    # ===========
    grid_path = op.join(out_dir, 'grid.tar')
    if not op.exists(grid_path) or not lazy:
        grid_paths = glob.glob(op.join(features_dir, "*_grid.tar"))
        with tarfile.open(grid_path, "w") as tarfout:
            for grid_path in grid_paths:
                with tarfile.open(grid_path, "r") as tarfin:
                    for tarinfo in tarfin:
                        tarfout.addfile(
                            tarinfo=tarinfo,
                            fileobj=tarfin.extractfile(tarinfo))
