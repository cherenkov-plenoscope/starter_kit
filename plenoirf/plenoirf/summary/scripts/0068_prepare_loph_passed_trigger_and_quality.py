#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import tempfile

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

_passed_quality_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0067_passing_quality")
)

passed_idx_sets = {}
for sk in irf_config["config"]["sites"]:
    passed_idx_sets[sk] = {}
    for pk in irf_config["config"]["particles"]:
        passed_trigger_and_quality = spt.intersection(
            [
                _passed_trigger_indices[sk][pk]["passed_trigger"][spt.IDX],
                _passed_quality_indices[sk][pk]["passed_quality"][spt.IDX]
            ]
        )
        passed_idx_sets[sk][pk] = set(passed_trigger_and_quality)


for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        raw_loph_run = os.path.join(
            pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar",
        )

        loph_chunk_dir = os.path.join(pa["out_dir"], sk, pk, "chunks")

        with tempfile.TemporaryDirectory(prefix="irf_summary_0068") as tmp_dir:

            loph_run_passed = os.path.join(tmp_dir, "cherenkov.phs.loph.tar")

            pl.photon_stream.loph.read_filter_write(
                in_path=raw_loph_run,
                out_path=loph_run_passed,
                identity_set=passed_idx_sets[sk][pk],
            )

            pl.photon_stream.loph.split_into_chunks(
                loph_path=loph_run_passed,
                out_dir=loph_chunk_dir,
                chunk_prefix="",
                num_events_in_chunk=256,
            )
