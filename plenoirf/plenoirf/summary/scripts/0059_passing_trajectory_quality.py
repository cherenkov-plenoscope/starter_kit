#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        idx_common = spt.intersection(
            [
                passing_trigger[sk][pk]["passed_trigger"]["idx"],
                passing_quality[sk][pk]["passed_quality"]["idx"],
            ]
        )
        event_table = spt.cut_and_sort_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_common,
        )

        event_frame = irf.reconstruction.trajectory_quality.make_rectangular_table(
            event_table=event_table,
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        # estimate_quality
        # ----------------

        quality = irf.reconstruction.trajectory_quality.estimate_trajectory_quality(
            event_frame=event_frame,
            quality_features=irf.reconstruction.trajectory_quality.QUALITY_FEATURES,
        )

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "trajectory_quality.json"),
            {
                "comment": (
                    "Quality of reconstructed trajectory. "
                    "0 is worst, 1 is best."
                ),
                spt.IDX: event_frame[spt.IDX],
                "unit": "1",
                "quality": quality,
            },
        )

        # apply cut
        # ---------
        mask = quality >= sum_config["quality"]["min_trajectory_quality"]
        idx_passed = event_frame[spt.IDX][mask]

        irf.json_numpy.write(
            path=os.path.join(
                site_particle_dir, "passed_trajectory_quality.json"
            ),
            out_dict={spt.IDX: idx_passed},
        )
