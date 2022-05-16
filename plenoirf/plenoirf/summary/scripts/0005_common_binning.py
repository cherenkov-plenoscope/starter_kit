#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy
import binning_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

# energy
# ------
energy = {}
for scenario_key in sum_config["energy_binning"]["fine"]:
    edges, num_bins = irf.utils.power10space_bin_edges(
        binning=sum_config["energy_binning"],
        fine=sum_config["energy_binning"]["fine"][scenario_key],
    )

    assert len(edges) >= 2
    assert np.all(np.gradient(edges) > 0.0)

    energy[scenario_key] = {
        "key": scenario_key,
        "edges": edges,
        "num_bins": num_bins,
        "centers": binning_utils.centers(edges),
        "width": binning_utils.widths(edges),
        "start": edges[0],
        "stop": edges[-1],
        "limits": [edges[0], edges[-1]],
        "unit": "GeV",
    }

json_numpy.write(os.path.join(pa["out_dir"], "energy.json"), energy)
