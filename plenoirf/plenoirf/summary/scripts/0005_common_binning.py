#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy

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
        fine=sum_config["energy_binning"]["fine"][scenario_key]
    )

    energy[scenario_key] = {
        "edges": edges,
        "num_bins": num_bins,
        "centers": irf.utils.bin_centers(edges)
    }

json_numpy.write(os.path.join(pa["out_dir"], "energy.json"), energy)
