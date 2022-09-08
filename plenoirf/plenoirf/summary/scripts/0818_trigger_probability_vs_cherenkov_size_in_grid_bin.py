#!/usr/bin/python
import sys
import os
from os.path import join as opj
import numpy as np
import sparse_numeric_table as spt
import plenoirf as irf
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]


passing_trigger = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

grid_bin_area_m2 = irf_config["grid_geometry"]["bin_area"]
density_bin_edges_per_m2 = np.geomspace(1e-2, 1e5, 7 * 5 + 1)


for sk in SITES:
    for pk in PARTICLES:
        site_particle_dir = opj(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        key = "trigger_probability_vs_cherenkov_size_in_grid_bin"

        mask_pasttrigger = spt.make_mask_of_right_in_left(
            left_indices=event_table["trigger"][spt.IDX],
            right_indices=passing_trigger[sk][pk]["idx"],
        )

        num_thrown = np.histogram(
            event_table["cherenkovsizepart"]["num_photons"] / grid_bin_area_m2,
            bins=density_bin_edges_per_m2,
        )[0]

        num_pasttrigger = np.histogram(
            event_table["cherenkovsizepart"]["num_photons"] / grid_bin_area_m2,
            bins=density_bin_edges_per_m2,
            weights=mask_pasttrigger,
        )[0]

        trigger_probability = irf.utils._divide_silent(
            numerator=num_pasttrigger, denominator=num_thrown, default=np.nan
        )

        trigger_probability_unc = irf.utils._divide_silent(
            numerator=np.sqrt(num_pasttrigger),
            denominator=num_pasttrigger,
            default=np.nan,
        )

        json_numpy.write(
            os.path.join(site_particle_dir, key + ".json"),
            {
                "Cherenkov_density_bin_edges_per_m2": density_bin_edges_per_m2,
                "unit": "1",
                "mean": trigger_probability,
                "relative_uncertainty": trigger_probability_unc,
            },
        )
