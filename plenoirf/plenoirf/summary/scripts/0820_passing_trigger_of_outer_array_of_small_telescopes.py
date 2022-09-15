#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import json_numpy
import binning_utils

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

os.makedirs(pa["out_dir"], exist_ok=True)

trigger_vs_cherenkov_density = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0818_trigger_probability_vs_cherenkov_size_in_grid_bin",
    )
)
KEY = "passing_trigger_if_only_accepting_not_rejecting"

grid_bin_area_m2 = irf_config["grid_geometry"]["bin_area"]

telescope_mirror_diameter_m = 7.1
telescopes = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, -1],
    [0, 1],
    [1, -1],
    [1, 0],
    [1, 1],
]

NUM_BINS_ON_EDGE = 25
CENTER_BIN = NUM_BINS_ON_EDGE // 2

plenoscope_mirror_diameter_m = (
    irf_config["grid_geometry"]["bin_width"]
    / irf_config["config"]["grid"]["bin_width_overhead"]
)

telescope_mirror_area_m2 = np.pi * (0.5 * telescope_mirror_diameter_m) ** 2
plenoscope_mirror_area_m2 = np.pi * (0.5 * plenoscope_mirror_diameter_m) ** 2

prng = np.random.Generator(np.random.generator.PCG64(sum_config["random_seed"]))

for sk in SITES:
    for pk in PARTICLES:

        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)

        trigger = {
            "probability": trigger_vs_cherenkov_density[sk][pk][KEY]["mean"],
            "cherenkov_density_per_m2": binning_utils.centers(
                bin_edges=trigger_vs_cherenkov_density[sk][pk][KEY][
                    "Cherenkov_density_bin_edges_per_m2"
                ]
            ),
        }

        trigger["probability"] = irf.utils.fill_nans_from_end(
            arr=trigger["probability"],
            val=1.0,
        )

        trigger["probability"] = irf.utils.fill_nans_from_start(
            arr=trigger["probability"],
            val=0.0,
        )

        grid_reader = irf.grid.GridReader(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                sk,
                pk,
                "grid_roi_pasttrigger.tar",
            )
        )

        idx_pasttrigger_outer_telescope_array = []

        for shower in grid_reader:
            shower_idx, grid_cherenkov_intensity = shower
            assert grid_cherenkov_intensity.shape == (NUM_BINS_ON_EDGE, NUM_BINS_ON_EDGE)
            grid_cherenkov_density_per_m2 = grid_cherenkov_intensity / grid_bin_area_m2

            for telescope in telescopes:
                ix = telescope[0] + CENTER_BIN
                iy = telescope[1] + CENTER_BIN
                telescope_position_cherenkov_density_per_m2 = grid_cherenkov_density_per_m2[
                    ix, iy
                ]

                telescope_trigger_probability = np.interp(
                    telescope_position_cherenkov_density_per_m2 * (telescope_mirror_area_m2 / plenoscope_mirror_area_m2),
                    xp=trigger["cherenkov_density_per_m2"],
                    fp=trigger["probability"],
                )

                if telescope_trigger_probability >= prng.uniform():
                    idx_pasttrigger_outer_telescope_array.append(shower_idx)
                    break

        json_numpy.write(
            path=os.path.join(sk_pk_dir, "idx.json"),
            out_dict=idx_pasttrigger_outer_telescope_array,
        )
