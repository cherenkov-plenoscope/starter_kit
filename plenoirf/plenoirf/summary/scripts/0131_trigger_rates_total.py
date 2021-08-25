#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

cosmic_rates = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0105_trigger_rates_for_cosmic_particles")
)
nsb_rates = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0120_trigger_rates_for_night_sky_background"
    )
)

SITES = irf_config["config"]["sites"]

trigger_thresholds = np.array(sum_config["trigger"]["ratescan_thresholds_pe"])
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]

assert analysis_trigger_threshold in trigger_thresholds
analysis_trigger_threshold_idx = irf.utils.find_closest_index_in_array_for_value(
    arr=trigger_thresholds, val=analysis_trigger_threshold
)

trigger_rates = {}
for sk in SITES:
    os.makedirs(os.path.join(pa["out_dir"], sk), exist_ok=True)
    trigger_rates[sk] = {}
    trigger_rates[sk]["night_sky_background"] = nsb_rates[sk]["night_sky_background_rates"]["mean"]

    for cosmic_key in irf_config["config"]["particles"]:
        trigger_rates[sk][cosmic_key] = cosmic_rates[sk][cosmic_key]["integral_rate"]["mean"]

    json_numpy.write(
        os.path.join(pa["out_dir"], sk, "trigger_rates_by_origin.json"),
        {
            "comment": (
                "Trigger-rates by origin VS. trigger-threshold. "
                "Including the analysis_trigger_threshold."
            ),
            "analysis_trigger_threshold_idx": analysis_trigger_threshold_idx,
            "unit": "s$^{-1}$",
            "origins": trigger_rates[sk],
        },
    )
