#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_table as spt
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

MAX_CHERENKOV_IN_NSB_PE = sum_config[
    'night_sky_background'][
    'max_num_true_cherenkov_photons']
TIME_SLICE_DURATION = 0.5e-9
NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE = 100
NUM_TIME_SLICES_PER_EVENT = (
    NUM_TIME_SLICES_IN_LIGHTFIELDSEQUENCE -
    irf_config['config']['sum_trigger']['integration_time_slices']
)
EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION

energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins'] + 1
)
trigger_thresholds = sum_config['trigger']['ratescan_thresholds_pe']
trigger_modus = sum_config["trigger"]["modus"]

tmp_nsb = {}

for site_key in irf_config['config']['sites']:
    tmp_nsb[site_key] = {}
    for particle_key in irf_config['config']['particles']:

        airshower_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE
        )

        idx_nsb = airshower_table['trigger'][spt.IDX][
            airshower_table['trigger']['num_cherenkov_pe'] <=
            MAX_CHERENKOV_IN_NSB_PE
        ]
        nsb_table = spt.cut_level_on_indices(
            table=airshower_table,
            structure=irf.table.STRUCTURE,
            level_key='trigger',
            indices=idx_nsb
        )

        tmp_nsb[site_key][particle_key] = []
        for threshold in trigger_thresholds:
            idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=nsb_table,
                threshold=threshold,
                modus=trigger_modus,
            )
            tmp_nsb[site_key][particle_key].append({
                'num_nsb_exposures': len(idx_nsb),
                'num_nsb_triggers': len(idx_detected)
            })

nsb_response = {}
for site_key in irf_config['config']['sites']:
    num_nsb_exposures = 0
    num_nsb_triggers_vs_threshold = np.zeros(
        len(trigger_thresholds),
        dtype=np.int
    )
    for particle_key in irf_config['config']['particles']:
        num_nsb_exposures += tmp_nsb[
            site_key][
            particle_key][
            0][
            'num_nsb_exposures']
        for tt, threshold in enumerate(trigger_thresholds):
            num_nsb_triggers_vs_threshold[tt] += tmp_nsb[
                site_key][
                particle_key][
                tt][
                'num_nsb_triggers']

    nsb_rate = (
        num_nsb_triggers_vs_threshold /
        (num_nsb_exposures*EXPOSURE_TIME_PER_EVENT)
    )
    relative_uncertainty = irf.analysis.effective_quantity._divide_silent(
        numerator=np.sqrt(num_nsb_triggers_vs_threshold),
        denominator=num_nsb_triggers_vs_threshold,
        default=np.nan
    )

    nsb_response[site_key] = {
        "rate": nsb_rate,
        "rate_relative_uncertainty": relative_uncertainty,
        "unit": "s$^{-1}$",
    }

Qout = {
    "trigger_thresholds": {
        "value": trigger_thresholds,
        "unit": "p.e."
    },
    "night_sky_background_rates": nsb_response,
}

with open(os.path.join(pa['out_dir'], 'acceptance_trigger.json'), 'wt') as f:
    f.write(json.dumps(Qout, indent=4, cls=irf.json_numpy.Encoder))
