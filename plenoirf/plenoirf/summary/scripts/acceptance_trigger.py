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
NUM_TIME_SLICES_PER_EVENT = (
    100 -
    irf_config['config']['sum_trigger']['integration_time_slices']
)
EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION
NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2

MAX_SOURCE_ANGLE_DEG = sum_config[
    'gamma_ray_source_direction'][
    'max_angle_relative_to_pointing_deg']

energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins']
)
trigger_thresholds = sum_config['trigger']['ratescan_thresholds_pe']
trigger_modus = sum_config["trigger"]["modus"]

cosmic_response = {}
_tmp_nsb_response = {}

for site_key in irf_config['config']['sites']:
    cosmic_response[site_key] = {}
    _tmp_nsb_response[site_key] = {}
    for particle_key in irf_config['config']['particles']:
        print(site_key, particle_key)

        cosmic_response[site_key][particle_key] = {}

        diffuse_particle_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE
        )

        # point source
        # ------------
        idx_in_possible_onregion = irf.analysis.effective_quantity.cut_primary_direction_within_angle(
            primary_table=diffuse_particle_table['primary'],
            radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
            azimuth_deg=irf_config[
                'config']['plenoscope_pointing']['azimuth_deg'],
            zenith_deg=irf_config[
                'config']['plenoscope_pointing']['zenith_deg'],
        )

        point_particle_table = spt.cut_table_on_indices(
            table=diffuse_particle_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_in_possible_onregion
        )

        energy_GeV = point_particle_table['primary']['energy_GeV']
        quantity_scatter = point_particle_table['grid']['area_thrown_m2']
        num_grid_cells_above_lose_threshold = point_particle_table[
            'grid']['num_bins_above_threshold']

        _point = {
            "value": [],
            "relative_uncertainty": [],
            "unit": "m$^{2}$",
            "axis_0": "trigger_thresholds",
            "axis_1": "energy",
        }
        for threshold in trigger_thresholds:
            mask_detected = spt.make_mask_of_right_in_left(
                left_indices=point_particle_table['primary'][spt.IDX],
                right_indices=irf.analysis.light_field_trigger_modi.make_indices(
                    trigger_table=point_particle_table['trigger'],
                    threshold=threshold,
                    modus=trigger_modus,
                )
            )
            (
                _q_eff,
                _q_unc
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=energy_GeV,
                mask_detected=mask_detected,
                quantity_scatter=quantity_scatter,
                num_grid_cells_above_lose_threshold=
                    num_grid_cells_above_lose_threshold,
                total_num_grid_cells=NUM_GRID_BINS,
            )
            _point['value'].append(_q_eff)
            _point['relative_uncertainty'].append(_q_unc)
        cosmic_response[site_key][particle_key]['point'] = _point

        # diffuse source
        # --------------
        energy_GeV = diffuse_particle_table['primary']['energy_GeV']
        quantity_scatter = (
            diffuse_particle_table['grid']['area_thrown_m2'] *
            diffuse_particle_table['primary']['solid_angle_thrown_sr']
        )
        num_grid_cells_above_lose_threshold = diffuse_particle_table[
            'grid'][
            'num_bins_above_threshold']

        _diffuse = {
            "value": [],
            "relative_uncertainty": [],
            "unit": "m$^{2}$ sr",
            "axis_0": "trigger_thresholds",
            "axis_1": "energy",
        }
        for threshold in trigger_thresholds:
            mask_detected = spt.make_mask_of_right_in_left(
                left_indices=diffuse_particle_table['primary'][spt.IDX],
                right_indices=irf.analysis.light_field_trigger_modi.make_indices(
                    trigger_table=diffuse_particle_table['trigger'],
                    threshold=threshold,
                    modus=trigger_modus,
                )
            )
            (
                _q_eff,
                _q_unc
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=energy_GeV,
                mask_detected=mask_detected,
                quantity_scatter=quantity_scatter,
                num_grid_cells_above_lose_threshold=
                    num_grid_cells_above_lose_threshold,
                total_num_grid_cells=NUM_GRID_BINS,
            )
            _diffuse['value'].append(_q_eff)
            _diffuse['relative_uncertainty'].append(_q_unc)
            cosmic_response[site_key][particle_key]['diffuse'] = _diffuse

        # acceidental triggers in night-sky-background
        # --------------------------------------------

        nsb_thrown_indices = diffuse_particle_table['trigger'][spt.IDX][
            diffuse_particle_table['trigger']['num_cherenkov_pe'] <=
            MAX_CHERENKOV_IN_NSB_PE
        ]
        nsb_thrown_trigger_table = spt.cut_level_on_indices(
            table=diffuse_particle_table,
            structure=irf.table.STRUCTURE,
            level_key='trigger',
            indices=nsb_thrown_indices
        )

        _tmp_nsb_response[site_key][particle_key] = []
        for threshold in trigger_thresholds:
            nsb_triggered_in_thrown_indices = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=nsb_thrown_trigger_table,
                threshold=threshold,
                modus=trigger_modus,
            )
            _tmp_nsb_response[site_key][particle_key].append({
                'num_nsb_exposures': len(nsb_thrown_indices),
                'num_nsb_triggers': len(nsb_triggered_in_thrown_indices)
            })

nsb_response = {}
for site_key in irf_config['config']['sites']:
    num_nsb_exposures = 0
    num_nsb_triggers_vs_threshold = np.zeros(
        len(trigger_thresholds),
        dtype=np.int
    )
    for particle_key in irf_config['config']['particles']:
        num_nsb_exposures += _tmp_nsb_response[
            site_key][
            particle_key][
            0][
            'num_nsb_exposures']
        for tt, threshold in enumerate(trigger_thresholds):
            num_nsb_triggers_vs_threshold[tt] += _tmp_nsb_response[
                site_key][
                particle_key][
                tt][
                'num_nsb_triggers']

    nsb_rate = (
        num_nsb_triggers_vs_threshold /
        (num_nsb_exposures*EXPOSURE_TIME_PER_EVENT)
    )
    relative_uncertainty = (
        np.sqrt(num_nsb_triggers_vs_threshold) /
        num_nsb_triggers_vs_threshold
    )

    nsb_response[site_key] = {
        "rate": nsb_rate,
        "rate_relative_uncertainty": relative_uncertainty,
        "unit": "s$^{-1}$",
    }

Qout = {
    "energy_bin_edges": {
        "value": energy_bin_edges,
        "unit": "GeV"
    },
    "trigger_thresholds": {
        "value": trigger_thresholds,
        "unit": "p.e."
    },
    "cosmic_response": cosmic_response,
    "night_sky_background_response": nsb_response,
}

with open(os.path.join(pa['out_dir'], 'acceptance_trigger.json'), 'wt') as f:
    f.write(json.dumps(Qout, indent=4, cls=irf.json_numpy.Encoder))
