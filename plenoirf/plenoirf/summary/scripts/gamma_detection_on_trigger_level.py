#!/usr/bin/python
import sys
import magnetic_deflection as mdfl
import plenoirf as irf
import sparse_table as spt
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2
MAX_SOURCE_ANGLE_DEG = sum_config[
    'gamma_ray_source_direction'][
    'max_angle_relative_to_pointing_deg']
pointing_azimuth_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'azimuth_deg']
pointing_zenith_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'zenith_deg']
analysis_trigger_threshold = sum_config['trigger']['threshold_pe']
trigger_modus = sum_config["trigger"]["modus"]
num_energy_bins = sum_config['energy_binning']['num_bins']
energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    num_energy_bins + 1
)
max_relative_leakage = sum_config['quality']['max_relative_leakage']
min_reconstructed_photons = sum_config['quality']['min_reconstructed_photons']
psf_68_deg = 0.8

fc16by9 = sum_config['plot']['16_by_9']
fc5by4 = fc16by9.copy()
fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

area_gamma_detection_on_trigger_level = {}

for site_key in irf_config['config']['sites']:
    area_gamma_detection_on_trigger_level[site_key] = {}
    diffuse_gamma_table = spt.read(
        path=os.path.join(
            pa['run_dir'],
            'event_table',
            site_key,
            'gamma',
            'event_table.tar'
        ),
        structure=irf.table.STRUCTURE
    )

    indices_onregion = irf.analysis.effective_quantity.cut_primary_direction_within_angle(
        primary_table=diffuse_gamma_table['primary'],
        radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
        azimuth_deg=pointing_azimuth_deg,
        zenith_deg=pointing_zenith_deg,
    )

    # thrown
    gamma_table_thrown = spt.cut_table_on_indices(
        table=diffuse_gamma_table,
        structure=irf.table.STRUCTURE,
        common_indices=indices_onregion,
        level_keys=None
    )

    # detected
    indices_passed_trigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=diffuse_gamma_table['trigger'],
        threshold=analysis_trigger_threshold,
        modus=trigger_modus,
    )
    indices_has_features = diffuse_gamma_table['features'][spt.IDX]

    '''
    indices_quality = irf.analysis.effective_quantity.cut_quality(
        feature_table=diffuse_gamma_table['features'],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )
    '''
    indices_candidates = spt.intersection([
        indices_onregion,
        indices_passed_trigger,
        indices_has_features,
        #indices_quality,
    ])

    gamma_table_candidates = spt.cut_table_on_indices(
        table=diffuse_gamma_table,
        structure=irf.table.STRUCTURE,
        common_indices=indices_candidates,
        level_keys=None
    )
    gamma_table_candidates = spt.sort_table_on_common_indices(
        table=gamma_table_candidates,
        common_indices=indices_candidates
    )

    indices_detected_in_onregion = []

    for evt in range(gamma_table_candidates['features'].shape[0]):

        true_cx, true_cy = mdfl.discovery._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(gamma_table_candidates['primary']['azimuth_rad'][evt]),
            zenith_deg=np.rad2deg(gamma_table_candidates['primary']['zenith_rad'][evt])
        )

        (rec_cx, rec_cy) = irf.analysis.gamma_direction.estimate(
            light_front_cx=gamma_table_candidates['features']['light_front_cx'][evt],
            light_front_cy=gamma_table_candidates['features']['light_front_cy'][evt],
            image_infinity_cx_mean=gamma_table_candidates['features']['image_infinity_cx_mean'][evt],
            image_infinity_cy_mean=gamma_table_candidates['features']['image_infinity_cy_mean'][evt],
        )

        delta_cx = true_cx - rec_cx
        delta_cy = true_cy - rec_cy

        delta_c = np.hypot(delta_cx, delta_cy)
        delta_c_deg = np.rad2deg(delta_c)

        if delta_c_deg <= psf_68_deg:
            indices_detected_in_onregion.append(gamma_table_candidates['primary'][spt.IDX][evt])

    indices_detected_in_onregion = np.array(indices_detected_in_onregion)

    mask_detected = spt.make_mask_of_right_in_left(
        left_indices=gamma_table_thrown['primary'][spt.IDX],
        right_indices=indices_detected_in_onregion,
    )
    (_q_eff, _q_unc) = irf.analysis.effective_quantity.effective_quantity_for_grid(
        energy_bin_edges_GeV=energy_bin_edges,
        energy_GeV=gamma_table_thrown['primary']['energy_GeV'],
        mask_detected=mask_detected,
        quantity_scatter=gamma_table_thrown['grid']['area_thrown_m2'],
        num_grid_cells_above_lose_threshold=gamma_table_thrown['grid']['num_bins_above_threshold'],
        total_num_grid_cells=NUM_GRID_BINS,
    )

    area_gamma_detection_on_trigger_level[site_key]['point'] = {}
    area_gamma_detection_on_trigger_level[site_key]['point'] = {
        "value": _q_eff,
        "relative_uncertainty": _q_unc,
    }

with open(os.path.join(pa['out_dir'], 'gamma_detection_on_trigger_level.json'), 'wt') as f:
    f.write(
        json.dumps(
            area_gamma_detection_on_trigger_level,
            indent=4,
            cls=irf.json_numpy.Encoder
        )
    )
