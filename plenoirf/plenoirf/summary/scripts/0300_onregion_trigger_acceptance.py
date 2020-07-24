#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])
np.random.seed(sum_config['random_seed'])

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

energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins']['trigger_acceptance_onregion'] + 1
)
max_relative_leakage = sum_config['quality']['max_relative_leakage']
min_reconstructed_photons = sum_config['quality']['min_reconstructed_photons']

_psf = irf.json_numpy.read_tree(
    os.path.join(pa['summary_dir'], "0200_gamma_point_spread_function")
)
fix_onregion_radius_deg = {}
for site_key in irf_config['config']['sites']:
    fix_onregion_radius_deg[site_key] = _psf[
        site_key][
        'gamma'][
        'containment_angle_for_fix_onregion'][
        'containment_angle']
    assert 'deg' == _psf[
        site_key]['gamma']['containment_angle_for_fix_onregion']['unit']

cosmic_ray_keys = list(irf_config['config']['particles'].keys())
cosmic_ray_keys.remove('gamma')

for site_key in irf_config['config']['sites']:

    actual_onregion_over_possible_onregion_factor = (
        fix_onregion_radius_deg[site_key]**2 /
        MAX_SOURCE_ANGLE_DEG**2
    )

    for particle_key in irf_config['config']['particles']:
        site_particle_dir = os.path.join(pa['out_dir'], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        # point source
        # ------------
        table_diffuse = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'
            ),
            structure=irf.table.STRUCTURE
        )

        idx_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
            primary_table=table_diffuse['primary'],
            radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
            azimuth_deg=pointing_azimuth_deg,
            zenith_deg=pointing_zenith_deg,
        )

        # thrown
        table_point = spt.cut_table_on_indices(
            table=table_diffuse,
            structure=irf.table.STRUCTURE,
            common_indices=idx_onregion,
            level_keys=None
        )

        # detected
        idx_trigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=table_point['trigger'],
            threshold=analysis_trigger_threshold,
            modus=trigger_modus,
        )
        idx_quality = irf.analysis.cuts.cut_quality(
            feature_table=table_point['features'],
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )
        idx_has_features = table_point['features'][spt.IDX]

        idx_candidates = spt.intersection([
            idx_trigger,
            idx_has_features,
            idx_quality,
        ])

        candidate_table = spt.cut_table_on_indices(
            table=table_point,
            structure=irf.table.STRUCTURE,
            common_indices=idx_candidates,
            level_keys=None
        )
        candidate_table = spt.sort_table_on_common_indices(
            table=candidate_table,
            common_indices=idx_candidates
        )

        idx_detected_in_onregion = \
            irf.analysis.cuts.cut_reconstructed_source_in_true_onregion(
                table=candidate_table,
                radial_angle_onregion_deg=fix_onregion_radius_deg[site_key],
            )

        mask_detected = spt.make_mask_of_right_in_left(
            left_indices=table_point['primary'][spt.IDX],
            right_indices=idx_detected_in_onregion,
        )
        (
            _q_eff,
            _q_unc
        ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
            energy_bin_edges_GeV=energy_bin_edges,
            energy_GeV=table_point['primary']['energy_GeV'],
            mask_detected=mask_detected,
            quantity_scatter=table_point['grid']['area_thrown_m2'],
            num_grid_cells_above_lose_threshold=table_point[
                'grid'][
                'num_bins_above_threshold'],
            total_num_grid_cells=NUM_GRID_BINS,
        )

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "point.json"),
            {
                "comment": (
                    "Effective area "
                    "for a point source, reconstructed in onregion. "
                    "VS energy-bins"),
                "unit": "m$^{2}$",
                "mean": _q_eff,
                "relative_uncertainty": _q_unc,
            }
        )

        # diffuse source
        # ---------------

        # thrown
        table_diffuse = table_diffuse

        # detected
        idx_trigger = irf.analysis.light_field_trigger_modi.make_indices(
            trigger_table=table_diffuse['trigger'],
            threshold=analysis_trigger_threshold,
            modus=trigger_modus,
        )
        idx_quality = irf.analysis.cuts.cut_quality(
            feature_table=table_diffuse['features'],
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )
        idx_features = table_diffuse['features'][spt.IDX]

        idx_candidates = spt.intersection([
            idx_trigger,
            idx_features,
            idx_quality,
        ])
        table_candidates = spt.cut_table_on_indices(
            table=table_diffuse,
            structure=irf.table.STRUCTURE,
            common_indices=idx_candidates,
            level_keys=None
        )
        table_candidates = spt.sort_table_on_common_indices(
            table=table_candidates,
            common_indices=idx_candidates
        )
        idx_in_possible_onregion = \
            irf.analysis.cuts.cut_reconstructed_source_in_possible_onregion(
                feature_table=table_candidates['features'],
                radial_angle_to_put_possible_onregion_deg=MAX_SOURCE_ANGLE_DEG,
            )

        mask_detected_in_possible_onregion = spt.make_mask_of_right_in_left(
            left_indices=table_diffuse['primary'][spt.IDX],
            right_indices=idx_in_possible_onregion,
        )
        (
            _q_eff_all_possible_onregion,
            _q_unc
        ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
            energy_bin_edges_GeV=energy_bin_edges,
            energy_GeV=table_diffuse['primary']['energy_GeV'],
            mask_detected=mask_detected_in_possible_onregion,
            quantity_scatter=(
                table_diffuse['grid']['area_thrown_m2'] *
                table_diffuse['primary']['solid_angle_thrown_sr']
            ),
            num_grid_cells_above_lose_threshold=table_diffuse[
                'grid'][
                'num_bins_above_threshold'],
            total_num_grid_cells=NUM_GRID_BINS,
        )
        _q_eff = (
            _q_eff_all_possible_onregion *
            actual_onregion_over_possible_onregion_factor
        )

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source, reconstructed in onregion. "
                    "VS energy-bins"),
                "unit": "m$^{2}$ sr",
                "mean": _q_eff,
                "relative_uncertainty": _q_unc,
            }
        )
