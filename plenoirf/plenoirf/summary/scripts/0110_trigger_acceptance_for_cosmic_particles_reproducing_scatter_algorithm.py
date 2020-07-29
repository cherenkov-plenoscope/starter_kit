#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os

import matplotlib
matplotlib.use('Agg')

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
energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins']['trigger_acceptance'] + 1
)
trigger_thresholds = sum_config['trigger']['ratescan_thresholds_pe']
trigger_modus = sum_config["trigger"]["modus"]

phd_scatter_limits = {
    "gamma": {
        "energy_GeV":           [0.23, 0.8, 3.0, 35,   81,   432,  1000],
        "max_scatter_radius_m": [150,  150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy_GeV":           [0.23, 1.0,  10,  100,  1000],
        "max_scatter_radius_m": [150,  150,  500, 1100, 2600],
    },
    "proton": {
        "energy_GeV":           [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    }
}
phd_scatter_limits['helium'] = phd_scatter_limits['proton'].copy()


def d_grasp(table_level, column_key):
    out = {}
    for row in table_level:
        out[row[spt.IDX]] = row[column_key]
    return out


def d_set(values, idxs):
    assert len(values) == len(idxs)
    out = {}
    for ii in range(len(idxs)):
        out[idxs[ii]] = values[ii]
    return out


def cut_airshower_with_valid_scatter(event_table, scatter_limit):
    '''
    all wrt to 'primary'-level
    '''
    d_energy = d_grasp(event_table['primary'], 'energy_GeV')

    _max_scatter_radius = np.interp(
        x=event_table['primary']['energy_GeV'],
        xp=scatter_limit['energy_GeV'],
        fp=scatter_limit['max_scatter_radius_m'],
    )

    d_max_scatter_radius = d_set(
        _max_scatter_radius,
        event_table['primary']['idx']
    )

    _scatter_radius = np.hypot(
        event_table['core']['core_x_m'],
        event_table['core']['core_y_m'],
    )
    d_scatter_radius = d_set(
        _scatter_radius,
        event_table['core']['idx']
    )

    d_scatter_inside = {}
    for idx in event_table['primary']['idx']:
        if idx in d_scatter_radius:
            if d_scatter_radius[idx] <= d_max_scatter_radius[idx]:
                d_scatter_inside[idx] = 1
            else:
                d_scatter_inside[idx] = 0
        else:
            d_scatter_inside[idx] = 1

    idx_scatter_thrown = []
    for idx in d_scatter_inside:
        if d_scatter_inside[idx] == 1:
            idx_scatter_thrown.append(idx)
    return np.array(idx_scatter_thrown)



for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        site_particle_dir = os.path.join(pa['out_dir'], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        _diffuse_particle_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE
        )

        idx_valid_scatt = cut_airshower_with_valid_scatter(
            event_table=_diffuse_particle_table,
            scatter_limit=phd_scatter_limits[particle_key]
        )

        diffuse_particle_table_scatter = spt.cut_table_on_indices(
            table=_diffuse_particle_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_valid_scatt
        )

        # point source
        # ------------
        idx_possible_onregion = \
            irf.analysis.cuts.cut_primary_direction_within_angle(
                primary_table=diffuse_particle_table_scatter['primary'],
                radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
                azimuth_deg=pointing_azimuth_deg,
                zenith_deg=pointing_zenith_deg,
            )

        point_particle_table = spt.cut_table_on_indices(
            table=diffuse_particle_table_scatter,
            structure=irf.table.STRUCTURE,
            common_indices=idx_possible_onregion
        )

        energy_GeV_wrt_primary = point_particle_table['primary']['energy_GeV']

        _max_scatter_radius_wrt_primary = np.interp(
            x=point_particle_table['primary']['energy_GeV'],
            xp=phd_scatter_limits[particle_key]['energy_GeV'],
            fp=phd_scatter_limits[particle_key]['max_scatter_radius_m'],
        )
        _max_scatter_area_wrt_primary = np.pi*_max_scatter_radius_wrt_primary**2.0
        quantity_scatter_wrt_primary = 1.0*_max_scatter_area_wrt_primary

        value = []
        relative_uncertainty = []
        for threshold in trigger_thresholds:
            idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=point_particle_table['trigger'],
                threshold=threshold,
                modus=trigger_modus,
            )
            mask_detected_wrt_primary = spt.make_mask_of_right_in_left(
                left_indices=point_particle_table['primary'][spt.IDX],
                right_indices=idx_detected,
            )
            (
                _q_eff,
                _q_unc
            ) = irf.analysis.effective_quantity.effective_quantity_for_scatter(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=energy_GeV_wrt_primary,
                mask_detected=mask_detected_wrt_primary,
                quantity_scatter=quantity_scatter_wrt_primary,
            )
            value.append(_q_eff)
            relative_uncertainty.append(_q_unc)

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "point.json"),
            {
                "comment": (
                    "Effective area for a point source. "
                    "VS trigger-ratescan-thresholds VS energy-bins. "
                    "algorithm: scatter."),
                "energy_bin_edges_GeV": energy_bin_edges,
                "trigger": sum_config['trigger'],
                "unit": "m$^{2}$",
                "mean": value,
                "relative_uncertainty": relative_uncertainty,
            }
        )

        # diffuse source
        # --------------
        energy_GeV_wrt_primary = diffuse_particle_table_scatter[
            'primary'][
            'energy_GeV']

        _max_scatter_radius_wrt_primary = np.interp(
            x=diffuse_particle_table_scatter['primary']['energy_GeV'],
            xp=phd_scatter_limits[particle_key]['energy_GeV'],
            fp=phd_scatter_limits[particle_key]['max_scatter_radius_m'],
        )
        _max_scatter_area_wrt_primary = np.pi*_max_scatter_radius_wrt_primary**2.0
        area_scatter_wrt_primary = _max_scatter_area_wrt_primary

        quantity_scatter_wrt_primary = (
            area_scatter_wrt_primary *
            diffuse_particle_table_scatter['primary']['solid_angle_thrown_sr']
        )

        value = []
        relative_uncertainty = []
        for threshold in trigger_thresholds:
            idx_detected = irf.analysis.light_field_trigger_modi.make_indices(
                trigger_table=diffuse_particle_table_scatter['trigger'],
                threshold=threshold,
                modus=trigger_modus,
            )
            mask_detected_wrt_primary = spt.make_mask_of_right_in_left(
                left_indices=diffuse_particle_table_scatter['primary'][spt.IDX],
                right_indices=idx_detected,
            )
            (
                _q_eff,
                _q_unc
            ) = irf.analysis.effective_quantity.effective_quantity_for_scatter(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=energy_GeV_wrt_primary,
                mask_detected=mask_detected_wrt_primary,
                quantity_scatter=quantity_scatter_wrt_primary,
            )
            value.append(_q_eff)
            relative_uncertainty.append(_q_unc)

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS trigger-ratescan-thresholds VS energy-bins"
                    "algorithm: scatter."),
                "energy_bin_edges_GeV": energy_bin_edges,
                "trigger": sum_config['trigger'],
                "unit": "m$^{2}$ sr",
                "mean": value,
                "relative_uncertainty": relative_uncertainty,
            }
        )
