#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os
import json
import magnetic_deflection as mdfl
import cosmic_fluxes
import pandas as pd

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

out_dir = os.path.join(pa['summary_dir'], 'trigger_ratescan')
os.makedirs(out_dir, exist_ok=True)

cosmic_rays = {
    "proton": {"color": "red"},
    "helium": {"color": "orange"},
    "electron": {"color": "blue"}
}

coarse_energy_bin_edges = sum_config['energy_bin_edges_GeV']
NUM_COARSE_ENERGY_BINS = len(coarse_energy_bin_edges) - 1

MAX_CHERENKOV_IN_NSB_PE = 0

TIME_SLICE_DURATION = irf_config[
    'merlict_propagation_config'][
    'photon_stream'][
    'time_slice_duration']

NUM_TIME_SLICES_PER_EVENT = (
    100 -
    irf_config['config']['sum_trigger']['integration_time_slices']
)

NOMINAL_THRESHOLD = irf_config['config']['sum_trigger']['threshold_pe']

trigger_thresholds_pe = np.arange(
    start=int(np.floor(0.5*NOMINAL_THRESHOLD)),
    stop=int(np.floor(2.0*NOMINAL_THRESHOLD)),
    step=int(np.floor(0.1*NOMINAL_THRESHOLD))
)

NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2
EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION

# setup possible plenoscope-trigger-combinations
INTENSITY_RATIOS_BETWEEN_FOCI = np.linspace(1.0, 1.2, 5)

NUM_TRIGGER_FOCI_SIMULATED = np.sum(
    ["focus_" in k for k in irf.table.STRUCTURE['trigger'].keys()]
)

TRIGGER_FOCUS_COMBINATIONS = []
for accepting_focus in np.arange(NUM_TRIGGER_FOCI_SIMULATED-1, 3, -1):
    for rejecting_focus in np.arange(accepting_focus-1, accepting_focus-4, -1):
        TRIGGER_FOCUS_COMBINATIONS.append([accepting_focus, rejecting_focus])

trigger_modi = []

# add telescope-modi
for accepting_focus in range(NUM_TRIGGER_FOCI_SIMULATED):
    m = {
        "accepting_focus": accepting_focus,
        "rejecting_focus": -1,
        "use_rejection_focus": 0,
        "intensity_ratio_between_foci": 0,
    }
    trigger_modi.append(m)
'''
# add plenoscope-modi
for focus_combination in TRIGGER_FOCUS_COMBINATIONS:
    for intensity_ratio in INTENSITY_RATIOS_BETWEEN_FOCI:
        m = {
            "accepting_focus": focus_combination[0],
            "rejecting_focus": focus_combination[1],
            "use_rejection_focus": 1,
            "intensity_ratio_between_foci": intensity_ratio,
        }
        trigger_modi.append(m)
'''
trigger_modi = spt.dict_to_recarray(trigger_modi)


def divide_silent(numerator, denominator, default):
    valid = denominator != 0
    division = np.ones(shape=numerator.shape)*default
    division[valid] = numerator[valid]/denominator[valid]
    return division


def effective_quantity_for_grid(
    energy_bin_edges_GeV,
    energy_GeV,
    mask_detected,
    quantity_scatter,
    num_grid_cells_above_lose_threshold,
    total_num_grid_cells,
):
    """
    Returns the effective quantity and its uncertainty.

    Parameters
    ----------
    energy_bin_edges_GeV            Array of energy-bin-edges in GeV

    energy_GeV                      Array(num. thrown airshower)
                                    The energy of each airshower.

    mask_detected                   Array(num. thrown airshower)
                                    A flag/weight for each airshower marking
                                    its detection.

    quantity_scatter                Array(num. thrown airshower)
                                    The scatter-quantity for each airshower.
                                    This is area/m^2 for point like sources, or
                                    acceptance/m^2 sr for diffuse sources.

    num_grid_cells_above_lose_threshold     Array(num. thrown airshower)
                                            Num. of grid cells passing the lose
                                            threshold of the grid for each
                                            airshower.

    total_num_grid_cells            Int
                                    The total number of grid-cells.
    """

    quantity_detected = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(
            mask_detected*
            num_grid_cells_above_lose_threshold*
            quantity_scatter
        )
    )[0]

    count_thrown = total_num_grid_cells*np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV
    )[0]

    effective_quantity = divide_silent(
        numerator=quantity_detected,
        denominator=count_thrown,
        default=0.0
    )

    # uncertainty
    # according to Werner EffAreaComment.pdf 2020-03-21 17:35

    A_square = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(mask_detected*num_grid_cells_above_lose_threshold**2)
    )[0]

    A = np.histogram(
        energy_GeV,
        bins=energy_bin_edges_GeV,
        weights=(mask_detected*num_grid_cells_above_lose_threshold)
    )[0]

    effective_quantity_uncertainty = divide_silent(
        numerator=np.sqrt(A_square),
        denominator=A,
        default=np.nan
    )

    return effective_quantity, effective_quantity_uncertainty


def make_trigger_mask(
    trigger_table,
    threshold,
    accepting_focus,
    rejecting_focus,
    intensity_ratio_between_foci,
    use_rejection_focus,
):
    tt = trigger_table
    KEY = 'focus_{:02d}_response_pe'
    accepting_key = KEY.format(accepting_focus)
    accepting_mask = tt[accepting_key] >= threshold

    if use_rejection_focus:
        rejecting_key = KEY.format(rejecting_focus)

        rejecting_mask = tt[rejecting_key] < (
            tt[accepting_key]/intensity_ratio_between_foci)

        trigger_mask = accepting_mask*rejecting_mask
    else:
        trigger_mask = accepting_mask

    return trigger_mask


def cut_gamma_rays_on_trigger_level(
    gamma_table,
    radius_possible_on_regions_deg,
):
    cut_idxs_primary = gamma_table['primary'][spt.IDX]
    cut_idxs_grid = gamma_table['grid'][spt.IDX]
    cut_idxs_trigger = gamma_table['trigger'][spt.IDX]

    # true source direction is inside a valid on-region
    _off_deg = mdfl.discovery._angle_between_az_zd_deg(
        az1_deg=np.rad2deg(gamma_table['primary']['azimuth_rad']),
        zd1_deg=np.rad2deg(gamma_table['primary']['zenith_rad']),
        az2_deg=irf_config['config']['plenoscope_pointing']['azimuth_deg'],
        zd2_deg=irf_config['config']['plenoscope_pointing']['zenith_deg'])
    _off_mask = (_off_deg <= radius_possible_on_regions_deg)

    cut_idxs_comming_from_possible_on_region = (
        gamma_table['primary'][spt.IDX][_off_mask]
    )

    cut_idx_detected = spt.intersection([
        cut_idxs_primary,
        cut_idxs_grid,
        cut_idxs_trigger,
        cut_idxs_comming_from_possible_on_region
    ])

    return spt.cut_table_on_indices(
        table=gamma_table,
        structure=irf.table.STRUCTURE,
        common_indices=spt.dict_to_recarray({spt.IDX: cut_idx_detected}),
        level_keys=['primary', 'grid', 'trigger']
    )

acceptance = {}

def write_acceptance_to_path(trigger_modi, trigger_thresholds_pe, acceptance, path):
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, "trigger_modi.json"), "wt") as f:
        f.write(json.dumps(0))


for site_key in irf_config['config']['sites']:
    acceptance[site_key] = {}

    # read all particle-tables for site
    # ---------------------------------
    particle_tables = {}
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'
            ),
            structure=irf.table.STRUCTURE
        )
        particle_tables[particle_key] = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray(
                {spt.IDX: event_table['trigger'][spt.IDX]}
            ),
            level_keys=['primary', 'cherenkovsize', 'grid', 'core', 'trigger']
        )

    # night-sky-background-table
    # --------------------------
    nsb_table = {}
    for particle_key in irf_config['config']['particles']:
        mask_nsb = (
            particle_tables[particle_key]['trigger']['num_cherenkov_pe'] <=
            MAX_CHERENKOV_IN_NSB_PE
        )
        mask_cosmic_rays = np.logical_not(mask_nsb)

        nsb_table[particle_key] = {}
        for level in particle_tables[particle_key]:
            nsb_table[particle_key][level] = particle_tables[
                particle_key][level][mask_nsb]

    # nsb on trigger-level
    # --------------------
    nsb_exposure_time = 0.0
    for particle_key in irf_config['config']['particles']:
        nsb_exposure_time += (
            nsb_table[particle_key]['primary'].shape[0]*
            EXPOSURE_TIME_PER_EVENT
        )
    num_nsb_triggers = np.zeros(
        shape=(
            len(trigger_modi),
            len(trigger_thresholds_pe),
        ),
        dtype=np.int
    )
    for particle_key in irf_config['config']['particles']:
        for tm in range(len(trigger_modi)):
            for tt in range(len(trigger_thresholds_pe)):
                nsb_trigger_mask = make_trigger_mask(
                    trigger_table=nsb_table[particle_key]['trigger'],
                    threshold=trigger_thresholds_pe[tt],
                    accepting_focus=trigger_modi[tm]['accepting_focus'],
                    rejecting_focus=trigger_modi[tm]['rejecting_focus'],
                    intensity_ratio_between_foci=trigger_modi[tm][
                        'intensity_ratio_between_foci'],
                    use_rejection_focus=trigger_modi[tm]['use_rejection_focus'],
                ).astype(np.int)
                num_nsb_triggers[tm, tt] += np.sum(nsb_trigger_mask)

    acceptance[site_key]['night_sky_background'] = {}

    acceptance[
        site_key][
        'night_sky_background'][
        'rate'] = num_nsb_triggers/nsb_exposure_time

    acceptance[
        site_key][
        'night_sky_background'][
        'rate_uncertainty'] = divide_silent(
        numerator=np.sqrt(num_nsb_triggers),
        denominator=num_nsb_triggers,
        default=np.nan
    )

    # gamma-rays on trigger-level
    # ---------------------------
    _fov_radius_deg = 0.5*irf_config[
        'light_field_sensor_geometry']['max_FoV_diameter_deg']

    gamma_table = cut_gamma_rays_on_trigger_level(
        gamma_table=particle_tables['gamma'],
        radius_possible_on_regions_deg=(_fov_radius_deg - 0.5),
    )

    acceptance[site_key]['gamma'] = {}
    acceptance[site_key]['gamma']['area'] = np.zeros(
        shape=(
            len(trigger_modi),
            len(trigger_thresholds_pe),
            NUM_COARSE_ENERGY_BINS,
        )
    )

    acceptance[site_key]['gamma']['area_uncertainty'] = np.zeros(
        shape=(
            len(trigger_modi),
            len(trigger_thresholds_pe),
            NUM_COARSE_ENERGY_BINS,
        )
    )

    for tm in range(len(trigger_modi)):
        for tt in range(len(trigger_thresholds_pe)):
            print('gamma', tm, tt)
            num_gamma = gamma_table['primary'].shape[0]

            gamma_mask_detected = make_trigger_mask(
                trigger_table=gamma_table['trigger'],
                threshold=trigger_thresholds_pe[tt],
                accepting_focus=trigger_modi[tm]['accepting_focus'],
                rejecting_focus=trigger_modi[tm]['rejecting_focus'],
                intensity_ratio_between_foci=trigger_modi[tm][
                    'intensity_ratio_between_foci'],
                use_rejection_focus=trigger_modi[tm]['use_rejection_focus'],
            ).astype(np.int)

            a, b = effective_quantity_for_grid(
                energy_bin_edges_GeV=coarse_energy_bin_edges,
                energy_GeV=gamma_table['primary']['energy_GeV'],
                mask_detected=gamma_mask_detected,
                quantity_scatter=gamma_table['grid']['area_thrown_m2'],
                num_grid_cells_above_lose_threshold=gamma_table[
                    'grid']['num_bins_above_threshold'],
                total_num_grid_cells=NUM_GRID_BINS,
            )

            acceptance[site_key]['gamma']['area'][tm, tt, :] = a
            acceptance[site_key]['gamma']['area_uncertainty'][tm, tt, :] = b

    # cosmic-rays on trigger-level
    # ----------------------------
    for particle_key in cosmic_rays:

        num_airshower = particle_tables[particle_key]['primary'].shape[0]

        q_max = (
            particle_tables[particle_key]['primary']['solid_angle_thrown_sr']*
            particle_tables[particle_key]['grid']['area_thrown_m2']
        )

        w_grid_trials = np.ones(num_airshower)*NUM_GRID_BINS
        w_grid_intense = particle_tables[
            particle_key]['grid']['num_bins_above_threshold']

        acceptance[site_key][particle_key] = {}

        acceptance[site_key][particle_key]['acceptance'] = np.zeros(
            shape=(
                len(trigger_modi),
                len(trigger_thresholds_pe),
                NUM_COARSE_ENERGY_BINS,
            )
        )
        acceptance[site_key][particle_key]['acceptance_uncertainty'] = np.zeros(
            shape=(
                len(trigger_modi),
                len(trigger_thresholds_pe),
                NUM_COARSE_ENERGY_BINS,
            )
        )

        for tm in range(len(trigger_modi)):
            for tt in range(len(trigger_thresholds_pe)):
                print(particle_key, tm, tt)

                particle_mask_detected = make_trigger_mask(
                    trigger_table=particle_tables[particle_key]['trigger'],
                    threshold=trigger_thresholds_pe[tt],
                    accepting_focus=trigger_modi[tm]['accepting_focus'],
                    rejecting_focus=trigger_modi[tm]['rejecting_focus'],
                    intensity_ratio_between_foci=trigger_modi[tm][
                        'intensity_ratio_between_foci'],
                    use_rejection_focus=trigger_modi[tm]['use_rejection_focus'],
                ).astype(np.int)

                particle_scatter = (
                    particle_tables[
                        particle_key]['grid']['area_thrown_m2']*
                    particle_tables[
                        particle_key]['primary']['solid_angle_thrown_sr']
                )

                a, b = effective_quantity_for_grid(
                    energy_bin_edges_GeV=coarse_energy_bin_edges,
                    energy_GeV=particle_tables[particle_key]['primary']['energy_GeV'],
                    mask_detected=particle_mask_detected,
                    quantity_scatter=particle_scatter,
                    num_grid_cells_above_lose_threshold=particle_tables[
                        particle_key]['grid']['num_bins_above_threshold'],
                    total_num_grid_cells=NUM_GRID_BINS,
                )

                acceptance[site_key][particle_key]['acceptance'][tm, tt, :] = a
                acceptance[site_key][particle_key]['acceptance_uncertainty'][tm, tt, :]  = b

