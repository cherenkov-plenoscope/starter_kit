#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os
import json
import cosmic_fluxes
import pandas as pd
import multiprocessing

NUM_JOBS = 8

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

out_dir = os.path.join(pa['summary_dir'], 'trigger_mode_and_threshold_scan')
os.makedirs(out_dir, exist_ok=True)

cosmic_rays = list(irf_config['config']['particles'].keys())
cosmic_rays.remove('gamma')

coarse_energy_bin_edges = sum_config['energy_bin_edges_GeV']
NUM_COARSE_ENERGY_BINS = len(coarse_energy_bin_edges) - 1

nominal_threshold = irf_config['config']['sum_trigger']['threshold_pe']

trigger_thresholds = np.arange(
    start=int(np.floor(0.8*nominal_threshold)),
    stop=int(np.floor(2.0*nominal_threshold)),
    step=10
)


def num_trigger_foci_simulated():
    return np.sum(
        ["focus_" in k for k in irf.table.STRUCTURE['trigger'].keys()]
    )


def list_trigger_focus_combinations(
    lowest_accepting_focus=3,
    lowest_rejecting_focus=0,
):
    num_foci = num_trigger_foci_simulated()
    focus_combinations = []
    for accepting_focus in np.arange(num_foci-1, lowest_accepting-1, -1):
        for rejecting_focus in np.arange(accepting_focus-1, lowest_rejecting_focus-1, -1):
            focus_combinations.append([accepting_focus, rejecting_focus])
    return focus_combinations


def list_trigger_modi(
    intensity_ratios_between_foci=np.linspace(1.0, 1.1, 2),
    add_telescope_mode=True,
    add_light_field_mode=True,
    lowest_accepting_focus=3,
    lowest_rejecting_focus=0,
):
    trigger_modi = []

    if add_telescope_mode:
        for accepting_focus in range(num_trigger_foci_simulated()):
            m = {
                "accepting_focus": int(accepting_focus),
                "rejecting_focus": -1,
                "use_rejection_focus": 0,
                "intensity_ratio_between_foci": 0,
            }
            trigger_modi.append(m)

    if add_light_field_mode
        focus_combinations = list_trigger_focus_combinations(
            lowest_accepting_focus=lowest_accepting_focus,
            lowest_rejecting_focus=lowest_rejecting_focus
        )
        for focus_combination in focus_combinations:
            for intensity_ratio in intensity_ratios_between_foci:
                m = {
                    "accepting_focus": int(focus_combination[0]),
                    "rejecting_focus": int(focus_combination[1]),
                    "use_rejection_focus": 1,
                    "intensity_ratio_between_foci": float(intensity_ratio),
                }
                trigger_modi.append(m)

    return trigger_modi


def make_jobs(
    map_dir,
    trigger_modi,
    trigger_thresholds,
    energy_bin_edges,
    max_cherenkov_in_nsb_pe=0.0,

):
    combinations = []
    for tm in range(len(trigger_modi)):
        for tt in range(len(trigger_thresholds)):
            combi = {
                'idx_modus': tm,
                'idx_threshold': tt,
                'modus': trigger_modi[tm],
                'threshold': trigger_thresholds[tt],
            }
            combinations.append(combi)
    bundles = irf.bundle.bundle_jobs(
        jobs=combinations,
        desired_num_bunbles=NUM_JOBS
    )
    jobs = []
    for job_idx, bundle in enumerate(bundles):
        job = {}
        job['idx'] = job_idx
        job['trigger_combinations'] = bundle
        job['irf_config'] = irf_config
        job['pa'] = pa
        job['map_dir'] = map_dir
        job['max_cherenkov_in_nsb_pe'] = max_cherenkov_in_nsb_pe
        job['energy_bin_edges'] = energy_bin_edges
        jobs.append(job)
    return jobs


def run_job(job):
    pa = job['pa']
    irf_config = job['irf_config']
    map_dir = job['map_dir']
    JSONL_NAME = "{:06d}_{:s}_{:s}.jsonl"
    os.makedirs(map_dir, exist_ok=True)

    max_cherenkov_in_nsb_pe = job['max_cherenkov_in_nsb_pe']
    NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2

    TIME_SLICE_DURATION = irf_config[
        'merlict_propagation_config'][
        'photon_stream'][
        'time_slice_duration']
    NUM_TIME_SLICES_PER_EVENT = (
        100 -
        irf_config['config']['sum_trigger']['integration_time_slices']
    )
    EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION

    energy_bin_edges = job['energy_bin_edges']

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
                level_keys=[
                    'primary',
                    'cherenkovsize',
                    'grid',
                    'core',
                    'trigger'
                ]
            )

        # night-sky-background-table
        # --------------------------
        nsb_table = {}
        for particle_key in irf_config['config']['particles']:
            mask_nsb = (
                particle_tables[particle_key]['trigger']['num_cherenkov_pe'] <=
                max_cherenkov_in_nsb_pe
            )
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

        nsb_records = []
        for particle_key in irf_config['config']['particles']:
            for trigger_combi in job['trigger_combinations']:
                nsb_trigger_mask = irf.analysis.light_field_trigger_mask(
                    trigger_table=nsb_table[particle_key]['trigger'],
                    threshold=trigger_combi['threshold'],
                    modus=trigger_combi['modus'],
                ).astype(np.int)
                num_nsb_trigger = np.sum(nsb_trigger_mask)

                with np.errstate(divide='ignore', invalid='ignore'):
                    rate_uncertainty = np.sqrt(num_nsb_trigger)/num_nsb_trigger

                nsb_records.append({
                    'idx_threshold': int(trigger_combi['idx_threshold']),
                    'idx_modus': int(trigger_combi['idx_modus']),

                    'value': float(num_nsb_trigger/nsb_exposure_time),
                    'relative_uncertainty': float(rate_uncertainty),
                    'unit': "s$^{-1}$"
                })
        fname = JSONL_NAME.format(job['idx'], site_key, 'nsb')
        with open(os.path.join(map_dir, fname), "wt") as f:
            for record in nsb_records:
                f.write(json.dumps(record)+'\n')

        # gamma-rays on trigger-level
        # ---------------------------
        _fov_radius_deg = 0.5*irf_config[
            'light_field_sensor_geometry']['max_FoV_diameter_deg']

        gamma_idx_thrown = irf.analysis.cut_primary_direction_within_angle(
            primary_table=particle_tables['gamma']['primary'],
            radial_angle_deg=_fov_radius_deg - 0.5,
            azimuth_deg=irf_config['config']['plenoscope_pointing']['azimuth_deg'],
            zenith_deg=irf_config['config']['plenoscope_pointing']['zenith_deg'],
        )

        gamma_table = spt.cut_table_on_indices(
            table=particle_tables['gamma'],
            structure=irf.table.STRUCTURE,
            common_indices=gamma_idx_thrown,
            level_keys=['primary', 'grid', 'trigger']
        )

        gamma_records = []
        for trigger_combi in job['trigger_combinations']:
            gamma_mask_detected = irf.analysis.light_field_trigger_mask(
                trigger_table=gamma_table['trigger'],
                threshold=trigger_combi['threshold'],
                modus=trigger_combi['modus'],
            ).astype(np.int)

            Aeff, Aeff_unc = irf.analysis.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=gamma_table['primary']['energy_GeV'],
                mask_detected=gamma_mask_detected,
                quantity_scatter=gamma_table['grid']['area_thrown_m2'],
                num_grid_cells_above_lose_threshold=gamma_table[
                    'grid']['num_bins_above_threshold'],
                total_num_grid_cells=NUM_GRID_BINS,
            )

            gamma_records.append({
                'idx_threshold': int(trigger_combi['idx_threshold']),
                'idx_modus': int(trigger_combi['idx_modus']),

                'value': Aeff.tolist(),
                'relative_uncertainty': Aeff_unc.tolist(),
                'unit': "m$^{2}$",
            })

        fname = JSONL_NAME.format(job['idx'], site_key, 'gamma')
        with open(os.path.join(map_dir, fname), "wt") as f:
            for record in gamma_records:
                f.write(json.dumps(record)+'\n')

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

            cosmic_ray_records = []
            for trigger_combi in job['trigger_combinations']:
                particle_mask_detected = irf.analysis.light_field_trigger_mask(
                    trigger_table=particle_tables[particle_key]['trigger'],
                    threshold=trigger_combi['threshold'],
                    modus=trigger_combi['modus'],
                ).astype(np.int)

                particle_scatter = (
                    particle_tables[
                        particle_key]['grid']['area_thrown_m2']*
                    particle_tables[
                        particle_key]['primary']['solid_angle_thrown_sr']
                )

                acc, acc_unc = irf.analysis.effective_quantity_for_grid(
                    energy_bin_edges_GeV=energy_bin_edges,
                    energy_GeV=particle_tables[particle_key]['primary']['energy_GeV'],
                    mask_detected=particle_mask_detected,
                    quantity_scatter=particle_scatter,
                    num_grid_cells_above_lose_threshold=particle_tables[
                        particle_key]['grid']['num_bins_above_threshold'],
                    total_num_grid_cells=NUM_GRID_BINS,
                )
                cosmic_ray_records.append({
                    'idx_threshold': int(trigger_combi['idx_threshold']),
                    'idx_modus': int(trigger_combi['idx_modus']),

                    'value': acc.tolist(),
                    'relative_uncertainty': acc_unc.tolist(),
                    'unit': "m$^{2}$ sr",
                })

            fname = JSONL_NAME.format(job['idx'], site_key, particle_key)
            with open(os.path.join(map_dir, fname), "wt") as f:
                for record in cosmic_ray_records:
                    f.write(json.dumps(record)+'\n')


pool = multiprocessing.Pool(NUM_JOBS)
pool.map(run_job, jobs)


def reduce_jobs(map_dir, )

    scan = {}

    channels = list(irf_config['config']['particles'].keys()) + ['nsb']

    for site_key in irf_config['config']['sites']:
        scan[site_key] = {}
        for channel in channels:
            scan[site_key][channel] = []
            for tm in range(len(trigger_modi)):
                scan[site_key][channel].append(
                    [{} for tt in range(len(trigger_thresholds_pe))]
                )
            for job in jobs:

                fname = "{:06d}_{:s}_{:s}.jsonl".format(
                    job['idx'],
                    site_key,
                    channel
                )
                with open(os.path.join(map_reduce_dir, fname), 'rt') as f:
                    for line in f:
                        record = json.loads(line)
                        tm = record['idx_modus']
                        tt = record['idx_threshold']

                        for item in ['value', 'relative_uncertainty', 'unit']:
                            scan[site_key][channel][tm][tt][item] = record[item]


with open(os.path.join(out_dir, 'trigger_mode_and_threshold_scan.json'), "wt") as f:
    f.write(json.dumps(scan))

with open(os.path.join(out_dir, 'trigger_modi.json'), "wt") as f:
    f.write(json.dumps(trigger_modi, indent=4))

with open(os.path.join(out_dir, 'trigger_thresholds.json'), "wt") as f:
    f.write(json.dumps(trigger_thresholds_pe.tolist(), indent=4))
