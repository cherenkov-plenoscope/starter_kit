#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_table as spt
import os
import json
import magnetic_deflection as mdfl
import cosmic_fluxes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

MAX_CHERENKOV_IN_NSB_PE = 10
TIME_SLICE_DURATION = 0.5e-9
NUM_TIME_SLICES_PER_EVENT = 100

NUM_ENERGY_BINS = 11
energy_bin_edges = np.geomspace(0.5, 1000, NUM_ENERGY_BINS + 1)
energy_bin_width = np.gradient(energy_bin_edges)

cosmic_rays = {
    "proton": {},
    "helium": {},
    "electron": {}
}

TRIGGER_CHANNEL = 'response_pe'  # 'refocus_2_respnse_pe' 'response_pe'

_cosmic_ray_raw_fluxes = {}
with open(os.path.join(pa['summary_dir'], "proton_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["proton"] = json.loads(f.read())
with open(os.path.join(pa['summary_dir'], "helium_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["helium"] = json.loads(f.read())
with open(os.path.join(pa['summary_dir'], "electron_positron_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["electron"] = json.loads(f.read())
for p in cosmic_rays:
    cosmic_rays[p]['differential_flux'] = np.interp(
        x=energy_bin_edges,
        xp=_cosmic_ray_raw_fluxes[p]['energy']['values'],
        fp=_cosmic_ray_raw_fluxes[p]['differential_flux']['values'])

with open(os.path.join(pa['summary_dir'], "gamma_sources.json"), "rt") as f:
    gamma_sources = json.loads(f.read())

for source in gamma_sources:
    if source['source_name'] == '3FGL J2254.0+1608':
        reference_gamma_source = source
differential_flux_gamma_source_per_m2_per_GeV = cosmic_fluxes.flux_of_fermi_source(
    fermi_source=reference_gamma_source,
    energy=energy_bin_edges)


def make_trigger_mask(trigger_table, threshold):
    TRIGGER_THRESHOLD_2OVER0_RATIO = 1.06
    tt = trigger_table

    t2_high = tt['refocus_2_respnse_pe'] >= threshold
    t0_low = tt['refocus_0_respnse_pe'] < (tt['refocus_2_respnse_pe']/TRIGGER_THRESHOLD_2OVER0_RATIO)
    return np.logical_and(t2_high, t0_low)
    #return t2_high


geomagnetic_cutoff_fraction = 0.05
# geomagnetic cutoff
# ------------------
airshower_rates = {}
for p in cosmic_rays:
    airshower_rates[p] = cosmic_rays[p]
    below_cutoff = energy_bin_edges < 10.0
    airshower_rates[p]['differential_flux'][below_cutoff] = (
        geomagnetic_cutoff_fraction*
        airshower_rates[p]['differential_flux'][below_cutoff]
    )


EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION


NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2

# TRIGGER_THRESHOLD_2 = irf_config['config']['sum_trigger']['patch_threshold']
TRIGGER_THRESHOLD = 103
ONREGION_RADIUS_DEG = 0.8

ONREGION_SOLID_ANGLE_SR = irf.map_and_reduce._cone_solid_angle(
    cone_radial_opening_angle_rad=np.deg2rad(ONREGION_RADIUS_DEG))
FIELD_OF_VIEW_SOLID_ANGLE_SR = irf.map_and_reduce._cone_solid_angle(
    cone_radial_opening_angle_rad=np.deg2rad(3.25))
solid_angle_ratio_on_region = ONREGION_SOLID_ANGLE_SR/FIELD_OF_VIEW_SOLID_ANGLE_SR


trigger_thresholds_pe = np.arange(
    start=85,
    stop=TRIGGER_THRESHOLD+20,
    step=1)
NUM_THRESHOLDS = trigger_thresholds_pe.shape[0]


fig = plt.figure(figsize=(16, 9), dpi=100)
ax = fig.add_axes((.1, .1, .8, .8))
for particle_key in airshower_rates:
    ax.plot(
        energy_bin_edges,
        airshower_rates[particle_key]['differential_flux'],
        label=particle_key)
ax.set_xlabel('energy / GeV')
ax.set_ylabel('differential flux of airshowers / m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.loglog()
fig.savefig(
    os.path.join(
        pa['summary_dir'],
        'airshower_differential_flux.png'
    )
)
plt.close(fig)

channels = {}

for site_key in irf_config['config']['sites']:

    # read all tables
    # ----------------
    tables = {}
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)
        tables[particle_key] = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray(
                {spt.IDX: event_table['trigger'][spt.IDX]}),
            level_keys=['primary', 'cherenkovsize', 'grid', 'core', 'trigger'])

    # split nsb and cosmic-rays
    # -------------------------
    nsb_table = {}
    cosmic_table = {}
    for particle_key in irf_config['config']['particles']:
        mask_nsb = (
            tables[particle_key]['trigger']['num_cherenkov_pe'] <=
            MAX_CHERENKOV_IN_NSB_PE
        )
        mask_cosmic_rays = np.logical_not(mask_nsb)

        nsb_table[particle_key] = {}
        cosmic_table[particle_key] = {}
        for level in tables[particle_key]:
            nsb_table[particle_key][level] = tables[particle_key][level][mask_nsb]
            cosmic_table[particle_key][level] = tables[particle_key][level][mask_cosmic_rays]


    # nsb-channel
    # -----------
    num_nsb_triggers = np.zeros(NUM_THRESHOLDS, dtype=np.int)
    nsb_exposure_time = 0.0

    for particle_key in irf_config['config']['particles']:
        nsb_exposure_time += (
            nsb_table[particle_key]['primary'].shape[0]*
            EXPOSURE_TIME_PER_EVENT
        )
        for t in range(NUM_THRESHOLDS):
            nsb_trigger_mask = make_trigger_mask(
                trigger_table=nsb_table[particle_key]['trigger'],
                threshold=trigger_thresholds_pe[t],
            ).astype(np.int)
            num_nsb_triggers[t] += np.sum(nsb_trigger_mask)

    channels['nsb'] = {}
    channels['nsb']['rate'] = num_nsb_triggers/nsb_exposure_time


    # trigger area eff gamma
    # ----------------------
    cut_idxs_primary = cosmic_table['gamma']['primary'][spt.IDX]
    cut_idxs_grid = cosmic_table['gamma']['grid'][spt.IDX]
    cut_idxs_trigger = cosmic_table['gamma']['trigger'][spt.IDX]

    _off_deg = mdfl.discovery._angle_between_az_zd_deg(
        az1_deg=np.rad2deg(cosmic_table['gamma']['primary']['azimuth_rad']),
        zd1_deg=np.rad2deg(cosmic_table['gamma']['primary']['zenith_rad']),
        az2_deg=irf_config['config']['plenoscope_pointing']['azimuth_deg'],
        zd2_deg=irf_config['config']['plenoscope_pointing']['zenith_deg'])
    _off_mask = (_off_deg <= 3.25 - 1.0)

    cut_idxs_in_possible_on_region = (
        cosmic_table['gamma']['primary'][spt.IDX][_off_mask]
    )

    cut_idx_detected = spt.intersection([
        cut_idxs_primary,
        cut_idxs_grid,
        cut_idxs_trigger,
        cut_idxs_in_possible_on_region])

    gamma_table = mrg_pri_grd = spt.cut_table_on_indices(
            table=cosmic_table['gamma'],
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray({spt.IDX: cut_idx_detected}),
            level_keys=['primary', 'grid', 'trigger'])

    gamma_f_detected = make_trigger_mask(
        trigger_table=gamma_table['trigger'],
        threshold=TRIGGER_THRESHOLD
    ).astype(np.int)

    gamma_energies = gamma_table['primary']['energy_GeV']

    gamma_q_max = gamma_table['grid']['area_thrown_m2']
    gamma_w_grid_trials = np.ones(gamma_energies.shape[0])*NUM_GRID_BINS
    gamma_w_grid_intense = gamma_table['grid']['num_bins_above_threshold']

    gamma_q_detected = np.histogram(
        gamma_energies,
        weights=gamma_q_max*gamma_f_detected*gamma_w_grid_intense,
        bins=energy_bin_edges)[0]

    gamma_c_thrown = np.histogram(
        gamma_energies,
        weights=gamma_w_grid_trials,
        bins=energy_bin_edges)[0]

    gamma_q_effective = gamma_q_detected/gamma_c_thrown
    gamma_inan = np.isnan(gamma_q_effective)
    gamma_q_effective[gamma_inan] = 0.0

    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes((.1, .1, .8, .8))
    ax.plot(
        energy_bin_edges[:-1],
        gamma_q_effective)
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel('area / m^2')
    ax.set_ylim([2e2, 2e6])
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.loglog()
    fig.savefig(
        os.path.join(
            pa['summary_dir'],
            '{:s}_gamma_area.png'.format(site_key)))
    plt.close(fig)

    diff_trigger_rates = differential_flux_gamma_source_per_m2_per_GeV[:-1]*gamma_q_effective
    ON_REGION_CONTAINMENT = 0.68
    diff_trigger_rates *= ON_REGION_CONTAINMENT
    print('gamma-on: ', np.sum(diff_trigger_rates))
    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes((.1, .1, .8, .8))
    ax.plot(
        energy_bin_edges[:-1],
        diff_trigger_rates)
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel('differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
    ax.set_ylim([1e-6, 1e3])
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.loglog()
    fig.savefig(
        os.path.join(
            pa['summary_dir'],
            '{:s}_{:s}_differential_trigger_rate.png'.format(
                site_key,
                'gamma')))
    plt.close(fig)



    # cosmic-ray-channels
    # -------------------

    for particle_key in airshower_rates:

        num_airshower = cosmic_table[particle_key]['primary'].shape[0]

        energies = cosmic_table[particle_key]['primary']['energy_GeV']

        q_max = (
            cosmic_table[particle_key]['primary']['solid_angle_thrown_sr']*
            cosmic_table[particle_key]['grid']['area_thrown_m2']
        )

        w_grid_trials = np.ones(num_airshower)*NUM_GRID_BINS
        w_grid_intense = cosmic_table[particle_key]['grid']['num_bins_above_threshold']

        integrated_rates = np.zeros(NUM_THRESHOLDS)
        for t in range(NUM_THRESHOLDS):

            f_detected = make_trigger_mask(
                trigger_table=cosmic_table[particle_key]['trigger'],
                threshold=trigger_thresholds_pe[t]
            ).astype(np.int)

            q_detected = np.histogram(
                energies,
                weights=q_max*f_detected*w_grid_intense,
                bins=energy_bin_edges)[0]

            c_thrown = np.histogram(
                energies,
                weights=w_grid_trials,
                bins=energy_bin_edges)[0]

            q_effective = q_detected/c_thrown
            inan = np.isnan(q_effective)
            q_effective[inan] = 0.0

            diff_trigger_rates = (
                q_effective*
                airshower_rates[particle_key]['differential_flux'][:-1]
            )

            trigger_rates = diff_trigger_rates*energy_bin_width[:-1]

            integrated_rates[t] = np.sum(trigger_rates)

            if trigger_thresholds_pe[t] == TRIGGER_THRESHOLD:
                emask = np.logical_and(energies >= 5., energies < 10.)
                cell_ratio = np.sum(w_grid_intense[emask])/np.sum(w_grid_trials[emask])
                trg_ratio = np.sum(f_detected[emask])/f_detected[emask].shape[0]
                print(
                    site_key,
                    particle_key,
                    "thr", trigger_thresholds_pe[t],
                    "cell. ratio {:.3f}u".format(1e6*cell_ratio),
                    "trg. ratio {:.3f}".format(trg_ratio),
                    trg_ratio*cell_ratio*1e6
                )

                fig = plt.figure(figsize=(16, 9), dpi=100)
                ax = fig.add_axes((.1, .1, .8, .8))
                ax.plot(
                    energy_bin_edges[:-1],
                    q_effective)
                ax.set_xlabel('energy / GeV')
                ax.set_ylabel('acceptance / m$^2$ sr')
                ax.set_ylim([1e0, 1e5])
                ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
                ax.loglog()
                fig.savefig(
                    os.path.join(
                        pa['summary_dir'],
                        '{:s}_{:s}_acceptance.png'.format(
                            site_key,
                            particle_key)))
                plt.close(fig)

                print(
                    '{:s}-on: '.format(particle_key),
                    np.sum(diff_trigger_rates*solid_angle_ratio_on_region)
                )
                fig = plt.figure(figsize=(16, 9), dpi=100)
                ax = fig.add_axes((.1, .1, .8, .8))
                ax.plot(
                    energy_bin_edges[:-1],
                    diff_trigger_rates*solid_angle_ratio_on_region)
                ax.set_xlabel('energy / GeV')
                ax.set_ylabel('differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
                ax.set_ylim([1e-6, 1e3])
                ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
                ax.loglog()
                fig.savefig(
                    os.path.join(
                        pa['summary_dir'],
                        '{:s}_{:s}_differential_trigger_rate.png'.format(
                            site_key,
                            particle_key)))
                plt.close(fig)

        channels[particle_key] = {}
        channels[particle_key]['rate'] = integrated_rates


    fig = plt.figure(figsize=(16, 9), dpi=100)
    ax = fig.add_axes((.1, .1, .8, .8))
    ax.plot(
        trigger_thresholds_pe,
        channels['nsb']['rate'] +
        channels['electron']['rate'] +
        channels['proton']['rate'] +
        channels['helium']['rate'],
        'k',
        label='night-sky + cosmic-rays')
    ax.plot(
        trigger_thresholds_pe,
        channels['nsb']['rate'],
        'k:',
        label='night-sky')

    ax.plot(
        trigger_thresholds_pe,
        channels['proton']['rate'],
        color='r',
        label='proton')
    ax.plot(
        trigger_thresholds_pe,
        channels['electron']['rate'],
        color='b',
        label='electron')
    ax.plot(
        trigger_thresholds_pe,
        channels['helium']['rate'],
        color='orange',
        label='helium')

    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel(r'trigger-rate / s$^{-1}$')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    ax.axvline(
        x=TRIGGER_THRESHOLD,
        color='k',
        linestyle='-',
        alpha=0.25)
    fig.savefig(
        os.path.join(pa['summary_dir'], 'ratescan_{:s}.png'.format(site_key))
    )
    plt.close(fig)
