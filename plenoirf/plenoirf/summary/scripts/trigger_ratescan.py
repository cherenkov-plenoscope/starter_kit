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

PLOT_FIGSIZE = (16/2, 9/2)
PLOT_DPI = 200
PLOT_ENERGY_MIN = 1e-0
PLOT_ENERGY_MAX = 1e3


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

out_dir = os.path.join(pa['summary_dir'], 'trigger_rate_scan')
os.makedirs(out_dir, exist_ok=True)

ON_REGION_CONTAINMENT = 0.68

MAX_CHERENKOV_IN_NSB_PE = 0
TIME_SLICE_DURATION = 0.5e-9
NUM_TIME_SLICES_PER_EVENT = (
    100 -
    irf_config['config']['sum_trigger']['integration_time_slices'])

ANALYSIS_ENERGY_MIN = PLOT_ENERGY_MIN
ANALYSIS_ENERGY_MAX = PLOT_ENERGY_MAX

_weight_lower_edge = 0.0
_weight_upper_edge = 1.0 - _weight_lower_edge

NUM_COARSE_ENERGY_BINS = 24
coarse_energy_bin_edges = np.geomspace(
    ANALYSIS_ENERGY_MIN,
    ANALYSIS_ENERGY_MAX,
    NUM_COARSE_ENERGY_BINS + 1)
coarse_energy_bin_width = coarse_energy_bin_edges[1:] - coarse_energy_bin_edges[:-1]
coarse_energy_bin_centers = (
    _weight_lower_edge*coarse_energy_bin_edges[:-1] +
    _weight_upper_edge*coarse_energy_bin_edges[1:]
)

NUM_FINE_ENERGY_BINS = 1337
fine_energy_bin_edges = np.geomspace(
    ANALYSIS_ENERGY_MIN,
    ANALYSIS_ENERGY_MAX,
    NUM_FINE_ENERGY_BINS + 1)

fine_energy_bin_width = fine_energy_bin_edges[1:] - fine_energy_bin_edges[:-1]
fine_energy_bin_centers = (
    _weight_lower_edge*fine_energy_bin_edges[:-1] +
    _weight_upper_edge*fine_energy_bin_edges[1:]
)

cosmic_rays = {
    "proton": {"color": "red"},
    "helium": {"color": "orange"},
    "electron": {"color": "blue"}
}

_cosmic_ray_raw_fluxes = {}
with open(os.path.join(pa['summary_dir'], "proton_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["proton"] = json.loads(f.read())
with open(os.path.join(pa['summary_dir'], "helium_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["helium"] = json.loads(f.read())
with open(os.path.join(pa['summary_dir'], "electron_positron_flux.json"), "rt") as f:
    _cosmic_ray_raw_fluxes["electron"] = json.loads(f.read())
for p in cosmic_rays:
    cosmic_rays[p]['differential_flux'] = np.interp(
        x=fine_energy_bin_centers,
        xp=_cosmic_ray_raw_fluxes[p]['energy']['values'],
        fp=_cosmic_ray_raw_fluxes[p]['differential_flux']['values']
    )

# geomagnetic cutoff
# ------------------
geomagnetic_cutoff_fraction = 0.05
below_cutoff = fine_energy_bin_centers < 10.0
airshower_rates = {}
for p in cosmic_rays:
    airshower_rates[p] = cosmic_rays[p]
    airshower_rates[p]['differential_flux'][below_cutoff] = (
        geomagnetic_cutoff_fraction*
        airshower_rates[p]['differential_flux'][below_cutoff]
    )

fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
ax = fig.add_axes((.1, .1, .8, .8))
for particle_key in airshower_rates:
    ax.plot(
        fine_energy_bin_centers,
        airshower_rates[particle_key]['differential_flux'],
        label=particle_key,
        color=cosmic_rays[particle_key]['color'],
    )
ax.set_xlabel('energy / GeV')
ax.set_ylabel('differential flux of airshowers / m$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.loglog()
ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
ax.legend()
fig.savefig(
    os.path.join(
        out_dir,
        'airshower_differential_flux.png'
    )
)
plt.close(fig)


with open(os.path.join(pa['summary_dir'], "gamma_sources.json"), "rt") as f:
    gamma_sources = json.loads(f.read())

for source in gamma_sources:
    if source['source_name'] == '3FGL J2254.0+1608':
        reference_gamma_source = source

gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
    fermi_source=reference_gamma_source,
    energy=fine_energy_bin_centers
)

fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
ax = fig.add_axes((.1, .1, .8, .8))
ax.plot(
    fine_energy_bin_centers,
    gamma_dF_per_m2_per_s_per_GeV,
    'k'
)
ax.set_xlabel('energy / GeV')
ax.set_ylabel('differential flux of gamma-rays / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$')
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.loglog()
ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
fig.savefig(
    os.path.join(
        out_dir,
        'gamma_ray_flux.png'
    )
)
plt.close(fig)


def make_trigger_mask_telescope(trigger_table, threshold):
    on_level = 7
    KEY = 'focus_{:02d}_response_pe'
    on_key = KEY.format(on_level)
    on_mask = trigger_table[on_key] >= threshold
    return on_mask

def make_trigger_mask_plenoscope(trigger_table, threshold):
    FOCUS_RATIO = 1.06
    tt = trigger_table

    on_level = 7
    veto_level = 4

    KEY = 'focus_{:02d}_response_pe'
    on_key = KEY.format(on_level)
    veto_key = KEY.format(veto_level)

    on_mask = tt[on_key] >= threshold
    veto_mask = tt[veto_key] < (tt[on_key]/FOCUS_RATIO)

    return veto_mask*on_mask


make_trigger_mask = make_trigger_mask_telescope


trigger_config = {
    "chile": {"threshold": 60, "on_focus": 5, "veto_focus": 3},
    "namibia": {"threshold": 60, "on_focus": 5, "veto_focus": 3},
}

EXPOSURE_TIME_PER_EVENT = NUM_TIME_SLICES_PER_EVENT*TIME_SLICE_DURATION

NUM_GRID_BINS = irf_config['grid_geometry']['num_bins_diameter']**2

ONREGION_RADIUS_DEG = 0.8

ONREGION_SOLID_ANGLE_SR = irf.map_and_reduce._cone_solid_angle(
    cone_radial_opening_angle_rad=np.deg2rad(ONREGION_RADIUS_DEG)
)
FIELD_OF_VIEW_SOLID_ANGLE_SR = irf.map_and_reduce._cone_solid_angle(
    cone_radial_opening_angle_rad=np.deg2rad(3.25)
)
SOLID_ANGLE_RATIO_ON_REGION = (
    ONREGION_SOLID_ANGLE_SR/
    FIELD_OF_VIEW_SOLID_ANGLE_SR
)

trigger_thresholds_pe = np.arange(
    start=20,
    stop=140,
    step=1
)

channels = {}

for site_key in irf_config['config']['sites']:

    NUM_THRESHOLDS = trigger_thresholds_pe.shape[0]

    print('site', site_key)
    print('-----------------')
    # read all tables
    # ----------------
    cosmic_table = {}
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)
        cosmic_table[particle_key] = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray(
                {spt.IDX: event_table['trigger'][spt.IDX]}),
            level_keys=['primary', 'cherenkovsize', 'grid', 'core', 'trigger'])

    # split nsb and cosmic-rays
    # -------------------------
    nsb_table = {}
    for particle_key in irf_config['config']['particles']:
        mask_nsb = (
            cosmic_table[particle_key]['trigger']['num_cherenkov_pe'] <=
            MAX_CHERENKOV_IN_NSB_PE
        )

        nsb_table[particle_key] = {}
        for level in cosmic_table[particle_key]:
            nsb_table[particle_key][level] = cosmic_table[particle_key][level][mask_nsb]

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


    channels['gamma'] = {}
    channels['gamma']['rate'] = []
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
    _fov_radius_deg = 0.5*irf_config[
        'light_field_sensor_geometry']['max_FoV_diameter_deg']
    _off_mask = (_off_deg <= _fov_radius_deg - 1.0)

    cut_idxs_comming_from_possible_on_region = (
        cosmic_table['gamma']['primary'][spt.IDX][_off_mask]
    )

    cut_idx_detected = spt.intersection([
        cut_idxs_primary,
        cut_idxs_grid,
        cut_idxs_trigger,
        cut_idxs_comming_from_possible_on_region
    ])

    gamma_table = mrg_pri_grd = spt.cut_table_on_indices(
        table=cosmic_table['gamma'],
        structure=irf.table.STRUCTURE,
        common_indices=spt.dict_to_recarray({spt.IDX: cut_idx_detected}),
        level_keys=['primary', 'grid', 'trigger']
    )

    for t in range(NUM_THRESHOLDS):

        gamma_f_detected = make_trigger_mask(
            trigger_table=gamma_table['trigger'],
            threshold=trigger_thresholds_pe[t]
        ).astype(np.int)

        gamma_energies = gamma_table['primary']['energy_GeV']

        gamma_q_max = gamma_table['grid']['area_thrown_m2']
        gamma_w_grid_trials = np.ones(gamma_energies.shape[0])*NUM_GRID_BINS
        gamma_w_grid_intense = gamma_table['grid']['num_bins_above_threshold']

        gamma_q_detected = np.histogram(
            gamma_energies,
            weights=gamma_q_max*gamma_f_detected*gamma_w_grid_intense,
            bins=coarse_energy_bin_edges)[0]

        gamma_c_thrown = np.histogram(
            gamma_energies,
            weights=gamma_w_grid_trials,
            bins=coarse_energy_bin_edges)[0]

        gamma_c_thrown_valid = gamma_c_thrown > 0
        coarse_gamma_effective_area_m2 = np.zeros(NUM_COARSE_ENERGY_BINS)
        coarse_gamma_effective_area_m2[gamma_c_thrown_valid] = (
            gamma_q_detected[gamma_c_thrown_valid]/
            gamma_c_thrown[gamma_c_thrown_valid]
        )

        gamma_effective_area_m2 = np.interp(
            x=fine_energy_bin_centers,
            xp=coarse_energy_bin_centers,
            fp=coarse_gamma_effective_area_m2
        )

        gamma_dT_per_s_per_GeV = (
            gamma_dF_per_m2_per_s_per_GeV*
            gamma_effective_area_m2
        )

        gamma_dT_onregion_per_s_per_GeV = (
            gamma_dT_per_s_per_GeV*
            ON_REGION_CONTAINMENT
        )

        gamma_T_onregion_per_s = (
            gamma_dT_onregion_per_s_per_GeV*
            fine_energy_bin_width
        )

        channels['gamma']['rate'].append(np.sum(gamma_T_onregion_per_s))

        if trigger_thresholds_pe[t] == trigger_config[site_key]['threshold']:
            fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
            ax = fig.add_axes((.1, .1, .8, .8))
            ax.plot(
                fine_energy_bin_centers,
                gamma_effective_area_m2,
                color='k',
            )
            ax.set_xlabel('energy / GeV')
            ax.set_ylabel('area / m${^2}$')
            ax.set_ylim([2e2, 2e6])
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.loglog()
            ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
            fig.savefig(
                os.path.join(
                    out_dir,
                    '{:s}_gamma_area.png'.format(site_key)))
            plt.close(fig)

            print('gamma-on: ', np.sum(gamma_T_onregion_per_s))
            fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
            ax = fig.add_axes((.1, .1, .8, .8))
            ax.plot(
                fine_energy_bin_centers,
                gamma_dT_onregion_per_s_per_GeV,
                color='k',
            )
            ax.set_xlabel('energy / GeV')
            ax.set_ylabel('on-region differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
            ax.set_ylim([1e-6, 1e3])
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            ax.loglog()
            ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
            fig.savefig(
                os.path.join(
                    out_dir,
                    '{:s}_{:s}_differential_trigger_rate.png'.format(
                        site_key,
                        'gamma')))
            plt.close(fig)

    channels['gamma']['rate'] = np.array(channels['gamma']['rate'])

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
                bins=coarse_energy_bin_edges)[0]

            c_thrown = np.histogram(
                energies,
                weights=w_grid_trials,
                bins=coarse_energy_bin_edges)[0]

            c_thrown_valid = c_thrown > 0
            coarse_cosmic_effective_acceptance_m2_sr = np.zeros(shape=c_thrown.shape)
            coarse_cosmic_effective_acceptance_m2_sr[c_thrown_valid] = (
                q_detected[c_thrown_valid]/c_thrown[c_thrown_valid]
            )

            cosmic_effective_acceptance_m2_sr = np.interp(
                x=fine_energy_bin_centers,
                xp=coarse_energy_bin_centers,
                fp=coarse_cosmic_effective_acceptance_m2_sr
            )

            cosmic_dT_per_s_per_GeV = (
                cosmic_effective_acceptance_m2_sr*
                airshower_rates[particle_key]['differential_flux']
            )

            cosmic_dT_onregion_per_s_per_GeV = (
                SOLID_ANGLE_RATIO_ON_REGION*
                cosmic_dT_per_s_per_GeV
            )

            cosmic_T_per_s = cosmic_dT_per_s_per_GeV*fine_energy_bin_width

            integrated_rates[t] = np.sum(cosmic_T_per_s)

            if trigger_thresholds_pe[t] == trigger_config[site_key]['threshold']:

                fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
                ax = fig.add_axes((.1, .1, .8, .8))
                ax.plot(
                    fine_energy_bin_centers,
                    cosmic_effective_acceptance_m2_sr,
                    color='k',
                )
                ax.set_xlabel('energy / GeV')
                ax.set_ylabel('acceptance / m$^2$ sr')
                ax.set_ylim([1e0, 1e5])
                ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
                ax.loglog()
                ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
                fig.savefig(
                    os.path.join(
                        out_dir,
                        '{:s}_{:s}_acceptance.png'.format(
                            site_key,
                            particle_key
                        )
                    )
                )
                plt.close(fig)

                print(
                    '{:s}-on: '.format(particle_key),
                    integrated_rates[t]*SOLID_ANGLE_RATIO_ON_REGION
                )
                fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
                ax = fig.add_axes((.1, .1, .8, .8))
                ax.plot(
                    fine_energy_bin_centers,
                    cosmic_dT_onregion_per_s_per_GeV,
                    color='k',
                )
                ax.set_xlabel('energy / GeV')
                ax.set_ylabel('on-region differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
                ax.set_ylim([1e-6, 1e3])
                ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
                ax.loglog()
                ax.set_xlim([PLOT_ENERGY_MIN, PLOT_ENERGY_MAX])
                fig.savefig(
                    os.path.join(
                        out_dir,
                        '{:s}_{:s}_differential_trigger_rate.png'.format(
                            site_key,
                            particle_key
                        )
                    )
                )
                plt.close(fig)

        channels[particle_key] = {}
        channels[particle_key]['rate'] = integrated_rates


    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
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

    for particle_key in cosmic_rays:
        ax.plot(
            trigger_thresholds_pe,
            channels[particle_key]['rate'],
            color=cosmic_rays[particle_key]['color'],
            label=particle_key)

    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel(r'trigger-rate / s$^{-1}$')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    ax.axvline(
        x=trigger_config[site_key]['threshold'],
        color='k',
        linestyle='-',
        alpha=0.25)
    ax.set_ylim([1e2, 1e7])
    fig.savefig(
        os.path.join(
            out_dir,
            'ratescan_{:s}.png'.format(site_key)
        )
    )
    plt.close(fig)


    fig = plt.figure(figsize=PLOT_FIGSIZE, dpi=PLOT_DPI)
    ax = fig.add_axes((.1, .1, .8, .8))
    signal_vs_threshold = channels['gamma']['rate']
    background_vs_threshold = SOLID_ANGLE_RATIO_ON_REGION*(
        channels['nsb']['rate'] +
        channels['electron']['rate'] +
        channels['proton']['rate'] +
        channels['helium']['rate']
    )
    ax.plot(
        trigger_thresholds_pe,
        signal_vs_threshold/background_vs_threshold,
        'k',
    )
    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel(r'gamma/background / 1')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.axvline(
        x=trigger_config[site_key]['threshold'],
        color='k',
        linestyle='-',
        alpha=0.25
    )
    ax.set_ylim([1e-4, 1e-1])
    fig.savefig(
        os.path.join(
            out_dir,
            'gamma_over_background_{:s}.png'.format(site_key)
        )
    )
    plt.close(fig)