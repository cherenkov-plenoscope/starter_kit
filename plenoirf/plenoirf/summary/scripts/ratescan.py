#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_table as spt
import cosmic_fluxes
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

acceptance_trigger_path = os.path.join(
    pa['summary_dir'],
    "acceptance_trigger",
    "acceptance_trigger.json"
)
with open(acceptance_trigger_path, 'rt') as f:
    acceptance = json.loads(f.read())

trigger_thresholds = np.array(sum_config['trigger_thresholds_pe'])
nominal_trigger_threshold_idx = sum_config['nominal_trigger_threshold_idx']
nominal_trigger_threshold = trigger_thresholds[nominal_trigger_threshold_idx]
NUM_THRESHOLDS = len(trigger_thresholds)

ON_REGION_CONTAINMENT = 0.68

energy_bin_edges = np.array(acceptance['energy_bin_edges']['value'])
ENERGY_MIN = np.min(energy_bin_edges)
ENERGY_MAX = np.max(energy_bin_edges)
NUM_COARSE_ENERGY_BINS = energy_bin_edges.shape[0] - 1

_weight_lower_edge = 0.5
_weight_upper_edge = 1.0 - _weight_lower_edge

energy_bin_width = energy_bin_edges[1:] - energy_bin_edges[:-1]
energy_bin_centers = (
    _weight_lower_edge*energy_bin_edges[:-1] +
    _weight_upper_edge*energy_bin_edges[1:]
)

NUM_FINE_ENERGY_BINS = 1337
fine_energy_bin_edges = np.geomspace(
    ENERGY_MIN,
    ENERGY_MAX,
    NUM_FINE_ENERGY_BINS + 1
)

fine_energy_bin_width = fine_energy_bin_edges[1:] - fine_energy_bin_edges[:-1]
fine_energy_bin_centers = (
    _weight_lower_edge*fine_energy_bin_edges[:-1] +
    _weight_upper_edge*fine_energy_bin_edges[1:]
)

# cosmic-ray-flux
# ----------------
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
geomagnetic_cutoff_fraction = 0.05
below_cutoff = fine_energy_bin_centers < 10.0
airshower_rates = {}
for p in cosmic_rays:
    airshower_rates[p] = cosmic_rays[p]
    airshower_rates[p]['differential_flux'][below_cutoff] = (
        geomagnetic_cutoff_fraction*
        airshower_rates[p]['differential_flux'][below_cutoff]
    )

# gamma-ray-flux
# ---------------
with open(os.path.join(pa['summary_dir'], "gamma_sources.json"), "rt") as f:
    gamma_sources = json.loads(f.read())
for source in gamma_sources:
    if source['source_name'] == '3FGL J2254.0+1608':
        reference_gamma_source = source
gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
    fermi_source=reference_gamma_source,
    energy=fine_energy_bin_centers
)


fig = irf.summary.figure.figure(sum_config['figure_16_9'])
ax = fig.add_axes((.1, .1, .8, .8))
for particle_key in airshower_rates:
    ax.plot(
        fine_energy_bin_centers,
        airshower_rates[particle_key]['differential_flux'],
        label=particle_key,
        color=cosmic_rays[particle_key]['color'],
    )
ax.set_xlabel('energy / GeV')
ax.set_ylabel(
    'differential flux of airshowers / '+
    'm$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$'
)
ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
ax.loglog()
ax.set_xlim([ENERGY_MIN, ENERGY_MAX])
ax.legend()
fig.savefig(
    os.path.join(
        pa['out_dir'],
        'airshower_differential_flux.jpg'
    )
)
plt.close(fig)

trigger_rates = {}

for site_key in irf_config['config']['sites']:
    trigger_rates[site_key] = {}

    # night-sky-background
    # --------------------

    trigger_rates[site_key]['night_sky_background'] = np.array(acceptance[
        'night_sky_background_response'][
        site_key][
        'rate'])

    # gamma-ray
    # ---------
    trigger_rates[site_key]['gamma'] = np.zeros(NUM_THRESHOLDS)

    _gamma_effective_area_m2 = np.array(acceptance[
        'cosmic_response'][
        site_key][
        'gamma'][
        'point'][
        'value']
    )
    for tt in range(NUM_THRESHOLDS):
        gamma_effective_area_m2 = np.interp(
            x=fine_energy_bin_centers,
            xp=energy_bin_centers,
            fp=_gamma_effective_area_m2[tt, :]
        )
        gamma_dT_per_s_per_GeV = (
            gamma_dF_per_m2_per_s_per_GeV*
            gamma_effective_area_m2
        )
        gamma_T_per_s = np.sum(gamma_dT_per_s_per_GeV*fine_energy_bin_width)
        trigger_rates[site_key]['gamma'][tt] = gamma_T_per_s


        fig = irf.summary.figure.figure(sum_config['figure_16_9'])
        ax = fig.add_axes((.1, .1, .8, .8))
        ax.plot(
            fine_energy_bin_centers,
            gamma_effective_area_m2,
            'k'
        )
        ax.loglog()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('energy / GeV')
        ax.set_ylabel('area / m2')
        ax.set_ylim([1e2, 1e6])
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_gamma_{:d}.jpg'.format(site_key, tt)
            )
        )
        plt.close(fig)
        print(tt, trigger_thresholds[tt])



    # cosmic-rays
    # -----------
    for cosmic_key in cosmic_rays:
        trigger_rates[site_key][cosmic_key] = np.zeros(NUM_THRESHOLDS)

        _cosmic_effective_acceptance_m2_sr = np.array(acceptance[
            'cosmic_response'][
            site_key][
            cosmic_key][
            'diffuse'][
            'value']
        )
        for tt in range(NUM_THRESHOLDS):

            cosmic_effective_acceptance_m2_sr = np.interp(
                x=fine_energy_bin_centers,
                xp=energy_bin_centers,
                fp=_cosmic_effective_acceptance_m2_sr[tt, :]
            )
            cosmic_dT_per_s_per_GeV = (
                cosmic_effective_acceptance_m2_sr*
                airshower_rates[particle_key]['differential_flux']
            )
            cosmic_T_per_s = cosmic_dT_per_s_per_GeV*fine_energy_bin_width
            trigger_rates[site_key][cosmic_key][tt] = np.sum(cosmic_T_per_s)

            fig = irf.summary.figure.figure(sum_config['figure_16_9'])
            ax = fig.add_axes((.1, .1, .8, .8))
            ax.plot(
                fine_energy_bin_centers,
                cosmic_effective_acceptance_m2_sr,
                'k'
            )
            ax.loglog()
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.set_xlabel('energy / GeV')
            ax.set_ylabel('acceptance / m2 sr')
            ax.set_ylim([1e-1, 1e5])
            ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
            fig.savefig(
                os.path.join(
                    pa['out_dir'],
                    '{:s}_{:s}_{:d}.jpg'.format(site_key, cosmic_key, tt)
                )
            )
            plt.close(fig)
            print(tt, trigger_thresholds[tt])



for site_key in trigger_rates:
    for particle_key in trigger_rates[site_key]:
        trigger_rates[site_key][particle_key] = (
            trigger_rates[site_key][particle_key].tolist()
        )
with open(os.path.join(pa['out_dir'], 'trigger_rates.json'), 'wt') as f:
    f.write(json.dumps(trigger_rates))
for site_key in trigger_rates:
    for particle_key in trigger_rates[site_key]:
        trigger_rates[site_key][particle_key] = np.array(
            trigger_rates[site_key][particle_key]
        )

for site_key in irf_config['config']['sites']:
    tr = trigger_rates[site_key]

    fig = irf.summary.figure.figure(sum_config['figure_16_9'])
    ax = fig.add_axes((.1, .1, .8, .8))
    ax.plot(
        trigger_thresholds,
        tr['night_sky_background'] +
        tr['electron'] +
        tr['proton'] +
        tr['helium'],
        'k',
        label='night-sky + cosmic-rays')
    ax.plot(
        trigger_thresholds,
        tr['night_sky_background'],
        'k:',
        label='night-sky')

    for cosmic_key in cosmic_rays:
        ax.plot(
            trigger_thresholds,
            tr[cosmic_key],
            color=cosmic_rays[cosmic_key]['color'],
            label=cosmic_key
        )

    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel(r'trigger-rate / s$^{-1}$')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    ax.axvline(
        x=nominal_trigger_threshold,
        color='k',
        linestyle='-',
        alpha=0.25)
    ax.set_ylim([1e0, 1e7])
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_ratescan.jpg'.format(site_key)
        )
    )
    plt.close(fig)