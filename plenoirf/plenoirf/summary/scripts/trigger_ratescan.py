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

trigger_thresholds = np.array(sum_config['trigger']['ratescan_thresholds_pe'])
collection_trigger_threshold = irf_config['config']['sum_trigger']['threshold_pe']
analysis_trigger_threshold = sum_config['trigger']['threshold_pe']
num_trigger_thresholds = len(trigger_thresholds)

energy_lower = sum_config['energy_binning']['lower_edge_GeV']
energy_upper = sum_config['energy_binning']['upper_edge_GeV']

energy_bin_edges = np.array(acceptance['energy_bin_edges']['value'])
energy_bin_centers = irf.summary.bin_centers(energy_bin_edges)

fine_energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins_fine'] + 1
)

fine_energy_bin_width = irf.summary.bin_width(fine_energy_bin_edges)
fine_energy_bin_centers = irf.summary.bin_centers(fine_energy_bin_edges)

# cosmic-ray-flux
# ----------------
airshower_fluxes = irf.summary.read_airshower_differential_flux(
    summary_dir=pa['summary_dir'],
    energy_bin_centers=fine_energy_bin_centers,
    sites=irf_config['config']['sites'],
    geomagnetic_cutoff_fraction=sum_config[
        'airshower_flux'][
        'fraction_of_flux_below_geomagnetic_cutoff'],
)
_cosmic_ray_colors = {
    "proton": "red",
    "helium": "orange",
    "electron": "blue",
}

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

for site_key in irf_config['config']['sites']:
    fig = irf.summary.figure.figure(sum_config['figure_16_9'])
    ax = fig.add_axes((.1, .1, .8, .8))
    for particle_key in airshower_fluxes[site_key]:
        ax.plot(
            fine_energy_bin_centers,
            airshower_fluxes[site_key][particle_key]['differential_flux'],
            label=particle_key,
            color=_cosmic_ray_colors[particle_key],
        )
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel(
        'differential flux of airshowers / ' +
        'm$^{-2}$ s$^{-1}$ sr$^{-1}$ (GeV)$^{-1}$'
    )
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.loglog()
    ax.set_xlim([energy_lower, energy_upper])
    ax.legend()
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_airshower_differential_flux.jpg'.format(site_key)
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
    trigger_rates[site_key]['gamma'] = np.zeros(num_trigger_thresholds)

    _gamma_effective_area_m2 = np.array(acceptance[
        'cosmic_response'][
        site_key][
        'gamma'][
        'point'][
        'value']
    )
    for tt in range(num_trigger_thresholds):
        gamma_effective_area_m2 = np.interp(
            x=fine_energy_bin_centers,
            xp=energy_bin_centers,
            fp=_gamma_effective_area_m2[tt, :]
        )
        gamma_dT_per_s_per_GeV = (
            gamma_dF_per_m2_per_s_per_GeV *
            gamma_effective_area_m2
        )
        gamma_T_per_s = np.sum(gamma_dT_per_s_per_GeV*fine_energy_bin_width)
        trigger_rates[site_key]['gamma'][tt] = gamma_T_per_s

        fig = irf.summary.figure.figure(sum_config['figure_16_9'])
        ax = fig.add_axes((.1, .1, .8, .8))
        ax.plot(
            fine_energy_bin_centers,
            gamma_dT_per_s_per_GeV,
            color='k'
        )
        ax.set_title('trigger-threshold: {:d}p.e.'.format(trigger_thresholds[tt]))
        ax.set_xlabel('energy / GeV')
        ax.set_ylabel('differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.loglog()
        ax.set_xlim([energy_lower, energy_upper])
        ax.set_ylim([1e-6, 1e2])
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_{:s}_{:d}_differential_trigger_rate.jpg'.format(
                    site_key,
                    'gamma',
                    tt
                )
            )
        )
        plt.close(fig)

    # cosmic-rays
    # -----------
    for cosmic_key in airshower_fluxes[site_key]:
        trigger_rates[site_key][cosmic_key] = np.zeros(num_trigger_thresholds)

        _cosmic_effective_acceptance_m2_sr = np.array(acceptance[
            'cosmic_response'][
            site_key][
            cosmic_key][
            'diffuse'][
            'value']
        )
        for tt in range(num_trigger_thresholds):

            cosmic_effective_acceptance_m2_sr = np.interp(
                x=fine_energy_bin_centers,
                xp=energy_bin_centers,
                fp=_cosmic_effective_acceptance_m2_sr[tt, :]
            )
            cosmic_dT_per_s_per_GeV = (
                cosmic_effective_acceptance_m2_sr *
                airshower_fluxes[site_key][cosmic_key]['differential_flux']
            )
            cosmic_T_per_s = cosmic_dT_per_s_per_GeV*fine_energy_bin_width
            trigger_rates[site_key][cosmic_key][tt] = np.sum(cosmic_T_per_s)


with open(os.path.join(pa['out_dir'], 'trigger_rates.json'), 'wt') as f:
    f.write(json.dumps(trigger_rates, cls=irf.json_numpy.Encoder))


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

    for cosmic_key in airshower_fluxes[site_key]:
        ax.plot(
            trigger_thresholds,
            tr[cosmic_key],
            color=_cosmic_ray_colors[cosmic_key],
            label=cosmic_key
        )

    ax.semilogy()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel('trigger-threshold / photo-electrons')
    ax.set_ylabel('trigger-rate / s$^{-1}$')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.legend(loc='best', fontsize=10)
    ax.axvline(
        x=analysis_trigger_threshold,
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
