#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import cosmic_fluxes
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)


onregion_acceptance = irf.json_numpy.read_tree(
    os.path.join(
        pa['summary_dir'],
        "0300_onregion_trigger_acceptance"
    )
)
onregion_rates = irf.json_numpy.read_tree(
    os.path.join(
        pa['summary_dir'],
        "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)

energy_lower = sum_config['energy_binning']['lower_edge_GeV']
energy_upper = sum_config['energy_binning']['upper_edge_GeV']
energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config['energy_binning']['num_bins']['trigger_acceptance_onregion'] + 1
)
energy_bin_centers = irf.summary.bin_centers(energy_bin_edges)

fine_energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config['energy_binning']['num_bins']['interpolation'] + 1
)
fine_energy_bin_centers = irf.summary.bin_centers(fine_energy_bin_edges)

detection_threshold_std = sum_config[
    'on_off_measuremnent'][
    'detection_threshold_std']
on_over_off_ratio = sum_config[
    'on_off_measuremnent'][
    'on_over_off_ratio']
observation_time_s = 50*3600
n_points_to_plot = 7

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

cosmic_ray_keys = list(irf_config['config']['particles'].keys())
cosmic_ray_keys.remove('gamma')

fermi_broadband = irf.analysis.fermi_lat_integral_spectral_exclusion_zone()
assert fermi_broadband[
    'energy']['unit_tex'] == "GeV"
assert fermi_broadband[
    'differential_flux']['unit_tex'] == "m$^{-2}$ s$^{-1}$ GeV$^{-1}$"

_, gamma_name = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa['summary_dir'],
    gamma_ray_reference_source=sum_config['gamma_ray_reference_source'],
    energy_supports_GeV=fine_energy_bin_centers,
)

# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

# background rates
# ----------------
cosmic_ray_rate_onregion = {}
electron_rate_onregion = {}
for site_key in irf_config['config']['sites']:

    electron_rate_onregion[site_key] = onregion_rates[
            site_key][
            'electron'][
            'integral_rate'][
            'mean']

    cosmic_ray_rate_onregion[site_key] = 0
    for cosmic_ray_key in cosmic_ray_keys:
        cosmic_ray_rate_onregion[site_key] += onregion_rates[
            site_key][
            cosmic_ray_key][
            'integral_rate'][
            'mean']

for site_key in irf_config['config']['sites']:

    # integral spectral exclusion zone
    # --------------------------------

    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    # Crab reference fluxes
    for i in range(4):
        scale_factor = np.power(10., (-1)*i)
        _energy_GeV = np.array(crab_flux['energy']['values'])
        _flux_per_GeV_per_m2_per_s = np.array(
            crab_flux['differential_flux']['values']
        )
        ax.plot(
            _energy_GeV,
            _flux_per_GeV_per_m2_per_s*scale_factor,
            color='k',
            linestyle='--',
            label='{:.3f} Crab'.format(scale_factor),
            alpha=1./(1.+i)
        )

    ax.plot(
        fermi_broadband['energy']['values'],
        fermi_broadband['differential_flux']['values'],
        'k',
        label='Fermi-LAT 10 y'
    )

    # plenoscope
    gamma_effective_area_m2 = np.array(onregion_acceptance[
        site_key][
        'gamma'][
        'point'][
        'mean'])

    (
        isez_energy_GeV,
        isez_differential_flux_per_GeV_per_m2_per_s
    ) = irf.analysis.estimate_integral_spectral_exclusion_zone(
        gamma_effective_area_m2=gamma_effective_area_m2,
        energy_bin_centers_GeV=energy_bin_centers,
        background_rate_in_onregion_per_s=cosmic_ray_rate_onregion[site_key],
        onregion_over_offregion_ratio=on_over_off_ratio,
        observation_time_s=observation_time_s,
        num_points=n_points_to_plot
    )
    ax.plot(
        isez_energy_GeV,
        isez_differential_flux_per_GeV_per_m2_per_s,
        'r',
        label='Portal {:2.0f} h, trigger'.format(observation_time_s/3600.)
    )

    # plenoscope rejecting all hadrons
    (
        e_isez_energy_GeV,
        e_isez_differential_flux_per_GeV_per_m2_per_s
    ) = irf.analysis.estimate_integral_spectral_exclusion_zone(
        gamma_effective_area_m2=gamma_effective_area_m2,
        energy_bin_centers_GeV=energy_bin_centers,
        background_rate_in_onregion_per_s=electron_rate_onregion[site_key],
        onregion_over_offregion_ratio=on_over_off_ratio,
        observation_time_s=observation_time_s,
        num_points=n_points_to_plot
    )
    ax.plot(
        e_isez_energy_GeV,
        e_isez_differential_flux_per_GeV_per_m2_per_s,
        'r:',
        label='Portal {:2.0f} h, trigger, rejecting all hadrons'.format(
            observation_time_s/3600.
        )
    )

    ax.set_xlim([1e-1, 1e4])
    ax.set_ylim([1e-16, 1e-0])
    ax.loglog()
    ax.legend(loc='best', fontsize=10)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('Energy / GeV')
    ax.set_ylabel('Differential flux / m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$')
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_integral_spectral_exclusion_zone.jpg'.format(site_key)
        )
    )
    plt.close(fig)

    # differential trigger-rates
    # --------------------------
    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    text_y = 0.7
    for particle_key in irf_config['config']['particles']:
        ax.plot(
            fine_energy_bin_centers,
            onregion_rates[
                site_key][
                particle_key][
                'differential_rate'][
                'mean'],
            color=sum_config['plot']['particle_colors'][particle_key]
        )
        ax.text(
            0.8,
            0.1 + text_y,
            particle_key,
            color=particle_colors[particle_key],
            transform=ax.transAxes
        )
        ir = onregion_rates[
                site_key][
                particle_key][
                'integral_rate'][
                'mean']
        ax.text(
            0.9,
            0.1 + text_y,
            "{: 8.1f} s$^{{-1}}$".format(ir),
            color='k',
            family='monospace',
            transform=ax.transAxes
        )
        text_y += 0.06

    ax.set_title('trigger, onregion, ' + gamma_name)
    ax.set_xlim([energy_lower, energy_upper])
    ax.set_ylim([1e-3, 1e5])
    ax.loglog()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel('Energy / GeV')
    ax.set_ylabel('Differential trigger-rate / s$^{-1}$ (GeV)$^{-1}$')
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_differential_trigger_rates_in_onregion.jpg'.format(site_key)
        )
    )
    plt.close(fig)
