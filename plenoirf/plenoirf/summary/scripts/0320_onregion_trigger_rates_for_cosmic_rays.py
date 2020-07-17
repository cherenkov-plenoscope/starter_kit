#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
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

acceptance_trigger_in_onregion_path = os.path.join(
    pa['summary_dir'],
    "acceptance_trigger_in_onregion",
    "acceptance_trigger_in_onregion.json"
)
with open(acceptance_trigger_in_onregion_path, 'rt') as f:
    acceptance = json.loads(f.read())

energy_lower_GeV = sum_config['energy_binning']['lower_edge_GeV']
energy_upper_GeV = sum_config['energy_binning']['upper_edge_GeV']

energy_bin_edges_GeV = np.array(acceptance['energy_bin_edges']['value'])
assert acceptance['energy_bin_edges']['unit'] == "GeV"
energy_bin_centers_GeV = irf.summary.bin_centers(energy_bin_edges_GeV)

fine_energy_bin_edges_GeV = np.geomspace(
    energy_lower_GeV,
    energy_upper_GeV,
    sum_config['energy_binning']['num_bins_fine'] + 1
)

fine_energy_bin_width = irf.summary.bin_width(fine_energy_bin_edges_GeV)
fine_energy_bin_centers_GeV = irf.summary.bin_centers(fine_energy_bin_edges_GeV)

detection_threshold_std = sum_config['on_off_measuremnent']['detection_threshold_std']

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

observation_time_s = 50*3600

n_points_to_plot = 17

crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

# cosmic-ray-flux
# ----------------
airshower_fluxes = irf.summary.read_airshower_differential_flux(
    summary_dir=pa['summary_dir'],
    energy_bin_centers=fine_energy_bin_centers_GeV,
    sites=irf_config['config']['sites'],
    geomagnetic_cutoff_fraction=sum_config[
        'airshower_flux'][
        'fraction_of_flux_below_geomagnetic_cutoff'],
)

_crab_nebula = '3FGL J0534.5+2201'
_phd_thesis_reference_source = '3FGL J2254.0+1608'
# gamma-ray-flux
# ---------------
with open(os.path.join(pa['summary_dir'], "gamma_sources.json"), "rt") as f:
    gamma_sources = json.loads(f.read())
for source in gamma_sources:
    if source['source_name'] == _phd_thesis_reference_source:
        reference_gamma_source = source
gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
    fermi_source=reference_gamma_source,
    energy=fine_energy_bin_centers_GeV
)

onregion_rates = {}
differential_trigger_rates = {}
differential_trigger_rates['energy_bin_edges'] = {
    "value": fine_energy_bin_edges_GeV,
    "unit": "GeV"
}

for site_key in irf_config['config']['sites']:
    onregion_rates[site_key] = {}
    differential_trigger_rates[site_key] = {}

    # gamma-ray
    # ---------
    _gamma_effective_area_m2 = np.array(acceptance[
        'cosmic_response'][
        site_key][
        'gamma'][
        'point'][
        'value']
    )
    gamma_effective_area_m2 = np.interp(
        x=fine_energy_bin_centers_GeV,
        xp=energy_bin_centers_GeV,
        fp=_gamma_effective_area_m2
    )
    gamma_dT_per_s_per_GeV = (
        gamma_dF_per_m2_per_s_per_GeV *
        gamma_effective_area_m2
    )
    differential_trigger_rates[site_key]['gamma'] = gamma_dT_per_s_per_GeV

    gamma_T_per_s = np.sum(gamma_dT_per_s_per_GeV*fine_energy_bin_width)
    onregion_rates[site_key]['gamma'] = gamma_T_per_s

    # cosmic-rays
    # -----------
    background_rate_in_onregion = 0
    for cosmic_key in airshower_fluxes[site_key]:
        _cosmic_effective_acceptance_m2_sr = np.array(acceptance[
            'cosmic_response'][
            site_key][
            cosmic_key][
            'diffuse'][
            'value']
        )

        cosmic_effective_acceptance_m2_sr = np.interp(
            x=fine_energy_bin_centers_GeV,
            xp=energy_bin_centers_GeV,
            fp=_cosmic_effective_acceptance_m2_sr
        )
        cosmic_dT_per_s_per_GeV = (
            cosmic_effective_acceptance_m2_sr *
            airshower_fluxes[site_key][cosmic_key]['differential_flux']
        )
        differential_trigger_rates[
            site_key][
            cosmic_key] = cosmic_dT_per_s_per_GeV

        cosmic_T_per_s = cosmic_dT_per_s_per_GeV*fine_energy_bin_width
        onregion_rates[site_key][cosmic_key] = np.sum(cosmic_T_per_s)

        background_rate_in_onregion += onregion_rates[site_key][cosmic_key]

    # integral spectral exclusion zone
    # --------------------------------
    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    # Crab reference fluxes
    for i in range(4):
        scale_factor = np.power(10., (-1)*i)

        _energy = np.array(crab_flux['energy']['values'])
        _flux = np.array(crab_flux['differential_flux']['values'])

        ax.plot(
            _energy,
            _flux*scale_factor,
            color='k',
            linestyle='--',
            label='{:.3f} Crab'.format(scale_factor),
            alpha=1./(1.+i)
        )

    fermi_broadband = irf.analysis.fermi_lat_integral_spectral_exclusion_zone()
    assert fermi_broadband[
        'energy']['unit_tex'] == "GeV"
    assert fermi_broadband[
        'differential_flux']['unit_tex'] == "m$^{-2}$ s$^{-1}$ GeV$^{-1}$"
    ax.plot(
        fermi_broadband['energy']['values'],
        fermi_broadband['differential_flux']['values'],
        'k',
        label='Fermi-LAT 10y'
    )


    # plenoscope
    (
        isez_energy_GeV,
        isez_differential_flux_per_GeV_per_m2_per_s
    ) = irf.analysis.estimate_integral_spectral_exclusion_zone(
        gamma_effective_area_m2=_gamma_effective_area_m2,
        energy_bin_centers_GeV=energy_bin_centers_GeV,
        background_rate_in_onregion_per_s=background_rate_in_onregion,
        onregion_over_offregion_ratio=on_over_off_ratio,
        observation_time_s=observation_time_s,
        num_points=n_points_to_plot
    )
    ax.plot(
        isez_energy_GeV,
        isez_differential_flux_per_GeV_per_m2_per_s,
        'r',
        label='Portal {:2.0f}h, trigger'.format(observation_time_s/3600.)
    )

    # plenoscope rejecting all hadrons
    (
        e_isez_energy_GeV,
        e_isez_differential_flux_per_GeV_per_m2_per_s
    ) = irf.analysis.estimate_integral_spectral_exclusion_zone(
        gamma_effective_area_m2=_gamma_effective_area_m2,
        energy_bin_centers_GeV=energy_bin_centers_GeV,
        background_rate_in_onregion_per_s=onregion_rates[site_key]['electron'],
        onregion_over_offregion_ratio=on_over_off_ratio,
        observation_time_s=observation_time_s,
        num_points=n_points_to_plot
    )
    ax.plot(
        e_isez_energy_GeV,
        e_isez_differential_flux_per_GeV_per_m2_per_s,
        'r:',
        label='Portal {:2.0f}h, trigger, rejecting all hadrons'.format(observation_time_s/3600.)
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



opath = os.path.join(pa['out_dir'], "background_rates_in_onregion.json")
with open(opath, 'wt') as f:
    f.write(json.dumps(onregion_rates, indent=4, cls=irf.json_numpy.Encoder))


opath = os.path.join(
    pa['out_dir'],
    'differential_trigger_rates_in_onregion.json'.format(site_key)
)
with open(opath, 'wt') as f:
    f.write(json.dumps(differential_trigger_rates, indent=4, cls=irf.json_numpy.Encoder))



for site_key in irf_config['config']['sites']:

    # differential trigger-rates
    # --------------------------
    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    text_y = 0.7
    for particle_key in irf_config['config']['particles']:
        ax.plot(
            irf.summary.bin_centers(
                differential_trigger_rates["energy_bin_edges"]["value"]
            ),
            differential_trigger_rates[site_key][particle_key],
            color=sum_config['plot']['particle_colors'][particle_key]
        )
        ax.text(
            0.9,
            0.1 + text_y,
            particle_key,
            color=particle_colors[particle_key],
            transform=ax.transAxes
        )
        text_y += 0.06


    ax.set_xlim([energy_lower_GeV, energy_upper_GeV])
    ax.set_ylim([1e-3, 1e4])
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
