#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_table as spt
import cosmic_fluxes
import os
import json
import gamma_limits_sensitivity as gls
import scipy

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

energy_lower_TeV = 1e-3*energy_lower_GeV
energy_upper_TeV = 1e-3*energy_upper_GeV


energy_bin_edges_GeV = np.array(acceptance['energy_bin_edges']['value'])
assert acceptance['energy_bin_edges']['unit'] == "GeV"
energy_bin_centers_GeV = irf.summary.bin_centers(energy_bin_edges_GeV)

fine_energy_bin_edges_GeV = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    sum_config['energy_binning']['num_bins_fine'] + 1
)

fine_energy_bin_width = irf.summary.bin_width(fine_energy_bin_edges_GeV)
fine_energy_bin_centers_GeV = irf.summary.bin_centers(fine_energy_bin_edges_GeV)

on_over_off_ratio = sum_config['on_off_measuremnent']['on_over_off_ratio']
detection_threshold_std = sum_config['on_off_measuremnent']['detection_threshold_std']

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

observation_time_s = 50*3600

n_points_to_plot = 27

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

# gamma-ray-flux
# ---------------
with open(os.path.join(pa['summary_dir'], "gamma_sources.json"), "rt") as f:
    gamma_sources = json.loads(f.read())
for source in gamma_sources:
    if source['source_name'] == '3FGL J0534.5+2201':
        reference_gamma_source = source
gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
    fermi_source=reference_gamma_source,
    energy=fine_energy_bin_centers_GeV
)

onregion_rates = {}

for site_key in irf_config['config']['sites']:
    onregion_rates[site_key] = {}

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
    gamma_T_per_s = np.sum(gamma_dT_per_s_per_GeV*fine_energy_bin_width)
    onregion_rates[site_key]['gamma'] = gamma_T_per_s

    background_rate_in_onregion = 0

    # cosmic-rays
    # -----------
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
        cosmic_T_per_s = cosmic_dT_per_s_per_GeV*fine_energy_bin_width
        onregion_rates[site_key][cosmic_key] = np.sum(cosmic_T_per_s)

        background_rate_in_onregion += onregion_rates[site_key][cosmic_key]


    # integral spectral exclusion zone
    # --------------------------------
    log10_energy_bin_centers_GeV_TeV = np.log10(1e-3*energy_bin_centers_GeV)
    _gamma_effective_area_cm2 = 1e2*1e2*_gamma_effective_area_m2

    acp_aeff_scaled = scipy.interpolate.interpolate.interp1d(
        x=log10_energy_bin_centers_GeV_TeV,
        y=_gamma_effective_area_cm2,
        bounds_error=False,
        fill_value=0.
    )

    acp_sigma_bg = background_rate_in_onregion
    acp_alpha = on_over_off_ratio

    fig = irf.summary.figure.figure(fig_16_by_9)
    ax = fig.add_axes((.1, .1, .8, .8))

    waste_figure = plt.figure()

    acp_energy_range = gls.get_energy_range(acp_aeff_scaled)
    acp_energy_x, acp_dn_de_y = gls.plot_sens_spectrum_figure(
        sigma_bg=acp_sigma_bg,
        alpha=acp_alpha,
        t_obs=observation_time_s,
        a_eff_interpol=acp_aeff_scaled,
        e_0=acp_energy_range[0]*5.,
        n_points_to_plot=n_points_to_plot,
        fmt='r',
        label=''
    )
    ax.plot(
        acp_energy_x*1e3,
        acp_dn_de_y*1e-3*1e4,
        'r',
        label='Portal {:2.0f}h'.format(observation_time_s/3600.)
    )

    # Crab reference fluxes
    for i in range(4):
        scale_factor = np.power(10., (-1)*i)
        log_resolution = 0.2

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
