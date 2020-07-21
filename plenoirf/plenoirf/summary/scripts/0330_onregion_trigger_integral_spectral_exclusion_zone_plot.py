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

detection_threshold_std = sum_config[
    'on_off_measuremnent'][
    'detection_threshold_std']
on_over_off_ratio = sum_config[
    'on_off_measuremnent'][
    'on_over_off_ratio']
observation_time_s = 50*3600
n_points_to_plot = 7

fig_16_by_9 = sum_config['plot']['16_by_9']

cosmic_ray_keys = list(irf_config['config']['particles'].keys())
cosmic_ray_keys.remove('gamma')

fermi_broadband = irf.analysis.fermi_lat_integral_spectral_exclusion_zone()
assert fermi_broadband[
    'energy']['unit_tex'] == "GeV"
assert fermi_broadband[
    'differential_flux']['unit_tex'] == "m$^{-2}$ s$^{-1}$ GeV$^{-1}$"

# gamma-ray-flux of crab-nebula
# -----------------------------
crab_flux = cosmic_fluxes.read_crab_nebula_flux_from_resources()

internal_sed_style = irf.analysis.spectral_energy_distribution.PLENOIRF_SED_STYLE

output_sed_styles = {
    'plenoirf': irf.analysis.spectral_energy_distribution.PLENOIRF_SED_STYLE,
    'science': irf.analysis.spectral_energy_distribution.SCIENCE_SED_STYLE,
    'fermi': irf.analysis.spectral_energy_distribution.FERMI_SED_STYLE,
}

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

x_lim_GeV = np.array([1e-1, 1e4])
y_lim_per_m2_per_s_per_GeV = np.array([1e-0, 1e-16])

for site_key in irf_config['config']['sites']:

    components = []

    # Crab reference fluxes
    # ---------------------
    for i in range(4):
        com = {}
        scale_factor = np.power(10., (-1)*i)
        com['energy'] = np.array(crab_flux['energy']['values'])
        com['differential_flux'] = scale_factor*np.array(
            crab_flux['differential_flux']['values']
        )
        com['label'] = '{:.3f} Crab'.format(scale_factor)
        com['color'] = 'k'
        com['alpha'] = 1./(1.+i)
        com['linestyle'] = '--'
        components.append(com.copy())

    # Fermi-LAT broadband
    # -------------------
    com = {}
    com['energy'] = np.array(fermi_broadband['energy']['values'])
    com['differential_flux'] = np.array(
        fermi_broadband['differential_flux']['values'])
    com['label'] = 'Fermi-LAT 10 y'
    com['color'] = 'k'
    com['alpha'] = 1.0
    com['linestyle'] = '-'
    components.append(com)

    # plenoscope
    # ----------
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
    com = {}
    com['energy'] = isez_energy_GeV
    com['differential_flux'] = isez_differential_flux_per_GeV_per_m2_per_s
    com['label'] = 'Portal {:2.0f} h, trigger'.format(observation_time_s/3600.)
    com['color'] = 'r'
    com['alpha'] = 1.0
    com['linestyle'] = '-'
    components.append(com)

    # plenoscope no hadrons
    # ---------------------
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
    com = {}
    com['energy'] = e_isez_energy_GeV
    com['differential_flux'] = e_isez_differential_flux_per_GeV_per_m2_per_s
    com['label'] = 'Portal {:2.0f} h, trigger, rejecting all hadrons'.format(
            observation_time_s/3600.)
    com['color'] = 'r'
    com['alpha'] = 0.5
    com['linestyle'] = '--'
    components.append(com)

    for sed_style_key in output_sed_styles:
        sed_style = output_sed_styles[sed_style_key]

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes((.1, .1, .8, .8))

        for com in components:

            _energy, _dFdE = irf.analysis.spectral_energy_distribution.convert_units_style(
                x=com['energy'],
                y=com['differential_flux'],
                in_style=internal_sed_style,
                out_style=sed_style,
            )
            ax.plot(
                _energy,
                _dFdE,
                label=com['label'],
                color=com['color'],
                alpha=com['alpha'],
                linestyle=com['linestyle'],
            )

        _x_lim, _y_lim = irf.analysis.spectral_energy_distribution.convert_units_style(
            x=x_lim_GeV,
            y=y_lim_per_m2_per_s_per_GeV,
            in_style=internal_sed_style,
            out_style=sed_style,
        )

        ax.set_xlim(np.sort(_x_lim))
        ax.set_ylim(np.sort(_y_lim))
        ax.loglog()
        ax.legend(loc='best', fontsize=10)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.set_xlabel(sed_style['x_label']+" / "+sed_style['x_unit'])
        ax.set_ylabel(sed_style['y_label']+" / "+sed_style['y_unit'])
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_integral_spectral_exclusion_zone_style_{:s}.jpg'.format(
                    site_key,
                    sed_style_key
                )
            )
        )
        plt.close(fig)
