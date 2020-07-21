#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

onregion_rates = irf.json_numpy.read_tree(
    os.path.join(
        pa['summary_dir'],
        "0320_onregion_trigger_rates_for_cosmic_rays"
    )
)

energy_lower = sum_config['energy_binning']['lower_edge_GeV']
energy_upper = sum_config['energy_binning']['upper_edge_GeV']

fine_energy_bin_edges = np.geomspace(
    energy_lower,
    energy_upper,
    sum_config['energy_binning']['num_bins']['interpolation'] + 1
)
fine_energy_bin_centers = irf.summary.bin_centers(fine_energy_bin_edges)

fig_16_by_9 = sum_config['plot']['16_by_9']
particle_colors = sum_config['plot']['particle_colors']

cosmic_ray_keys = list(irf_config['config']['particles'].keys())
cosmic_ray_keys.remove('gamma')

_, gamma_name = irf.summary.make_gamma_ray_reference_flux(
    summary_dir=pa['summary_dir'],
    gamma_ray_reference_source=sum_config['gamma_ray_reference_source'],
    energy_supports_GeV=fine_energy_bin_centers,
)

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
