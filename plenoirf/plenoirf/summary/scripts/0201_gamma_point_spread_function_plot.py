#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

psf = irf.json_numpy.read_tree(
    os.path.join(pa['summary_dir'], "0200_gamma_point_spread_function")
)

num_energy_bins = sum_config[
    'energy_binning']['num_bins']['point_spread_function']
energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    num_energy_bins + 1
)

theta_square_bin_edges_deg2 = np.linspace(
    0,
    sum_config['point_spread_function']['theta_square']['max_angle_deg']**2,
    sum_config['point_spread_function']['theta_square']['num_bins']
)
fov_radius_deg = 0.5*irf_config[
    'light_field_sensor_geometry'][
    'max_FoV_diameter_deg']

num_c_bins = 50
c_bin_edges_deg = np.linspace(
    -fov_radius_deg,
    fov_radius_deg,
    num_c_bins + 1
)

fc16by9 = sum_config['plot']['16_by_9']
fc5by4 = fc16by9.copy()
fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

for site_key in irf_config['config']['sites']:
    particle_key = 'gamma'

    # theta square vs energy
    # ----------------------
    for energy_bin in range(num_energy_bins):

        delta_hist = np.array(psf[
            site_key][
            particle_key][
            'theta_square_histogram_vs_energy'][
            'mean'][
            energy_bin])
        num_airshower = np.sum(delta_hist)
        delta_hist = delta_hist/num_airshower

        delta_hist_unc = np.array(psf[
            site_key][
            particle_key][
            'theta_square_histogram_vs_energy'][
            'relative_uncertainty'][
            energy_bin])

        containment_angle_deg = psf[
            site_key][
            particle_key][
            'containment_angle_vs_energy'][
            'mean'][
            energy_bin]

        containment_angle_rel_unc_deg = psf[
            site_key][
            particle_key][
            'containment_angle_vs_energy'][
            'relative_uncertainty'][
            energy_bin]

        fig_title = "".join([
            'num. airshower {: 6d}, '.format(num_airshower),
            'energy {: 7.1f} - '.format(energy_bin_edges[energy_bin]),
            '{: 7.1f}GeV'.format(energy_bin_edges[energy_bin + 1]),
        ])

        fig = irf.summary.figure.figure(fc16by9)
        ax = fig.add_axes((.1, .1, .8, .8))
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=theta_square_bin_edges_deg2,
            bincounts=delta_hist,
            linestyle='k-',
            bincounts_upper=delta_hist*(1 + delta_hist_unc),
            bincounts_lower=delta_hist*(1 - delta_hist_unc),
            face_color='k',
            face_alpha=0.25
        )
        ax.set_title(fig_title, family='monospace')
        ax.axvline(
            x=containment_angle_deg**2,
            color='k',
            linestyle='-',
            alpha=0.75
        )
        _ca2 = containment_angle_deg**2
        _ca_unc = containment_angle_rel_unc_deg
        ax.fill(
            [
                _ca2*(1 - _ca_unc),
                _ca2*(1 + _ca_unc),
                _ca2*(1 + _ca_unc),
                _ca2*(1 - _ca_unc),
            ],
            [0, 0, 1, 1],
            color='k',
            alpha=0.1
        )
        ax.semilogy()
        ax.set_xlabel('$\\theta^{2}$ / deg$^{2}$')
        ax.set_ylabel('relative intensity / 1')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.set_xlim([0, theta_square_bin_edges_deg2[-1]])
        ax.set_ylim([1e-4, 1])
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_gamma_{:06d}_theta_square.jpg'.format(
                    site_key,
                    energy_bin
                )
            )
        )
        plt.close(fig)

        # psf vs energy
        # ----------------------
        dev_deg = np.array(psf[
            site_key][
            particle_key][
            'point_spread_distribution_vs_energy'][
            'deviation'][
            energy_bin])

        if len(dev_deg) > 0:
            img = np.histogram2d(
                dev_deg[:, 0],
                dev_deg[:, 1],
                c_bin_edges_deg
            )[0]
        else:
            img = np.zeros(shape=(num_c_bins, num_c_bins))

        fig = irf.summary.figure.figure(fc5by4)
        ax = fig.add_axes((.1, .1, .8, .8))
        ax.pcolormesh(
            c_bin_edges_deg,
            c_bin_edges_deg,
            img,
            norm=plt_colors.PowerNorm(gamma=0.5),
            cmap='Blues',
            vmin=None,
            vmax=None
        )
        ax.set_title(fig_title, family='monospace')
        ax.set_xlabel('cx / deg')
        ax.set_ylabel('cy / deg')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_gamma_{:06d}_psf.jpg'.format(
                    site_key,
                    energy_bin
                )
            )
        )
        plt.close(fig)

    psf_68_radius_deg = np.array(psf[
        site_key][
        particle_key][
        'containment_angle_vs_energy'][
        'mean'])
    psf_68_radius_deg_unc = np.array(psf[
        site_key][
        particle_key][
        'containment_angle_vs_energy'][
        'relative_uncertainty'])

    fix_onregion_radius = np.array(psf[
        site_key][
        particle_key][
        'containment_angle_for_fix_onregion'][
        'containment_angle'])

    fig = irf.summary.figure.figure(fc16by9)
    ax = fig.add_axes((.1, .1, .8, .8))
    irf.summary.figure.ax_add_hist(
        ax=ax,
        bin_edges=energy_bin_edges,
        bincounts=psf_68_radius_deg,
        linestyle='k-',
        bincounts_upper=psf_68_radius_deg*(1 + psf_68_radius_deg_unc),
        bincounts_lower=psf_68_radius_deg*(1 - psf_68_radius_deg_unc),
        face_color='k',
        face_alpha=0.25
    )
    ax.semilogx()
    ax.set_xlabel('energy / GeV')
    ax.set_ylabel('$\\theta$ / deg')
    ax.set_ylim([0, fov_radius_deg])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.axhline(
        y=fix_onregion_radius,
        color='k',
        linestyle='--',
        alpha=0.75
    )
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(
        os.path.join(
            pa['out_dir'],
            '{:s}_gamma_psf_radial.jpg'.format(site_key)
        )
    )
    plt.close(fig)
