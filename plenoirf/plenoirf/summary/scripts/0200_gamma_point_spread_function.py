#!/usr/bin/python
import sys
import numpy as np
import magnetic_deflection as mdfl
import plenoirf as irf
import sparse_table as spt
import os
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as plt_colors

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

MAX_SOURCE_ANGLE_DEG = sum_config[
    'gamma_ray_source_direction'][
    'max_angle_relative_to_pointing_deg']
pointing_azimuth_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'azimuth_deg']
pointing_zenith_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'zenith_deg']
analysis_trigger_threshold = sum_config['trigger']['threshold_pe']
trigger_modus = sum_config["trigger"]["modus"]
num_energy_bins = sum_config['energy_binning']['num_bins_coarse']
energy_bin_edges = np.geomspace(
    sum_config['energy_binning']['lower_edge_GeV'],
    sum_config['energy_binning']['upper_edge_GeV'],
    num_energy_bins + 1
)
max_relative_leakage = sum_config['quality']['max_relative_leakage']
min_reconstructed_photons = sum_config['quality']['min_reconstructed_photons']

fc16by9 = sum_config['plot']['16_by_9']
fc5by4 = fc16by9.copy()
fc5by4['cols'] = fc16by9['cols']*(9/16)*(5/4)

fov_radius_deg = 0.5*irf_config[
    'light_field_sensor_geometry'][
    'max_FoV_diameter_deg']

delta_c2_bin_edges_deg = np.linspace(0, 3.25**2, 32)

for site_key in irf_config['config']['sites']:
    diffuse_gamma_table = spt.read(
        path=os.path.join(
            pa['run_dir'],
            'event_table',
            site_key,
            'gamma',
            'event_table.tar'
        ),
        structure=irf.table.STRUCTURE
    )

    indices_passed_trigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=diffuse_gamma_table['trigger'],
        threshold=analysis_trigger_threshold,
        modus=trigger_modus,
    )

    indices_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
        primary_table=diffuse_gamma_table['primary'],
        radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
        azimuth_deg=pointing_azimuth_deg,
        zenith_deg=pointing_zenith_deg,
    )

    indices_quality = irf.analysis.cuts.cut_quality(
        feature_table=diffuse_gamma_table['features'],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )

    psf_68_radius_deg = []
    psf_68_radius_deg_unc = []

    for energy_bin in range(num_energy_bins):
        indices_energy_bin = irf.analysis.cuts.cut_energy_bin(
            primary_table=diffuse_gamma_table['primary'],
            lower_energy_edge_GeV=energy_bin_edges[energy_bin],
            upper_energy_edge_GeV=energy_bin_edges[energy_bin + 1],
        )

        indices_gammas = spt.intersection([
            indices_passed_trigger,
            indices_onregion,
            indices_quality,
            indices_energy_bin,
        ])

        num_airshower = len(indices_gammas)

        gamma_table = spt.cut_table_on_indices(
            table=diffuse_gamma_table,
            structure=irf.table.STRUCTURE,
            common_indices=indices_gammas,
            level_keys=None
        )
        gamma_table = spt.sort_table_on_common_indices(
            table=gamma_table,
            common_indices=indices_gammas
        )

        gt = gamma_table
        deltas_deg = []
        psf = []
        for evt in range(gamma_table['features'].shape[0]):

            true_cx, true_cy = mdfl.discovery._az_zd_to_cx_cy(
                azimuth_deg=np.rad2deg(gt['primary']['azimuth_rad'][evt]),
                zenith_deg=np.rad2deg(gt['primary']['zenith_rad'][evt])
            )

            (rec_cx, rec_cy) = irf.analysis.gamma_direction.estimate(
                light_front_cx=gt['features']['light_front_cx'][evt],
                light_front_cy=gt['features']['light_front_cy'][evt],
                image_infinity_cx_mean=gt['features']['image_infinity_cx_mean'][evt],
                image_infinity_cy_mean=gt['features']['image_infinity_cy_mean'][evt],
            )

            delta_cx = true_cx - rec_cx
            delta_cy = true_cy - rec_cy
            psf.append([delta_cx, delta_cy])
            delta_c = np.hypot(delta_cx, delta_cy)
            delta_c_deg = np.rad2deg(delta_c)
            deltas_deg.append(delta_c_deg)

        deltas_deg = np.array(deltas_deg)
        psf = np.array(psf)

        delta_hist = np.histogram(
            deltas_deg**2,
            bins=delta_c2_bin_edges_deg
        )[0]
        delta_hist_unc = irf.analysis.effective_quantity._divide_silent(
            np.sqrt(delta_hist),
            delta_hist,
            np.nan
        )

        psf_r2_deg2 = irf.analysis.gamma_direction.integration_width_for_containment(
            bin_counts=delta_hist,
            bin_edges=delta_c2_bin_edges_deg,
            containment=0.68
        )
        psf_68_radius_deg.append(np.sqrt(psf_r2_deg2))
        if num_airshower > 0:
            _psf_68_radius_deg_unc = np.sqrt(num_airshower)/num_airshower
        else:
            _psf_68_radius_deg_unc = np.nan
        psf_68_radius_deg_unc.append(_psf_68_radius_deg_unc)

        fig = irf.summary.figure.figure(fc16by9)
        ax = fig.add_axes((.1, .1, .8, .8))
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=delta_c2_bin_edges_deg,
            bincounts=delta_hist,
            linestyle='k-',
            bincounts_upper=delta_hist*(1 + delta_hist_unc),
            bincounts_lower=delta_hist*(1 - delta_hist_unc),
            face_color='k',
            face_alpha=0.25
        )
        ax.set_title(
            'num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f}GeV'.format(
                num_airshower,
                energy_bin_edges[energy_bin],
                energy_bin_edges[energy_bin + 1]
            ),
            family='monospace'
        )
        ax.axvline(
            x=psf_r2_deg2,
            color='k',
            linestyle='-',
            alpha=0.25)
        ax.semilogy()
        ax.set_xlabel('$\\delta^{2}$ / deg$^{2}$')
        ax.set_ylabel('intensity / 1')
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.set_xlim([delta_c2_bin_edges_deg[0], delta_c2_bin_edges_deg[-1]])
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                '{:s}_gamma_{:06d}_psf_radial.jpg'.format(
                    site_key,
                    energy_bin
                )
            )
        )
        plt.close(fig)

        psf = np.rad2deg(psf)
        num_c_bins = 50
        psf_c_bin_edges = np.linspace(-3.5, 3.5, num_c_bins + 1)

        if len(psf) > 0:
            img = np.histogram2d(psf[:, 0], psf[:, 1], psf_c_bin_edges)[0]
        else:
            img = np.zeros(shape=(num_c_bins, num_c_bins))

        fig = irf.summary.figure.figure(fc5by4)
        ax = fig.add_axes((.1, .1, .8, .8))

        ax.pcolormesh(
            psf_c_bin_edges,
            psf_c_bin_edges,
            img,
            norm=plt_colors.PowerNorm(gamma=0.5),
            cmap='Blues',
            vmin=None,
            vmax=None
        )
        ax.set_title(
            'num. airshower {: 6d}, energy {: 7.1f} - {: 7.1f}GeV'.format(
                num_airshower,
                energy_bin_edges[energy_bin],
                energy_bin_edges[energy_bin + 1]
            ),
            family='monospace'
        )
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

    psf_68_path = os.path.join(
        pa['out_dir'],
        '{:s}_gamma_psf_radial'.format(site_key)
    )
    psf_68_radius_deg = np.array(psf_68_radius_deg)
    psf_68_radius_deg_unc = np.array(psf_68_radius_deg_unc)
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
    ax.set_ylabel('$\\delta$ / deg')
    ax.set_ylim([0, fov_radius_deg])
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(psf_68_path+'.jpg')
    plt.close(fig)

    with open(psf_68_path+'.json', 'wt') as f:
        spsf = {
            "energy_bin_edges_GeV": energy_bin_edges,
            "delta_deg": psf_68_radius_deg,
            "delta_deg_relative_uncertainty": psf_68_radius_deg_unc,
        }
        f.write(json.dumps(spsf, indent=4, cls=irf.json_numpy.Encoder))
