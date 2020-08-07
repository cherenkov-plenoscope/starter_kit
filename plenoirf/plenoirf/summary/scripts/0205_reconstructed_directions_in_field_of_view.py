#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import os
import magnetic_deflection

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

pointing_azimuth_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'azimuth_deg']
pointing_zenith_deg = irf_config[
    'config'][
    'plenoscope_pointing'][
    'zenith_deg']
fov_radius_deg = 0.5 * irf_config[
    'light_field_sensor_geometry'][
    'max_FoV_diameter_deg']

num_cradial_bins = 23
max_cradial2_deg2 = (1.5*fov_radius_deg)**2

cradial2_bin_edges_deg2 = np.linspace(
    0,
    max_cradial2_deg2,
    num_cradial_bins+1
)

fig_16_by_9 = sum_config['plot']['16_by_9']

for site_key in irf_config['config']['sites']:
    for particle_key in irf_config['config']['particles']:
        event_table = spt.read(
            path=os.path.join(
                pa['run_dir'],
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'
            ),
            structure=irf.table.STRUCTURE
        )



        # thrown
        # ======
        delta_deg = magnetic_deflection.discovery._angle_between_az_zd_deg(
            az1_deg=np.rad2deg(event_table['primary']['azimuth_rad']),
            zd1_deg=np.rad2deg(event_table['primary']['zenith_rad']),
            az2_deg=pointing_azimuth_deg,
            zd2_deg=pointing_zenith_deg
        )
        bin_counts_thrown = np.histogram(
            delta_deg**2,
            bins=cradial2_bin_edges_deg2
        )[0]
        bin_counts_thrown_rel = bin_counts_thrown/np.sum(bin_counts_thrown)
        bin_counts_thrown_unc = np.sqrt(bin_counts_thrown)/bin_counts_thrown

        # reconstructed
        # =============
        features = event_table['features']
        reconstructed_cx_cy = []
        for evt in range(features.shape[0]):

            _rec_cx, _rec_cy = irf.analysis.gamma_direction.estimate(
                light_front_cx=features['light_front_cx'][evt],
                light_front_cy=features['light_front_cy'][evt],
                image_infinity_cx_mean=features['image_infinity_cx_mean'][evt],
                image_infinity_cy_mean=features['image_infinity_cy_mean'][evt],
            )
            reconstructed_cx_cy.append([_rec_cx, _rec_cy])
        reconstructed_cx_cy = np.array(reconstructed_cx_cy)

        offaxis = np.hypot(
            reconstructed_cx_cy[:, 0],
            reconstructed_cx_cy[:, 1]
        )
        offaxis_deg = np.rad2deg(offaxis)
        offaxis2_deg2 = offaxis_deg**2.0

        bin_counts = np.histogram(
            offaxis2_deg2,
            bins=cradial2_bin_edges_deg2
        )[0]

        bin_counts_rel = bin_counts/np.sum(bin_counts)
        bin_counts_unc = np.sqrt(bin_counts)/bin_counts

        ymax = 0.2
        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes([0.1, 0.13, 0.8, 0.8])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=cradial2_bin_edges_deg2,
            bincounts=bin_counts_thrown_rel,
            linestyle=':',
            linecolor='k',
            bincounts_upper=bin_counts_thrown_rel * (1 + bin_counts_thrown_unc),
            bincounts_lower=bin_counts_thrown_rel * (1 - bin_counts_thrown_unc),
            face_color='k',
            face_alpha=.3
        )

        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=cradial2_bin_edges_deg2,
            bincounts=bin_counts_rel,
            linestyle='-',
            linecolor='k',
            bincounts_upper=bin_counts_rel * (1 + bin_counts_unc),
            bincounts_lower=bin_counts_rel * (1 - bin_counts_unc),
            face_color='k',
            face_alpha=.3
        )
        ax.text(
            x=0.8,
            y=0.8,
            s=": thrown",
            color='k',
            transform=ax.transAxes)
        ax.text(
            x=0.8,
            y=0.85,
            s="- reconstructed",
            color='k',
            transform=ax.transAxes)
        ax.text(
            x=1.05*fov_radius_deg**2,
            y=0.95*ymax,
            s="field-of-view's radius",
            color='k'
        )

        ax.set_xlabel('(offaxis angle)$^2$ / (1$^{\\circ}$)$^2$')
        ax.set_ylabel('relative intensity / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.set_xlim([0, np.max(cradial2_bin_edges_deg2)])
        ax.set_ylim([0, ymax])
        ax.axvline(fov_radius_deg**2, color='k')
        fig.savefig(
            os.path.join(
                pa['out_dir'],
                site_key + "_" + particle_key + ".jpg"
            )
        )
        plt.close(fig)