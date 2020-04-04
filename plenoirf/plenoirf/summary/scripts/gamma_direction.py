#!/usr/bin/python
import sys
import plenoirf as irf
import magnetic_deflection as mdfl
import os
import sklearn
from sklearn import neural_network
from sklearn import gaussian_process
from sklearn import tree
import pandas as pd
import sparse_table as spt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
run_dir = argv[1]
summary_dir = argv[2]

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)


def one_sigma_68_containment(residual_angles):
    sorted_angles = np.sort(residual_angles)
    return sorted_angles[int(sorted_angles.shape[0]*0.67)]


def make_pfs_features(psf_df):
    return np.array([
        #np.log10(psf_df['features.num_photons']),
        #np.log10(psf_df['features.image_smallest_ellipse_object_distance']),
        #psf_df['features.image_smallest_ellipse_solid_angle'],
        psf_df['features.image_infinity_cx_mean'],
        psf_df['features.image_infinity_cy_mean'],
        psf_df['features.image_infinity_cx_std'],
        psf_df['features.image_infinity_cy_std'],
        psf_df['features.light_front_cx'],
        psf_df['features.light_front_cy'],
        psf_df['features.image_half_depth_shift_cx'],
        psf_df['features.image_half_depth_shift_cy'],
    ]).T



for site_key in irf_config['config']['sites']:
    for particle_key in ['gamma']:
        prefix_str = '{:s}_{:s}'.format(site_key, particle_key)

        event_table = spt.read(
            path=os.path.join(
                run_dir,
                'event_table',
                site_key,
                particle_key,
                'event_table.tar'),
            structure=irf.table.STRUCTURE)

        # cut
        # ===

        # only with features
        idxs_primary = event_table['primary'][spt.IDX]
        idxs_features = event_table['features'][spt.IDX]

        _off_deg = mdfl.discovery._great_circle_distance_alt_zd_deg(
            az1_deg=np.rad2deg(event_table['primary']['azimuth_rad']),
            zd1_deg=np.rad2deg(event_table['primary']['zenith_rad']),
            az2_deg=irf_config['config']['plenoscope_pointing']['azimuth_deg'],
            zd2_deg=irf_config['config']['plenoscope_pointing']['zenith_deg'])
        _off_mask= (_off_deg <= 3.25 - 1.0)
        idxs_in_possible_on_region = event_table['primary'][spt.IDX][_off_mask]

        idxs_in_energy_bin = event_table['primary'][spt.IDX][
            event_table['primary']['energy_GeV'] < 1.6]

        cut_idxs = spt.intersection([
            idxs_primary,
            idxs_features,
            idxs_in_possible_on_region,
            idxs_in_energy_bin])

        events = spt.cut_table_on_indices(
            table=event_table,
            structure=irf.table.STRUCTURE,
            common_indices=spt.dict_to_recarray({spt.IDX: cut_idxs})
        )

        psf_events = spt.make_rectangular(events)

        # reconstruction
        # ==============
        psf_test_sample_fraction = 0.5

        (psf_train, psf_test) = sklearn.model_selection.train_test_split(
            psf_events,
            test_size=psf_test_sample_fraction,
            random_state=13)

        x_psf_train = make_pfs_features(psf_train)
        y_psf_train = np.array(mdfl.discovery._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(psf_train['primary.azimuth_rad']),
            zenith_deg=np.rad2deg(psf_train['primary.zenith_rad']))).T

        x_psf_test = make_pfs_features(psf_test)
        y_psf_test = np.array(mdfl.discovery._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(psf_test['primary.azimuth_rad'].values),
            zenith_deg=np.rad2deg(psf_test['primary.zenith_rad'].values))).T

        # scaling
        x_psf_scaler = sklearn.preprocessing.StandardScaler()
        x_psf_scaler.fit(x_psf_train)
        x_psf_train_s = x_psf_scaler.transform(x_psf_train)
        x_psf_test_s = x_psf_scaler.transform(x_psf_test)

        y_psf_scaler = sklearn.preprocessing.StandardScaler()
        y_psf_scaler.fit(y_psf_train)
        y_psf_train_s = y_psf_scaler.transform(y_psf_train)
        y_psf_test_s = y_psf_scaler.transform(y_psf_test)

        psf_mlp = sklearn.neural_network.MLPRegressor(
            solver='lbfgs',
            alpha=1e-2,
            hidden_layer_sizes=(5, 3),
            random_state=123,
            verbose=False,
            max_iter=3000)
        psf_mlp.fit(x_psf_train_s, y_psf_train_s)

        psf_tree = tree.DecisionTreeRegressor()
        psf_tree.fit(x_psf_train_s, y_psf_train_s)

        cxcy_reconstructed_mlp = y_psf_scaler.inverse_transform(
            psf_mlp.predict(x_psf_test_s))

        cxcy_reconstructed_tree = y_psf_scaler.inverse_transform(
            psf_tree.predict(x_psf_test_s))

        primary_cx = y_psf_test[:, 0]
        primary_cy = y_psf_test[:, 1]

        primary_cx_reco = cxcy_reconstructed_mlp[:, 0]
        primary_cy_reco = cxcy_reconstructed_mlp[:, 1]

        """
        primary_cx_reco = ( - 0.5*(
            psf_test['features.image_infinity_cx_mean'] +
            psf_test['features.light_front_cx'])
        )
        primary_cy_reco = ( - 0.5*(
            psf_test['features.image_infinity_cy_mean'] +
            psf_test['features.light_front_cy'])
        )
        """

        delta_cx = primary_cx - primary_cx_reco
        delta_cy = primary_cy - primary_cy_reco

        _psf_center = sklearn.cluster.MeanShift(bandwidth=1).fit(
            np.c_[delta_cx, delta_cy])

        delta_cx -= _psf_center.cluster_centers_[0][0]
        delta_cy -= _psf_center.cluster_centers_[0][1]

        delta_c = np.hypot(delta_cx, delta_cy)

        print(site_key, one_sigma_68_containment(np.rad2deg(delta_c)), "deg")

        num_bin_edges = 32
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.hist2d(
            np.rad2deg(primary_cx),
            np.rad2deg(primary_cy),
            bins=np.linspace(-4, 4, num_bin_edges))
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(summary_dir, "{:s}_gamma_true.jpg".format(site_key)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.hist2d(
            np.rad2deg(primary_cx_reco),
            np.rad2deg(primary_cy_reco),
            bins=np.linspace(-4, 4, num_bin_edges))
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(summary_dir, "{:s}_gamma_reco.jpg".format(site_key)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.hist2d(
            np.rad2deg(delta_cx),
            np.rad2deg(delta_cy),
            bins=np.linspace(-4, 4, num_bin_edges))
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(summary_dir, "{:s}_gamma_psf.jpg".format(site_key)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.hist2d(
            np.rad2deg(primary_cx),
            np.rad2deg(primary_cx_reco),
            bins=np.linspace(-4, 4, num_bin_edges))
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(
                summary_dir,
                "{:s}_cx_true_vs_reco.jpg".format(site_key)))

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.9, 0.9])
        ax.hist2d(
            np.rad2deg(primary_cy),
            np.rad2deg(primary_cy_reco),
            bins=np.linspace(-4, 4, num_bin_edges))
        ax.set_aspect('equal')
        fig.savefig(
            os.path.join(
                summary_dir,
                "{:s}_cy_true_vs_reco.jpg".format(site_key)))
