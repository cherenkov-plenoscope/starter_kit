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
import json


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
assert len(argv) == 3
run_dir = argv[1]
summary_dir = argv[2]

irf_config = irf.summary.read_instrument_response_config(run_dir=run_dir)
sum_config = irf.summary.read_summary_config(summary_dir=summary_dir)


def make_gh_features(gh_df):

    _slope = np.hypot(
        gh_df['features.paxel_intensity_median_x'],
        gh_df['features.paxel_intensity_median_y'])
    norm_paxel_intensity_offset = np.sqrt(_slope)/np.sqrt(71./2.)


    _cx_img_lf_diff = gh_df['features.image_infinity_cx_mean'] - gh_df['features.light_front_cx']
    _cy_img_lf_diff = gh_df['features.image_infinity_cy_mean'] - gh_df['features.light_front_cy']
    _c_diff = np.hypot(_cx_img_lf_diff, _cy_img_lf_diff)


    return np.array([
        np.log10(gh_df['features.num_photons']),

        np.log10(gh_df['features.image_smallest_ellipse_object_distance']) - 3,
        np.log10(gh_df['features.image_smallest_ellipse_solid_angle']) + 7,
        np.log10(gh_df['features.image_smallest_ellipse_half_depth']),

        norm_paxel_intensity_offset,
        _c_diff,

        np.log10(gh_df['features.paxel_intensity_peakness_std_over_mean']),
        np.log10(gh_df['features.paxel_intensity_peakness_max_over_mean']),
        gh_df['features.aperture_num_islands_watershed_rel_thr_2'],
        gh_df['features.aperture_num_islands_watershed_rel_thr_4'],
        gh_df['features.aperture_num_islands_watershed_rel_thr_8'],
        gh_df['features.image_num_islands'],

        gh_df['features.image_infinity_cx_mean'],
        gh_df['features.image_infinity_cy_mean'],
        gh_df['features.image_infinity_cx_std'],
        gh_df['features.image_infinity_cy_std'],
        gh_df['features.light_front_cx'],
        gh_df['features.light_front_cy'],
        gh_df['features.image_half_depth_shift_cx'],
        gh_df['features.image_half_depth_shift_cy'],
        gh_df['features.image_infinity_num_photons_on_edge_field_of_view'],

        gh_df['features.image_half_depth_shift_cx'],
        gh_df['features.image_half_depth_shift_cy'],
    ]).T


for site_key in irf_config['config']['sites']:

    gamma_max_energy = 10.

    # prepare gamma-sample
    # ====================
    event_table = spt.read(
        path=os.path.join(
            run_dir,
            'event_table',
            site_key,
            'gamma',
            'event_table.tar'),
        structure=irf.table.STRUCTURE)

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
        event_table['primary']['energy_GeV'] < gamma_max_energy]

    cut_idxs = spt.intersection([
        idxs_primary,
        idxs_features,
        idxs_in_possible_on_region,
        idxs_in_energy_bin])

    gamma_table = spt.cut_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=spt.dict_to_recarray({spt.IDX: cut_idxs})
    )

    gamma_df = spt.make_rectangular(gamma_table)
    gamma_df['gammaness'] = np.ones(gamma_df.shape[0], dtype=np.float)

    # prepare hadron-sample
    # =====================

    # proton
    _full_proton_table = irf.summary.read_event_table_cache(
        summary_dir=summary_dir,
        run_dir=run_dir,
        site_key=site_key,
        particle_key='proton')
    proton_table = irf.table.cut(
        event_table=_full_proton_table,
        indices=irf.table.level_indices(_full_proton_table['features']))
    proton_df = irf.table.combine_in_rectangular_dataframe(proton_table)

    # helium
    _full_helium_table = irf.summary.read_event_table_cache(
        summary_dir=summary_dir,
        run_dir=run_dir,
        site_key=site_key,
        particle_key='helium')
    helium_table = irf.table.cut(
        event_table=_full_helium_table,
        indices=irf.table.level_indices(_full_helium_table['features']))
    helium_df = irf.table.combine_in_rectangular_dataframe(helium_table)
    hadron_df = pd.concat([proton_df, helium_df])
    hadron_df['gammaness'] = np.zeros(hadron_df.shape[0], dtype=np.float)


    # split test and train
    # ====================
    full_df = pd.concat([gamma_df, hadron_df])

    (gh_train, gh_test) = sklearn.model_selection.train_test_split(
        full_df,
        test_size=0.5,
        random_state=13)

    x_gh_train = make_gh_features(gh_train)
    y_gh_train = np.array(gh_train['gammaness'].values)
    y_gh_train = y_gh_train.reshape((y_gh_train.shape[0], 1))

    x_gh_test = make_gh_features(gh_test)
    y_gh_test = np.array(gh_test['gammaness'].values)
    y_gh_test = y_gh_test.reshape((y_gh_test.shape[0], 1))

    # scaling
    x_gh_scaler = sklearn.preprocessing.StandardScaler()
    x_gh_scaler.fit(x_gh_train)
    x_gh_train_s = x_gh_scaler.transform(x_gh_train)
    x_gh_test_s = x_gh_scaler.transform(x_gh_test)

    y_gh_scaler = sklearn.preprocessing.StandardScaler()
    y_gh_scaler.fit(y_gh_train)
    y_gh_train_s = y_gh_scaler.transform(y_gh_train)
    y_gh_test_s = y_gh_scaler.transform(y_gh_test)

    # learn
    # =====

    gh_mlp = sklearn.neural_network.MLPRegressor(
        solver='lbfgs',
        alpha=1e-2,
        hidden_layer_sizes=(15),
        random_state=123,
        verbose=False,
        max_iter=3000)
    gh_mlp.fit(x_gh_train_s, y_gh_train_s[:, 0])

    gh_tree = tree.DecisionTreeRegressor()
    gh_tree = gh_tree.fit(x_gh_train_s, y_gh_train_s)

    # benchmark
    # =========
    fpr_gh, tpr_gh, thresholds_gh = sklearn.metrics.roc_curve(
        y_true=y_gh_test,
        y_score=y_gh_scaler.inverse_transform(
            gh_mlp.predict(x_gh_test_s)))

    auc_gh = sklearn.metrics.roc_auc_score(
        y_true=y_gh_test,
        y_score=y_gh_scaler.inverse_transform(
            gh_mlp.predict(x_gh_test_s)))

    roc_gh = {
        "false_positive_rate": fpr_gh.tolist(),
        "true_positive_rate": tpr_gh.tolist(),
        "gamma_hadron_threshold": thresholds_gh.tolist(),
        "area_under_curve": float(auc_gh),
        "num_events_for_training": int(x_gh_train.shape[0]),
        "max_gamma_ray_energy": float(gamma_max_energy),
    }
    roc_gh_path = os.path.join(
        summary_dir,
        "{:s}_roc_gamma_hadron".format(site_key))
    with open(roc_gh_path+".json", "wt") as fout:
        fout.write(json.dumps(roc_gh, indent=4))

    print(site_key, 'auc.', auc_gh)

    fig = plt.figure(figsize=(10, 10), dpi=200)
    ax = fig.add_axes([.2, .2, .72, .72])
    ax.plot(fpr_gh, tpr_gh, 'k')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_title('area under curve {:.2f}'.format(auc_gh))
    ax.set_xlabel('false positive rate / 1\nproton acceptance')
    ax.set_ylabel('true positive rate / 1\ngamma-ray acceptance')
    ax.semilogx()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(roc_gh_path+".jpg")
    plt.close('all')
