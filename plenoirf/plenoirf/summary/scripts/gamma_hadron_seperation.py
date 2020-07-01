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
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa['run_dir'])
sum_config = irf.summary.read_summary_config(summary_dir=pa['summary_dir'])

os.makedirs(pa['out_dir'], exist_ok=True)

FOV_RADIUS_DEG = (
    0.5 *
    irf_config['light_field_sensor_geometry']['max_FoV_diameter_deg']
)

MAX_LEAKAGE_PE = 50

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
        # gh_df['features.image_infinity_num_photons_on_edge_field_of_view'],

        gh_df['features.image_half_depth_shift_cx'],
        gh_df['features.image_half_depth_shift_cy'],
    ]).T


for site_key in irf_config['config']['sites']:

    gamma_max_energy = 10.

    # prepare gamma-sample
    # ====================
    event_table = spt.read(
        path=os.path.join(
            pa['run_dir'],
            'event_table',
            site_key,
            'gamma',
            'event_table.tar'),
        structure=irf.table.STRUCTURE)

    # only with features
    idxs_features = event_table['features'][spt.IDX]

    idxs_in_possible_on_region = irf.analysis.effective_quantity.cut_primary_direction_within_angle(
        primary_table=event_table['primary'],
        radial_angle_deg=FOV_RADIUS_DEG - 0.25,
        azimuth_deg=irf_config[
            'config']['plenoscope_pointing']['azimuth_deg'],
        zenith_deg=irf_config[
            'config']['plenoscope_pointing']['zenith_deg'],
    )

    idxs_in_energy_bin = event_table['primary'][spt.IDX][
        event_table['primary']['energy_GeV'] < gamma_max_energy]

    _mask_no_leakage = event_table[
        'features'][
        'image_infinity_num_photons_on_edge_field_of_view'] < MAX_LEAKAGE_PE
    idxs_no_leakage = event_table['features'][spt.IDX][_mask_no_leakage]

    cut_idxs = spt.intersection([
        idxs_features,
        idxs_no_leakage,
        idxs_in_possible_on_region,
        idxs_in_energy_bin
    ])

    gamma_table = spt.cut_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=cut_idxs
    )
    gamma_table = spt.sort_table_on_common_indices(
        table=gamma_table,
        common_indices=cut_idxs
    )

    gamma_df = spt.make_rectangular_DataFrame(gamma_table)
    gamma_df['gammaness'] = np.ones(gamma_df.shape[0], dtype=np.float)

    # prepare hadron-sample
    # =====================

    # proton
    _full_proton_table = spt.read(
        path=os.path.join(
            pa['run_dir'],
            'event_table',
            site_key,
            'proton',
            'event_table.tar'
        ),
        structure=irf.table.STRUCTURE
    )
    _mask_no_leakage = _full_proton_table[
        'features'][
        'image_infinity_num_photons_on_edge_field_of_view'] < MAX_LEAKAGE_PE
    idxs_no_leakage = _full_proton_table['features'][spt.IDX][_mask_no_leakage]
    proton_table = spt.cut_table_on_indices(
        table=_full_proton_table,
        structure=irf.table.STRUCTURE,
        common_indices=idxs_no_leakage
    )
    proton_table = spt.sort_table_on_common_indices(
        table=proton_table,
        common_indices=idxs_no_leakage
    )
    proton_df = spt.make_rectangular_DataFrame(proton_table)

    # helium
    _full_helium_table = spt.read(
        path=os.path.join(
            pa['run_dir'],
            'event_table',
            site_key,
            'helium',
            'event_table.tar'
        ),
        structure=irf.table.STRUCTURE
    )
    _mask_no_leakage = _full_helium_table[
        'features'][
        'image_infinity_num_photons_on_edge_field_of_view'] < MAX_LEAKAGE_PE
    idxs_no_leakage = _full_helium_table['features'][spt.IDX][_mask_no_leakage]
    helium_table = spt.cut_table_on_indices(
        table=_full_helium_table,
        structure=irf.table.STRUCTURE,
        common_indices=idxs_no_leakage
    )
    helium_table = spt.sort_table_on_common_indices(
        table=helium_table,
        common_indices=idxs_no_leakage
    )
    helium_df = spt.make_rectangular_DataFrame(helium_table)

    # hadron
    # ------
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
        "false_positive_rate": fpr_gh,
        "true_positive_rate": tpr_gh,
        "gamma_hadron_threshold": thresholds_gh,
        "area_under_curve": auc_gh,
        "num_events_for_training": x_gh_train.shape[0],
        "max_gamma_ray_energy": gamma_max_energy,
    }
    roc_gh_path = os.path.join(
        pa['out_dir'],
        "{:s}_roc_gamma_hadron".format(site_key))
    with open(roc_gh_path+".json", "wt") as fout:
        fout.write(json.dumps(roc_gh, indent=4, cls=irf.json_numpy.Encoder))

    print(site_key, 'auc.', auc_gh)

    fig = irf.summary.figure.figure(sum_config['figure_16_9'])
    ax = fig.add_axes([.2, .2, .72, .72])
    ax.plot(fpr_gh, tpr_gh, 'k')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_title('area under curve {:.2f}'.format(auc_gh))
    ax.set_xlabel('false positive rate / 1\nhadron-acceptance')
    ax.set_ylabel('true positive rate / 1\ngamma-ray-acceptance')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_aspect('equal')
    plt.savefig(roc_gh_path+".jpg")
    plt.close('all')
