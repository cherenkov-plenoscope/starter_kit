import os
from os.path import join
from subprocess import call
import plenopy as pl
import tempfile
import numpy as np
import pytest
import corsika_wrapper as cw
import shutil


@pytest.fixture(scope='session')
def tmp(tmpdir_factory):
    fn = tmpdir_factory.mktemp('merlict_plenopy')
    call([
        join('build', 'merlict', 'merlict-plenoscope-calibration'),
        '--scenery', join('resources', 'iact', 'MAGIC_1', 'scenery'),
        '--num_mega_photons', '5',
        '--output', join(fn, 'light_field_geometry')
    ])
    return fn


def test_light_field_geometry_with_plenopy(tmp):
    lfg = pl.LightFieldGeometry(join(tmp, 'light_field_geometry'))
    assert lfg.number_pixel == 1039
    assert lfg.number_paxel == 19
    assert lfg.number_lixel == 1039*19

    assert lfg.expected_aperture_radius_of_imaging_system == 8.5
    assert lfg.expected_focal_length_of_imaging_system == 17
    assert np.isclose(
        lfg.sensor_plane2imaging_system.housing_overhead,
        1.2,
        rtol=1e-6)
    assert np.isclose(
        lfg.sensor_plane2imaging_system.max_FoV_diameter,
        np.deg2rad(3.5),
        atol=np.rad2deg(1e-3))

    assert np.isclose(lfg.lixel_outer_radius, 0.0037088257)

    assert np.isclose(np.mean(lfg.efficiency), 0.2957, atol=0.03)

    assert np.isclose(np.mean(lfg.cx_mean), 0.0, atol=1e-5)
    assert np.isclose(np.mean(lfg.cy_mean), 0.0, atol=1e-5)


def test_plot_light_field_geometry_with_plenopy(tmp):
    lfg = pl.LightFieldGeometry(join(tmp, 'light_field_geometry'))
    pl.plot.light_field_geometry.save_all(
        lfg,
        join(tmp, 'light_field_geometry', 'plot'))

    expected_images = [
        'c_mean_vs_c_std.jpg',
        'cx_cy_mean_hist2d.jpg',
        'cx_mean.jpg',
        'cx_std.jpg',
        'cy_mean.jpg',
        'cy_std.jpg',
        'eta_mean.jpg',
        'eta_std.jpg',
        'overview_cx_mean.jpg',
        'overview_cx_mean_zoom_center.jpg',
        'overview_cx_mean_zoom_pos_x.jpg',
        'overview_cx_mean_zoom_pos_y.jpg',
        'overview_cx_std.jpg',
        'overview_cx_std_zoom_center.jpg',
        'overview_cx_std_zoom_pos_x.jpg',
        'overview_cx_std_zoom_pos_y.jpg',
        'overview_cy_mean.jpg',
        'overview_cy_mean_zoom_center.jpg',
        'overview_cy_mean_zoom_pos_x.jpg',
        'overview_cy_mean_zoom_pos_y.jpg',
        'overview_cy_std.jpg',
        'overview_cy_std_zoom_center.jpg',
        'overview_cy_std_zoom_pos_x.jpg',
        'overview_cy_std_zoom_pos_y.jpg',
        'overview_eta_mean.jpg',
        'overview_eta_mean_zoom_center.jpg',
        'overview_eta_mean_zoom_pos_x.jpg',
        'overview_eta_mean_zoom_pos_y.jpg',
        'overview_t_mean_aperture.jpg',
        'overview_t_mean_aperture_zoom_center.jpg',
        'overview_t_mean_aperture_zoom_pos_x.jpg',
        'overview_t_mean_aperture_zoom_pos_y.jpg',
        'overview_t_mean_image.jpg',
        'overview_t_mean_image_zoom_center.jpg',
        'overview_t_mean_image_zoom_pos_x.jpg',
        'overview_t_mean_image_zoom_pos_y.jpg',
        'overview_x_mean.jpg',
        'overview_x_mean_zoom_center.jpg',
        'overview_x_mean_zoom_pos_x.jpg',
        'overview_x_mean_zoom_pos_y.jpg',
        'overview_x_std.jpg',
        'overview_x_std_zoom_center.jpg',
        'overview_x_std_zoom_pos_x.jpg',
        'overview_x_std_zoom_pos_y.jpg',
        'overview_y_mean.jpg',
        'overview_y_mean_zoom_center.jpg',
        'overview_y_mean_zoom_pos_x.jpg',
        'overview_y_mean_zoom_pos_y.jpg',
        'overview_y_std.jpg',
        'overview_y_std_zoom_center.jpg',
        'overview_y_std_zoom_pos_x.jpg',
        'overview_y_std_zoom_pos_y.jpg',
        't_mean.jpg',
        't_std.jpg',
        'x_mean.jpg',
        'x_std.jpg',
        'x_y_mean_hist2d.jpg',
        'y_mean.jpg',
        'y_std.jpg',
    ]

    for image_filename in expected_images:
        assert os.path.exists(
            join(tmp, 'light_field_geometry', 'plot', image_filename))


def test_corsika_simulation(tmp):
    card = cw.read_steering_card(
        join('resources', 'acp', '71m', 'calibration_gamma_event.txt'))
    rc = cw.corsika(
        card,
        join(tmp, 'calibration_gamma.evtio'),
        save_stdout=True)
    assert rc == 0
    assert os.path.exists(join(tmp, 'calibration_gamma.evtio'))


def test_propagation_with_mctracer(tmp):
    assert os.path.exists(join(tmp, 'calibration_gamma.evtio'))
    rc = call([
        join('build', 'merlict', 'merlict-plenoscope-propagation'),
        '--lixel', join(tmp, 'light_field_geometry'),
        '--config', join(
            'resources', 'acp',
            'merlict_propagation_config_no_night_sky_background.json'),
        '--input', join(tmp, 'calibration_gamma.evtio'),
        '--output', join(tmp, 'calibration_gamma.acp'),
        '--all_truth',
        '--random_seed', '1'
    ])
    assert rc == 0


def test_read_run_with_plenopy(tmp):
    assert os.path.exists(join(tmp, 'calibration_gamma.acp'))
    run = pl.Run(join(tmp, 'calibration_gamma.acp'))
    assert len(run) > 1
    np.random.seed(0)

    direction_from_image_correct = 0
    direction_from_arrivale_times_correct = 0
    plane_fit_correct = 0

    for event in run:
        # ASSERT the incoming directions of the photons are as expected
        # -------------------------------------------------------------
        m = event.simulation_truth.event.corsika_event_header.momentum().copy()
        assert m[2] >= 0.0  # the negative z-component
        m[2] *= -1.0
        momentum_vector_in_corsika_frame = m

        approx_direction_of_photons_in_corsika_frame = (
            momentum_vector_in_corsika_frame/np.linalg.norm(
                momentum_vector_in_corsika_frame))
        # photons are running down towards the observation level.

        approx_incoming_direction_of_photons_in_plenoscope_frame = (
            -1.0 * approx_direction_of_photons_in_corsika_frame)
        # incoming directions for the plenoscope are pointing upwards into
        # the sky.

        (
            arrival_slices, lixel_ids
        ) = pl.photon_stream.cython_reader.arrival_slices_and_lixel_ids(
            event.raw_sensor_response)

        cx_mean = np.mean(run.light_field_geometry.cx_mean[lixel_ids])
        cy_mean = np.mean(run.light_field_geometry.cy_mean[lixel_ids])
        cx_std = np.std(run.light_field_geometry.cx_mean[lixel_ids])
        cy_std = np.std(run.light_field_geometry.cy_mean[lixel_ids])

        expected_cx_mean = (
            approx_incoming_direction_of_photons_in_plenoscope_frame[0])
        expected_cy_mean = (
            approx_incoming_direction_of_photons_in_plenoscope_frame[1])

        if (
            np.isclose(expected_cx_mean, cx_mean, atol=np.deg2rad(0.3)) and
            np.isclose(expected_cy_mean, cy_mean, atol=np.deg2rad(0.3))
        ):
            direction_from_image_correct += 1

        # ASSERT arrival_slices are as expected
        # -------------------------------------
        arrival_times_in_sensors = (
            arrival_slices*event.raw_sensor_response.time_slice_duration)

        arrival_times_in_sensors -= np.median(arrival_times_in_sensors)
        # absolute arrival time is not relevant

        arrival_times_on_principal_aperture_plane = (
            arrival_times_in_sensors -
            run.light_field_geometry.time_delay_mean[lixel_ids])

        speed_of_light = 3e8

        arrival_path_delays_on_principal_aperture_plane = (
            arrival_times_on_principal_aperture_plane*speed_of_light)

        zs = arrival_path_delays_on_principal_aperture_plane
        xs = run.light_field_geometry.x_mean[lixel_ids]
        ys = run.light_field_geometry.y_mean[lixel_ids]

        xyzs = np.c_[xs, ys, zs]

        plane_model, inlier = pl.tools.ransac_3d_plane.fit(
            xyz_point_cloud=xyzs,
            max_number_itarations=1000,
            min_number_points_for_plane_fit=3,
            max_orthogonal_distance_of_inlier=(
                event.raw_sensor_response.time_slice_duration*speed_of_light))

        if inlier.sum() >= 0.5*event.raw_sensor_response.number_photons:
            plane_fit_correct += 1

        if (
            np.isclose(
                expected_cx_mean, plane_model[0], atol=np.deg2rad(0.5))
            and
            np.isclose(
                expected_cy_mean, plane_model[1], atol=np.deg2rad(0.5))
        ):
            direction_from_arrivale_times_correct += 1

    number_events = len(run)
    assert direction_from_image_correct >= 0.75*number_events
    assert direction_from_arrivale_times_correct >= 0.34*number_events
    assert plane_fit_correct >= 0.34*number_events
