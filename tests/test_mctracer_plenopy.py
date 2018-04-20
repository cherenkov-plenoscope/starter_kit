import os
from os.path import join
from subprocess import call
import plenopy as pl
import tempfile
import numpy as np
import pytest


@pytest.fixture(scope='session')
def tmp(tmpdir_factory):
    fn = tmpdir_factory.mktemp('mctracer_plenopy')
    call([
        join('build', 'mctracer', 'mctPlenoscopeCalibration'),
        '--scenery', join('resources', 'iact', 'MAGIC_1', 'scenery'),
        '--number_mega_photons', '5',
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
    plotter = pl.plot.light_field_geometry_2.PlotLightFieldGeometry(
        lfg, join(tmp, 'light_field_geometry', 'plot'))
    plotter.save()

    expected_images = [
        'c_mean_vs_c_std.png',
        'overview_cx_stddev_zoom_pos_y.png',
        'overview_time_delay_to_img_zoom_pos_x.png',
        'overview_y_mean_zoom_center.png',
        'cx_cy_mean_hist2d.png',
        'overview_cy_mean.png',
        'overview_time_delay_to_img_zoom_pos_y.png',
        'overview_y_mean_zoom_pos_x.png',
        'cx_mean.png',
        'overview_cy_mean_zoom_center.png',
        'overview_time_delay_to_pap.png',
        'overview_y_mean_zoom_pos_y.png',
        'cx_stddev.png',
        'overview_cy_mean_zoom_pos_x.png',
        'overview_time_delay_to_pap_zoom_center.png',
        'overview_y_stddev.png',
        'cy_mean.png',
        'overview_cy_mean_zoom_pos_y.png',
        'overview_time_delay_to_pap_zoom_pos_x.png',
        'overview_y_stddev_zoom_center.png',
        'cy_stddev.png',
        'overview_cy_stddev.png',
        'overview_time_delay_to_pap_zoom_pos_y.png',
        'overview_y_stddev_zoom_pos_x.png',
        'efficiency_error.png',
        'overview_cy_stddev_zoom_center.png',
        'overview_x_mean.png',
        'overview_y_stddev_zoom_pos_y.png',
        'efficiency.png',
        'overview_cy_stddev_zoom_pos_x.png',
        'overview_x_mean_zoom_center.png',
        'time_delay_mean.png',
        'overview_cx_mean.png',
        'overview_cy_stddev_zoom_pos_y.png',
        'overview_x_mean_zoom_pos_x.png',
        'time_stddev.png',
        'overview_cx_mean_zoom_center.png',
        'overview_efficiency.png',
        'overview_x_mean_zoom_pos_y.png',
        'x_mean.png',
        'overview_cx_mean_zoom_pos_x.png',
        'overview_efficiency_zoom_center.png',
        'overview_x_stddev.png',
        'x_stddev.png',
        'overview_cx_mean_zoom_pos_y.png',
        'overview_efficiency_zoom_pos_x.png',
        'overview_x_stddev_zoom_center.png',
        'x_y_mean_hist2d.png',
        'overview_cx_stddev.png',
        'overview_efficiency_zoom_pos_y.png',
        'overview_x_stddev_zoom_pos_x.png',
        'y_mean.png',
        'overview_cx_stddev_zoom_center.png',
        'overview_time_delay_to_img.png',
        'overview_x_stddev_zoom_pos_y.png',
        'y_stddev.png',
        'overview_cx_stddev_zoom_pos_x.png',
        'overview_time_delay_to_img_zoom_center.png',
        'overview_y_mean.png',
    ]

    for image_filename in expected_images:
        assert os.path.exists(
            join(tmp, 'light_field_geometry', 'plot', image_filename))
