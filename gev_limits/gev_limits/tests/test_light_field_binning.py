import gev_limits as gli
import numpy as np


def test_understand_np_digitize():
    assert np.digitize(-1.5, bins=[-1, 0, 1]) == 0
    assert np.digitize(-0.5, bins=[-1, 0, 1]) == 1
    assert np.digitize(0.5, bins=[-1, 0, 1]) == 2
    assert np.digitize(1.5, bins=[-1, 0, 1]) == 3


def test_photons_to_light_field():
    photons = gli.light_field.PhotonObservables(
        x=[23, 5],
        y=[-24, 19],
        cx=np.deg2rad([0.5, -0.5]),
        cy=np.deg2rad([0.1, 0.2]),
        relative_arrival_times=[1e-9, -1e-9])
    plenoscope = gli.light_field.init_Plenoscope(
        aperture_radius=35.5,
        num_paxel_on_diagonal=9,
        field_of_view_radius_deg=3.25,
        num_pixel_on_diagonal=97,
        time_radius=25e-9,
        num_time_slices=100)
    lfs = gli.light_field.photons_to_light_field_sequence(
        photons=photons,
        plenoscope=plenoscope)
    assert lfs.shape[0] == 2
    assert lfs.shape[1] == 5
    photons_back = gli.light_field.light_field_sequence_to_photons(
        light_field_sequence=lfs,
        plenoscope=plenoscope)
    delta_xy = plenoscope.aperture_radius/plenoscope.num_paxel_on_diagonal
    for i in range(len(photons.x)):
        assert np.abs(photons_back.x[i] - photons.x[i]) < delta_xy
    for i in range(len(photons.y)):
        assert np.abs(photons_back.y[i] - photons.y[i]) < delta_xy
    delta_cxy = np.deg2rad(
        plenoscope.field_of_view_radius_deg/plenoscope.num_pixel_on_diagonal)
    for i in range(len(photons.cx)):
        assert np.abs(photons_back.cx[i] - photons.cx[i]) < delta_cxy
    for i in range(len(photons.cy)):
        assert np.abs(photons_back.cy[i] - photons.cy[i]) < delta_cxy


def test_photons_to_light_field_out_of_xy_bin_range():
    plenoscope = gli.light_field.init_Plenoscope(
        aperture_radius=35.5,
        num_paxel_on_diagonal=9,
        field_of_view_radius_deg=3.25,
        num_pixel_on_diagonal=97,
        time_radius=25e-9,
        num_time_slices=100)
    photons = gli.light_field.PhotonObservables(
        x=[50, 0],
        y=[0, -100],
        cx=np.deg2rad([0.5, -0.5]),
        cy=np.deg2rad([0.1, 0.2]),
        relative_arrival_times=[1e-9, -1e-9])
    lfs = gli.light_field.photons_to_light_field_sequence(
        photons=photons,
        plenoscope=plenoscope)
    assert lfs.shape[0] == 0
    photons = gli.light_field.PhotonObservables(
        x=[10, 0],
        y=[0, -10],
        cx=np.deg2rad([6.5, -0.5]),
        cy=np.deg2rad([0.1, 22.2]),
        relative_arrival_times=[1e-9, -1e-9])
    lfs = gli.light_field.photons_to_light_field_sequence(
        photons=photons,
        plenoscope=plenoscope)
    assert lfs.shape[0] == 0
    photons = gli.light_field.PhotonObservables(
        x=[10, 0],
        y=[0, -10],
        cx=np.deg2rad([0.5, -0.5]),
        cy=np.deg2rad([0.1, 0.2]),
        relative_arrival_times=[100e-9, -100e-9])
    lfs = gli.light_field.photons_to_light_field_sequence(
        photons=photons,
        plenoscope=plenoscope)
    assert lfs.shape[0] == 0
