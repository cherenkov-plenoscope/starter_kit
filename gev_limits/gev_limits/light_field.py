from collections import namedtuple
import numpy as np


PhotonObservables = namedtuple(
    'PhotonObservables',
    ['x', 'y', 'cx', 'cy', 'relative_arrival_times'])


def cut_PhotonObservables(photons, mask):
    return PhotonObservables(
        x=photons.x[mask],
        y=photons.y[mask],
        cx=photons.cx[mask],
        cy=photons.cy[mask],
        relative_arrival_times=photons.relative_arrival_times[mask])


Plenoscope = namedtuple(
    'Plenoscope',
    [
        'x',
        'y',
        'cx',
        'cy',
        't',
        'x_bin_edges',
        'y_bin_edges',
        'cx_bin_edges',
        'cy_bin_edges',
        't_bin_edges',
        'aperture_radius',
        'num_paxel_on_diagonal',
        'field_of_view_radius_deg',
        'num_pixel_on_diagonal',
        'time_radius',
        'num_time_slices'
    ])


def init_Plenoscope(
    aperture_radius=35.5,
    num_paxel_on_diagonal=9,
    field_of_view_radius_deg=3.25,
    num_pixel_on_diagonal=int(np.round(6.5/0.0667)),
    time_radius=25e-9,
    num_time_slices=100,
):
    cxy_bin_edges = np.linspace(
        -np.deg2rad(field_of_view_radius_deg),
        +np.deg2rad(field_of_view_radius_deg),
        num_pixel_on_diagonal + 1)
    cxy_bin_centers = (cxy_bin_edges[0: -1] + cxy_bin_edges[1:])/2

    xy_bin_edges = np.linspace(
        -aperture_radius,
        +aperture_radius,
        num_paxel_on_diagonal + 1)
    xy_bin_centers = (xy_bin_edges[0: -1] + xy_bin_edges[1:])/2

    t_bin_edges = np.linspace(
        -time_radius,
        time_radius,
        num_time_slices + 1)
    t_bin_centers = (t_bin_edges[0: -1] + t_bin_edges[1:])/2

    return Plenoscope(
        cx=cxy_bin_centers,
        cx_bin_edges=cxy_bin_edges,
        cy=cxy_bin_centers,
        cy_bin_edges=cxy_bin_edges,
        x=xy_bin_centers,
        x_bin_edges=xy_bin_edges,
        y=xy_bin_centers,
        y_bin_edges=xy_bin_edges,
        t=t_bin_centers,
        t_bin_edges=t_bin_edges,
        aperture_radius=aperture_radius,
        num_paxel_on_diagonal=num_paxel_on_diagonal,
        field_of_view_radius_deg=field_of_view_radius_deg,
        num_pixel_on_diagonal=num_pixel_on_diagonal,
        time_radius=time_radius,
        num_time_slices=num_time_slices,)


def photons_to_light_field_sequence(photons, plenoscope):
    lfg = plenoscope
    ph = photons
    cx_idx = np.digitize(ph.cx, bins=lfg.cx_bin_edges)
    cx_valid = (cx_idx > 0)*(cx_idx < lfg.cx_bin_edges.shape[0])
    cy_idx = np.digitize(ph.cy, bins=lfg.cy_bin_edges)
    cy_valid = (cy_idx > 0)*(cy_idx < lfg.cy_bin_edges.shape[0])
    x_idx = np.digitize(ph.x, bins=lfg.x_bin_edges)
    x_valid = (x_idx > 0)*(x_idx < lfg.x_bin_edges.shape[0])
    y_idx = np.digitize(ph.y, bins=lfg.y_bin_edges)
    y_valid = (y_idx > 0)*(y_idx < lfg.y_bin_edges.shape[0])
    t_idx = np.digitize(ph.relative_arrival_times, bins=lfg.t_bin_edges)
    t_valid = (t_idx > 0)*(t_idx < lfg.t_bin_edges.shape[0])
    valid = cx_valid * cy_valid * x_valid * y_valid * t_valid
    photons_column_wise = np.array(
        [
            cx_idx[valid] - 1,
            cy_idx[valid] - 1,
            x_idx[valid] - 1,
            y_idx[valid] - 1,
            t_idx[valid] - 1
        ],
        dtype=np.uint8)
    photons_row_wise = photons_column_wise.T
    return photons_row_wise


def light_field_sequence_to_photons(light_field_sequence, plenoscope):
    lfg = plenoscope
    lfs = light_field_sequence
    return PhotonObservables(
        x=lfg.x[lfs[:, 2]],
        y=lfg.y[lfs[:, 3]],
        cx=lfg.cx[lfs[:, 0]],
        cy=lfg.cy[lfs[:, 1]],
        relative_arrival_times=lfg.t[lfs[:, 4]])


def extract_basic_features(
    photons,
    plenoscope
):
    feat = {}
    feat['mean_cx'] = np.mean(photons.cx)
    feat['mean_cy'] = np.mean(photons.cy)
    x_bins = np.histogram(photons.x, bins=plenoscope.x_bin_edges)[0]
    feat['slope_x'] = np.polyfit(plenoscope.x, x_bins, 1)[0]
    y_bins = np.histogram(photons.y, bins=plenoscope.y_bin_edges)[0]
    feat['slope_y'] = np.polyfit(plenoscope.y, y_bins, 1)[0]
    return feat
