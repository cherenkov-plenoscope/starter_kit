from collections import namedtuple
import numpy as np
from scipy import signal
from scipy import spatial
import matplotlib.pyplot as plt


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


def angle_in_between(v0, v1):
    return np.arccos(
        np.dot(v0, v1)/(np.linalg.norm(v0)*np.linalg.norm(v1)))


def euclidean_similarity(v0, v1):
    v_diff = v0 - v1
    diff = np.linalg.norm(v_diff)
    max_diff = np.sqrt(np.linalg.norm(v0)**2 + np.linalg.norm(v1)**2)
    return 1. - diff/max_diff


def euclidean_similarity_intensity_independent(v0, v1):
    v00 = v0/np.linalg.norm(v0)
    v11 = v1/np.linalg.norm(v1)
    return euclidean_similarity(v00, v11)


def slice_into_image_sequence(lfs, plenoscope, paxel_radius=1):
    image_sequences = []
    p = plenoscope
    pr = paxel_radius
    for i, x in enumerate(p.x):
        for j, y in enumerate(p.y):
            mask = (
                (lfs[:, 2]>i-pr)*(lfs[:, 2]<i+pr)*
                (lfs[:, 3]>j-pr)*(lfs[:, 3]<j+pr))
            image_sequences.append(
                np.array([lfs[mask, 0], lfs[mask, 1], lfs[mask, 4]]).T)
    return image_sequences


def diluted_images(
    lfs,
    plenoscope,
    paxel_dilution_radius=1,
    pixel_dilution_mask=(1./273.)*np.array(
    [
        [ 0,  4,  7,  4,  0],
        [ 4, 16, 26, 16,  4],
        [ 7, 26, 41, 26,  7],
        [ 4, 16, 26, 16,  4],
        [ 0,  4,  7,  4,  0]
    ])
):
    images = slice_into_image_sequence(
        lfs=lfs,
        plenoscope=plenoscope,
        paxel_radius=paxel_dilution_radius)
    imgs = []
    for image in images:
        img = np.histogram2d(
            image[:, 0],
            image[:, 1],
            bins=np.arange(plenoscope.num_pixel_on_diagonal+1))[0]
        img = signal.convolve2d(img, pixel_dilution_mask)
        imgs.append(img)
    return imgs


def similarity_light_field(lfs0, lfs1, plenoscope):
    imgs0 = diluted_images(lfs0, plenoscope)
    imgs1 = diluted_images(lfs1, plenoscope)
    sims = []
    weights = []
    for i in range(len(imgs0)):
        img0 = imgs0[i]
        img1 = imgs1[i]
        num_ph_0 = np.sum(img0)
        num_ph_1 = np.sum(img1)
        if num_ph_0 > 0 and num_ph_1 > 0:
            sim_i = euclidean_similarity_intensity_independent(
                img0.flatten(),
                img1.flatten())
            sims.append(sim_i)
        else:
            sims.append(0.)
        weights.append(num_ph_0 + num_ph_1)
    return np.average(sims, weights=weights)


def similarity_image(
    lfs0,
    lfs1,
    plenoscope,
    pixel_dilution_mask=(1./273.)*np.array(
        [
            [ 0,  4,  7,  4,  0],
            [ 4, 16, 26, 16,  4],
            [ 7, 26, 41, 26,  7],
            [ 4, 16, 26, 16,  4],
            [ 0,  4,  7,  4,  0]
        ])
):
    img0 = np.histogram2d(
        lfs0[:, 0],
        lfs0[:, 1],
        bins=np.arange(plenoscope.num_pixel_on_diagonal+1))[0]
    img1 = np.histogram2d(
        lfs1[:, 0],
        lfs1[:, 1],
        bins=np.arange(plenoscope.num_pixel_on_diagonal+1))[0]
    img0 = signal.convolve2d(img0, pixel_dilution_mask)
    img1 = signal.convolve2d(img1, pixel_dilution_mask)
    sim = euclidean_similarity_intensity_independent(
        img0.flatten(),
        img1.flatten())
    """
    print("sim", sim)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.pcolor(img0)
    ax2.pcolor(img1)
    plt.show()
    """
    return sim



def similarity_aperture(
    lfs0,
    lfs1,
    plenoscope,
    paxel_dilution_mask=(1./16.)*np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])
):
    ap0 = np.histogram2d(
        lfs0[:, 2],
        lfs0[:, 3],
        bins=np.arange(plenoscope.num_paxel_on_diagonal+1))[0]
    ap1 = np.histogram2d(
        lfs1[:, 2],
        lfs1[:, 3],
        bins=np.arange(plenoscope.num_paxel_on_diagonal+1))[0]
    ap0 = signal.convolve2d(ap0, paxel_dilution_mask)
    ap1 = signal.convolve2d(ap1, paxel_dilution_mask)
    sim = euclidean_similarity_intensity_independent(
        ap0.flatten(),
        ap1.flatten())
    """
    print("sim", sim)
    f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    ax1.pcolor(ap0)
    ax2.pcolor(ap1)
    plt.show()
    """
    return sim