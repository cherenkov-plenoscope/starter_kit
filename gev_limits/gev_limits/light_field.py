from collections import namedtuple
import numpy as np
from scipy import signal
from scipy import spatial
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 3e8

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


def euclidean_similarity(v0, v1):
    v_diff = v0 - v1
    diff = np.linalg.norm(v_diff)
    max_diff = np.sqrt(np.linalg.norm(v0)**2 + np.linalg.norm(v1)**2)
    return 1. - diff/max_diff


def euclidean_similarity_intensity_independent(v0, v1):
    v00 = v0/np.linalg.norm(v0)
    v11 = v1/np.linalg.norm(v1)
    return euclidean_similarity(v00, v11)


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


ImagePhotons = namedtuple(
    'ImagePhotons',
    ['x', 'y', 'dx', 'dy', 'dz'])


def get_image_photons(lfs, plenoscope, FOCAL_LENGTH=1.):
    cxs = -plenoscope.cx[lfs[:, 0]]
    cys = -plenoscope.cy[lfs[:, 1]]
    xs = plenoscope.x[lfs[:, 2]]
    ys = plenoscope.y[lfs[:, 3]]

    focal_x = -np.tan(cxs)*FOCAL_LENGTH
    focal_y = -np.tan(cys)*FOCAL_LENGTH
    num_rays = cxs.shape[0]
    aperture_pos = np.array([xs, ys, np.zeros(num_rays)]).T
    focal_pos = np.array([focal_x, focal_y, FOCAL_LENGTH*np.ones(num_rays)]).T
    img_ray_dir = focal_pos - aperture_pos
    img_ray_dir_lengths = np.sqrt(np.sum(img_ray_dir**2, axis=1))
    img_ray_dir = img_ray_dir/img_ray_dir_lengths[:, np.newaxis]

    return ImagePhotons(
        x=xs,
        y=ys,
        dx=img_ray_dir[:, 0],
        dy=img_ray_dir[:, 1],
        dz=img_ray_dir[:, 2])


def get_refocused_cx_cy(image_photons, object_distance, FOCAL_LENGTH=1.):
    sensor_distance = 1/(1/FOCAL_LENGTH - 1/object_distance)
    alpha = sensor_distance/image_photons.dz
    sens_x = image_photons.x + image_photons.dx*alpha
    sens_y = image_photons.y + image_photons.dy*alpha
    sens_cxs = -np.arctan(sens_x/FOCAL_LENGTH)
    sens_cys = -np.arctan(sens_y/FOCAL_LENGTH)
    return sens_cxs, sens_cys


def get_image_from_image_photons(
    image_photons,
    object_distance,
    cx_bin_edges,
    cy_bin_edges,
):
    sens_cxs, sens_cys = get_refocused_cx_cy(
        image_photons=image_photons,
        object_distance=object_distance)
    return np.histogram2d(
        -sens_cxs,
        -sens_cys,
        bins=(cx_bin_edges, cy_bin_edges))[0]


def get_refocus_stack(
    image_photons,
    object_distances,
    cx_bin_edges,
    cy_bin_edges
):
    imgs = []
    for object_distance in object_distances:
        img = get_image_from_image_photons(
            image_photons=image_photons,
            object_distance=object_distance,
            cx_bin_edges=cx_bin_edges,
            cy_bin_edges=cy_bin_edges)
        imgs.append(img)
    return np.array(imgs)


GAUSS_2D_3X3 = (1./16.)*np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])

def smear_out_refocus_stack(
    refocus_stack,
    kernel=GAUSS_2D_3X3
):
    cimgs = []
    for i in range(refocus_stack.shape[0]):
        img = refocus_stack[i, :, :]
        cimg = signal.convolve2d(img, kernel)
        cimgs.append(cimg)
    return np.array(cimgs)


def get_light_front(lfs, plenoscope):
    xs = plenoscope.x[lfs[:, 2]]
    ys = plenoscope.y[lfs[:, 3]]
    ts = plenoscope.t[lfs[:, 4]]
    zs = SPEED_OF_LIGHT*ts
    return xs, ys, zs


def add_to_tpap_to_get_timg(cx, cy, x, y):
    # cz = np.sqrt(1. - cx**2 - cy**2)
    delta_path_length = cx*x + cy*y
    delta_path_time = delta_path_length/SPEED_OF_LIGHT
    return -1.*delta_path_time
