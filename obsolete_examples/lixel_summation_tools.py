import plenopy as pl
import numpy as np


def make_summation_mask_for_pixel(
    pixel_id,
    object_distance,
    light_field_geometry,
):
    number_nearest_neighbor_pixels = 1
    c_epsilon = 2*np.deg2rad(0.11)
    image_rays = pl.image.ImageRays(light_field_geometry)
    cx, cy = image_rays.cx_cy_in_object_distance(object_distance)
    cxy = np.vstack((cx, cy)).T
    distances, pixel_indicies = image_rays.pixel_pos_tree.query(
        x=cxy,
        k=number_nearest_neighbor_pixels)
    lixel_summation = [[] for i in range(light_field_geometry.number_lixel)]
    for lix in range(light_field_geometry.number_lixel):
        if distances[lix] <= c_epsilon:
            lixel_summation[pixel_indicies[lix]].append(lix)

    lixel_summation = pl.trigger.lixel_summation_to_sparse_matrix(
        lixel_summation=lixel_summation,
        number_lixel=light_field_geometry.number_lixel,
        number_pixel=light_field_geometry.number_pixel)

    mask = lixel_summation[pixel_id]
    mask = mask.todense()
    mask = 1.0 * mask
    z = np.zeros(light_field_geometry.number_lixel)
    for i in range(light_field_geometry.number_lixel):
        z[i] = mask.T[i]
    mask = z
    return mask


def add_aperture_plane_to_ax(ax, color='k'):
    c = color
    ax.plot([-1, 1], [0, 0], color=c)
    N = 25
    s = 1/N
    x_starts = np.linspace(-1, 1, N) - s
    x_ends = np.linspace(-1, 1, N)
    for i in range(N):
        ax.plot([x_starts[i], x_ends[i]], [-s, 0], color=c)


def add_rays_to_ax(ax, object_distance, color='k', linewidth=1):
    c = color
    N = 4
    x_starts = np.linspace(-0.9, 0.9, N)
    y_starts = np.zeros(N)

    x_ends = -x_starts*100
    y_ends = 2*object_distance*np.ones(N)*100

    for i in range(N):
        ax.plot(
            [x_starts[i], x_ends[i]],
            [y_starts[i], y_ends[i]],
            color=c,
            linewidth=linewidth)
