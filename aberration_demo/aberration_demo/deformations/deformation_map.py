import os
import numpy as np
import json_numpy
import copy
import binning_utils
import scipy
from scipy.interpolate import interp2d as scipy_interpolate_interp2d
import perlin_noise


def init_from_z_map(z_map, mirror_diameter_m):
    """
    Parameters
    ----------
    z_map : np.array 2D / m
        Deformation amplitudes along the z-axis of the mirror in units of
        meters.
    mirror_diameter_m : float / m
        Diameter of the map/mirror. Better make this a little bit bigger than
        the actual diameter of the mirror.
    """
    assert z_map.shape[0] == z_map.shape[1]
    cc = {}
    cc["pixel_bin"] = binning_utils.Binning(
        bin_edges=np.linspace(
            -mirror_diameter_m / 2, mirror_diameter_m / 2, z_map.shape[0] + 1
        )
    )
    cc["z"] = scipy_interpolate_interp2d(
        x=cc["pixel_bin"]["centers"],
        y=cc["pixel_bin"]["centers"],
        z=z_map,
        kind="cubic",
    )
    return cc


def init_from_perlin_noise(
    mirror_diameter_m,
    amplitude_m,
    offset_m,
    perlin_noise_octaves,
    perlin_noise_seed,
    perlin_noise_num_bins_on_edge,
):
    png = perlin_noise.PerlinNoise(
        octaves=perlin_noise_octaves, seed=perlin_noise_seed,
    )

    N = perlin_noise_num_bins_on_edge
    z_map = np.zeros(shape=(N, N), dtype=np.float32,)

    for x in range(N):
        for y in range(N):
            z_map[x, y] = png.noise([x / N, y / N])

    z_map *= amplitude_m
    z_map += offset_m

    return init_from_z_map(z_map=z_map, mirror_diameter_m=mirror_diameter_m,)


def init_zero(mirror_diameter_m, num_bins_on_edge=8):
    z_map = np.zeros(shape=(num_bins_on_edge, num_bins_on_edge))
    return init_from_z_map(z_map=z_map, mirror_diameter_m=mirror_diameter_m,)


def evaluate(deformation_map, x_m, y_m):
    d = np.median(deformation_map["pixel_bin"]["widths"])
    mi = deformation_map["pixel_bin"]["limits"][0] - d
    ma = deformation_map["pixel_bin"]["limits"][1] + d
    assert mi < x_m < ma
    assert mi < y_m < ma
    return deformation_map["z"](x_m, y_m)[0]
