import os
import numpy as np
import json_numpy
import copy
import binning_utils
import scipy
from scipy.interpolate import interp2d as scipy_interpolate_interp2d
import skimage
from skimage import io as skimage_io
import pkg_resources

EXAMPLE_DEFORMATION_MAP_PATH = pkg_resources.resource_filename(
    "aberration_demo", "deformations/resources/example_deformation_map"
)


def read(path):
    z_map = skimage_io.imread(os.path.join(path, "z.png"))
    original_dtype = z_map.dtype
    z_map = z_map.astype(np.float32)
    z_map /= np.iinfo(original_dtype).max

    if len(z_map.shape) == 3:
        z_map = (1 / 3) * z_map[:, :, 0] + z_map[:, :, 1] + z_map[:, :, 2]

    with open(os.path.join(path, "scale.json"), "rt") as f:
        scale = json_numpy.loads(f.read())

    return init(z_map=z_map, **scale)


def init(
    z_map, mirror_diameter_m, intensity_per_m, z_0_offset,
):
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
        z=(z_map - z_0_offset) / intensity_per_m,
        kind="cubic",
    )
    return cc


def init_zero(mirror_diameter_m,):
    z_map = np.zeros(shape=(80, 80))
    return init(
        z_map=z_map,
        mirror_diameter_m=mirror_diameter_m,
        intensity_per_m=1.0,
        z_0_offset=0.0,
    )


def evaluate(deformation_map, x_m, y_m):
    d = np.median(deformation_map["pixel_bin"]["widths"])
    mi = deformation_map["pixel_bin"]["limits"][0] - d
    ma = deformation_map["pixel_bin"]["limits"][1] + d
    assert mi < x_m < ma
    assert mi < y_m < ma
    return deformation_map["z"](x_m, y_m)[0]
