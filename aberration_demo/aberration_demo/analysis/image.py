from . import analysis
import plenopy
import numpy as np


def compute_image(
    light_field_geometry, light_field, object_distance, bins, prng
):
    photon_arrival_times_s, photon_lixel_ids = light_field

    image_rays = plenopy.image.ImageRays(
        light_field_geometry=light_field_geometry
    )

    (image_beams_cx, image_beams_cy,) = image_rays.cx_cy_in_object_distance(
        object_distance
    )
    image_beams_cx_std = light_field_geometry.cx_std
    image_beams_cy_std = light_field_geometry.cy_std

    weights = np.zeros(light_field_geometry.number_lixel, dtype=np.uint)
    for lixel_id in photon_lixel_ids:
        weights[lixel_id] += 1

    img = analysis.histogram2d_std(
        x=image_beams_cx,
        y=image_beams_cy,
        x_std=image_beams_cx_std,
        y_std=image_beams_cy_std,
        weights=weights,
        bins=bins,
        prng=prng,
        num_sub_samples=10,
    )[0]

    return img


def write_image(path, image):
    imo = image.astype(np.float32)
    x = np.array([imo.shape[0]], dtype=np.uint64)
    y = np.array([imo.shape[1]], dtype=np.uint64)
    with open(path, "wb") as f:
        f.write(x.tobytes())
        f.write(y.tobytes())
        f.write(imo.flatten(order="C").tobytes())


def read_image(path):
    with open(path, "rb") as f:
        x = np.fromstring(f.read(8), dtype=np.uint64)[0]
        y = np.fromstring(f.read(8), dtype=np.uint64)[0]
        img = np.fromstring(f.read(), dtype=np.float32)
    img = np.reshape(img, (x, y), order="C")
    return img
