import os
import json_numpy
import numpy as np
import plenopy
import aberration_demo
from .. import mesh
from .. import light_field
from .. import merlict


EXAMPLE_CONFIG = {
    "seed": 1337,
    "light_field_geometry_path": os.path.join(
        ".", "run", "light_field_geometry"
    ),
    "merlict_propagate_photons_path": os.path.join(
        "build", "merlict", "merlict-plenoscope-raw-photon-propagation"
    ),
    "merlict_propagate_config_path": os.path.join(
        "resources",
        "acp",
        "merlict_propagation_config_no_night_sky_background.json",
    ),
    "aperture_radius_m": 50,
    "emission_distance_to_aperture_m": 1e3,
    "intensity": 1000,
}


def init(work_dir, config=None):
    if config is None:
        config = EXAMPLE_CONFIG
    os.makedirs(name=work_dir, exist_ok=True)
    json_numpy.write(os.path.join(work_dir, "config.json"), config)


def run(work_dir):
    config = json_numpy.read(os.path.join(work_dir, "config.json"))

    phantom_source_meshes_path = os.path.join(
        work_dir, "phantom_source_meshes.json"
    )
    phantom_source_meshes_img_path = os.path.join(
        work_dir, "phantom_source_meshes_img.json"
    )
    (
        phantom_source_meshes,
        phantom_source_meshes_img,
    ) = _make_mesh_for_phantom_source(intensity=config["intensity"])

    json_numpy.write(phantom_source_meshes_path, phantom_source_meshes)
    json_numpy.write(phantom_source_meshes_img_path, phantom_source_meshes_img)

    phantom_source_light_field_path = os.path.join(
        work_dir, "phantom_source_light_field"
    )

    if not os.path.exists(phantom_source_light_field_path):
        prng = np.random.Generator(np.random.MT19937(seed=config["seed"]))

        light_fields = light_field.make_light_fields_from_meshes(
            meshes=phantom_source_meshes,
            aperture_radius=config["aperture_radius_m"],
            prng=prng,
            emission_distance_to_aperture=config[
                "emission_distance_to_aperture_m"
            ],
        )

        (
            event,
            light_field_geometry,
        ) = merlict.make_plenopy_event_and_read_light_field_geometry(
            light_fields=light_fields,
            light_field_geometry_path=config["light_field_geometry_path"],
            merlict_propagate_photons_path=config[
                "merlict_propagate_photons_path"
            ],
            merlict_propagate_config_path=config[
                "merlict_propagate_config_path"
            ],
            random_seed=config["seed"],
        )

        (
            arrival_times_s,
            photo_sesnor_ids,
        ) = plenopy.light_field_sequence.photon_arrival_times_and_lixel_ids(
            raw_sensor_response=event.raw_sensor_response
        )

        write_light_field(
            path=phantom_source_light_field_path,
            arrival_times_s=arrival_times_s,
            photo_sesnor_ids=photo_sesnor_ids,
        )


TRUE_DEPTH = {
    "triangle": 2.5e3,
    "spiral": 4.2e3,
    "sun": 7.1e3,
    "smiley": 11.9e3,
    "cross": 20.0e3,
}

# np.geomspace(2.5, 20, 5)
# 1.4865088937534012
# 33.635856610148586
DEPTH_STEPS = np.geomspace(1.4865088937534012, 33.635856610148586, 7)


def _make_mesh_for_phantom_source(intensity=360):
    RR = 1.0
    intensity = 1e4 * intensity

    Mimg = []
    Mimg.append(
        mesh.triangle(
            pos=[-1.0, +1.3, TRUE_DEPTH["triangle"]],
            radius=1.8,
            density=intensity * (TRUE_DEPTH["triangle"] / 1e3 ** RR),
        )
    )
    Mimg.append(
        mesh.spiral(
            pos=[-1.0, -1.3, TRUE_DEPTH["spiral"]],
            turns=2.5,
            outer_radius=1.7,
            density=intensity * (TRUE_DEPTH["spiral"] / 1e3 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.sun(
            pos=[1.7, 0.0, TRUE_DEPTH["sun"]],
            num_flares=11,
            radius=1.0,
            density=intensity * (TRUE_DEPTH["sun"] / 1e3 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.smiley(
            pos=[-1.0, +1.3, TRUE_DEPTH["smiley"]],
            radius=0.9,
            density=intensity * (TRUE_DEPTH["smiley"] / 1e3 ** RR),
            fn=50,
        )
    )
    Mimg.append(
        mesh.cross(
            pos=[+1.3, -1.3, TRUE_DEPTH["cross"]],
            radius=0.7,
            density=intensity * (TRUE_DEPTH["cross"] / 1e3 ** RR),
        )
    )

    # transform to cartesian scenery
    # ------------------------------
    Mscn = []
    for mimg in Mimg:
        mscn = mesh.transform_image_to_scneney(mesh=mimg)
        Mscn.append(mscn)

    return Mscn, Mimg


def write_light_field(path, arrival_times_s, photo_sesnor_ids):
    os.makedirs(path, exist_ok=True)
    assert photo_sesnor_ids.dtype == np.uint32
    assert arrival_times_s.dtype == np.float32
    at_path = os.path.join(path, "arrival_times_s.float32")
    pi_path = os.path.join(path, "photo_sesnor_ids.uint32")
    with open(at_path, "wb") as f:
        f.write(arrival_times_s.tobytes())
    with open(pi_path, "wb") as f:
        f.write(photo_sesnor_ids.tobytes())


def read_light_field(path):
    at_path = os.path.join(path, "arrival_times_s.float32")
    pi_path = os.path.join(path, "photo_sesnor_ids.uint32")
    with open(at_path, "rb") as f:
        arrival_times_s = np.fromstring(f.read(), dtype=np.float32)
    with open(pi_path, "rb") as f:
        photo_sesnor_ids = np.fromstring(f.read(), dtype=np.uint32)
    return arrival_times_s, photo_sesnor_ids


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

    img = aberration_demo.analysis.histogram2d_std(
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
