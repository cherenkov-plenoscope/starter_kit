import os
import json_numpy
import numpy as np
from .. import mesh
from .. import light_field


EXAMPLE_CONFIG = {
    "seed": 1337,
    "light_field_geometry_path": os.path.join(".", "run", "light_field_geometry"),
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
}


def init(work_dir, config=None):
    if config is None:
        config = EXAMPLE_CONFIG
    os.makedirs(name=work_dir, exist_ok=True)
    json_numpy.write(os.path.join(work_dir, "config.json"), config)


def run(work_dir):
    config = json_numpy.read(os.path.join(work_dir, "config.json"))

    phantom_source_meshes_path = os.path.join(work_dir, "phantom_source_meshes.json")

    phantom_source_meshes = None
    if not os.path.exists(phantom_source_meshes_path):
        phantom_source_meshes = _make_mesh_for_phantom_source()
        json_numpy.write(phantom_source_meshes_path, phantom_source_meshes)

    phantom_source_light_field_path = os.path.join(work_dir, "phantom_source_light_field")

    if not os.path.exists(phantom_source_light_field_path):
        prng = np.random.Generator(np.random.MT19937(seed=config["seed"]))

        if not phantom_source_meshes:
            phantom_source_meshes = json_numpy.read(phantom_source_meshes_path)

        light_fields = light_field.make_light_fields_from_meshes(
            meshes=phantom_source_meshes,
            aperture_radius=config["aperture_radius_m"],
            prng=prng,
            emission_distance_to_aperture=config["emission_distance_to_aperture_m"],
        )

        (
            event,
            light_field_geometry,
        ) = phantom_source.merlict.make_plenopy_event_and_read_light_field_geometry(
        light_fields=light_fields,
        light_field_geometry_path=config["light_field_geometry_path"],
        merlict_propagate_photons_path=config["merlict_propagate_photons_path"],
        merlict_propagate_config_path=config["merlict_propagate_config_path"],
        random_seed=config["seed"],
        )

        arrival_times_s, photo_sesnor_ids = event.photon_arrival_times_and_lixel_ids()

        write_light_field(
            path=phantom_source_light_field_path,
            arrival_times_s=arrival_times_s,
            photo_sesnor_ids=photo_sesnor_ids,
        )


def _make_mesh_for_phantom_source(intensity=1e4 * 120):
    RR=1.0

    Mimg = []
    Mimg.append(
        mesh.triangle(
            pos=[-1.0, +1.3, 2.5e3], radius=1.8, density=intensity * (2.5 ** RR),
        )
    )
    Mimg.append(
        mesh.spiral(
            pos=[-1.0, -1.3, 4.2e3],
            turns=2.5,
            outer_radius=1.7,
            density=intensity * (4.2 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.sun(
            pos=[1.7, 0.0, 7.1e3],
            num_flares=11,
            radius=1.0,
            density=intensity * (7.1 ** RR),
            fn=110,
        )
    )
    Mimg.append(
        mesh.smiley(
            pos=[-1.0, +1.3, 11.9e3],
            radius=0.9,
            density=intensity * (11.9 ** RR),
            fn=50,
        )
    )
    Mimg.append(
        mesh.cross(
            pos=[+1.3, -1.3, 20.0e3], radius=0.7, density=intensity * (20.0 ** RR),
        )
    )

    # transform to cartesian scenery
    # ------------------------------
    Mscn = []
    for mimg in Mimg:
        mscn = mesh.transform_image_to_scneney(mesh=mimg)
        Mscn.append(mscn)

    return Mscn


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
        arrival_times_s = np.fromstring( f.read(), dtype=np.float32)
    with open(pi_path, "rb") as f:
        photo_sesnor_ids = np.fromstring( f.read(), dtype=np.uint32)
    arrival_times_s, photo_sesnor_ids