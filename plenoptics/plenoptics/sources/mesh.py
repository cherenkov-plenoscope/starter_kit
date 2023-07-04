import phantom_source
import numpy as np
import tempfile
import json_utils
import os
from .. import utils


EXAMPLE_MESH_CONFIG = {
    "type": "mesh",
    "meshes": [
        phantom_source.mesh.triangle(pos=[0, 0, 1e4], radius=100, density=1)
    ],
    "seed": 42,
}


def make_response_to_mesh(
    mesh_config,
    light_field_geometry_path,
    merlict_config,
    emission_distance_to_aperture_m=1e3,
):
    instgeom = utils.get_instrument_geometry_from_light_field_geometry(
        light_field_geometry_path=light_field_geometry_path
    )
    illum_radius = (
        1.5 * instgeom["expected_imaging_system_max_aperture_radius"]
    )

    prng = np.random.Generator(np.random.PCG64(mesh_config["seed"]))

    light_fields = phantom_source.light_field.make_light_fields_from_meshes(
        meshes=mesh_config["meshes"],
        aperture_radius=illum_radius,
        prng=prng,
        emission_distance_to_aperture=emission_distance_to_aperture_m,
    )
    merlict_random_seed = prng.integers(low=0, high=2 ** 32)

    with tempfile.TemporaryDirectory(
        prefix="plenoscope-aberration-demo_"
    ) as tmp_dir:
        merlict_plenoscope_propagator_config_path = os.path.join(
            tmp_dir, "merlict_propagation_config.json"
        )
        json_utils.write(
            merlict_plenoscope_propagator_config_path,
            merlict_config["merlict_propagation_config"],
        )

        (
            event,
            light_field_geometry,
        ) = phantom_source.merlict.make_plenopy_event_and_read_light_field_geometry(
            light_fields=light_fields,
            light_field_geometry_path=light_field_geometry_path,
            merlict_propagate_photons_path=merlict_config["executables"][
                "merlict_plenoscope_raw_photon_propagation_path"
            ],
            merlict_propagate_config_path=merlict_plenoscope_propagator_config_path,
            random_seed=merlict_random_seed,
            work_dir=None,
        )
        return event.raw_sensor_response


def make_source_config_from_job(job):
    phantom_cfg = json_utils.tree.read(
        os.path.join(job["work_dir"], "config", "observations", "phantom")
    )
    source_config = {
        "type": "mesh",
        "meshes": phantom_cfg["phantom_source_meshes"],
        "seed": job["number"],
    }
    return source_config
