import plenoirf
import os
import plenopy
import tempfile
import phantom_source
import numpy as np
import json_numpy
from ... import calibration_source


def make_response_to_source(
    source_config, light_field_geometry_path, merlict_config,
):
    if source_config["type"] == "star":
        return make_response_to_star(
            star_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "mesh":
        return make_response_to_mesh(
            mesh_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "point":
        return make_response_to_point(
            point_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    else:
        raise AssertionError("Type of source is not known")


EXAMPLE_STAR_CONFIG = {
    "type": "star",
    "cx_deg": 0.0,
    "cy_deg": 1.0,
    "areal_photon_density_per_m2": 20,
    "seed": 122,
}


def make_response_to_star(
    star_config, light_field_geometry_path, merlict_config,
):
    instgeom = get_instrument_geometry_from_light_field_geometry(
        light_field_geometry_path=light_field_geometry_path
    )
    illum_radius = (
        1.5 * instgeom["expected_imaging_system_max_aperture_radius"]
    )
    illum_area = np.pi * illum_radius ** 2
    num_photons = int(
        np.round(star_config["areal_photon_density_per_m2"] * illum_area)
    )

    prng = np.random.Generator(np.random.PCG64(star_config["seed"]))

    with tempfile.TemporaryDirectory(
        prefix="plenoscope-aberration-demo_"
    ) as tmp_dir:
        star_light_path = os.path.join(tmp_dir, "star_light.tar")

        calibration_source.write_photon_bunches(
            cx=np.deg2rad(star_config["cx_deg"]),
            cy=np.deg2rad(star_config["cy_deg"]),
            size=num_photons,
            path=star_light_path,
            prng=prng,
            aperture_radius=illum_radius,
            BUFFER_SIZE=10000,
        )

        run_path = os.path.join(tmp_dir, "run")

        merlict_plenoscope_propagator_config_path = os.path.join(
            tmp_dir, "merlict_propagation_config.json"
        )
        json_numpy.write(
            merlict_plenoscope_propagator_config_path,
            merlict_config["merlict_propagation_config"],
        )

        plenoirf.production.merlict.plenoscope_propagator(
            corsika_run_path=star_light_path,
            output_path=run_path,
            light_field_geometry_path=light_field_geometry_path,
            merlict_plenoscope_propagator_path=merlict_config["executables"][
                "merlict_plenoscope_propagation_path"
            ],
            merlict_plenoscope_propagator_config_path=merlict_plenoscope_propagator_config_path,
            random_seed=star_config["seed"],
            photon_origins=True,
            stdout_path=run_path + ".o",
            stderr_path=run_path + ".e",
        )

        run = plenopy.Run(path=run_path)
        event = run[0]
        return event.raw_sensor_response


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
    instgeom = get_instrument_geometry_from_light_field_geometry(
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
        json_numpy.write(
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


EXAMPLE_POINT_CONFIG = {
    "type": "point",
    "cx_deg": 0.0,
    "cy_deg": 1.0,
    "depth_m": 10e3,
    "areal_photon_density_per_m2": 20,
    "seed": 122,
}


def make_response_to_point(
    point_config,
    light_field_geometry_path,
    merlict_config,
    point_source_apparent_radius_deg=0.005,
    emission_distance_to_aperture_m=1e3,
):
    instgeom = get_instrument_geometry_from_light_field_geometry(
        light_field_geometry_path=light_field_geometry_path
    )
    illum_radius = (
        1.5 * instgeom["expected_imaging_system_max_aperture_radius"]
    )
    illum_area = np.pi * illum_radius ** 2
    num_photons = point_config["areal_photon_density_per_m2"] * illum_area

    _source_edge_length = (
        np.deg2rad(point_source_apparent_radius_deg) * point_config["depth_m"]
    )

    SOME_FACORT_TO_GET_DENSITY_RIGHT = 2.16

    density = phantom_source.light_field.get_edge_density_from_number_photons(
        number_photons=num_photons / SOME_FACORT_TO_GET_DENSITY_RIGHT,
        edge_length=_source_edge_length,
        distance_to_aperture=point_config["depth_m"],
        aperture_radius=illum_radius,
    )

    mesh_img = phantom_source.mesh.triangle(
        pos=[
            point_config["cx_deg"],
            point_config["cy_deg"],
            point_config["depth_m"],
        ],
        radius=point_source_apparent_radius_deg,
        density=density,
    )
    mesh_scn = phantom_source.mesh.transform_image_to_scneney(mesh=mesh_img)

    mesh_config = {}
    mesh_config["type"] = "mesh"
    mesh_config["meshes"] = [mesh_scn]
    mesh_config["seed"] = point_config["seed"]

    return make_response_to_mesh(
        mesh_config=mesh_config,
        light_field_geometry_path=light_field_geometry_path,
        merlict_config=merlict_config,
        emission_distance_to_aperture_m=emission_distance_to_aperture_m,
    )


def get_instrument_geometry_from_light_field_geometry(
    light_field_geometry=None, light_field_geometry_path=None,
):
    if light_field_geometry_path:
        assert light_field_geometry is None
        geom_path = os.path.join(
            light_field_geometry_path, "light_field_sensor_geometry.header.bin"
        )
        geom_header = plenopy.corsika.utils.hr.read_float32_header(geom_path)
        geom = plenopy.light_field_geometry.PlenoscopeGeometry(raw=geom_header)
    else:
        geom = light_field_geometry.sensor_plane2imaging_system
    return class_members_to_dict(c=geom)


def class_members_to_dict(c):
    member_keys = []
    for key in dir(c):
        if not callable(getattr(c, key)):
            if not str.startswith(key, "__"):
                member_keys.append(key)
    out = {}
    for key in member_keys:
        out[key] = getattr(c, key)
    return out
