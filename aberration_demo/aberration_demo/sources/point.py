import phantom_source
import numpy as np
from . import mesh
from .. import utils


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
    instgeom = utils.get_instrument_geometry_from_light_field_geometry(
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

    return mesh.make_response_to_mesh(
        mesh_config=mesh_config,
        light_field_geometry_path=light_field_geometry_path,
        merlict_config=merlict_config,
        emission_distance_to_aperture_m=emission_distance_to_aperture_m,
    )
