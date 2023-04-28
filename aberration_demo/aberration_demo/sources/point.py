import phantom_source
import numpy as np
import json_numpy
import corsika_primary
import os
import plenopy
from . import mesh
from .. import utils


EXAMPLE_POINT_CONFIG = {
    "type": "point",
    "cx_deg": 0.0,
    "cy_deg": 1.0,
    "object_distance_m": 10e3,
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
        np.deg2rad(point_source_apparent_radius_deg)
        * point_config["object_distance_m"]
    )

    SOME_FACORT_TO_GET_DENSITY_RIGHT = 2.16

    density = phantom_source.light_field.get_edge_density_from_number_photons(
        number_photons=num_photons / SOME_FACORT_TO_GET_DENSITY_RIGHT,
        edge_length=_source_edge_length,
        distance_to_aperture=point_config["object_distance_m"],
        aperture_radius=illum_radius,
    )

    mesh_img = phantom_source.mesh.triangle(
        pos=[
            point_config["cx_deg"],
            point_config["cy_deg"],
            point_config["object_distance_m"],
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


def make_source_config_from_job(job):
    prng = np.random.Generator(np.random.PCG64(job["number"]))

    point_cfg = json_numpy.read(
        os.path.join(job["work_dir"], "config", "observations", "point.json")
    )
    (cx_deg, cy_deg,) = corsika_primary.random.distributions.draw_x_y_in_disc(
        prng=prng, radius=point_cfg["max_angle_off_optical_axis_deg"]
    )
    object_distance_m = corsika_primary.random.distributions.draw_power_law(
        prng=prng,
        lower_limit=point_cfg["min_object_distance_m"],
        upper_limit=point_cfg["max_object_distance_m"],
        power_slope=-1,
        num_samples=1,
    )[0]

    source_config = {
        "type": "point",
        "cx_deg": cx_deg,
        "cy_deg": cy_deg,
        "object_distance_m": object_distance_m,
        "areal_photon_density_per_m2": point_cfg[
            "areal_photon_density_per_m2"
        ],
        "seed": job["number"],
    }
    return source_config


def analysis_run_job(job):
    nkey = "{:06d}".format(job["number"])

    indir = os.path.join(
        job["work_dir"],
        "responses",
        job["instrument_key"],
        job["observation_key"],
    )

    outdir = os.path.join(
        job["work_dir"],
        "analysis",
        job["instrument_key"],
        job["observation_key"],
    )

    os.makedirs(outdir, exist_ok=True)

    inpath = os.path.join(indir, nkey)
    truth = json_numpy.read(inpath + ".json")
    with open(inpath, "rb") as f:
        raw_sensor_response = plenopy.raw_light_field_sensor_response.read(f)

    (
        _,
        beam_ids,
    ) = plenopy.light_field_sequence.photon_arrival_times_and_lixel_ids(
        raw_sensor_response=raw_sensor_response
    )

    participating_beams = phantom_source.depth.make_participating_beams_from_lixel_ids(
        beam_ids=beam_ids
    )

    prng = np.random.Generator(np.random.PCG64(job["number"]))

    light_field_geometry = plenopy.LightFieldGeometry(
        os.path.join(
            job["work_dir"],
            "instruments",
            job["instrument_key"],
            "light_field_geometry",
        )
    )

    cfg_analysis = json_numpy.read(
        os.path.join(job["work_dir"], "config", "analysis", "point.json")
    )

    image_binning = phantom_source.depth.make_image_binning(
        field_of_view_deg=cfg_analysis["field_of_view_deg"],
        num_pixel_on_edge=cfg_analysis["num_pixel_on_edge"],
    )

    report = phantom_source.depth.estimate_focus(
        light_field_geometry=light_field_geometry,
        participating_beams=participating_beams,
        prng=prng,
        image_binning=image_binning,
        max_object_distance_m=1.25 * cfg_analysis["max_object_distance_m"],
        min_object_distance_m=0.75 * cfg_analysis["min_object_distance_m"],
        image_containment_percentile=cfg_analysis[
            "image_containment_percentile"
        ],
        oversampling_beam_spread=cfg_analysis["oversampling_beam_spread"],
    )

    report["cx_deg"] = truth["cx_deg"]
    report["cy_deg"] = truth["cy_deg"]
    report["object_distance_m"] = truth["object_distance_m"]

    outpath = os.path.join(outdir, nkey + ".json")
    json_numpy.write(outpath + ".incomplete", report)
    os.rename(outpath + ".incomplete", outpath)

    try:
        phantom_source.depth.plot_report(
            report=report, path=outpath + ".jpg",
        )
    except:
        pass
