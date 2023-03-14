import os
import numpy as np
import json_numpy
import plenopy
import corsika_primary
import binning_utils
import network_file_system as nfs
from . import mesh
from . import light_field
from . import merlict

EXAMPLE_CONFIG = {
    "random_seed": 1,
    "light_field_geometry_path": "",
    "merlict_propagate_photons_path": os.path.join(
        "build", "merlict", "merlict-plenoscope-raw-photon-propagation"
    ),
    "merlict_propagate_config_path": os.path.join(
        "resources",
        "acp",
        "merlict_propagation_config_no_night_sky_background.json",
    ),
    "num_photons": 1e4,
    "min_object_distance_m": 2.7e3,
    "max_object_distance_m": 27.0e3,
    "num_pixel_on_edge": 1024,
    "image_containment_percentile": 95,
    "auto_focus_step_rate": 0.5,
    "oversampling_beam_spread": 1000,
    "num_estimates": 128,
}


def init(work_dir, config=EXAMPLE_CONFIG):
    """
    Estimate the resolution in depth of the Cherenkov-plenoscope.
    Init a map-and-reduce work_dir to run many refocussings.
    """
    os.makedirs(name=work_dir, exist_ok=True)
    nfs_json_numpy_write(
        os.path.join(work_dir, "config.json"), config, indent=4
    )


def make_jobs(work_dir):
    work_dir = os.path.abspath(work_dir)
    config = json_numpy.read(os.path.join(work_dir, "config.json"))

    lfg = plenopy.LightFieldGeometry(config["light_field_geometry_path"])

    focal_length_m = (
        lfg.sensor_plane2imaging_system.expected_imaging_system_focal_length
    )
    aperture_radius_m = (
        1.3
        * lfg.sensor_plane2imaging_system.expected_imaging_system_max_aperture_radius
    )

    field_of_view_deg = np.rad2deg(
        lfg.sensor_plane2imaging_system.max_FoV_diameter
    )

    prng = np.random.Generator(np.random.MT19937(seed=config["random_seed"]))

    jobs = []
    for uid in range(config["num_estimates"]):

        object_distance_m = prng.uniform(
            low=config["min_object_distance_m"],
            high=config["max_object_distance_m"],
        )

        cx_deg, cy_deg = corsika_primary.random.distributions.draw_x_y_in_disc(
            prng=prng, radius=0.5 * field_of_view_deg,
        )

        job = {
            "work_dir": work_dir,
            "uid": uid,
            "object_distance_m": object_distance_m,
            "aperture_radius_m": aperture_radius_m,
            "field_of_view_deg": field_of_view_deg,
            "cx_deg": cx_deg,
            "cy_deg": cy_deg,
        }
        jobs.append(job)

    return jobs


def run_job(job):
    config = json_numpy.read(os.path.join(job["work_dir"], "config.json"))

    # image_binning
    # -------------
    c_radius_rad = np.deg2rad(0.5 * job["field_of_view_deg"])
    c_bin_rag = binning_utils.Binning(
        bin_edges=np.linspace(
            -c_radius_rad, c_radius_rad, config["num_pixel_on_edge"] + 1
        )
    )
    image_binning = {
        "cx": c_bin_rag,
        "cy": c_bin_rag,
    }

    job_random_seed = config["random_seed"] + job["uid"]

    prng = np.random.Generator(np.random.MT19937(seed=job_random_seed))

    report, img = estimate_resolution(
        cx_deg=job["cx_deg"],
        cy_deg=job["cy_deg"],
        object_distance_m=job["object_distance_m"],
        aperture_radius_m=job["aperture_radius_m"],
        image_binning=image_binning,
        max_object_distance_m=config["max_object_distance_m"],
        min_object_distance_m=config["min_object_distance_m"],
        prng=prng,
        light_field_geometry_path=config["light_field_geometry_path"],
        merlict_propagate_photons_path=config[
            "merlict_propagate_photons_path"
        ],
        merlict_propagate_config_path=config["merlict_propagate_config_path"],
        image_containment_percentile=config["image_containment_percentile"],
        auto_focus_step_rate=config["auto_focus_step_rate"],
        oversampling_beam_spread=config["oversampling_beam_spread"],
        num_photons=config["num_photons"],
    )

    uid_str = "{:06d}".format(job["uid"])

    report["cx_deg"] = job["cx_deg"]
    report["cy_deg"] = job["cy_deg"]
    report["object_distance_m"] = job["object_distance_m"]

    nfs_json_numpy_write(
        os.path.join(job["work_dir"], uid_str + ".json"), report, indent=4,
    )


def make_participating_beams_from_lixel_ids(beam_ids):
    participating_beams = {}
    for beam_id in beam_ids:
        if beam_id not in participating_beams:
            participating_beams[beam_id] = 0
        participating_beams[beam_id] += 1
    return participating_beams


def get_num_photons_in_participating_beams(participating_beams):
    return np.sum(
        [participating_beams[beam_id] for beam_id in participating_beams]
    )


def make_image(
    image_beams,
    light_field_geometry,
    participating_beams,
    object_distance,
    image_binning,
    oversampling,
    prng,
):
    img = np.zeros(
        shape=(image_binning["cx"]["num"], image_binning["cy"]["num"])
    )

    img_cx, img_cy = image_beams.cx_cy_in_object_distance(object_distance)
    img_cx_std = light_field_geometry.cx_std
    img_cy_std = light_field_geometry.cy_std

    for beam_id in participating_beams:
        num_photons = participating_beams[beam_id]

        cx_hits = prng.normal(
            loc=img_cx[beam_id],
            scale=img_cx_std[beam_id],
            size=oversampling * num_photons,
        )

        cy_hits = prng.normal(
            loc=img_cy[beam_id],
            scale=img_cy_std[beam_id],
            size=oversampling * num_photons,
        )

        img += (1 / oversampling) * np.histogram2d(
            cx_hits,
            cy_hits,
            bins=(image_binning["cx"]["edges"], image_binning["cy"]["edges"]),
        )[0]

    return img


def count_pixels_containing_percentile(image, percentile):
    assert 0.0 < percentile <= 100.0
    assert np.all(image >= 0.0)

    I = image.flatten()
    S = np.sum(I)
    a = np.flip(np.argsort(I))

    fraction = 0.0
    targeted_fraction = percentile / 100
    n = 0
    while fraction < targeted_fraction:
        s = I[a[n]]
        fraction += s / S
        n += 1
    return n


def estimate_depth_from_participating_beams(
    prng,
    image_beams,
    light_field_geometry,
    participating_beams,
    image_binning,
    max_object_distance,
    min_object_distance,
    image_containment_percentile=95,
    auto_focus_step_rate=0.5,
    oversampling_beam_spread=1000,
    num_max_iterations=1000,
):
    """
    auto-focus
    """
    assert 0.0 < auto_focus_step_rate < 1.0
    assert num_max_iterations > 0
    assert 0 < image_containment_percentile <= 100
    assert max_object_distance > 0
    assert min_object_distance > 0
    assert min_object_distance < max_object_distance
    assert oversampling_beam_spread >= 1

    afrate = auto_focus_step_rate
    r = {}

    obj_hi = max_object_distance
    img_hi = make_image(
        image_beams=image_beams,
        light_field_geometry=light_field_geometry,
        participating_beams=participating_beams,
        object_distance=obj_hi,
        image_binning=image_binning,
        oversampling=oversampling_beam_spread,
        prng=prng,
    )
    n_hi = count_pixels_containing_percentile(
        image=img_hi, percentile=image_containment_percentile
    )

    obj_lo = min_object_distance
    img_lo = make_image(
        image_beams=image_beams,
        light_field_geometry=light_field_geometry,
        participating_beams=participating_beams,
        object_distance=obj_lo,
        image_binning=image_binning,
        oversampling=oversampling_beam_spread,
        prng=prng,
    )
    n_lo = count_pixels_containing_percentile(
        image=img_lo, percentile=image_containment_percentile
    )
    r["num_photons"] = get_num_photons_in_participating_beams(
        participating_beams=participating_beams
    )
    r["focus"] = False
    r["iteration"] = 0
    while not r["focus"]:
        r["iteration"] += 1
        if r["iteration"] > num_max_iterations:
            raise RuntimeError(json_numpy.dumps(r))

        obj_mi = np.mean([obj_lo, obj_hi])
        img_mi = make_image(
            image_beams=image_beams,
            light_field_geometry=light_field_geometry,
            participating_beams=participating_beams,
            object_distance=obj_mi,
            image_binning=image_binning,
            oversampling=oversampling_beam_spread,
            prng=prng,
        )
        n_mi = count_pixels_containing_percentile(
            image=img_mi, percentile=image_containment_percentile
        )

        r["reco_object_distance_high_m"] = obj_hi
        r["reco_object_distance_m"] = obj_mi
        r["reco_object_distance_low_m"] = obj_lo
        r["spread_in_image_high"] = n_hi
        r["spread_in_image"] = n_mi
        r["spread_in_image_low"] = n_lo

        if n_hi <= n_lo and n_mi < n_lo:
            obj_lo = afrate * obj_mi + (1 - afrate) * obj_lo
            img_lo = make_image(
                image_beams=image_beams,
                light_field_geometry=light_field_geometry,
                participating_beams=participating_beams,
                object_distance=obj_lo,
                image_binning=image_binning,
                oversampling=oversampling_beam_spread,
                prng=prng,
            )
            n_lo = count_pixels_containing_percentile(
                image=img_lo, percentile=image_containment_percentile
            )
        elif n_mi < n_hi and n_lo <= n_hi:
            obj_hi = afrate * obj_mi + (1 - afrate) * obj_hi
            img_hi = make_image(
                image_beams=image_beams,
                light_field_geometry=light_field_geometry,
                participating_beams=participating_beams,
                object_distance=obj_hi,
                image_binning=image_binning,
                oversampling=oversampling_beam_spread,
                prng=prng,
            )
            n_hi = count_pixels_containing_percentile(
                image=img_hi, percentile=image_containment_percentile
            )
        else:
            r["focus"] = True

    return r, img_mi


def estimate_resolution(
    cx_deg,
    cy_deg,
    object_distance_m,
    aperture_radius_m,
    image_binning,
    max_object_distance_m,
    min_object_distance_m,
    prng,
    light_field_geometry_path,
    merlict_propagate_photons_path,
    merlict_propagate_config_path,
    image_containment_percentile,
    auto_focus_step_rate,
    oversampling_beam_spread,
    num_photons,
    num_max_iterations=100,
    point_source_apparent_radius_deg=0.01,
    emission_distance_to_aperture_m=1e3,
):
    _source_edge_length = (
        np.deg2rad(point_source_apparent_radius_deg) * object_distance_m
    )

    density = light_field.get_edge_density_from_number_photons(
        number_photons=num_photons,
        edge_length=_source_edge_length,
        distance_to_aperture=object_distance_m,
        aperture_radius=aperture_radius_m,
    )

    # create response
    # ---------------
    mesh_img = mesh.triangle(
        pos=[cx_deg, cy_deg, object_distance_m],
        radius=point_source_apparent_radius_deg,
        density=density,
    )
    mesh_scn = mesh.transform_image_to_scneney(mesh=mesh_img)
    light_fields = light_field.make_light_fields_from_meshes(
        meshes=[mesh_scn],
        aperture_radius=aperture_radius_m,
        prng=prng,
        emission_distance_to_aperture=emission_distance_to_aperture_m,
    )

    merlict_random_seed = prng.integers(low=0, high=2 ** 32)
    (
        event,
        light_field_geometry,
    ) = merlict.make_plenopy_event_and_read_light_field_geometry(
        light_fields=light_fields,
        light_field_geometry_path=light_field_geometry_path,
        merlict_propagate_photons_path=merlict_propagate_photons_path,
        merlict_propagate_config_path=merlict_propagate_config_path,
        random_seed=merlict_random_seed,
    )
    _beam_t, beam_ids = event.photon_arrival_times_and_lixel_ids()

    participating_beams = make_participating_beams_from_lixel_ids(
        beam_ids=beam_ids
    )

    image_beams = plenopy.image.ImageRays(
        light_field_geometry=light_field_geometry
    )
    report, img = estimate_depth_from_participating_beams(
        prng=prng,
        image_beams=image_beams,
        light_field_geometry=light_field_geometry,
        participating_beams=participating_beams,
        image_binning=image_binning,
        max_object_distance=max_object_distance_m,
        min_object_distance=min_object_distance_m,
        image_containment_percentile=image_containment_percentile,
        auto_focus_step_rate=auto_focus_step_rate,
        oversampling_beam_spread=oversampling_beam_spread,
        num_max_iterations=num_max_iterations,
    )
    return report, img


def nfs_json_numpy_write(path, obj, indent=4):
    tmp_path = path + ".incomplete"
    json_numpy.write(tmp_path, obj, indent=indent)
    nfs.move(tmp_path, path)
