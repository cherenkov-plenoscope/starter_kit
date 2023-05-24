import phantom_source
import numpy as np
import json_numpy
import corsika_primary
import os
import plenopy
import binning_utils
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

    participating_beams = make_participating_beams_from_lixel_ids(
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

    image_binning = make_image_binning(
        field_of_view_deg=cfg_analysis["field_of_view_deg"],
        num_pixel_on_edge=cfg_analysis["num_pixel_on_edge"],
    )

    report = estimate_focus(
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
        plot_report(
            report=report, path=outpath + ".jpg",
        )
    except:
        pass


def make_image_binning(field_of_view_deg, num_pixel_on_edge):
    c_radius_rad = np.deg2rad(0.5 * field_of_view_deg)
    c_bin_rag = binning_utils.Binning(
        bin_edges=np.linspace(
            -c_radius_rad, c_radius_rad, num_pixel_on_edge + 1
        )
    )
    image_binning = {
        "cx": c_bin_rag,
        "cy": c_bin_rag,
    }
    return image_binning


def make_participating_beams_from_lixel_ids(beam_ids):
    participating_beams = {}
    for beam_id in beam_ids:
        if beam_id not in participating_beams:
            participating_beams[beam_id] = 0
        participating_beams[beam_id] += 1
    return participating_beams


def plot_report(report, path):
    _ssort = np.argsort(report["depth_m"])
    depth_m = np.array(report["depth_m"])[_ssort]
    spread = np.array(report["spreads_pixel_per_photon"])[_ssort]
    numpho = report["num_photons"]
    spread_lim = [np.min(spread * numpho), np.max(spread * numpho)]

    fig = seb.figure()
    ax = seb.add_axes(fig=fig, span=[0.2, 0.2, 0.7, 0.7])
    seb.ax_add_grid(ax=ax, add_minor=True)
    ax.plot(
        depth_m, spread * numpho, "kx",
    )
    ax.plot(
        depth_m, spread * numpho, "k-", linewidth=0.5, alpha=0.5,
    )
    reco_depth_m = depth_m[np.argmin(spread)]
    ax.plot(
        [reco_depth_m, reco_depth_m],
        spread_lim,
        "k-",
        linewidth=0.5,
        alpha=0.5,
    )
    true_depth_m = report["object_distance_m"]
    ax.plot(
        [true_depth_m, true_depth_m],
        spread_lim,
        "b-",
        linewidth=0.5,
        alpha=0.5,
    )
    ax.loglog()
    ax.set_xlabel("depth / m")
    ax.set_ylabel("spread / pixel")
    fig.savefig(path)
    seb.close(fig)


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
    img_cx, img_cy = image_beams.cx_cy_in_object_distance(object_distance)
    img_cx_std = light_field_geometry.cx_std
    img_cy_std = light_field_geometry.cy_std

    cx_hits = []
    cy_hits = []
    for beam_id in participating_beams:
        num_photons = participating_beams[beam_id]
        cx_h = prng.normal(
            loc=img_cx[beam_id],
            scale=img_cx_std[beam_id],
            size=oversampling * num_photons,
        )
        cy_h = prng.normal(
            loc=img_cy[beam_id],
            scale=img_cy_std[beam_id],
            size=oversampling * num_photons,
        )
        cx_hits.append(cx_h)
        cy_hits.append(cy_h)

    if cx_hits:
        all_cx_hits = np.concatenate(cx_hits)
        all_cy_hits = np.concatenate(cy_hits)
    else:
        all_cx_hits = []
        all_cy_hits = []

    img = (1 / oversampling) * np.histogram2d(
        all_cx_hits,
        all_cy_hits,
        bins=(image_binning["cx"]["edges"], image_binning["cy"]["edges"]),
    )[0]

    return img


def estimate_inverse_photon_density_pixel_per_photon(image, percentile):
    assert 0.0 < percentile <= 100.0
    assert np.all(image >= 0.0)

    I = image.flatten()
    S = np.sum(I)
    a = np.flip(np.argsort(I))
    num_photons = 0.0
    fraction = 0.0
    targeted_fraction = percentile / 100
    num_pixel = 0
    while fraction < targeted_fraction:
        s = I[a[num_pixel]]
        num_photons += s
        fraction += s / S
        num_pixel += 1
    return num_pixel / num_photons


def estimate_depth_from_participating_beams(
    prng,
    image_beams,
    light_field_geometry,
    participating_beams,
    image_binning,
    max_object_distance_m,
    min_object_distance_m,
    image_containment_percentile,
    oversampling_beam_spread,
):
    assert oversampling_beam_spread > 0
    assert 0 < image_containment_percentile <= 100
    assert max_object_distance_m > 0
    assert min_object_distance_m > 0
    assert min_object_distance_m < max_object_distance_m

    r = {}
    r["num_photons"] = get_num_photons_in_participating_beams(
        participating_beams=participating_beams
    )
    r["focus"] = False
    r["num_iterations"] = 0
    r["depth_m"] = []
    r["spreads_pixel_per_photon"] = []

    num_initial_estimates = 9
    depths_range_ratio = max_object_distance_m / min_object_distance_m

    if r["num_photons"] < 2:
        r["depth_m"].append(float("nan"))
        r["spreads_pixel_per_photon"].append(float("nan"))
        return r

    # rough
    # -----
    r["depth_m"] = list(
        np.geomspace(
            min_object_distance_m,
            max_object_distance_m,
            num_initial_estimates,
        )
    )

    for i in range(len(r["depth_m"])):
        img = make_image(
            image_beams=image_beams,
            light_field_geometry=light_field_geometry,
            participating_beams=participating_beams,
            object_distance=r["depth_m"][i],
            image_binning=image_binning,
            oversampling=oversampling_beam_spread,
            prng=prng,
        )
        spread = estimate_inverse_photon_density_pixel_per_photon(
            image=img, percentile=image_containment_percentile
        )
        r["spreads_pixel_per_photon"].append(spread)
    reco_depth_m = r["depth_m"][np.argmin(r["spreads_pixel_per_photon"])]

    # fine iteration
    # --------------
    for it in range(12):
        next_depths_m = estimate_next_focus_depth_m(
            depths_m=r["depth_m"],
            spreads_pixel_per_photon=r["spreads_pixel_per_photon"],
            next_depths_radius_num_points=3,
        )
        for n in range(len(next_depths_m)):
            n_depth_m = next_depths_m[n]
            n_img = make_image(
                image_beams=image_beams,
                light_field_geometry=light_field_geometry,
                participating_beams=participating_beams,
                object_distance=n_depth_m,
                image_binning=image_binning,
                oversampling=oversampling_beam_spread,
                prng=prng,
            )
            n_spread = estimate_inverse_photon_density_pixel_per_photon(
                image=n_img, percentile=image_containment_percentile
            )
            r["depth_m"].append(n_depth_m)
            r["spreads_pixel_per_photon"].append(n_spread)

        reco_depth_m = r["depth_m"][np.argmin(r["spreads_pixel_per_photon"])]
        r["num_iterations"] += 1

    r["focus"] = True

    return r


def estimate_next_focus_depth_m(
    depths_m, spreads_pixel_per_photon, next_depths_radius_num_points,
):
    assert next_depths_radius_num_points > 0

    _depths = np.array(depths_m)
    _spreads = np.array(spreads_pixel_per_photon)

    _ssort = np.argsort(_depths)
    depths = _depths[_ssort]
    spreads = _spreads[_ssort]

    assert len(depths) == len(spreads)
    assert len(depths) >= 3

    assert np.all(depths > 0.0)
    assert np.all(spreads > 0.0)

    i_min = np.argmin(spreads)
    i_start = i_min - next_depths_radius_num_points
    i_stop = i_min + next_depths_radius_num_points

    next_depths = []
    for i in np.arange(i_start, i_stop):
        if i >= 0 and (i + 1) < len(depths):
            d_start = depths[i]
            d_stop = depths[i + 1]
            d_next = np.geomspace(d_start, d_stop, 3)[1]
            next_depths.append(d_next)

    next_depths = np.array(next_depths)
    return next_depths


def estimate_response_to_point_source(
    cx_deg,
    cy_deg,
    object_distance_m,
    aperture_radius_m,
    prng,
    light_field_geometry_path,
    merlict_propagate_photons_path,
    merlict_propagate_config_path,
    num_photons,
    work_dir=None,
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
        work_dir=work_dir,
    )
    (
        _beam_t,
        beam_ids,
    ) = plenopy.light_field_sequence.photon_arrival_times_and_lixel_ids(
        raw_sensor_response=event.raw_sensor_response
    )

    participating_beams = make_participating_beams_from_lixel_ids(
        beam_ids=beam_ids
    )
    return light_field_geometry, participating_beams


def estimate_focus(
    light_field_geometry,
    participating_beams,
    prng,
    image_binning,
    max_object_distance_m,
    min_object_distance_m,
    image_containment_percentile,
    oversampling_beam_spread,
):
    image_beams = plenopy.image.ImageRays(
        light_field_geometry=light_field_geometry
    )
    report = estimate_depth_from_participating_beams(
        prng=prng,
        image_beams=image_beams,
        light_field_geometry=light_field_geometry,
        participating_beams=participating_beams,
        image_binning=image_binning,
        max_object_distance_m=max_object_distance_m,
        min_object_distance_m=min_object_distance_m,
        image_containment_percentile=image_containment_percentile,
        oversampling_beam_spread=oversampling_beam_spread,
    )
    return report
