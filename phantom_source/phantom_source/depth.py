import os
import numpy as np
import json_numpy
import plenopy
import corsika_primary
import binning_utils
import network_file_system as nfs
import glob
import sebastians_matplotlib_addons as seb
import pandas as pd
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
    "num_photons": 1e5,
    "min_object_distance_m": 2.7e3,
    "max_object_distance_m": 27.0e3,
    "num_pixel_on_edge": 1024,
    "image_containment_percentile": 95,
    "depth_range_shrinking_rate": 0.5,
    "oversampling_beam_spread": 100,
    "num_estimates": 1280,
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


JOB_UID_STR_TEMPLATE = "{:06d}"

def run_job(job):
    uid_str = JOB_UID_STR_TEMPLATE.format(job["uid"])
    config = json_numpy.read(os.path.join(job["work_dir"], "config.json"))
    map_dir = os.path.join(job["work_dir"], "map")
    job_dir = os.path.join(map_dir, uid_str)
    os.makedirs(job_dir, exist_ok=True)

    job_random_seed = config["random_seed"] + job["uid"]
    prng = np.random.Generator(np.random.MT19937(seed=job_random_seed))

    participating_beams_path = os.path.join(job_dir, "participating_beams.json")
    if os.path.exists(participating_beams_path):
        participating_beams = read_participating_beams(participating_beams_path)
        light_field_geometry = plenopy.LightFieldGeometry(
            path=config["light_field_geometry_path"]
        )
    else:
        light_field_geometry, participating_beams = estimate_response_to_point_source(
            cx_deg=job["cx_deg"],
            cy_deg=job["cy_deg"],
            object_distance_m=job["object_distance_m"],
            aperture_radius_m=job["aperture_radius_m"],
            prng=prng,
            light_field_geometry_path=config["light_field_geometry_path"],
            merlict_propagate_photons_path=config[
                "merlict_propagate_photons_path"
            ],
            merlict_propagate_config_path=config["merlict_propagate_config_path"],
            num_photons=config["num_photons"],
        )
        write_participating_beams(
            os.path.join(job_dir, "participating_beams.json"),
            participating_beams,
        )

    result_path = os.path.join(job_dir, "result.json")
    if os.path.exists(result_path):
        pass
    else:
        image_binning = make_image_binning(
            field_of_view_deg=job["field_of_view_deg"],
            num_pixel_on_edge=config["num_pixel_on_edge"],
        )

        report = estimate_focus(
            light_field_geometry=light_field_geometry,
            participating_beams=participating_beams,
            prng=prng,
            image_binning=image_binning,
            max_object_distance_m=config["max_object_distance_m"],
            min_object_distance_m=config["min_object_distance_m"],
            image_containment_percentile=config["image_containment_percentile"],
            depth_range_shrinking_rate=config["depth_range_shrinking_rate"],
            oversampling_beam_spread=config["oversampling_beam_spread"],
        )

        report["cx_deg"] = job["cx_deg"]
        report["cy_deg"] = job["cy_deg"]
        report["object_distance_m"] = job["object_distance_m"]

        nfs_json_numpy_write(
            os.path.join(job_dir, "result.json"), report, indent=4,
        )


def plot_report(report, path):
    _ssort = np.argsort(report["depth_m"])
    depth_m = np.array(report["depth_m"])[_ssort]
    spread = np.array(report["spreads_pixel_per_photon"])[_ssort]
    numpho = report["num_photons"]
    spread_lim = [np.min(spread*numpho), np.max(spread*numpho)]

    fig = seb.figure()
    ax = seb.add_axes(fig=fig, span=[0.2, 0.2, 0.7, 0.7])
    seb.ax_add_grid(ax=ax, add_minor=True)
    ax.plot(
        depth_m,
        spread*numpho,
        "kx",
    )
    ax.plot(
        depth_m,
        spread*numpho,
        "k-",
        linewidth=0.5,
        alpha=0.5,
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


def reduce(work_dir):
    config = json_numpy.read(os.path.join(job["work_dir"], "config.json"))
    map_dir = os.path.join(work_dir, "map")

    results = []
    for uid in range(config["num_estimates"]):
        uid_str = JOB_UID_STR_TEMPLATE.format(job["uid"])
        result_path = os.path.join(map_dir, uid_str, "result.json")
        if os.path.exists(result_path):
            results.append(json_numpy.read(result_path))

    nfs_json_numpy_write(os.path.join(work_dir, "result.json"), results)


def make_participating_beams_from_lixel_ids(beam_ids):
    participating_beams = {}
    for beam_id in beam_ids:
        if beam_id not in participating_beams:
            participating_beams[beam_id] = 0
        participating_beams[beam_id] += 1
    return participating_beams


def write_participating_beams(path, participating_beams):
    out = {}
    for k in participating_beams:
        out[str(k)] = int(participating_beams[k])
    nfs_json_numpy_write(
        path,
        out,
        indent=None,
    )


def read_participating_beams(path):
    b = json_numpy.read(path)
    out = {}
    for k in b:
        out[np.uint32(k)] = np.uint32(b[k])
    return out


def get_num_photons_in_participating_beams(participating_beams):
    return np.sum(
        [participating_beams[beam_id] for beam_id in participating_beams]
    )


def make_image_old(
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

    img = (1 / oversampling) * np.histogram2d(
        np.concatenate(cx_hits),
        np.concatenate(cy_hits),
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
    image_containment_percentile=95,
    depth_range_shrinking_rate=0.5,
    oversampling_beam_spread=1000,
    num_max_iterations=100,
):
    assert 0.0 < depth_range_shrinking_rate < 1.0
    assert num_max_iterations > 0
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

    num_initial_estimates = 7
    depths_range_m = max_object_distance_m - min_object_distance_m

    # rough
    # -----
    r["depth_m"] = list(np.geomspace(
        min_object_distance_m,
        max_object_distance_m,
        num_initial_estimates,
    ))

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
    while depths_range_m / reco_depth_m > 1e-3:

        if r["num_iterations"] >= num_max_iterations:
            print("Estimating focus: Too many iterations.")
            return r

        depths_range_m = depth_range_shrinking_rate * depths_range_m

        next_depths_m = estimate_next_focus_depth_m(
            depths_m=r["depth_m"],
            spreads_pixel_per_photon=r["spreads_pixel_per_photon"],
            depths_range_m=depths_range_m,
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
    depths_m,
    spreads_pixel_per_photon,
    depths_range_m,
):
    assert depths_range_m > 0

    _depths = np.array(depths_m)
    _spreads = np.array(spreads_pixel_per_photon)

    _ssort = np.argsort(_depths)
    depths = _depths[_ssort]
    spreads = _spreads[_ssort]

    assert len(depths) == len(spreads)
    assert len(depths) >= 3

    assert np.all(depths > 0.0)
    assert np.all(spreads > 0.0)

    print("Estimate next focus: depths_m", depths)

    reco_depth_m = depths[np.argmin(spreads)]
    d_next_start = reco_depth_m - depths_range_m / 2
    d_next_stop = reco_depth_m + depths_range_m / 2

    next_depths = []

    for i in range(len(depths) - 1):
        d_start = depths[i]
        d_stop = depths[i + 1]
        # d_next = 0.5 * (d_stop + d_start)
        d_next = np.geomspace(d_start, d_stop, 3)[1]
        if d_next_start <= d_next <= d_next_stop:
            next_depths.append(d_next)

    return np.array(next_depths)




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
    return light_field_geometry, participating_beams


def estimate_focus(
    light_field_geometry,
    participating_beams,
    prng,
    image_binning,
    max_object_distance_m,
    min_object_distance_m,
    image_containment_percentile,
    depth_range_shrinking_rate,
    oversampling_beam_spread,
    num_max_iterations=100,
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
        depth_range_shrinking_rate=depth_range_shrinking_rate,
        oversampling_beam_spread=oversampling_beam_spread,
        num_max_iterations=num_max_iterations,
    )
    return report


def nfs_json_numpy_write(path, obj, indent=4):
    tmp_path = path + ".incomplete"
    json_numpy.write(tmp_path, obj, indent=indent)
    nfs.move(tmp_path, path)
