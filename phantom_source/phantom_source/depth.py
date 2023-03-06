import numpy as np
import json_numpy
import plenopy as pl
from . import mesh
from . import light_field
from . import merlict


def make_participating_beams_from_lixel_ids(beam_ids):
    participating_beams = {}
    for beam_id in beam_ids:
        if beam_id not in participating_beams:
            participating_beams[beam_id] = 0
        participating_beams[beam_id] += 1
    return participating_beams


def make_image(
    image_beams,
    participating_beams,
    object_distance,
    image_binning,
    oversampling,
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
    image_beams,
    participating_beams,
    image_binning,
    max_object_distance,
    min_object_distance,
    image_containment_percentile=95,
    step_rate=0.5,
    oversampling_beam_spread=1000,
    num_max_iterations=1000,
):
    r = {}

    obj_hi = max_object_distance
    img_hi = make_image(
        image_beams=image_beams,
        participating_beams=participating_beams,
        object_distance=obj_hi,
        image_binning=image_binning,
        oversampling=oversampling_beam_spread,
    )
    n_hi = count_pixels_containing_percentile(
        image=img_hi, percentile=image_containment_percentile
    )

    obj_lo = min_object_distance
    img_lo = make_image(
        image_beams=image_beams,
        participating_beams=participating_beams,
        object_distance=obj_lo,
        image_binning=image_binning,
        oversampling=oversampling_beam_spread,
    )
    n_lo = count_pixels_containing_percentile(
        image=img_lo, percentile=image_containment_percentile
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
            participating_beams=participating_beams,
            object_distance=obj_mi,
            image_binning=image_binning,
            oversampling=oversampling_beam_spread,
        )
        n_mi = count_pixels_containing_percentile(
            image=img_mi, percentile=image_containment_percentile
        )

        r["object_distance_high"] = obj_hi
        r["object_distance"] = obj_mi
        r["object_distance_low"] = obj_lo
        r["spread_in_image_high"] = n_hi
        r["spread_in_image"] = n_mi
        r["spread_in_image_low"] = n_lo

        if n_hi <= n_lo and n_mi < n_lo:
            obj_lo = step_rate * obj_mi + (1 - step_rate) * obj_lo
            img_lo = make_image(
                image_beams=image_beams,
                participating_beams=participating_beams,
                object_distance=obj_lo,
                image_binning=image_binning,
                oversampling=oversampling_beam_spread,
            )
            n_lo = count_pixels_containing_percentile(
                image=img_lo, percentile=image_containment_percentile
            )
        elif n_mi < n_hi and n_lo <= n_hi:
            obj_hi = step_rate * obj_mi + (1 - step_rate) * obj_hi
            img_hi = make_image(
                image_beams=image_beams,
                participating_beams=participating_beams,
                object_distance=obj_hi,
                image_binning=image_binning,
                oversampling=oversampling_beam_spread,
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
    step_rate,
    oversampling_beam_spread,
    num_max_iterations=100,
    point_source_radois_deg=0.01,
    emission_distance_to_aperture_m=1e3,
):
    # create response
    # ---------------
    mesh_img = mesh.triangle(
        pos=[cx_deg, cy_deg, object_distance_m],
        radius=point_source_radois_deg,
        density=DENSITY,
    )
    mesh_scn = mesh.transform_image_to_scneney(mesh=mesh_img)
    light_fields = light_field.make_light_fields_from_meshes(
        meshes=[mesh_scn],
        aperture_radius=aperture_radius_m,
        prng=prng,
        emission_distance_to_aperture=emission_distance_to_aperture_m,
    )

    merlict_random_seed = prng.integers(low=0, high=2**32)
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

    image_beams = pl.image.ImageRays(light_field_geometry=light_field_geometry)
    report, img = estimate_depth_from_participating_beams(
        image_beams=image_beams,
        participating_beams=participating_beams,
        image_binning=image_binning,
        max_object_distance=max_object_distance_m,
        min_object_distance=min_object_distance_m,
        image_containment_percentile=image_containment_percentile,
        step_rate=step_rate,
        oversampling_beam_spread=oversampling_beam_spread,
        num_max_iterations=num_max_iterations,
    )
    return report, img
