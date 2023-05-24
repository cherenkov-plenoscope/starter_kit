from . import image
from . import statistical_estimators
import corsika_primary as cpw
import os
import copy
import plenoirf
import numpy as np
import plenopy


def make_bin_edges_and_centers(bin_width, num_bins, first_bin_center):
    bin_edges = np.linspace(
        start=first_bin_center + bin_width * (-0.5),
        stop=first_bin_center + bin_width * (num_bins + 0.5),
        num=num_bins + 1,
    )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[0:-1])
    return bin_edges, bin_centers


def calibrate_plenoscope_response(
    raw_sensor_response, light_field_geometry, object_distance,
):
    image_rays = plenopy.image.ImageRays(
        light_field_geometry=light_field_geometry
    )

    time_bin_edges, time_bin_centers = make_bin_edges_and_centers(
        bin_width=raw_sensor_response["time_slice_duration"],
        num_bins=raw_sensor_response["number_time_slices"],
        first_bin_center=0.0,
    )

    out = {}
    out["time"] = {}
    out["time"]["bin_edges"] = time_bin_edges
    out["time"]["bin_centers"] = time_bin_centers

    isochor_image_seqence = plenopy.light_field_sequence.make_isochor_image(
        raw_sensor_response=raw_sensor_response,
        time_delay_image_mean=light_field_geometry.time_delay_image_mean,
    )

    out["time"]["weights"] = isochor_image_seqence.sum(axis=1)

    out["image_beams"] = {}
    out["image_beams"]["_weights"] = isochor_image_seqence.sum(axis=0)
    (
        out["image_beams"]["_cx"],
        out["image_beams"]["_cy"],
    ) = image_rays.cx_cy_in_object_distance(object_distance)
    out["image_beams"]["_cx_std"] = light_field_geometry.cx_std
    out["image_beams"]["_cy_std"] = light_field_geometry.cy_std

    valid_cxcy = np.logical_and(
        np.logical_not(np.isnan(out["image_beams"]["_cx"])),
        np.logical_not(np.isnan(out["image_beams"]["_cy"])),
    )
    valid_cxcy_std = np.logical_and(
        np.logical_not(np.isnan(out["image_beams"]["_cx_std"])),
        np.logical_not(np.isnan(out["image_beams"]["_cy_std"])),
    )
    valid = np.logical_and(valid_cxcy, valid_cxcy_std)
    out["image_beams"]["valid"] = valid
    out["image_beams"]["weights"] = out["image_beams"]["_weights"][valid]
    out["image_beams"]["cx"] = out["image_beams"]["_cx"][valid]
    out["image_beams"]["cy"] = out["image_beams"]["_cy"][valid]
    out["image_beams"]["cx_std"] = out["image_beams"]["_cx_std"][valid]
    out["image_beams"]["cy_std"] = out["image_beams"]["_cy_std"][valid]
    return out


def binning_image_bin_edges(binning):
    bb = binning
    cx_image_angle = np.deg2rad(
        bb["image"]["num_pixel_cx"] * bb["image"]["pixel_angle_deg"]
    )
    cy_image_angle = np.deg2rad(
        bb["image"]["num_pixel_cy"] * bb["image"]["pixel_angle_deg"]
    )

    cx_cen = np.deg2rad(bb["image"]["center"]["cx_deg"])
    cy_cen = np.deg2rad(bb["image"]["center"]["cy_deg"])

    cx_start = cx_cen - cx_image_angle / 2
    cx_stop = cx_cen + cx_image_angle / 2

    cy_start = cy_cen - cy_image_angle / 2
    cy_stop = cy_cen + cy_image_angle / 2

    cx_bin_edges = np.linspace(
        cx_start, cx_stop, bb["image"]["num_pixel_cx"] + 1
    )
    cy_bin_edges = np.linspace(
        cy_start, cy_stop, bb["image"]["num_pixel_cy"] + 1
    )

    return cx_bin_edges, cy_bin_edges


def analyse_response_to_calibration_source(
    image_center_cx_deg,
    image_center_cy_deg,
    raw_sensor_response,
    light_field_geometry,
    object_distance_m,
    containment_percentile,
    binning,
    prng,
):
    calibrated_response = calibrate_plenoscope_response(
        light_field_geometry=light_field_geometry,
        raw_sensor_response=raw_sensor_response,
        object_distance=object_distance_m,
    )

    cres = calibrated_response

    # print("image encirclement2d")
    psf_cx, psf_cy, psf_angle80 = encirclement2d(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        prng=prng,
        percentile=containment_percentile,
        num_sub_samples=1,
    )

    thisbinning = copy.deepcopy(binning)
    thisbinning["image"]["center"]["cx_deg"] = image_center_cx_deg
    thisbinning["image"]["center"]["cy_deg"] = image_center_cy_deg
    thisimg_bin_edges = binning_image_bin_edges(binning=thisbinning)

    # print("image histogram2d_std")
    imgraw = image.histogram2d_std(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        bins=thisimg_bin_edges,
        prng=prng,
        num_sub_samples=1000,
    )[0]

    # print("time encirclement1d")
    time_80_start, time_80_stop = encirclement1d(
        x=cres["time"]["bin_centers"],
        f=cres["time"]["weights"],
        percentile=containment_percentile,
    )
    # print("time full_width_half_maximum")
    (time_fwhm_start, time_fwhm_stop,) = full_width_half_maximum(
        x=cres["time"]["bin_centers"], f=cres["time"]["weights"],
    )

    # export
    out = {}
    out["statistics"] = {}
    out["statistics"]["image_beams"] = {}
    out["statistics"]["image_beams"][
        "total"
    ] = light_field_geometry.number_lixel
    out["statistics"]["image_beams"]["valid"] = np.sum(
        cres["image_beams"]["valid"]
    )
    out["statistics"]["photons"] = {}
    out["statistics"]["photons"]["total"] = raw_sensor_response[
        "number_photons"
    ]
    out["statistics"]["photons"]["valid"] = np.sum(
        cres["image_beams"]["weights"]
    )

    out["time"] = cres["time"]
    out["time"]["fwhm"] = {}
    out["time"]["fwhm"]["start"] = time_fwhm_start
    out["time"]["fwhm"]["stop"] = time_fwhm_stop
    out["time"]["containment80"] = {}
    out["time"]["containment80"]["start"] = time_80_start
    out["time"]["containment80"]["stop"] = time_80_stop

    out["image"] = {}
    out["image"]["angle80"] = psf_angle80
    out["image"]["binning"] = thisbinning
    out["image"]["raw"] = imgraw
    return out


def make_norm_image(image_response):
    norm_image = (
        image_response["image"]["raw"]
        / image_response["statistics"]["photons"]["valid"]
    )
    return norm_image
