import corsika_primary as cpw
import os
import copy
import plenoirf
import numpy as np
import plenopy
import scipy
from scipy import spatial
from scipy import stats


BINNING = {}
BINNING["image"] = {}
BINNING["image"]["center"] = {"cx_deg": 0.0, "cy_deg": 0.0}
BINNING["image"]["num_pixel_cx"] = 64
BINNING["image"]["num_pixel_cy"] = 64
BINNING["image"]["pixel_angle_deg"] = 0.0125


def make_bin_edges_and_centers(bin_width, num_bins, first_bin_center):
    bin_edges = np.linspace(
        start=first_bin_center + bin_width * (-0.5),
        stop=first_bin_center + bin_width * (num_bins + 0.5),
        num=num_bins + 1,
    )
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[0:-1])
    return bin_edges, bin_centers


def histogram2d_std(
    x, y, x_std, y_std, weights, bins, prng, num_sub_samples=10,
):
    num_samples = len(x)
    assert len(y) == num_samples
    assert len(x_std) == num_samples
    assert len(y_std) == num_samples
    assert len(weights) == num_samples
    assert num_sub_samples > 0

    bin_edges_x, bin_edges_y = bins
    assert np.all(np.gradient(bin_edges_x) > 0)
    assert np.all(np.gradient(bin_edges_y) > 0)

    num_bins_x = len(bin_edges_x) - 1
    num_bins_y = len(bin_edges_y) - 1
    assert num_bins_x > 0
    assert num_bins_y > 0

    counts = np.zeros(shape=(num_bins_x, num_bins_y))

    for i in range(num_samples):

        if weights[i] == 0:
            continue

        for s in range(num_sub_samples):

            rx = prng.normal(loc=x[i], scale=x_std[i])
            ry = prng.normal(loc=y[i], scale=y_std[i])

            ibx = np.digitize(x=rx, bins=bin_edges_x) - 1
            iby = np.digitize(x=ry, bins=bin_edges_y) - 1

            if 0 <= ibx < num_bins_x and 0 <= iby < num_bins_y:
                counts[ibx, iby] += weights[i] / num_sub_samples
    return counts, bins


def calibrate_plenoscope_response(
    event, light_field_geometry, object_distance,
):
    image_rays = plenopy.image.ImageRays(
        light_field_geometry=light_field_geometry
    )

    rsr = event.raw_sensor_response
    time_bin_edges, time_bin_centers = make_bin_edges_and_centers(
        bin_width=rsr.time_slice_duration,
        num_bins=rsr.number_time_slices,
        first_bin_center=0.0,
    )

    out = {}
    out["time"] = {}
    out["time"]["bin_edges"] = time_bin_edges
    out["time"]["bin_centers"] = time_bin_centers
    out["time"][
        "weights"
    ] = event.light_field_sequence_for_isochor_image().sum(axis=1)

    out["image_beams"] = {}
    out["image_beams"][
        "_weights"
    ] = event.light_field_sequence_for_isochor_image().sum(axis=0)
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


def encirclement2d(
    x,
    y,
    x_std,
    y_std,
    weights,
    prng,
    percentile=80,
    iteration_shrinking_factor=0.99,
    num_sub_samples=1,
):
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    assert not np.any(np.isnan(x_std))
    assert not np.any(np.isnan(y_std))
    assert np.all(x_std >= 0.0)
    assert np.all(y_std >= 0.0)
    assert not np.any(np.isnan(weights))
    assert np.all(weights >= 0.0)
    assert 0 < percentile <= 100
    assert num_sub_samples > 0
    assert 0 < iteration_shrinking_factor < 1.0

    xy = []
    for i in range(len(x)):
        for w in range(weights[i]):
            for s in range(num_sub_samples):
                rx = prng.normal(loc=x[i], scale=x_std[i])
                ry = prng.normal(loc=y[i], scale=y_std[i])
                xy.append([rx, ry])
    xy = np.array(xy)

    required_fraction = percentile / 100.0
    integral = xy.shape[0]
    if integral == 0:
        return float("nan"), float("nan"), float("nan")

    assert integral > 0
    center_x = np.median(xy[:, 0])
    center_y = np.median(xy[:, 1])
    radii = np.hypot((xy[:, 0] - center_x), (xy[:, 1] - center_y))
    radius = np.max(radii)

    tree = scipy.spatial.cKDTree(xy)

    num_loops = 0

    while True:
        overlap = len(tree.query_ball_point(x=[center_x, center_y], r=radius))
        if overlap / integral >= required_fraction:
            radius = radius * iteration_shrinking_factor
        else:
            break

        if num_loops > 1000:
            assert False, "Can not converge."
        num_loops += 1

    return center_x, center_y, radius


def encirclement1d(x, f, percentile=80, oversample=137):
    assert len(x) == len(f)
    assert len(x) >= 3
    assert np.all(np.gradient(x) > 0.0)
    assert percentile > 0
    assert oversample >= 1
    num_bins_fine = len(x) * oversample

    start_fraction = 0.5 - 0.5 * (percentile / 100.0)
    stop_fraction = 0.5 + 0.5 * (percentile / 100.0)

    xfine = np.linspace(x[0], x[-1], num_bins_fine,)
    ffine = np.interp(x=xfine, xp=x, fp=f)

    ffine = ffine / np.sum(ffine)
    cumffine = np.cumsum(ffine)

    imax = np.argmax(ffine)

    istart = int(imax)
    istop = int(imax)

    while True:
        if cumffine[istart] < start_fraction:
            break
        elif istart == 0:
            break
        else:
            istart -= 1

    while True:
        if cumffine[istop] > stop_fraction:
            break
        elif istop == num_bins_fine - 1:
            break
        else:
            istop += 1

    return xfine[istart], xfine[istop]


def full_width_half_maximum(x, f, oversample=137):
    assert len(x) == len(f)
    assert len(x) >= 3
    assert np.all(np.gradient(x) > 0.0)
    assert oversample >= 1
    num_bins_fine = len(x) * oversample

    xfine = np.linspace(x[0], x[-1], num_bins_fine,)
    # print("xfine", xfine[0], xfine[-1])

    ffine = np.interp(x=xfine, xp=x, fp=f)
    ffine = ffine / np.max(ffine)
    imax = np.argmax(ffine)

    istart = int(imax)
    istop = int(imax)
    while True:
        if ffine[istart] < 0.5:
            break
        elif istart == 0:
            break
        else:
            istart -= 1

    while True:
        if ffine[istop] < 0.5:
            break
        elif istop == num_bins_fine - 1:
            break
        else:
            istop += 1

    return xfine[istart], xfine[istop]


def analyse_response_to_calibration_source(
    off_axis_angle_deg,
    event,
    light_field_geometry,
    object_distance_m,
    containment_percentile,
    binning,
    prng,
):
    calibrated_response = calibrate_plenoscope_response(
        light_field_geometry=light_field_geometry,
        event=event,
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
    thisbinning["image"]["center"]["cx_deg"] = off_axis_angle_deg
    thisbinning["image"]["center"]["cy_deg"] = 0.0
    thisimg_bin_edges = binning_image_bin_edges(binning=thisbinning)

    # print("image histogram2d_std")
    imgraw = histogram2d_std(
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
    out["statistics"]["photons"][
        "total"
    ] = event.raw_sensor_response.number_photons
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
