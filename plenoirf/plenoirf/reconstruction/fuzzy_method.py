"""
First estimate for axis towards showe's core.
"""
import plenopy as pl
import numpy as np
import scipy
from ..summary import bin_centers as make_bin_centers


def compile_user_config(user_config):
    uc = user_config
    cfg = {}

    uimg = uc["image"]
    img = {}
    img["radius"] = np.deg2rad(uimg["radius_deg"])
    img["num_bins"] = uimg["num_bins"]
    img["c_bin_edges"] = np.linspace(
        -img["radius"], +img["radius"], img["num_bins"] + 1,
    )
    img["c_bin_centers"] = make_bin_centers(bin_edges=img["c_bin_edges"])
    _image_bins_per_rad = img["num_bins"] / (2.0 * img["radius"])
    img["smoothing_kernel_width"] = np.deg2rad(
        uimg["smoothing_kernel_width_deg"]
    )
    img["smoothing_kernel"] = pl.fuzzy.discrete_kernel.gauss2d(
        num_steps=int(
            np.round(img["smoothing_kernel_width"] * _image_bins_per_rad)
        )
    )
    cfg["image"] = img

    uazr = uc["azimuth_ring"]
    azr = {}
    azr["num_bins"] = uazr["num_bins"]
    azr["bin_edges"] = np.linspace(
        0.0, 2.0 * np.pi, azr["num_bins"], endpoint=False
    )
    azr["radius"] = np.deg2rad(uazr["radius_deg"])
    _ring_bins_per_rad = azr["num_bins"] / (2.0 * np.pi)
    azr["smoothing_kernel_width"] = np.deg2rad(
        uazr["smoothing_kernel_width_deg"]
    )
    azr["smoothing_kernel"] = pl.fuzzy.discrete_kernel.gauss1d(
        num_steps=int(
            np.round(_ring_bins_per_rad * azr["smoothing_kernel_width"])
        )
    )
    cfg["azimuth_ring"] = azr
    cfg["ellipse_model"] = dict(uc["ellipse_model"])
    return cfg


def estimate_main_axis_to_core(
    split_light_field,
    model_config,
    image_binning,
    image_smoothing_kernel,
    ring_binning,
    ring_smoothing_kernel,
):
    median_cx = split_light_field.median_cx
    median_cy = split_light_field.median_cy

    # make fuzzy image
    # ----------------
    # A probability-density for the shower's main-axis and primary particle's
    # direction.

    slf_model = pl.fuzzy.direction.estimate_model_from_light_field(
        split_light_field=split_light_field, model_config=model_config
    )
    fuzzy_image = pl.fuzzy.direction.make_image_from_model(
        split_light_field_model=slf_model, image_binning=image_binning,
    )
    fuzzy_image_smooth = scipy.signal.convolve2d(
        in1=fuzzy_image, in2=image_smoothing_kernel, mode="same"
    )
    reco_cx, reco_cy = pl.fuzzy.direction.argmax_image_cx_cy(
        image=fuzzy_image_smooth, image_binning=image_binning,
    )

    median_cx_std = np.std([a["median_cx"] for a in slf_model])
    median_cy_std = np.std([a["median_cy"] for a in slf_model])

    # make ring to find main-axis
    # ---------------------------

    azimuth_ring = pl.fuzzy.direction.project_image_onto_ring(
        image=fuzzy_image_smooth,
        image_binning=image_binning,
        ring_cx=median_cx,
        ring_cy=median_cy,
        ring_radius=ring_binning["radius"],
        ring_binning=ring_binning,
    )
    azimuth_ring_smooth = pl.fuzzy.direction.circular_convolve1d(
        in1=azimuth_ring, in2=ring_smoothing_kernel
    )
    azimuth_ring_smooth /= np.max(azimuth_ring_smooth)

    # analyse ring to find main-axis
    # ------------------------------

    # maximum
    main_axis_azimuth = ring_binning["bin_edges"][
        np.argmax(azimuth_ring_smooth)
    ]

    # relative uncertainty
    _unc = np.mean(azimuth_ring_smooth)
    main_axis_azimuth_uncertainty = _unc ** 2.0

    result = {}
    result["main_axis_support_cx"] = median_cx
    result["main_axis_support_cy"] = median_cy
    result["main_axis_support_uncertainty"] = np.hypot(
        median_cx_std, median_cy_std
    )
    result["main_axis_azimuth"] = float(main_axis_azimuth)
    result["main_axis_azimuth_uncertainty"] = main_axis_azimuth_uncertainty
    result["reco_cx"] = reco_cx
    result["reco_cy"] = reco_cy

    debug = {}
    debug["split_light_field_model"] = slf_model
    debug["fuzzy_image"] = fuzzy_image
    debug["fuzzy_image_smooth"] = fuzzy_image_smooth
    debug["azimuth_ring"] = azimuth_ring
    debug["azimuth_ring_smooth"] = azimuth_ring_smooth

    return result, debug
