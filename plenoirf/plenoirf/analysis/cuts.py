import sparse_numeric_table as spt
import magnetic_deflection as mdfl
import numpy as np
from . import gamma_direction
from .. import table as irf_table
import corsika_primary_wrapper as cpw

CUTS_EXAMPLE = {
    "passed_trigger": {"modus": {}, "threshold_pe": 125,},
    "true_source_in_possible_onregion": {
        "pointing_azimuth_deg": 0.0,
        "pointing_zenith_deg": 0.0,
        "radial_angle_to_put_possible_onregion_deg": 1.0,
    },
    "reconstructed_source_in_true_onregion": {
        "radial_angle_onregion_deg": 1.0,
    },
    "reconstructed_source_in_possible_onregion": {
        "radial_angle_onregion_deg": 1.0,
        "radial_angle_to_put_possible_onregion_deg": 3.0,
    },
    "has_features": {},
    "has_quality": {
        "max_relative_leakage": 0.1,
        "min_reconstructed_photons": 100,
    },
}


def cut_table(table, cuts):
    idxs = []

    if "passed_trigger" in cuts:
        cut = cuts["passed_trigger"]
        _idx = light_field_trigger_modi.make_indices(
            trigger_table=table["trigger"],
            threshold=cut["threshold_pe"],
            modus=cut["modus"],
        )
        idxs.append(_idx)

    if "true_source_in_possible_onregion" in cuts:
        cut = cuts["true_source_in_possible_onregion"]
        _idx = cut_primary_direction_within_angle(
            primary_table=table["primary"],
            radial_angle_deg=cut["radial_angle_deg"],
            azimuth_deg=cut["azimuth_deg"],
            zenith_deg=cut["zenith_deg"],
        )
        idxs.append(_idx)

    if "reconstructed_source_in_true_onregion" in cuts:
        cut = cuts["reconstructed_source_in_true_onregion"]
        _idx = _
        idxs.append(_idx)

    common_idxs = spt.intersection(idxs)


def cut_primary_direction_within_angle(
    primary_table, radial_angle_deg, azimuth_deg, zenith_deg,
):
    delta_deg = mdfl.discovery._angle_between_az_zd_deg(
        az1_deg=np.rad2deg(primary_table["azimuth_rad"]),
        zd1_deg=np.rad2deg(primary_table["zenith_rad"]),
        az2_deg=azimuth_deg,
        zd2_deg=zenith_deg,
    )
    inside = delta_deg <= radial_angle_deg
    idxs_inside = primary_table[spt.IDX][inside]
    return idxs_inside


def cut_quality(
    feature_table, max_relative_leakage, min_reconstructed_photons,
):
    ft = feature_table
    # size
    # ----
    mask_sufficient_size = ft["num_photons"] >= min_reconstructed_photons
    idxs_sufficient_size = ft[spt.IDX][mask_sufficient_size]

    # leakage
    # -------
    relative_leakage = (
        ft["image_smallest_ellipse_num_photons_on_edge_field_of_view"]
        / ft["num_photons"]
    )
    mask_acceptable_leakage = relative_leakage <= max_relative_leakage
    idxs_acceptable_leakage = ft[spt.IDX][mask_acceptable_leakage]

    return spt.intersection([idxs_sufficient_size, idxs_acceptable_leakage])


def cut_energy_bin(
    primary_table, lower_energy_edge_GeV, upper_energy_edge_GeV,
):
    mask_energy_bin = (
        primary_table["energy_GeV"] >= lower_energy_edge_GeV
    ) * (primary_table["energy_GeV"] < upper_energy_edge_GeV)
    return primary_table[spt.IDX][mask_energy_bin]


def cut_core_radius_bin(
    core_table,
    lower_core_radius_edge_m,
    upper_core_radius_edge_m,
):
    core_radius_m = np.hypot(core_table["core_x_m"], core_table["core_y_m"])
    mask = (
        (core_radius_m >= lower_core_radius_edge_m) *
        (core_radius_m < upper_core_radius_edge_m)
    )
    return core_table[spt.IDX][mask]


def cut_reconstructed_source_in_true_onregion(
    table, radial_angle_onregion_deg,
):
    ta = spt.cut_table_on_indices(
        table=table,
        structure=irf_table.STRUCTURE,
        common_indices=table["features"][spt.IDX],
        level_keys=["primary", "features"],
    )
    ta = spt.sort_table_on_common_indices(
        table=ta, common_indices=table["features"][spt.IDX],
    )

    idx = []
    for evt in range(table["features"].shape[0]):
        (true_cx, true_cy) = mdfl.discovery._az_zd_to_cx_cy(
            azimuth_deg=np.rad2deg(ta["primary"]["azimuth_rad"][evt]),
            zenith_deg=np.rad2deg(ta["primary"]["zenith_rad"][evt]),
        )

        (rec_cx, rec_cy) = gamma_direction.estimate(
            light_front_cx=ta["features"]["light_front_cx"][evt],
            light_front_cy=ta["features"]["light_front_cy"][evt],
            image_infinity_cx_mean=ta["features"]["image_infinity_cx_mean"][
                evt
            ],
            image_infinity_cy_mean=ta["features"]["image_infinity_cy_mean"][
                evt
            ],
        )

        delta_cx = true_cx - rec_cx
        delta_cy = true_cy - rec_cy

        delta_c = np.hypot(delta_cx, delta_cy)
        delta_c_deg = np.rad2deg(delta_c)

        if delta_c_deg <= radial_angle_onregion_deg:
            idx.append(ta["primary"][spt.IDX][evt])

    return np.array(idx)


def cut_reconstructed_source_in_possible_onregion(
    feature_table, radial_angle_to_put_possible_onregion_deg,
):
    idx = []
    for evt in range(feature_table.shape[0]):
        (rec_cx, rec_cy) = gamma_direction.estimate(
            light_front_cx=feature_table["light_front_cx"][evt],
            light_front_cy=feature_table["light_front_cy"][evt],
            image_infinity_cx_mean=feature_table["image_infinity_cx_mean"][
                evt
            ],
            image_infinity_cy_mean=feature_table["image_infinity_cy_mean"][
                evt
            ],
        )

        delta_c = np.hypot(rec_cx, rec_cx)
        delta_c_deg = np.rad2deg(delta_c)

        if delta_c_deg <= radial_angle_to_put_possible_onregion_deg:
            idx.append(feature_table[spt.IDX][evt])

    return np.array(idx)
