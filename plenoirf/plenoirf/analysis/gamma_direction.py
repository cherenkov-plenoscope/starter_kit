import numpy as np


def integration_width_for_containment(bin_counts, bin_edges, containment):
    assert containment >= 0
    assert containment <= 1
    if np.sum(bin_counts) == 0:
        return np.nan
    integral = np.cumsum(bin_counts / np.sum(bin_counts))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:]) / 2
    x = np.linspace(
        np.min(bin_centers), np.max(bin_centers), 100 * bin_centers.shape[0]
    )
    f = np.interp(x=x, fp=integral, xp=bin_centers)
    return x[np.argmin(np.abs(f - containment))]


def estimate(
    light_front_cx,
    light_front_cy,
    image_infinity_cx_mean,
    image_infinity_cy_mean,
):
    rec_cx = -0.5 * (light_front_cx + image_infinity_cx_mean)
    rec_cy = -0.5 * (light_front_cy + image_infinity_cy_mean)
    return rec_cx, rec_cy


def momentum_to_cx_cy_wrt_aperture(
    primary, plenoscope_pointing,
):
    assert plenoscope_pointing["zenith_deg"] == 0.0
    assert plenoscope_pointing["azimuth_deg"] == 0.0
    WRT_APERTURE = -1.0
    momentum = np.array(
        [
            primary["momentum_x_GeV_per_c"],
            primary["momentum_y_GeV_per_c"],
            primary["momentum_z_GeV_per_c"],
        ]
    ).T
    momentum_norm = np.linalg.norm(momentum, axis=1)
    for m in range(len(momentum_norm)):
        momentum[m, :] /= momentum_norm[m]
    return WRT_APERTURE * momentum[:, 0], WRT_APERTURE * momentum[:, 1]
