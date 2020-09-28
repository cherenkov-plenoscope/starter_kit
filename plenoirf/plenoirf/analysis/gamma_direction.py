import numpy as np
from . import effective_quantity


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
    momentum_x_GeV_per_c,
    momentum_y_GeV_per_c,
    momentum_z_GeV_per_c,
    plenoscope_pointing,
):
    assert plenoscope_pointing["zenith_deg"] == 0.0
    assert plenoscope_pointing["azimuth_deg"] == 0.0
    WRT_APERTURE = -1.0
    momentum = np.array(
        [
            momentum_x_GeV_per_c,
            momentum_y_GeV_per_c,
            momentum_z_GeV_per_c,
        ]
    ).T
    momentum_norm = np.linalg.norm(momentum, axis=1)
    for m in range(len(momentum_norm)):
        momentum[m, :] /= momentum_norm[m]
    return WRT_APERTURE * momentum[:, 0], WRT_APERTURE * momentum[:, 1]



def histogram_point_spread_function(
    delta_c_deg,
    theta_square_bin_edges_deg2,
    psf_containment_factor,
):
    """
    angle between truth and reconstruction for each event.

    psf_containment_factor e.g. 0.68
    """
    num_airshower = delta_c_deg.shape[0]

    if num_airshower > 0:
        delta_hist = np.histogram(
            delta_c_deg ** 2, bins=theta_square_bin_edges_deg2
        )[0]
        delta_hist_unc = effective_quantity._divide_silent(
            np.sqrt(delta_hist), delta_hist, np.nan
        )
    else:
        delta_hist = np.zeros(
            len(theta_square_bin_edges_deg2) - 1, dtype=np.int64
        )
        delta_hist_unc = np.nan * np.ones(
            len(theta_square_bin_edges_deg2) - 1, dtype=np.float64
        )

    theta_square_deg2 = integration_width_for_containment(
        bin_counts=delta_hist,
        bin_edges=theta_square_bin_edges_deg2,
        containment=psf_containment_factor,
    )

    if num_airshower > 0:
        theta_square_deg2_relunc = np.sqrt(num_airshower) / num_airshower
    else:
        theta_square_deg2_relunc = np.nan

    return {
        "delta_hist": delta_hist,
        "delta_hist_relunc": delta_hist_unc,
        "containment_angle_deg": np.sqrt(theta_square_deg2),
        "containment_angle_deg_relunc": theta_square_deg2_relunc,
    }