import numpy as np
import scipy
from . import integral_sensitivity
from . import critical_rate


def _find_intersection_two_lines(b1, m1, b2, m2):
    """
    Find the intersection of two affine functions:
    f1(x) = m1 * x + b1, and f2(x) = m2 * x + b2 in x.

    Parameters
    ----------
    m1 : float
        Slope of f1(x).
    b1 : float
        Support of f1(x).
    m2 : float
        Slope of f2(x).
    b2 : float
        Support of f2(x).

    Returns
    -------
    x : float
        Intersection of f1(x) and f2(x).
     """
    return (b2 - b1) / (m1 - m2)


def _estimate_tangent_of_consecutive_power_laws(A_ns, G_ns):
    """
    Estimate the curve described by the intersection-points of two
    consecutive power-laws in a list of N power-laws [f_0(x), ..., f_N(x)].

    f_0(x) = A_0 * x ^ {G_0},
           .
           .
    f_N(x) = A_N * x ^ {G_N}

    Parameters
    ----------
    A_ns : list of floats
        N power-laws normalizations: A_ns = [A_0, A_1, A_2, ... , A_N]
    G_ns : list of floats
        N power-laws exponents: G_ns = [G_0, G_1, G_2, ... , G_N]

    Returns
    -------
    (x, y) : (array of floats, array of floats)
        List of N-1 intersections for N power-laws.
    """
    assert len(A_ns) == len(G_ns)
    num = len(A_ns)
    assert num >= 2

    log10_A_ns = np.log10(np.array(A_ns))
    G_ns = np.array(G_ns)

    x = []
    y = []
    for i in range(num - 1):
        log10_x = _find_intersection_two_lines(
            b1=log10_A_ns[i], m1=G_ns[i], b2=log10_A_ns[i + 1], m2=G_ns[i + 1],
        )
        log10_y = log10_A_ns[i] + G_ns[i] * (log10_x)
        x.append(10 ** log10_x)
        y.append(10 ** log10_y)
    return (np.array(x), np.array(y))


def _estimate_tangent_of_my_consecutive_power_laws(power_laws):
    (
        energy_GeV,
        diff_flux_per_m2_per_GeV_per_s,
    ) = _estimate_tangent_of_consecutive_power_laws(
        A_ns=[p["flux_density_per_m2_per_GeV_per_s"] for p in power_laws],
        G_ns=[p["spectral_index"] for p in power_laws],
    )
    return energy_GeV, diff_flux_per_m2_per_GeV_per_s


def estimate_integral_spectral_exclusion_zone(
    effective_area_bins_m2,
    effective_area_energy_bin_edges_GeV,
    background_rate_in_onregion_per_s,
    onregion_over_offregion_ratio,
    observation_time_s,
    instrument_systematic_uncertainty=0.0,
    num_points=137,
    gamma_range=[-5, -0.5],
    detection_threshold_std=5.0,
    method="LiMaEq17",
):
    critical_rate_per_s = critical_rate.estimate_critical_rate(
        hatR_B=background_rate_in_onregion_per_s,
        alpha=onregion_over_offregion_ratio,
        T_obs=observation_time_s,
        U_sys_rel_unc=instrument_systematic_uncertainty,
        S=detection_threshold_std,
        estimator_statistics=method,
    )

    critical_power_laws = integral_sensitivity.estimate_critical_power_laws(
        effective_area_bins_m2=effective_area_bins_m2,
        effective_area_energy_bin_edges_GeV=effective_area_energy_bin_edges_GeV,
        critical_rate_per_s=critical_rate_per_s,
        power_law_spectral_indices=np.linspace(
            gamma_range[0], gamma_range[1], num_points
        ),
    )

    return _estimate_tangent_of_my_consecutive_power_laws(
        power_laws=critical_power_laws
    )
