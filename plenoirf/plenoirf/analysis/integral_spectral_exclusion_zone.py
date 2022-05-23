import numpy as np
import scipy
from . import integral_sensitivity


def _find_intersection_two_lines(sup1, slope1, sup2, slope2):
    return (sup2 - sup1) / (slope1 - slope2)


def estimate_tangent_of_critical_power_laws(
    power_law_flux_densities,
    power_law_spectral_indices,
):
    """
    Estimate the curve described by the intersections of
    consecutive power-laws.

    Parameters
    ----------
    power_law_flux_densities : list / m^{-2} (GeV)^{-1} s^{-1}
        The power-laws flux-density.
    power_law_spectral_indices : list / 1
        The power-laws spectral indices.
    Returns
    -------
    (energy, diff_flux) : (array, array) / (GeV, m^{-2} (GeV)^{-1} s^{-1})
    """
    assert len(power_law_flux_densities) == len(power_law_spectral_indices)
    assert len(power_law_flux_densities) >= 2

    supports = np.log10(np.array(power_law_flux_densities))
    slopes = np.array(power_law_spectral_indices)

    energy = []
    diff_flux = []
    for ll in range(len(supports) - 1):
        log10_E = _find_intersection_two_lines(
            sup1=supports[ll],
            slope1=slopes[ll],
            sup2=supports[ll + 1],
            slope2=slopes[ll + 1],
        )
        _E = 10 ** log10_E
        log10_F = supports[ll] + slopes[ll] * (log10_E)
        _F = 10 ** log10_F
        energy.append(_E)
        diff_flux.append(_F)
    return np.array(energy), np.array(diff_flux)


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
):
    critical_rate_per_s = integral_sensitivity.estimate_critical_rate(
        background_rate_in_onregion_per_s=background_rate_in_onregion_per_s,
        onregion_over_offregion_ratio=onregion_over_offregion_ratio,
        observation_time_s=observation_time_s,
        instrument_systematic_uncertainty=instrument_systematic_uncertainty,
        detection_threshold_std=detection_threshold_std,
        method="LiMaEq17",
    )

    critical_power_laws = integral_sensitivity.estimate_critical_power_laws(
        effective_area_bins_m2=effective_area_bins_m2,
        effective_area_energy_bin_edges_GeV=effective_area_energy_bin_edges_GeV,
        critical_rate_per_s=critical_rate_per_s,
        power_law_spectral_indices=np.linspace(
            gamma_range[0], gamma_range[1], num_points
        ),
    )

    return estimate_tangent_of_critical_power_laws(
        power_law_flux_densities=[p["flux_density_per_m2_per_GeV_per_s"] for p in critical_power_laws],
        power_law_spectral_indices=[p["spectral_index"] for p in critical_power_laws],
    )
