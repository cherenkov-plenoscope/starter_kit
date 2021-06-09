import numpy as np
import gamma_limits_sensitivity as gls
import scipy
from . import integral_sensitivity

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
        method="LiMa_eq17",
    )

    powlaws = integral_sensitivity.estimate_critical_power_laws(
        effective_area_bins_m2=effective_area_bins_m2,
        effective_area_energy_bin_edges_GeV=effective_area_energy_bin_edges_GeV,
        critical_rate_per_s=critical_rate_per_s,
        power_law_spectral_indices=np.linspace(
            gamma_range[0],
            gamma_range[1],
            num_points
        ),
    )

    return integral_sensitivity.estimate_tangent_of_critical_power_laws(
        critical_power_laws=powlaws
    )


def estimate_integral_spectral_exclusion_zone_using_gls(
    effective_area_bins_m2,
    effective_area_energy_bin_edges_GeV,
    background_rate_in_onregion_per_s,
    onregion_over_offregion_ratio,
    observation_time_s,
    instrument_systematic_uncertainty=0.0,
    num_points=27,
    gamma_range=[-5, -0.5],
    detection_threshold_std=5.0,
):
    assert len(effective_area_bins_m2) == len(effective_area_energy_bin_edges_GeV)
    assert observation_time_s >= 0.0
    assert background_rate_in_onregion_per_s >= 0.0
    assert num_points > 0
    assert detection_threshold_std > 0.0

    log10_energy_bin_centers_TeV = np.log10(1e-3 * effective_area_energy_bin_edges_GeV)
    gamma_effective_area_cm2 = 1e2 * 1e2 * effective_area_bins_m2
    a_eff_interpol = scipy.interpolate.interpolate.interp1d(
        x=log10_energy_bin_centers_TeV,
        y=gamma_effective_area_cm2,
        bounds_error=False,
        fill_value=0.0,
    )

    _energy_range = gls.get_energy_range(a_eff_interpol)
    e_0 = _energy_range[0] * 5.0

    lambda_lim = observation_time_s * gls.sigma_lim_li_ma_criterion(
        sigma_bg=background_rate_in_onregion_per_s,
        alpha=onregion_over_offregion_ratio,
        t_obs=observation_time_s,
        threshold=detection_threshold_std,
    )

    lambda_lim_sys = (
        instrument_systematic_uncertainty *
        background_rate_in_onregion_per_s *
        observation_time_s
    )

    lambda_lim = np.max([lambda_lim, lambda_lim_sys])

    energy_range = [
        gls.sensitive_energy(gamma_range[0], a_eff_interpol),
        gls.sensitive_energy(gamma_range[1], a_eff_interpol),
    ]
    _energy_support_TeV = np.geomspace(
        energy_range[0], energy_range[1], num_points
    )
    _differential_flux_per_TeV_per_s_per_cm2 = np.array(
        [
            gls.integral_spectral_exclusion_zone(
                energy=energy,
                lambda_lim=lambda_lim,
                a_eff_interpol=a_eff_interpol,
                t_obs=observation_time_s,
                e_0=e_0,
            )
            for energy in _energy_support_TeV
        ]
    )

    energy_support_GeV = 1e3 * _energy_support_TeV
    differential_flux_per_GeV_per_s_per_m2 = (
        1e-3 * 1e4 * (_differential_flux_per_TeV_per_s_per_cm2)
    )

    return energy_support_GeV, differential_flux_per_GeV_per_s_per_m2
