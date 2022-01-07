import numpy as np
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
        method="LiMaEq17",
    )

    powlaws = integral_sensitivity.estimate_critical_power_laws(
        effective_area_bins_m2=effective_area_bins_m2,
        effective_area_energy_bin_edges_GeV=effective_area_energy_bin_edges_GeV,
        critical_rate_per_s=critical_rate_per_s,
        power_law_spectral_indices=np.linspace(
            gamma_range[0], gamma_range[1], num_points
        ),
    )

    return integral_sensitivity.estimate_tangent_of_critical_power_laws(
        critical_power_laws=powlaws
    )
