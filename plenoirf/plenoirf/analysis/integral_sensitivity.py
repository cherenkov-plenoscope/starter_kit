import numpy as np
import cosmic_fluxes
import lima1983analysis
from .. import utils

def estimate_detection_rate_per_s_for_power_law(
    effective_area_bins_m2,
    effective_area_energy_bin_centers_GeV,
    effective_area_energy_bin_width_GeV,
    flux_density_per_m2_per_GeV_per_s,
    spectral_index,
    pivot_energy_GeV,
):
    differential_flux_per_m2_per_s_per_GeV = cosmic_fluxes._power_law(
        energy=effective_area_energy_bin_centers_GeV,
        flux_density=flux_density_per_m2_per_GeV_per_s,
        spectral_index=spectral_index,
        pivot_energy=pivot_energy_GeV,
    )

    differential_rate_per_s_per_GeV = (
        differential_flux_per_m2_per_s_per_GeV * effective_area_bins_m2
    )

    rate_per_s = np.sum(
        differential_rate_per_s_per_GeV * effective_area_energy_bin_width_GeV
    )
    return rate_per_s


def relative_ratio(a, b):
    return np.abs(a - b)/(0.5 * (a + b))


def find_intersection_two_lines(sup1, slope1, sup2, slope2):
    return (sup2 - sup1) / (slope1 - slope2)


def estimate_tangent_of_critical_power_laws(critical_power_laws):
    # power-laws are lines in the log-log-scale
    supports = []
    slopes = []
    for plaw in critical_power_laws:
        supports.append(np.log10(plaw["flux_density_per_m2_per_GeV_per_s"]))
        slopes.append(plaw["spectral_index"])

    energy_GeV = []
    flux_density_per_m2_per_GeV_per_s = []
    for ll in range(len(supports) - 1):
        log10_E = find_intersection_two_lines(
            sup1=supports[ll],
            slope1=slopes[ll],
            sup2=supports[ll + 1],
            slope2=slopes[ll + 1],
        )
        _E = 10 ** log10_E
        log10_F = supports[ll] + slopes[ll]*(log10_E)
        _F = 10 ** log10_F
        energy_GeV.append(_E)
        flux_density_per_m2_per_GeV_per_s.append(_F)
    return np.array(energy_GeV), np.array(flux_density_per_m2_per_GeV_per_s)


def estimate_critical_power_laws(
    effective_area_bins_m2,
    effective_area_energy_bin_edges_GeV,
    critical_rate_per_s,
    power_law_spectral_indices,
    pivot_energy_gev=1.0,
    margin=1e-2,
    upper_flux_density_per_m2_per_GeV_per_s=1e6,
    max_num_iterations=10000,
):
    assert (
        len(effective_area_energy_bin_edges_GeV) ==
        len(effective_area_bins_m2) + 1
    )

    assert np.all(effective_area_bins_m2 >= 0.0)
    assert np.all(effective_area_energy_bin_edges_GeV > 0.0)
    assert np.all(np.gradient(effective_area_energy_bin_edges_GeV) > 0.0)

    effective_area_energy_bin_centers_GeV = utils.bin_centers(
        bin_edges=effective_area_energy_bin_edges_GeV,
    )
    effective_area_energy_bin_width_GeV = utils.bin_width(
        bin_edges=effective_area_energy_bin_edges_GeV,
    )

    power_laws = []

    for i, spectral_index in enumerate(power_law_spectral_indices):

        flux = upper_flux_density_per_m2_per_GeV_per_s

        iteration = 0
        while True:
            assert iteration < max_num_iterations

            detection_rate_per_s = estimate_detection_rate_per_s_for_power_law(
                effective_area_bins_m2=effective_area_bins_m2,
                effective_area_energy_bin_centers_GeV=effective_area_energy_bin_centers_GeV,
                effective_area_energy_bin_width_GeV=effective_area_energy_bin_width_GeV,
                flux_density_per_m2_per_GeV_per_s=flux,
                spectral_index=spectral_index,
                pivot_energy_GeV=pivot_energy_gev,
            )

            ratio = relative_ratio(detection_rate_per_s, critical_rate_per_s)

            if ratio < margin:
                break

            rr = ratio/3
            if detection_rate_per_s > critical_rate_per_s:
                flux *= (1-rr)
            else:
                flux *= (1+rr)

            iteration += 1

        law = {}
        law["spectral_index"] = float(spectral_index)
        law["pivot_energy_GeV"] = float(pivot_energy_gev)
        law["flux_density_per_m2_per_GeV_per_s"] = float(flux)

        power_laws.append(law)
    return power_laws


def estimate_critical_rate(
    background_rate_in_onregion_per_s,
    onregion_over_offregion_ratio,
    observation_time_s,
    instrument_systematic_uncertainty,
    detection_threshold_std,
    method="LiMaEq17",
):
    bg_rate_off_per_s = (
        background_rate_in_onregion_per_s / onregion_over_offregion_ratio
    )
    bg_count_off = bg_rate_off_per_s * observation_time_s

    if method == "sqrt":
        bg_count_off_std = np.sqrt(bg_count_off)
        bg_count_on_std = bg_count_off_std * onregion_over_offregion_ratio
        sig_count_stat_on = detection_threshold_std * bg_count_on_std
    elif method == "LiMaEq9":
        sig_count_stat_on = lima1983analysis.estimate_N_s_eq9(
            N_off=bg_count_off,
            alpha=onregion_over_offregion_ratio,
            S=detection_threshold_std)
    elif method == "LiMaEq17":
        sig_count_stat_on = lima1983analysis.estimate_N_s_eq17(
            N_off=bg_count_off,
            alpha=onregion_over_offregion_ratio,
            S=detection_threshold_std)
    else:
        raise KeyError("Unknown method: '{:s}'".format(method))

    sig_count_sys_on = (
        detection_threshold_std *
        instrument_systematic_uncertainty *
        background_rate_in_onregion_per_s *
        observation_time_s
    )

    sig_count_on = np.max([sig_count_stat_on, sig_count_sys_on])


    sig_rate_on_per_s = sig_count_on / observation_time_s
    return sig_rate_on_per_s
