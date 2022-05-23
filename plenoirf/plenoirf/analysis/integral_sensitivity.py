import numpy as np
import cosmic_fluxes
import lima1983analysis
import binning_utils


def estimate_detection_rate_per_s_for_power_law(
    effective_area_m2,
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
        differential_flux_per_m2_per_s_per_GeV * effective_area_m2
    )

    rate_per_s = np.sum(
        differential_rate_per_s_per_GeV * effective_area_energy_bin_width_GeV
    )
    return rate_per_s


def _relative_ratio(a, b):
    return np.abs(a - b) / (0.5 * (a + b))


def estimate_critical_power_laws(
    effective_area_m2,
    effective_area_energy_bin_edges_GeV,
    critical_rate_per_s,
    power_law_spectral_indices,
    pivot_energy_gev=1.0,
    margin=1e-2,
    upper_flux_density_per_m2_per_GeV_per_s=1e6,
    max_num_iterations=10000,
):
    assert (
        len(effective_area_energy_bin_edges_GeV)
        == len(effective_area_m2) + 1
    )

    assert np.all(effective_area_m2 >= 0.0)
    assert np.all(effective_area_energy_bin_edges_GeV > 0.0)
    assert np.all(np.gradient(effective_area_energy_bin_edges_GeV) > 0.0)

    effective_area_energy_bin_centers_GeV = binning_utils.centers(
        bin_edges=effective_area_energy_bin_edges_GeV,
    )
    effective_area_energy_bin_width_GeV = binning_utils.widths(
        bin_edges=effective_area_energy_bin_edges_GeV,
    )

    power_laws = []

    for i, spectral_index in enumerate(power_law_spectral_indices):

        flux = upper_flux_density_per_m2_per_GeV_per_s

        iteration = 0
        while True:
            assert iteration < max_num_iterations

            detection_rate_per_s = estimate_detection_rate_per_s_for_power_law(
                effective_area_m2=effective_area_m2,
                effective_area_energy_bin_centers_GeV=effective_area_energy_bin_centers_GeV,
                effective_area_energy_bin_width_GeV=effective_area_energy_bin_width_GeV,
                flux_density_per_m2_per_GeV_per_s=flux,
                spectral_index=spectral_index,
                pivot_energy_GeV=pivot_energy_gev,
            )

            ratio = _relative_ratio(detection_rate_per_s, critical_rate_per_s)

            if ratio < margin:
                break

            rr = ratio / 3
            if detection_rate_per_s > critical_rate_per_s:
                flux *= 1 - rr
            else:
                flux *= 1 + rr

            iteration += 1

        law = {}
        law["spectral_index"] = float(spectral_index)
        law["pivot_energy_GeV"] = float(pivot_energy_gev)
        law["flux_density_per_m2_per_GeV_per_s"] = float(flux)

        power_laws.append(law)
    return power_laws
