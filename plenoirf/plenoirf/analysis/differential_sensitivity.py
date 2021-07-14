import numpy as np
from . import integral_sensitivity


def estimate_differential_sensitivity(
    energy_bin_edges,
    signal_effective_area_vs_energy_m2,
    signal_critical_rate_vs_energy_per_s,
):
    """
    Returns the diff.Flux-sensitivity df/dE / m^{-2} s^{-1} (GeV)^{-1}
    required to reach the critical signal rate.

    Parameters
    ----------
    energy_bin_edges : array 1d
            The edges of the energy-bins in GeV.
    signal_effective_area_vs_energy_m2 : array 1d
            The effective collection area for the signal in m$^{2}$.
    signal_critical_rate_vs_energy_per_s : array 1d
            The minimal rate for signal to claim a detection in s$^{-1}$.
    """
    dfdE = []
    for e in range(len(energy_bin_edges) - 1):
        dE_start_GeV = energy_bin_edges[e]
        dE_stop_GeV = energy_bin_edges[e + 1]
        dE_width_GeV = dE_stop - dE_start
        assert dE_width_GeV > 0.0

        rate_per_s = signal_critical_rate_vs_energy_per_s[e]
        f_per_s_per_m2 = rate_per_s / signal_effective_area_vs_energy_m2[e]
        dfdE_per_s_per_m2_per_GeV = f_per_s_per_m2 / dE_width_GeV
        dfdE.append(dfdE)
    return np.array(dfdE)


def make_energy_confusion_matrix_for_scenario(
    energy_confusion_matrix,
    scenario="reco_sharp",
):
    if scenario == "true":
        cm = np.eye(N=energy_confusion_matrix.shape[0])
    elif scenario == "reco_broad":
        cm = np.array(energy_confusion_matrix)
    elif scenario == "reco_sharp":
        cm = np.diag(energy_confusion_matrix)
    else:
        raise KeyError("Unknown scenario: '{:s}'".format(scenario))
    return cm


def estimate_critical_rate_vs_energy(
    background_rate_in_onregion_vs_energy_per_s,
    onregion_over_offregion_ratio,
    observation_time_s,
    instrument_systematic_uncertainty,
    detection_threshold_std,
    method,
):
    bg_vs_energy_per_s = background_rate_in_onregion_vs_energy_per_s
    rate_per_s = np.nan * np.ones(shape=bg_vs_energy_per_s.shape)

    for ebin in range(len(bg_vs_energy_per_s)):
        if bg_vs_energy_per_s[ebin] > 0.0:
            rate_per_s[ebin] = integral_sensitivity.estimate_critical_rate(
                background_rate_in_onregion_per_s=bg_vs_energy_per_s[ebin],
                onregion_over_offregion_ratio=onregion_over_offregion_ratio,
                observation_time_s=observation_time_s,
                instrument_systematic_uncertainty=instrument_systematic_uncertainty,
                detection_threshold_std=detection_threshold_std,
                method=method,
            )
        else:
            rate_per_s[ebin] = float("nan")
    return rate_per_s
