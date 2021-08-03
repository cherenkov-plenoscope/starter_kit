import numpy as np
from . import integral_sensitivity


def estimate_differential_sensitivity(
    energy_bin_edges_GeV,
    signal_area_vs_energy_m2,
    signal_rate_vs_energy_per_s,
):
    """
    Returns the diff.Flux-sensitivity df/dE / m^{-2} s^{-1} (GeV)^{-1}
    required to reach the critical signal rate.

    Parameters
    ----------
    energy_bin_edges_GeV : array 1d
            The edges of the energy-bins in GeV.
    signal_area_vs_energy_m2 : array 1d
            The effective area for the signal in m$^{2}$.
    signal_rate_vs_energy_per_s : array 1d
            The minimal rate for signal to claim a detection in s$^{-1}$.
    """
    num_bins = len(energy_bin_edges_GeV) - 1
    dfdE_per_s_per_m2_per_GeV = np.nan * np.ones(num_bins)
    for e in range(num_bins):
        dE_GeV = energy_bin_edges_GeV[e + 1] - energy_bin_edges_GeV[e]
        assert dE_GeV > 0.0
        if signal_area_vs_energy_m2[e] > 0:
            f_per_s_per_m2 = (
                signal_rate_vs_energy_per_s[e] / signal_area_vs_energy_m2[e]
            )
        else:
            f_per_s_per_m2 = np.nan

        dfdE_per_s_per_m2_per_GeV[e] = f_per_s_per_m2 / dE_GeV
    return dfdE_per_s_per_m2_per_GeV


SCENARIOS = [
    "perfect_energy",
    "broad_spectrum",
    "line_spectrum",
    "bell_spectrum",
]


def make_energy_confusion_matrix_for_scenario(
    energy_confusion_matrix,
    scenario="line_spectrum",
):
    if scenario == "perfect_energy":
        cm = np.eye(N=energy_confusion_matrix.shape[0])
    elif scenario == "broad_spectrum":
        cm = np.array(energy_confusion_matrix)
    elif scenario == "line_spectrum":
        cm = np.eye(N=energy_confusion_matrix.shape[0]) * np.diag(
            energy_confusion_matrix
        )
    elif scenario == "bell_spectrum":
        mask = make_energy_confusion_matrix_for_bell_spectrum(
            energy_confusion_matrix=energy_confusion_matrix,
            containment=0.68,
        )
        cm = mask * energy_confusion_matrix
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



def make_energy_confusion_matrix_for_bell_spectrum(
    energy_confusion_matrix,
    containment=0.68
):
    # ax0 -> true
    # ax1 -> reco
    num_bins = energy_confusion_matrix.shape[0]
    mask = np.zeros(shape=(num_bins, num_bins), dtype=np.int)

    # estimate containment regions:
    for etrue in range(num_bins):
        prob = energy_confusion_matrix[etrue, :]
        if np.sum(prob) > 0.0:
            assert 0.99 < np.sum(prob) < 1.01
            ereco_best = np.argmax(prob)
            cont = prob[ereco_best]
            mask[etrue, ereco_best] = 1

            start = ereco_best - 1
            stop = ereco_best + 1
            while True:
                if start > 0:
                    cont += prob[start]
                    mask[etrue, start] = 1
                    start -= 1
                if cont > containment:
                    break
                if stop + 1 < num_bins:
                    cont += prob[stop]
                    mask[etrue, stop] = 1
                    stop += 1
                if cont > containment:
                    break
                if start == 0 and stop + 1 == num_bins:
                    break
    return mask
