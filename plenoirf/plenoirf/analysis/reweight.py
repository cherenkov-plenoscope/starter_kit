import numpy as np
from .. import utils


def reweight(
    initial_energies,
    initial_rates,
    target_energies,
    target_rates,
    event_energies,
):
    assert len(initial_energies) == len(initial_rates)
    assert len(target_energies) == len(target_rates)

    relative_target_rate = target_rates / np.sum(target_rates)
    relative_initial_rate = initial_rates / np.sum(initial_rates)
    relative_initial_aban = np.interp(
        x=event_energies, xp=initial_energies, fp=relative_initial_rate,
    )
    relative_target_aban = np.interp(
        x=event_energies, xp=target_energies, fp=relative_target_rate,
    )
    weights = utils._divide_silent(
        numerator=relative_target_aban,
        denominator=relative_initial_aban,
        default=0.0,
    )

    return weights / np.max(weights)


def estimate_binning(
    energies, start_energy, stop_energy,
):
    num_events = energies.shape[0]
    num_bins = int(np.round(num_events ** (1 / 3)))
    assert num_bins > 0
    num_bin_edges = num_bins + 1

    # geomspace
    geom_energy_bin_edges = np.geomspace(
        start_energy, stop_energy, num_bin_edges
    )
    geom_counts = np.histogram(energies, bins=geom_energy_bin_edges)[0]

    # linspace
    lins_energy_bin_edges = np.linspace(
        start_energy, stop_energy, num_bin_edges
    )
    lins_counts = np.histogram(energies, bins=lins_energy_bin_edges)[0]

    if np.std(geom_counts) < np.std(lins_counts):
        return "geomspace", num_bins
    else:
        return "linspace", num_bins


def geominterp(x, xp, fp):
    return np.interp(x=np.log(x), xp=np.log(xp), fp=fp)


def estimate_relative_rates(
    energies, start_energy, stop_energy,
):
    space_name, num_bins = estimate_binning(
        energies=energies, start_energy=start_energy, stop_energy=stop_energy
    )
    if space_name == "linspace":
        space = np.linspace
        interspace = np.interp
    elif space_name == "geomspace":
        space = np.geomspace
        interspace = geominterp
    energy_bin_edges = space(start_energy, stop_energy, num_bins + 1)
    energy_supports = space(start_energy, stop_energy, 2 * num_bins + 1)
    energy_bin_centers = []
    for i in range(2 * num_bins):
        if i % 2 != 0:
            energy_bin_centers.append(energy_supports[i])
    energy_bin_centers = np.array(energy_bin_centers)
    bin_counts = np.histogram(energies, bins=energy_bin_edges)[0]
    rate = interspace(x=energy_supports, xp=energy_bin_centers, fp=bin_counts)
    rate = rate / np.max(rate)
    return energy_supports, rate


def make_relative_rates_for_power_law(
    energy_start, energy_stop, num, slope,
):
    assert energy_stop > energy_start > 0
    energies = np.geomspace(energy_start, energy_stop, num)
    rates = energies ** slope
    relative_rates = rates / np.max(rates)
    return energies, relative_rates


def histogram_with_bin_wise_power_law_reweighting(
    a, bins, weights, target_power_law_slope, max_power_law_weight_factor
):
    """
    Returns the bin-counts and bin-edges of a histogram.
    Same as numpy.histogram(a, bins, weights).
    But this histogram can reweight the values 'a' in order to follow a desired
    power-law.

    Parameters
    ----------
    a : array, floats
            The values to be histogramed.
    bins : array, floats
            The bin-__edges__ of the histogram, len(bins) - 1  == num_bins.
    weights : array, floats
            The weights for the values. len(a) == len(weights).
    target_power_law_slope : float
            The spectral-index of the targeted power-law spectrum.
            If target_power_law_slope == 0.0, the result is expected to equal
            numpy.histogram(a, bins, weights).
    max_power_law_weight_factor : float
            The maximal acceptable scaling factor for a power-law related
            weight.
    """
    num_bins = len(bins) - 1
    assert num_bins >= 1

    # init bins
    # ---------
    bin_counts = np.nan * np.ones(num_bins)

    # fill bins
    # ---------
    for ibin in range(num_bins):
        a_start = bins[ibin]
        a_stop = bins[ibin + 1]
        assert a_stop > a_start, "Bin edges must be ascending."
        bin_mask = np.logical_and((a >= a_start), (a < a_stop))

        a_in_bin = a[bin_mask]
        weights_in_bin = weights[bin_mask]

        a_in_bin_energies, a_in_bin_rates = estimate_relative_rates(
            energies=a_in_bin,
            start_energy=bins[ibin],
            stop_energy=bins[ibin + 1],
        )

        target_energies, target_rates = make_relative_rates_for_power_law(
            energy_start=a_start,
            energy_stop=a_stop,
            num=len(a_in_bin_rates),
            slope=target_power_law_slope,
        )

        power_law_weights = reweight(
            initial_energies=a_in_bin_energies,
            initial_rates=a_in_bin_rates,
            target_energies=target_energies,
            target_rates=target_rates,
            event_energies=a_in_bin,
        )

        power_law_weights = power_law_weights / np.mean(power_law_weights)
        assert np.all(power_law_weights < max_power_law_weight_factor)
        assert np.all(power_law_weights > 1.0 / max_power_law_weight_factor)

        bin_counts[ibin] = np.sum(weights_in_bin * power_law_weights)

    return bin_counts, bins
