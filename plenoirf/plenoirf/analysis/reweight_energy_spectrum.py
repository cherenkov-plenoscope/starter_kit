import numpy as np
from .effective_quantity import _divide_silent


def reweight_energy_spectrum(
    initial_energy_bin_centers,
    initial_rate,
    target_energy_bin_centers,
    target_rate,
    energies_of_events,
):
    relative_target_rate = target_rate / np.sum(target_rate)
    relative_initial_rate = initial_rate / np.sum(initial_rate)
    relative_initial_aban = np.interp(
        x=energies_of_events,
        xp=initial_energy_bin_centers,
        fp=relative_initial_rate,
    )
    relative_target_aban = np.interp(
        x=energies_of_events,
        xp=target_energy_bin_centers,
        fp=relative_target_rate,
    )
    weights = _divide_silent(
        numerator=relative_target_aban,
        denominator=relative_initial_aban,
        default=0.0,
    )

    return weights / np.max(weights)
