import numpy as np

ERG_GEV = 624.15091


def differential_flux_to_spectral_energy_distribution(
    energy_GeV,
    differential_flux_per_GeV_per_m2_per_s
):
    return _differential_flux_to_spectral_energy_distribution(
        energy=energy_GeV,
        differential_flux=differential_flux_per_GeV_per_m2_per_s,
        base_energy_MeV=1e3,    # was GeV
        base_flux_area_cm2=1e4, # was m2
        base_flux_time_s=1.0,   # was s
        base_differential_energy_erg=ERG_GEV, # was (GeV)^{-1}
    )


def _differential_flux_to_spectral_energy_distribution(
    energy,
    differential_flux,
    base_energy_MeV=1e3,
    base_flux_area_cm2=1e4,
    base_flux_time_s=1.0,
    base_differential_energy_erg=ERG_GEV,
):
    '''
    x-axis: E / MeV

    y-axis: E^{2} Flux / erg cm^{-2} s^{-1}
    '''


    energy_MeV = energy*base_energy_MeV

    sed_E2_erg_per_cm2_per_s = 1


    return energy_MeV, sed_E2_erg_per_cm2_per_s
