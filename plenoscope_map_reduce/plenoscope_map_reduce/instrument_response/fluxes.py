import os
import json
import numpy as np
import pkg_resources
import astropy


def read_cosmic_proton_flux_from_resources():
    path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'proton_spec.dat'))
    proton_flux = np.genfromtxt(path)
    proton_flux[:, 0] *= 1  # in GeV
    proton_flux[:, 1] /= proton_flux[:, 0]**2.7
    return {
        "energy": {
            "values": proton_flux[:, 0].tolist(),
            "unit_tex": "GeV",
            "unit": "GeV",
        },
        "differential_flux": {
            "values": proton_flux[:, 1].tolist(),
            "unit_tex": "m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$",
            "unit": "per_m2_per_s_per_sr_per_GeV"
        },
        "title": "Precision measurement of the proton flux in primary cosmic "
        "rays from rigidity 1 GV to 1.8 TV with the Alpha Magnetic "
        "Spectrometer on the International Space Station",
    }


def read_cosmic_electron_positron_flux_from_resources():
    res_path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        os.path.join('resources', 'e_plus_e_minus_spec.dat'))
    electron_flux = np.genfromtxt(res_path)
    electron_flux[:, 0] *= 1  # in GeV
    electron_flux[:, 1] /= electron_flux[:, 0]**3.0
    return {
        "energy": {
            "values": electron_flux[:, 0].tolist(),
            "unit_tex": "GeV",
            "unit": "GeV",
        },
        "differential_flux": {
            "values": electron_flux[:, 1].tolist(),
            "unit_tex": "m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$",
            "unit": "per_m2_per_s_per_sr_per_GeV"
        },
        "title": "Precision measurement of the (e++ e-) flux in primary "
        "cosmic rays from 0.5 GeV to 1 TeV with the Alpha Magnetic "
        "Spectrometer on the International Space Station",
    }


def read_fermi_3rd_galactic_from_resources():
    fermi_3fgl_path = pkg_resources.resource_filename(
        'acp_instrument_sensitivity_function',
        resource_name=os.path.join(
            'resources',
            'FermiLAT_3FGL_gll_psc_v16.fit'))
    gamma_sources_raw = []
    fermi_keys = [
        "Source_Name",  # str
        'RAJ2000',  # deg
        'DEJ2000',  # deg
        'GLON',  # deg
        'GLAT',  # deg
        'SpectrumType',  # str
        'Pivot_Energy',  # MeV
        'Spectral_Index',  # 1
        'Flux_Density',  # photons cm^{-2} MeV^{-1} s^{-1}
        'beta',  # 1
        'Cutoff',  # MeV
        'Exp_Index',  # 1
        'Flux1000',  # photons cm^{-2} s^{-1}
    ]

    with astropy.io.fits.open(fermi_3fgl_path) as fits:
        num_sources = fits[1].header["NAXIS2"]
        for source_idx in range(num_sources):
            s = {}
            for fermi_key in fermi_keys:
                s[fermi_key] = fits[1].data[source_idx][fermi_key]
            gamma_sources_raw.append(s)

    gamma_sources = []
    for i in range(len(gamma_sources_raw)):
        raw = gamma_sources_raw[i]
        so = {}
        so['source_name'] = raw['Source_Name']
        so['right_ascension_j2000_deg'] = raw['RAJ2000']
        so['declination_j2000_deg'] = raw['DEJ2000']
        so['galactic_longitude_deg'] = raw['GLON']
        so['galactic_latitude_deg'] = raw['GLAT']
        so['spectrum_type'] = raw['SpectrumType']
        so['spectral_index'] = -1.0*raw['Spectral_Index']
        so['beta'] = -1.0*raw['beta']
        so['exp_index'] = raw['Exp_Index']
        so['pivot_energy_GeV'] = raw["Pivot_Energy"]*1e-3  # MeV to GeV
        so['cutoff_energy_GeV'] = raw["Cutoff"]*1e-3  # MeV to GeV
        so["flux1000_per_m2_per_s"] = raw["Flux1000"]*1e4  # cm^2 to m^2
        Flux_Density_tmp = raw["Flux_Density"]*1e4  # cm^2 to m^2
        # MeV^{-1} to GeV^{-1}
        so["flux_density_per_m2_per_GeV_per_s"] = Flux_Density_tmp*1e3
        gamma_sources.append(so)
    return gamma_sources


def _power_law_super_exp_cutoff_according_to_3fgl(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    cutoff_energy,
    exp_index
):
    '''
    according to 3FGL cat, but with already negative spectral_index
    '''
    return (flux_density*(energy/pivot_energy)**(spectral_index))*np.exp(
        (pivot_energy/cutoff_energy)**exp_index -
        (energy/cutoff_energy)**exp_index
    )


def _power_law_log_parabola_according_to_3fgl(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    beta
):
    '''
    according to 3fgl cat, but with already negative spectral_index and beta
    '''
    expo = (+spectral_index+beta*np.log10(energy/pivot_energy))
    return flux_density*(energy/pivot_energy)**expo


def _power_law(energy, flux_density, spectral_index, pivot_energy):
    return flux_density*(energy/pivot_energy)**(spectral_index)


def flux_of_fermi_source(fermi_source, energy):
    '''
    according to fermi-lat 3fgl
    '''
    fs = fermi_source
    if fs['spectrum_type'] == 'PowerLaw':
        fluxes = _power_law(
            energy=energy,
            flux_density=fs['flux_density_per_m2_per_GeV_per_s'],
            spectral_index=fs['spectral_index'],
            pivot_energy=fs['pivot_energy_GeV'])
    elif fs['spectrum_type'] == 'LogParabola':
        fluxes = _power_law_log_parabola_according_to_3fgl(
            energy=energy,
            flux_density=fs['flux_density_per_m2_per_GeV_per_s'],
            spectral_index=fs['spectral_index'],
            pivot_energy=fs['pivot_energy_GeV'],
            beta=fs['beta'])
    elif (
        fs['spectrum_type'] == 'PLExpCutoff' or
        fs['spectrum_type'] == 'PLSuperExpCutoff'
    ):
        fluxes = _power_law_super_exp_cutoff_according_to_3fgl(
            energy=energy,
            flux_density=fs['flux_density_per_m2_per_GeV_per_s'],
            spectral_index=fs['spectral_index'],
            pivot_energy=fs['pivot_energy_GeV'],
            cutoff_energy=fs['cutoff_energy_GeV'],
            exp_index=fs['exp_index'])
    else:
        raise KeyError(
            'Unknown spectrum_type: {:s}'.format(fs['spectrum_type']))

    return fluxes
