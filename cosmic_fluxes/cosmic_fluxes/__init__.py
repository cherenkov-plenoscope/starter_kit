import os
import numpy as np
from . import pulsars
from . import utils
from astropy.io import fits as astropy_io_fits


def extrapolate_with_power_law(
    original,
    stop_energy_GeV,
    spectral_index,
    num_points=10,
):
    assert original["energy"]["unit"] == "GeV"
    E_start = original["energy"]["values"][-1]
    dFdE_start = original["differential_flux"]["values"][-1]
    assert num_points > 1
    assert stop_energy_GeV > E_start
    extra_E = np.geomspace(E_start, stop_energy_GeV, num_points + 1)
    extra_E = extra_E[1:]  # exclude pivot energy.
    extra_dFdE = _power_law(
        energy=extra_E,
        flux_density=dFdE_start,
        spectral_index=spectral_index,
        pivot_energy=E_start,
    )
    original["energy"]["values"] = np.hstack(
        (original["energy"]["values"], extra_E)
    )
    original["differential_flux"]["values"] = np.hstack(
        (original["differential_flux"]["values"], extra_dFdE)
    )
    info = (
        "EXTRAPOLATED(FROM:{:e}GEV, TO:{:e}GEV, SPECTRAL_INDEX:{:e}), ".format(
            E_start, stop_energy_GeV, spectral_index
        )
    )
    original["title"] = info + original["title"]
    return original


def proton_aguilar2015precision():
    path = os.path.join(utils.get_resources_dir(), "proton_flux_ams02.csv")
    proton_flux = np.genfromtxt(path)
    proton_flux[:, 0] *= 1  # in GeV
    proton_flux[:, 1] /= proton_flux[:, 0] ** 2.7
    return {
        "energy": {
            "values": proton_flux[:, 0].tolist(),
            "unit": "GeV",
        },
        "differential_flux": {
            "values": proton_flux[:, 1].tolist(),
            "unit": "m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$",
        },
        "title": "Precision measurement of the proton flux in primary cosmic "
        "rays from rigidity 1 GV to 1.8 TV with the Alpha Magnetic "
        "Spectrometer on the International Space Station",
        "citekey": "aguilar2015precision",
    }


def electron_positron_aguilar2014precision():
    res_path = os.path.join(
        utils.get_resources_dir(), "electron_positron_flux_ams02.csv"
    )
    electron_flux = np.genfromtxt(res_path)
    electron_flux[:, 0] *= 1  # in GeV
    electron_flux[:, 1] /= electron_flux[:, 0] ** 3.0
    return {
        "energy": {
            "values": electron_flux[:, 0].tolist(),
            "unit": "GeV",
        },
        "differential_flux": {
            "values": electron_flux[:, 1].tolist(),
            "unit": "m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$",
        },
        "title": "Precision measurement of the (e++ e-) flux in primary "
        "cosmic rays from 0.5 GeV to 1 TeV with the Alpha Magnetic "
        "Spectrometer on the International Space Station",
        "citekey": "aguilar2014precision",
    }


def helium_patrignani2017helium():
    res_path = os.path.join(utils.get_resources_dir(), "helium_flux.csv")
    helium_flux = np.genfromtxt(res_path, delimiter=",")
    return {
        "energy": {
            "values": helium_flux[:, 0].tolist(),
            "unit": "GeV",
        },
        "differential_flux": {
            "values": helium_flux[:, 1].tolist(),
            "unit": "m$^{-2}$ s$^{-1}$ sr$^{-1}$ GeV$^{-1}$",
        },
        "title": "Cosmic-ray primary spectra "
        "Revised October 2017 by J.J. Beatty (Ohio State Univ.), J. Matthews "
        "(Louisiana State Univ.), and S.P. Wakely (Univ. of Chicago)",
        "citekey": "patrignani2017helium",
    }


def read_crab_nebula_flux_from_resources():
    res_path = os.path.join(
        utils.get_resources_dir(),
        "crab_nebula_spectral_energy_distribution_fermi_lat_and_magic_ct.csv",
    )

    # # E / GeV
    # E^2*dF/DE / TeV cm^-2 s^-1
    _raw = np.genfromtxt(res_path, delimiter=" ")

    _energy_GeV = _raw[:, 0]
    _energy_TeV = 1e-3 * _energy_GeV.copy()

    _differential_flux_TeV_per_cm2_per_s = _raw[:, 1].copy()

    _differential_flux_TeV_per_m2_per_s = (
        _differential_flux_TeV_per_cm2_per_s * 1e4
    )

    _differential_flux_per_TeV_per_m2_per_s = (
        _differential_flux_TeV_per_m2_per_s / _energy_TeV**2
    )

    _differential_flux_per_GeV_per_m2_per_s = 1e-3 * (
        _differential_flux_per_TeV_per_m2_per_s
    )

    return {
        "energy": {
            "values": _energy_GeV.tolist(),
            "unit": "GeV",
        },
        "differential_flux": {
            "values": _differential_flux_per_GeV_per_m2_per_s.tolist(),
            "unit": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
        },
        "title": "Crab-nebula, Fermi-LAT and MAGIC. ",
        "citekey": "aleksic2015measurement",
    }


"""
According to:

@article{acero2015fermi3fgl,
  Author = {
    Acero, F and Ackermann, M and Ajello, M and Albert, A and
    Atwood, WB and Axelsson, Magnus and Baldini, L and Ballet, J and
    Barbiellini, G and Bastieri, D and others},
  Journal = {The Astrophysical Journal Supplement Series},
  Number = {2},
  Pages = {23},
  Publisher = {IOP Publishing},
  Title = {Fermi large area telescope third source catalog},
  Volume = {218},
  Year = {2015}}
"""


GAMMA_SOURCES_DTYPES = {
    "source_name": str,
    "right_ascension_j2000_deg": float,
    "declination_j2000_deg": float,
    "galactic_longitude_deg": float,
    "galactic_latitude_deg": float,
    "spectrum_type": str,
    "spectral_index": float,
    "beta": float,
    "exp_index": float,
    "pivot_energy_GeV": float,
    "cutoff_energy_GeV": float,
    "flux1000_per_m2_per_s": float,
    "flux_density_per_m2_per_GeV_per_s": float,
}


def fermi_3fgl_catalog():
    fermi_3fgl_path = os.path.join(
        utils.get_resources_dir(), "fermi_lat_3fgl_gll_psc_v16.fits"
    )
    gamma_sources_raw = []
    fermi_keys = [
        "Source_Name",  # str
        "RAJ2000",  # deg
        "DEJ2000",  # deg
        "GLON",  # deg
        "GLAT",  # deg
        "SpectrumType",  # str
        "Pivot_Energy",  # MeV
        "Spectral_Index",  # 1
        "Flux_Density",  # photons cm^{-2} MeV^{-1} s^{-1}
        "beta",  # 1
        "Cutoff",  # MeV
        "Exp_Index",  # 1
        "Flux1000",  # photons cm^{-2} s^{-1}
    ]

    with astropy_io_fits.open(fermi_3fgl_path) as fits:
        num_sources = fits[1].header["NAXIS2"]
        for source_idx in range(num_sources):
            s = {}
            for fermi_key in fermi_keys:
                s[fermi_key] = fits[1].data[source_idx][fermi_key]
            gamma_sources_raw.append(s)

    tmp_gamma_sources = []
    for i in range(len(gamma_sources_raw)):
        raw = gamma_sources_raw[i]
        so = {}
        so["source_name"] = raw["Source_Name"]
        so["right_ascension_j2000_deg"] = raw["RAJ2000"]
        so["declination_j2000_deg"] = raw["DEJ2000"]
        so["galactic_longitude_deg"] = raw["GLON"]
        so["galactic_latitude_deg"] = raw["GLAT"]
        so["spectrum_type"] = raw["SpectrumType"]
        so["spectral_index"] = -1.0 * raw["Spectral_Index"]
        so["beta"] = -1.0 * raw["beta"]
        so["exp_index"] = raw["Exp_Index"]
        so["pivot_energy_GeV"] = raw["Pivot_Energy"] * 1e-3  # MeV to GeV
        so["cutoff_energy_GeV"] = raw["Cutoff"] * 1e-3  # MeV to GeV
        so["flux1000_per_m2_per_s"] = raw["Flux1000"] * 1e4  # cm^2 to m^2
        Flux_Density_tmp = raw["Flux_Density"] * 1e4  # cm^2 to m^2
        # MeV^{-1} to GeV^{-1}
        so["flux_density_per_m2_per_GeV_per_s"] = Flux_Density_tmp * 1e3
        tmp_gamma_sources.append(so)

    gamma_sources = []
    for i in range(len(tmp_gamma_sources)):
        tmp_source = tmp_gamma_sources[i]
        final_source = {}
        for key in tmp_source:
            final_source[key] = GAMMA_SOURCES_DTYPES[key](tmp_source[key])
        gamma_sources.append(final_source)
    return gamma_sources


def _power_law_super_exp_cutoff(
    energy,
    flux_density,
    spectral_index,
    pivot_energy,
    cutoff_energy,
    exp_index,
):
    expo = (pivot_energy / cutoff_energy) ** exp_index - (
        energy / cutoff_energy
    ) ** exp_index
    return (
        flux_density * (energy / pivot_energy) ** (spectral_index)
    ) * np.exp(expo)


def _power_law_log_parabola(
    energy, flux_density, spectral_index, pivot_energy, beta
):
    expo = +spectral_index + beta * np.log10(energy / pivot_energy)
    return flux_density * (energy / pivot_energy) ** expo


def _power_law(energy, flux_density, spectral_index, pivot_energy):
    return flux_density * (energy / pivot_energy) ** (spectral_index)


def flux_of_fermi_source(fermi_source, energy):
    fs = fermi_source
    if fs["spectrum_type"] == "PowerLaw":
        fluxes = _power_law(
            energy=energy,
            flux_density=fs["flux_density_per_m2_per_GeV_per_s"],
            spectral_index=fs["spectral_index"],
            pivot_energy=fs["pivot_energy_GeV"],
        )
    elif fs["spectrum_type"] == "LogParabola":
        fluxes = _power_law_log_parabola(
            energy=energy,
            flux_density=fs["flux_density_per_m2_per_GeV_per_s"],
            spectral_index=fs["spectral_index"],
            pivot_energy=fs["pivot_energy_GeV"],
            beta=fs["beta"],
        )
    elif (
        fs["spectrum_type"] == "PLExpCutoff"
        or fs["spectrum_type"] == "PLSuperExpCutoff"
    ):
        fluxes = _power_law_super_exp_cutoff(
            energy=energy,
            flux_density=fs["flux_density_per_m2_per_GeV_per_s"],
            spectral_index=fs["spectral_index"],
            pivot_energy=fs["pivot_energy_GeV"],
            cutoff_energy=fs["cutoff_energy_GeV"],
            exp_index=fs["exp_index"],
        )
    else:
        raise KeyError(
            "Unknown spectrum_type: {:s}".format(fs["spectrum_type"])
        )
    return fluxes
