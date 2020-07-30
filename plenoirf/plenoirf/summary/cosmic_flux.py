import numpy as np
import copy
import magnetic_deflection
import cosmic_fluxes
import json
from os.path import join as opj
import os


def _read_raw_cosmic_ray_differential_fluxes(summary_dir):
    cosmic_ray_raw_fluxes = {}
    with open(opj(summary_dir, "proton_flux.json"), "rt") as f:
        cosmic_ray_raw_fluxes["proton"] = json.loads(f.read())
    with open(opj(summary_dir, "helium_flux.json"), "rt") as f:
        cosmic_ray_raw_fluxes["helium"] = json.loads(f.read())
    with open(opj(summary_dir, "electron_positron_flux.json"), "rt") as f:
        cosmic_ray_raw_fluxes["electron"] = json.loads(f.read())
    return cosmic_ray_raw_fluxes


def _rigidity_to_total_energy(rigidity_GV):
    return rigidity_GV*1.0


def _read_airshower_differential_flux(
    summary_dir,
    energy_bin_centers,
    sites,
    geomagnetic_cutoff_fraction,
):
    # read raw
    _raw_cosmic_rays = _read_raw_cosmic_ray_differential_fluxes(summary_dir)

    # interpolate
    cosmic_rays = {}
    for particle_key in _raw_cosmic_rays:
        cosmic_rays[particle_key] = {}
        cosmic_rays[particle_key]['differential_flux'] = np.interp(
            x=energy_bin_centers,
            xp=_raw_cosmic_rays[particle_key]['energy']['values'],
            fp=_raw_cosmic_rays[particle_key]['differential_flux']['values']
        )

    # earth's geomagnetic cutoff
    airshowers = {}
    for site_key in sites:
        airshowers[site_key] = {}
        for particle_key in cosmic_rays:

            cutoff_energy = _rigidity_to_total_energy(
                rigidity_GV=sites[site_key]['geomagnetic_cutoff_rigidity_GV']
            )

            below_cutoff = energy_bin_centers < cutoff_energy
            airshowers[
                site_key][
                particle_key] = copy.deepcopy(cosmic_rays[particle_key])
            airshowers[
                site_key][
                particle_key][
                'differential_flux'][
                below_cutoff] *= geomagnetic_cutoff_fraction

    return airshowers


def read_airshower_differential_flux_zenith_compensated(
    run_dir,
    summary_dir,
    energy_bin_centers,
    sites,
    geomagnetic_cutoff_fraction,
):
    airshower_fluxes = _read_airshower_differential_flux(
        summary_dir=summary_dir,
        energy_bin_centers=energy_bin_centers,
        sites=sites,
        geomagnetic_cutoff_fraction=geomagnetic_cutoff_fraction
    )

    particles = []
    for site_key in sites:
        for particle_key in airshower_fluxes[site_key]:
            particles.append(particle_key)
    particles = list(set(particles))

    deflection_table = magnetic_deflection.read(
        work_dir=os.path.join(run_dir, 'magnetic_deflection'),
    )

    zenith_compensated_airshower_fluxes = {}
    for site_key in sites:
        zenith_compensated_airshower_fluxes[site_key] = {}
        for particle_key in particles:
            zenith_compensated_airshower_fluxes[site_key][particle_key] = {}

            flux = airshower_fluxes[
                site_key][
                particle_key][
                'differential_flux']

            zenith_deg = np.interp(
                x=energy_bin_centers,
                xp=deflection_table[
                    site_key][
                    particle_key][
                    'energy_GeV'],
                fp=deflection_table[
                    site_key][
                    particle_key][
                    'primary_zenith_deg'],
            )

            scaling = np.cos(np.deg2rad(zenith_deg))
            comp_flux = scaling*flux
            zenith_compensated_airshower_fluxes[
                site_key][
                particle_key][
                'differential_flux'] = comp_flux

    return zenith_compensated_airshower_fluxes


def make_gamma_ray_reference_flux(
    summary_dir,
    gamma_ray_reference_source,
    energy_supports_GeV,
):
    _grrs = gamma_ray_reference_source
    if _grrs["type"] == "3fgl":
        with open(os.path.join(summary_dir, "gamma_sources.json"), "rt") as f:
            _gamma_sources = json.loads(f.read())
        for _source in _gamma_sources:
            if _source['source_name'] == _grrs["name_3fgl"]:
                _reference_gamma_source = _source
        gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes.flux_of_fermi_source(
            fermi_source=_reference_gamma_source,
            energy=energy_supports_GeV
        )
        source_name = _grrs["name_3fgl"]

        return gamma_dF_per_m2_per_s_per_GeV, source_name

    elif _grrs["type"] == "generic_power_law":
        _gpl = _grrs["generic_power_law"]
        gamma_dF_per_m2_per_s_per_GeV = cosmic_fluxes._power_law(
            energy=energy_supports_GeV,
            flux_density=_gpl['flux_density_per_m2_per_s_per_GeV'],
            spectral_index=_gpl['spectral_index'],
            pivot_energy=_gpl['pivot_energy_GeV'],
        )
        source_name = ''.join([
            '$F = F_0 \\left( \\frac{E}{E_0}\\right) ^{\\gamma}$, ',
            '$F_0$ = {:1.2f} m$^{-2}$ (GeV)$^{-1}$ s$^{-1}$, '.format(
                _gpl['flux_density_per_m2_per_s_per_GeV']
            ),
            '$E_0$ = {:1.2f} GeV, '.format(_gpl['pivot_energy_GeV']),
            '$\\gamma = {:1.2f}$'.format(_gpl['spectral_index']),
        ])
        return gamma_dF_per_m2_per_s_per_GeV, source_name

    else:
        raise KeyError("'type' must either be '3fgl', or 'generic_power_law'.")
