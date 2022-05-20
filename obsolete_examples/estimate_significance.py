import numpy as np
import os
import json
import acp_instrument_sensitivity_function as isf


def cone_solid_angle(cone_radial_opening_angle):
    cap_hight = (1.0 - np.cos(cone_radial_opening_angle))
    return 2.0*np.pi*cap_hight

input_dir = "analysis_results"

with open(os.path.join(input_dir, "irf_onaxis_proton.json"), "rt") as f:
    acceptance_protons = json.loads(f.read())

with open(os.path.join(input_dir, "irf_onaxis_electron.json"), "rt") as f:
    acceptance_electrons = json.loads(f.read())

with open(os.path.join(input_dir, "irf_onaxis_gamma.json"), "rt") as f:
    acceptance_gammas = json.loads(f.read())

sat_key = "solid_angle_thrown"
assert acceptance_protons[sat_key] == acceptance_electrons[sat_key]
assert acceptance_protons[sat_key] == acceptance_gammas[sat_key]
solid_angle_thrown = acceptance_protons[sat_key]

ebe_key = "energy_bin_edges"
assert acceptance_protons[ebe_key] == acceptance_electrons[ebe_key]
assert acceptance_protons[ebe_key] == acceptance_gammas[ebe_key]
energy_bin_edges = acceptance_electrons["energy_bin_edges"]
energy_bin_start = energy_bin_edges[:-1]

on_region_radial_angle = np.deg2rad(.6)
on_region_solid_angle = cone_solid_angle(on_region_radial_angle)
solid_angle_ratio_on_region = on_region_solid_angle/solid_angle_thrown
on_region_gamma_containment = 0.68

on_over_off_exposure_ratio = 1/5

gamma_seperations = {"past_trigger": "trigger", "past_cuts": "g/h"}
below_cutoff_fractions = [0.1, 0.05, 0.0]
systematic_uncertainties = [0.001, 0.01, 0.03]

ams02_protons = isf.differential_flux_proton()
ams02_electrons = isf.differential_flux_electron_positron()
fermilat_gamma_ray_sources = isf.gamma_ray_sources()

geomagnetic_cutoff_energy = 10.

number_energy_bins = 30
energy_start = 2.5e-1
energy_stop = 1e3
hr_energy_bin_edges = np.geomspace(
    energy_start,
    energy_stop,
    number_energy_bins*10 + 1)
hr_energy_bin_start = hr_energy_bin_edges[:-1]
hr_energy_bin_range = (
    hr_energy_bin_edges[1:] -
    hr_energy_bin_edges[:-1])

source_name = "3FGL J2254.0+1608"
for gamma_ray_source in fermilat_gamma_ray_sources:
    if gamma_ray_source["name"]["value"] == source_name:
        break

table_column_width = 12
table_num_columns = 12
table_format = "{:<" + "{:d}".format(table_column_width) + "} "
table_format = table_num_columns*table_format

print(table_format.format(
    "below",
    "separation",
    "systematic",
    "significance",
    "",
    "trigger",
    "trigger",
    "trigger",
    "Del_B_on",
    "sys.",
    "stat.",
    "time to"))

print(table_format.format(
    "cutoff",
    "method",
    "uncertainty",
    "after",
    "",
    "rate",
    "rate",
    "rate",
    "",
    "Del_B_on",
    "Del_B_on",
    "stat."))

print(table_format.format(
    "fraction",
    "",
    "",
    "1s",
    "10s",
    "gamma",
    "e+/e-",
    "proton",
    "after 1s",
    "",
    "after 1s",
    "limit",
    ))

print(table_format.format(
    "%",
    "",
    "%",
    "",
    "",
    "s^{-1}",
    "s^{-1}",
    "s^{-1}",
    "",
    "",
    "",
    "s",
    ))

print("-"*table_column_width*table_num_columns)

for below_cutoff_fraction in below_cutoff_fractions:
    p_shower_flux = {}
    p_shower_flux["energy"] = np.array(ams02_protons["energy"]["values"])
    p_below_cutoff = p_shower_flux["energy"] < geomagnetic_cutoff_energy
    p_shower_flux["differential_flux"] = np.array(
        ams02_protons["differential_flux"]["values"])
    p_shower_flux["differential_flux"][p_below_cutoff] *= below_cutoff_fraction

    e_shower_flux = {}
    e_shower_flux["energy"] = np.array(ams02_electrons["energy"]["values"])
    e_below_cutoff = e_shower_flux["energy"] < geomagnetic_cutoff_energy
    e_shower_flux["differential_flux"] = np.array(
        ams02_electrons["differential_flux"]["values"])
    e_shower_flux["differential_flux"][e_below_cutoff] *= below_cutoff_fraction

    p_diffuse_dFdE = np.interp(
        x=hr_energy_bin_start,
        xp=p_shower_flux['energy'],
        fp=p_shower_flux['differential_flux'])
    e_diffuse_dFdE = np.interp(
        x=hr_energy_bin_start,
        xp=e_shower_flux['energy'],
        fp=e_shower_flux['differential_flux'])
    g_point_dFdE = isf.differential_flux_gamma_ray_source(
        energy=hr_energy_bin_start,
        source=gamma_ray_source)["differential_flux"]["values"]

    for gamma_seperation in gamma_seperations:
        _p_area = np.array(acceptance_protons["area_"+gamma_seperation])
        _e_area = np.array(acceptance_electrons["area_"+gamma_seperation])
        _g_area = np.array(acceptance_gammas["area_"+gamma_seperation])

        g_area = np.interp(
                x=hr_energy_bin_start,
                xp=energy_bin_start,
                fp=_g_area)
        p_area = np.interp(
                x=hr_energy_bin_start,
                xp=energy_bin_start,
                fp=_p_area)
        e_area = np.interp(
                x=hr_energy_bin_start,
                xp=energy_bin_start,
                fp=_e_area)
        p_acceptance = p_area*solid_angle_thrown
        e_acceptance = e_area*solid_angle_thrown

        for systematic_uncertainty in systematic_uncertainties:
            g_dTdE = g_point_dFdE * g_area * on_region_gamma_containment
            e_dTdE = e_diffuse_dFdE * e_acceptance * solid_angle_ratio_on_region
            p_dTdE = p_diffuse_dFdE * p_acceptance * solid_angle_ratio_on_region

            g_T = np.sum(g_dTdE*hr_energy_bin_range)
            e_T = np.sum(e_dTdE*hr_energy_bin_range)
            p_T = np.sum(p_dTdE*hr_energy_bin_range)

            S_on = g_T
            B_on = e_T + p_T
            B_off = B_on / on_over_off_exposure_ratio
            stat_Delta_B_off = np.sqrt(B_off)
            sys_Delta_B_off = systematic_uncertainty*B_off
            Delta_B_off = np.hypot(stat_Delta_B_off, sys_Delta_B_off)
            rel_Delta_B = Delta_B_off/B_off
            Delta_B_on = rel_Delta_B * B_on

            s = S_on / Delta_B_on

            S_on_10 = g_T*10
            B_on_10 = (e_T + p_T)*10
            B_off_10 = B_on_10 / on_over_off_exposure_ratio
            stat_Delta_B_off_10 = np.sqrt(B_off_10)
            sys_Delta_B_off_10 = systematic_uncertainty*B_off_10
            Delta_B_off_10 = np.hypot(stat_Delta_B_off_10, sys_Delta_B_off_10)
            rel_Delta_B_10 = Delta_B_off_10/B_off_10
            Delta_B_on_10 = rel_Delta_B_10 * B_on_10

            s_10 = S_on_10 / Delta_B_on_10

            B_counts_to_stat_limit = 1/(systematic_uncertainty**2)
            time_to_stat_limit = B_counts_to_stat_limit/B_off

            # lim = 1/sqrt(Counts)
            # sqrt(Counts)*lim = 1
            # sqrt(Counts) = 1/lim
            # Counts = 1/(lim**2)

            print(table_format.format(
                "{:6.0f}".format(below_cutoff_fraction*100),
                "{:<6s}".format(gamma_seperations[gamma_seperation]),
                "{:6.1f}".format(systematic_uncertainty*100),
                "{:6.1f}".format(s),
                "{:6.1f}".format(s_10),
                "{:6.1f}".format(g_T),
                "{:6.1f}".format(e_T),
                "{:6.1f}".format(p_T),
                "{:6.1f}".format(Delta_B_on),
                "{:6.1f}".format(sys_Delta_B_off*on_over_off_exposure_ratio),
                "{:6.1f}".format(stat_Delta_B_off*on_over_off_exposure_ratio),
                "{:6.1f}".format(time_to_stat_limit),
                ))
