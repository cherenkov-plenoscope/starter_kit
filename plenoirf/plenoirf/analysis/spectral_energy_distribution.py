import numpy as np


one_eV_in_J = 1.602176634e-19
one_erg_in_J = 1e-7
one_erg_in_eV = one_erg_in_J/one_eV_in_J

PLENOIRF_SED_STYLE = {
    "x_energy_in_eV": 1e9,
    "y_inverse_energy_in_eV": 1e9,
    "y_inverse_area_in_m2": 1.0,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": 1e9,
    "y_scale_energy_power": 0.0,
    "x_label": "Energy",
    "x_unit": "GeV",
    "y_label": "(differential flux)",
    "y_unit": "m$^{-2}$ s$^{-1}$ (GeV)$^{-1}$",
}

FERMI_SED_STYLE = {
    "x_energy_in_eV": 1e6,
    "y_inverse_energy_in_eV": one_erg_in_eV,
    "y_inverse_area_in_m2": 1e-4,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": one_erg_in_eV,
    "y_scale_energy_power": 2.0,
    "x_label": "Energy",
    "x_unit": "MeV",
    "y_label": "Energy$^{2}$ $\\times$ (differential flux)",
    "y_unit": "erg$^{2}$ (cm)$^{-2}$ s$^{-1}$ erg$^{-1}$",
}

SCIENCE_SED_STYLE = {
    "x_energy_in_eV": 1e6,
    "y_inverse_energy_in_eV": 1e6,
    "y_inverse_area_in_m2": 1e-4,
    "y_inverse_time_in_s": 1.0,
    "y_scale_energy_in_eV": 1e6,
    "y_scale_energy_power": 2.0,
    "x_label": "Energy",
    "x_unit": "MeV",
    "y_label": "Energy$^{2}$ $\\times$ (differential flux)",
    "y_unit": "(MeV)$^{2}$ (cm)$^{-2}$ s$^{-1}$ (MeV)$^{-1}$",
}

def convert_units_style(
    x,
    y,
    in_style,
    out_style,
):
    inp = in_style
    out = out_style
    return convert_units(
        x=x,
        y=y,
        x_energy_in_eV=inp['x_energy_in_eV'],
        y_inverse_energy_in_eV=inp['y_inverse_energy_in_eV'],
        y_inverse_area_in_m2=inp['y_inverse_area_in_m2'],
        y_inverse_time_in_s=inp['y_inverse_time_in_s'],
        y_scale_energy_in_eV=inp['y_scale_energy_in_eV'],
        y_scale_energy_power=inp['y_scale_energy_power'],

        target_x_energy_in_eV=out['x_energy_in_eV'],
        target_y_inverse_energy_in_eV=out['y_inverse_energy_in_eV'],
        target_y_inverse_area_in_m2=out['y_inverse_area_in_m2'],
        target_y_inverse_time_in_s=out['y_inverse_time_in_s'],
        target_y_scale_energy_in_eV=out['y_scale_energy_in_eV'],
        target_y_scale_energy_power=out['y_scale_energy_power'],
    )

def convert_units(
    x,
    y,

    x_energy_in_eV,
    y_inverse_energy_in_eV,
    y_inverse_area_in_m2,
    y_inverse_time_in_s,
    y_scale_energy_in_eV,
    y_scale_energy_power,

    target_x_energy_in_eV,
    target_y_inverse_energy_in_eV,
    target_y_inverse_area_in_m2,
    target_y_inverse_time_in_s,
    target_y_scale_energy_in_eV,
    target_y_scale_energy_power,
):
    '''
    typical SED-units:

    Fermi-LAT:
        x: E / MeV
        y: E$^{gamma}$ dFdE / erg$^{-1}$ cm$^{-2}$ s$^{-1}$ erg$^{gamma}$
        gamma = 2

    Science-Magazine:
        x: E / MeV
        y: E$^{gamma}$ dFdE / MeV$^{-1}$ cm$^{-2}$ s$^{-1}$ MeV$^{gamma}$
        gamma = 2

    E is energy, dFdE is differential flux explicit dF/dE.

    '''
    assert len(x) == len(y)

    # unscale power law from input y
    _x_in_units_of_y_scale_energy = x*(x_energy_in_eV/y_scale_energy_in_eV)
    _y = y / (_x_in_units_of_y_scale_energy**y_scale_energy_power)

    # convert energy axis to SI base units
    x_eV = x*x_energy_in_eV

    # convert diff. flux axis to SI base units
    y_per_m2_per_s_per_eV = _y / (
        y_inverse_energy_in_eV *
        y_inverse_area_in_m2 *
        y_inverse_time_in_s
    )

    # convert energy axis to target units
    x_target = x_eV/target_x_energy_in_eV

    # convert diff. flux axis to target units
    _y_target = y_per_m2_per_s_per_eV * (
        target_y_inverse_energy_in_eV *
        target_y_inverse_area_in_m2 *
        target_y_inverse_time_in_s
    )

    _x_in_units_of_target_y_scale_energy = x_eV/target_y_scale_energy_in_eV

    y_target = _y_target*(
        _x_in_units_of_target_y_scale_energy**target_y_scale_energy_power
    )

    return x_target, y_target
