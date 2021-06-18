import numpy as np
import spectral_energy_distribution_units as sed
from .. analysis import spectral_energy_distribution as sed_styles


def sensitivity_vs_observation_time(energy_GeV=25.0, instrument="cta_south"):
    if energy_GeV == 25.0:
        cta = np.array([
            [1.54448e+1, 6.15336e-9],
            [2.70159e+1, 3.52307e-9],
            [4.72564e+1, 2.01004e-9],
            [8.26598e+1, 1.16304e-9],
            [1.51192e+2, 7.59478e-10],
            [2.52859e+2, 4.64194e-10],
            [4.42192e+2, 3.32821e-10],
            [7.73277e+2, 2.42859e-10],
            [1.35223e+3, 1.80991e-10],
            [2.36467e+3, 1.33000e-10],
            [4.13503e+3, 1.00170e-10],
            [7.23083e+3, 7.54429e-11],
            [1.26442e+4, 5.74223e-11],
            [2.21098e+4, 4.47950e-11],
            [3.86615e+4, 3.48218e-11],
            [6.76010e+4, 2.82353e-11],
            [1.18198e+5, 2.37972e-11],
            [2.06657e+5, 2.07013e-11],
            [3.61304e+5, 1.87182e-11],
            [6.31644e+5, 1.77787e-11],
            [1.10428e+6, 1.65922e-11],
            [1.93045e+6, 1.64963e-11],
            [3.37468e+6, 1.65167e-11],
            [5.89939e+6, 1.65371e-11],
            [1.03133e+7, 1.60419e-11],
            [1.80291e+7, 1.59492e-11],
            [3.15173e+7, 1.59689e-11],
            [5.50965e+7, 1.59886e-11],
            [9.63193e+7, 1.55098e-11],
            [1.68375e+8, 1.58600e-11],
            [2.94335e+8, 1.62180e-11],
            [5.14544e+8, 1.60677e-11],
            [8.99491e+8, 1.60876e-11],
            [1.57243e+9, 1.61074e-11],
            [2.74882e+9, 1.61273e-11],
            [4.80530e+9, 1.61472e-11],
            [7.03254e+9, 1.61608e-11],
        ])

        fermi = np.array([
            [1.38090e+1, 8.70556e-5],
            [2.41550e+1, 4.90454e-5],
            [4.22516e+1, 2.82409e-5],
            [7.39021e+1, 1.70497e-5],
            [1.29273e+2, 9.51841e-6],
            [2.26128e+2, 5.35790e-6],
            [3.95531e+2, 3.15127e-6],
            [6.91860e+2, 1.80748e-6],
            [1.21024e+3, 1.00265e-6],
            [2.11687e+3, 5.94580e-7],
            [3.62533e+3, 3.38095e-7],
            [6.42264e+3, 1.87740e-7],
            [1.13295e+4, 1.11653e-7],
            [1.98172e+4, 6.48271e-8],
            [3.57111e+4, 4.11470e-8],
            [6.15767e+4, 1.78582e-8],
            [1.07062e+5, 1.36805e-8],
            [1.90551e+5, 7.89615e-9],
            [3.21777e+5, 4.03152e-9],
            [5.67650e+5, 2.24790e-9],
            [9.72135e+5, 1.29960e-9],
            [1.73689e+6, 7.16254e-10],
            [3.03804e+6, 4.25385e-10],
            [5.31425e+6, 2.38275e-10],
            [9.29559e+6, 1.37536e-10],
            [1.62600e+7, 7.80828e-11],
            [2.85500e+7, 4.95109e-11],
            [4.83210e+7, 2.90515e-11],
            [1.74532e+8, 7.88401e-12],
            [3.55583e+8, 3.88780e-12],
            [5.96381e+8, 2.46769e-12],
            [1.05790e+9, 1.44023e-12],
            [1.83739e+9, 7.12110e-13],
            [3.21409e+9, 3.92323e-13],
            [7.82200e+9, 1.89423e-13],
        ])
    else:
        assert False, "No such energy_GeV"

    style = {
        "x_energy_in_eV": 1e9,
        "y_inverse_energy_in_eV": sed_styles.one_erg_in_eV,
        "y_inverse_area_in_m2": 1e-4,
        "y_inverse_time_in_s": 1.0,
        "y_scale_energy_in_eV": sed_styles.one_erg_in_eV,
        "y_scale_energy_power": 2.0,
    }

    PIVOT_ENERGY_GEV = 25.0

    if instrument == "Fermi-LAT":
        _energy_GeV = PIVOT_ENERGY_GEV * np.ones(fermi.shape[0])
        _sens_E2_rg_per_cm2_per_s = fermi[:, 1]
        obstime_s = fermi[:, 0]
    elif instrument == "CTA-South":
        _energy_GeV = PIVOT_ENERGY_GEV * np.ones(cta.shape[0])
        _sens_E2_rg_per_cm2_per_s = cta[:, 1]
        obstime_s = cta[:, 0]

    energy, dfdE = sed.convert_units_with_style(
        x=_energy_GeV,
        y=_sens_E2_rg_per_cm2_per_s,
        input_style=style,
        target_style=sed_styles.PLENOIRF_SED_STYLE,
    )

    assert energy[0] == PIVOT_ENERGY_GEV

    return {
        "pivot_energy": {
            "values": PIVOT_ENERGY_GEV,
            "unit_tex": "GeV",
            "unit": "GeV",
        },
        "differential_flux": {
            "values": dfdE,
            "unit_tex": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
            "unit": "per_m2_per_s_per_GeV",
        },
        "observation_time": {
            "values": obstime_s,
            "unit_tex": "s",
            "unit": "s",
        },
        "reference": {
            "key": "funk2013comparison",
            "title": "Comparison of Fermi-LAT and CTA in the region between 10--100 GeV",
            "author": "Funk, Stefan and Hinton, JA and CTA Consortium and others",
            "journal": "Astroparticle Physics",
            "volume": "43",
            "pages": "348--355",
            "year": "2013",
            "publisher": "Elsevier",
        }
    }
