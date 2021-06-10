import numpy as np
import spectral_energy_distribution_units as sed
from .. analysis import spectral_energy_distribution as sed_styles

COLOR = "darkorange"
LABEL = "Fermi-LAT"


def differential_sensitivity():

    raw = np.array([
        [4.19738e+1, 1.89701e-11],
        [7.48560e+1, 1.55628e-11],
        [1.33496e+2, 1.30110e-11],
        [2.35880e+2, 1.16211e-11],
        [4.20677e+2, 9.35538e-12],
        [7.36614e+2, 7.39032e-12],
        [1.33813e+3, 5.56883e-12],
        [2.34321e+3, 4.19617e-12],
        [4.25618e+3, 3.54148e-12],
        [7.38463e+3, 2.77129e-12],
        [1.31686e+4, 2.47527e-12],
        [2.36959e+4, 2.57162e-12],
        [4.18593e+4, 2.93638e-12],
        [7.46174e+4, 3.82697e-12],
        [1.34232e+5, 5.22895e-12],
        [2.37020e+5, 9.22013e-12],
        [4.26270e+5, 1.64122e-11],
        [7.52640e+5, 3.09178e-11],
    ])

    fermi_lat_energy_MeV = raw[:, 0]
    fermi_lat_sed_erg_per_cm2_per_s = raw[:, 1]

    ss = sed_styles
    assert ss.FERMI_SED_STYLE["x_energy_in_eV"] == 1e6
    assert ss.FERMI_SED_STYLE["y_inverse_energy_in_eV"] == ss.one_erg_in_eV
    assert ss.FERMI_SED_STYLE["y_inverse_area_in_m2"] == 1e-4
    assert ss.FERMI_SED_STYLE["y_inverse_time_in_s"] == 1.0
    assert ss.FERMI_SED_STYLE["y_scale_energy_in_eV"] == ss.one_erg_in_eV
    assert ss.FERMI_SED_STYLE["y_scale_energy_power"] == 2.0

    energy, dfdE = sed.convert_units_with_style(
        x=fermi_lat_energy_MeV,
        y=fermi_lat_sed_erg_per_cm2_per_s,
        input_style=sed_styles.FERMI_SED_STYLE,
        target_style=sed_styles.PLENOIRF_SED_STYLE,
    )

    return {
        "energy": {
            "values": energy,
            "unit_tex": "GeV",
            "unit": "GeV",
        },
        "differential_flux": {
            "values": dfdE,
            "unit_tex": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
            "unit": "per_m2_per_s_per_GeV",
        },
        "reference": {
            "url": "https://www.slac.stanford.edu/exp/glast/groups/canda/lat_Performance.htm",
            "figure_name": "differential_flux_sensitivity_p8r3_source_v2_all_10yr_zmax100_n10.0_e1.50_ts25.png",
            "l": 0.0,
            "b": 0.0,
            "observation_time": 10 * 365 * 24 * 3600
        }
    }


def integral_sensitivity():
    """
    https://www.slac.stanford.edu/exp/glast/groups/canda/lat_Performance_files/
        broadband_flux_sensitivity_p8r3_source_v2_10yr_zmax100_n10.0_e1.50_ts25_000_090.txt
    """
    # broadband_flux_sensitivity_p8r2_source_v6_all_10yr_zmax100_n10.0_e1.50_ts25_000_090
    # (l, b) = (0, 90)
    # Column 0: Energy / MeV
    # Column 1: E^2 times Broadband Flux Sensitivity / erg cm^{-2} s^{-1}

    data = np.array(
        [
            [31.623, 6.9842e-12],
            [35.112, 5.9694e-12],
            [38.986, 5.1021e-12],
            [43.288, 4.3608e-12],
            [48.064, 3.7272e-12],
            [53.367, 3.1857e-12],
            [59.255, 2.7229e-12],
            [65.793, 2.3272e-12],
            [73.053, 1.9891e-12],
            [81.113, 1.7062e-12],
            [90.063, 1.4784e-12],
            [100, 1.2909e-12],
            [111.03, 1.1385e-12],
            [123.28, 1.0094e-12],
            [136.89, 8.9963e-13],
            [151.99, 8.0689e-13],
            [168.76, 7.2671e-13],
            [187.38, 6.5663e-13],
            [208.06, 5.976e-13],
            [231.01, 5.4388e-13],
            [256.5, 4.9804e-13],
            [284.8, 4.5803e-13],
            [316.23, 4.2124e-13],
            [351.12, 3.8926e-13],
            [389.86, 3.6176e-13],
            [432.88, 3.3621e-13],
            [480.64, 3.1245e-13],
            [533.67, 2.9342e-13],
            [592.55, 2.7556e-13],
            [657.93, 2.5879e-13],
            [730.53, 2.4386e-13],
            [811.13, 2.3143e-13],
            [900.63, 2.1963e-13],
            [1000, 2.0843e-13],
            [1110.3, 1.9901e-13],
            [1232.8, 1.9085e-13],
            [1368.9, 1.8303e-13],
            [1519.9, 1.7552e-13],
            [1687.6, 1.6985e-13],
            [1873.8, 1.646e-13],
            [2080.6, 1.5952e-13],
            [2310.1, 1.5494e-13],
            [2565, 1.5173e-13],
            [2848, 1.4858e-13],
            [3162.3, 1.4551e-13],
            [3511.2, 1.4345e-13],
            [3898.6, 1.4196e-13],
            [4328.8, 1.4048e-13],
            [4806.4, 1.3914e-13],
            [5336.7, 1.3914e-13],
            [5925.5, 1.3914e-13],
            [6579.3, 1.3923e-13],
            [7305.3, 1.4069e-13],
            [8111.3, 1.4217e-13],
            [9006.3, 1.4367e-13],
            [10000, 1.4518e-13],
            [11103, 1.4714e-13],
            [12328, 1.5025e-13],
            [13689, 1.5343e-13],
            [15199, 1.5687e-13],
            [16876, 1.6187e-13],
            [18738, 1.6703e-13],
            [20806, 1.7236e-13],
            [23101, 1.7853e-13],
            [25650, 1.8616e-13],
            [28480, 1.9412e-13],
            [31623, 2.0242e-13],
            [35112, 2.1251e-13],
            [38986, 2.2393e-13],
            [43288, 2.3596e-13],
            [48064, 2.4884e-13],
            [53367, 2.6496e-13],
            [59255, 2.8214e-13],
            [65793, 3.0042e-13],
            [73053, 3.224e-13],
            [81113, 3.4691e-13],
            [90063, 3.7328e-13],
            [1e05, 4.0471e-13],
            [1.1103e05, 4.4005e-13],
            [1.2328e05, 4.7872e-13],
            [1.3689e05, 5.2601e-13],
            [1.5199e05, 5.7797e-13],
            [1.6876e05, 6.3988e-13],
            [1.8738e05, 7.1048e-13],
            [2.0806e05, 7.9488e-13],
            [2.3101e05, 8.9208e-13],
            [2.565e05, 1.0115e-12],
            [2.848e05, 1.1537e-12],
            [3.1623e05, 1.3267e-12],
            [3.5112e05, 1.5408e-12],
            [3.8986e05, 1.8027e-12],
            [4.3288e05, 2.1091e-12],
            [4.8064e05, 2.4677e-12],
            [5.3367e05, 2.8871e-12],
            [5.9255e05, 3.3779e-12],
            [6.5793e05, 3.9521e-12],
            [7.3053e05, 4.6239e-12],
            [8.1113e05, 5.41e-12],
            [9.0063e05, 6.3296e-12],
            [1e06, 7.4055e-12],
        ]
    )

    """
    # 1 erg = 0.62415091 TeV
    ERG_OVER_GEV = 624.15091

    base_energy_x_in_GeV = 1e-3  # in MeV
    base_energy_y_in_GeV = ERG_OVER_GEV  # in ergs
    power_slope = 2.0
    base_area_in_m2 = 1e-4  # in cm2
    base_time_s = 1.0

    # convert energy axis to energy used on y axis
    data[:, 0] = data[:, 0]*base_energy_x_in_GeV
    data[:, 0] = data[:, 0]/base_energy_y_in_GeV

    # convert to normal flux, then convert to cm^2, convert to per TeV
    data[:, 1] = data[:, 1]/(data[:, 0]**power_slope)
    data[:, 1] = data[:, 1]/base_area_in_m2
    data[:, 1] = data[:, 1]/base_energy_y_in_GeV
    data[:, 1] = data[:, 1]/base_time_s

    # convert energy axis to GeV scale
    data[:, 0] = data[:, 0]*base_energy_y_in_GeV
    """

    fermi_lat_energy_MeV = data[:, 0]
    fermi_lat_sed_erg_per_cm2_pers = data[:, 1]
    ss = sed_styles
    assert ss.FERMI_SED_STYLE["x_energy_in_eV"] == 1e6
    assert ss.FERMI_SED_STYLE["y_inverse_energy_in_eV"] == ss.one_erg_in_eV
    assert ss.FERMI_SED_STYLE["y_inverse_area_in_m2"] == 1e-4
    assert ss.FERMI_SED_STYLE["y_inverse_time_in_s"] == 1.0
    assert ss.FERMI_SED_STYLE["y_scale_energy_in_eV"] == ss.one_erg_in_eV
    assert ss.FERMI_SED_STYLE["y_scale_energy_power"] == 2.0

    energy, dfdE = sed.convert_units_with_style(
        x=fermi_lat_energy_MeV,
        y=fermi_lat_sed_erg_per_cm2_pers,
        input_style=sed_styles.FERMI_SED_STYLE,
        target_style=sed_styles.PLENOIRF_SED_STYLE,
    )

    return {
        "energy": {"values": energy, "unit_tex": "GeV", "unit": "GeV",},
        "differential_flux": {
            "values": dfdE,
            "unit_tex": "m$^{-2}$ s$^{-1}$ GeV$^{-1}$",
            "unit": "per_m2_per_s_per_GeV",
        },
        "title": "Fermi-LAT broadband flux sensitivity 10years, "
        "P8R2, l=0deg, b=90deg.",
    }


def energy_resolution():
    raw = np.array([
        [2.80634e+1, 2.99275e-1],
        [3.13220e+1, 2.76430e-1],
        [5.46466e+1, 2.27752e-1],
        [9.73430e+1, 1.96304e-1],
        [1.73388e+2, 1.65978e-1],
        [3.08760e+2, 1.40522e-1],
        [5.49731e+2, 1.18062e-1],
        [9.67900e+2, 1.01219e-1],
        [1.76017e+3, 8.88718e-2],
        [3.09803e+3, 7.83964e-2],
        [5.57227e+3, 6.86705e-2],
        [9.69848e+3, 6.41874e-2],
        [1.72476e+4, 6.30755e-2],
        [3.06699e+4, 6.38362e-2],
        [5.45318e+4, 6.64697e-2],
        [9.48561e+4, 7.28480e-2],
        [1.72315e+5, 8.03507e-2],
        [3.02978e+5, 8.86019e-2],
        [5.32655e+5, 9.91003e-2],
        [9.56327e+5, 1.22708e-1],
        [1.71378e+6, 1.80772e-1],
        [2.97252e+6, 2.39959e-1],
    ])

    reco_energy_MeV = raw[:, 0]
    resolution_68 = raw[:, 1]

    res = {
        "reconstructed_energy": {
            "values": 1e-3 * reco_energy_MeV,
            "label": "reco. energy",
            "unit_tex": "GeV",
            "unit": "GeV",
        },
        "energy_resolution_68": {
            "values": resolution_68,
            "label": "energy (reco. - true) / true (68% containment)",
            "unit_tex": "1",
            "unit": "1",
        },
        "reference": {
            "url": (
                "https://www.slac.stanford.edu/exp/glast/"
                "groups/canda/lat_Performance.htm"
            ),
            "date": "2021-06-10",
            "production": "P8R3_SOURCE_V2",
            "figure_name": "gEdispAve68Energy_P8R3_SOURCE_V2fb_10MeV.png",
            "comment": (
                "Acceptance weighted (acc. weighted) energy resolution "
                "(i.e. 68% containment half width of the reconstructed "
                "incoming photon energy) as a function of energy; "
                "Black total curve. Not red front curve, not blue back curve."
            )
        }
    }
    return res
