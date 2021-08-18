#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import cosmic_fluxes
import os
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

onregion_acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fenergy_bin = energy_binning["interpolation"]

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

# cosmic-ray-flux
# ----------------
airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

# gamma-ray-flux of reference source
# ----------------------------------
gamma_source = json_numpy.read(
    os.path.join(
        pa["summary_dir"], "0009_flux_of_gamma_rays", "reference_source.json"
    )
)
gamma_dKdE = gamma_source["differential_flux"]["values"]
gamma_dKdE_au = np.zeros(shape=gamma_dKdE.shape)

comment_differential = "Differential trigger-rate, reconstructed in onregion."
comment_integral = "Integral trigger-rate, reconstructed in onregion."


def integrate_rate_where_known(dRdE, dRdE_au, E_edges):
    unknown = np.isnan(dRdE_au)

    _dRdE = dRdE.copy()
    _dRdE_au = dRdE_au.copy()

    _dRdE[unknown] = 0.0
    _dRdE_au[unknown] = 0.0

    T, T_au = irf.utils.integrate(f=_dRdE, f_au=_dRdE_au, x_edges=E_edges)
    return T, T_au


"""
A / m^{2}
Q / m^{2} sr

R / s^{-1}
dRdE / s^{-1} (GeV)^{-1}

F / s^{-1} m^{-2} (sr)^{-1}
dFdE / s^{-1} m^{-2} (sr)^{-1} (GeV)^{-1}

K / s^{-1} m^{-2}
dKdE / s^{-1} m^{-2} (GeV)^{-1}
"""

for sk in irf_config["config"]["sites"]:
    site_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(site_dir, exist_ok=True)

    # gamma-ray
    # ---------
    site_gamma_dir = os.path.join(site_dir, "gamma")
    os.makedirs(site_gamma_dir, exist_ok=True)

    T = np.zeros(shape=(num_bins_onregion_radius))
    T_au = np.zeros(shape=T.shape)
    dRdE = np.zeros(
        shape=(fenergy_bin["num_bins"], num_bins_onregion_radius)
    )
    dRdE_au = np.zeros(shape=dRdE.shape)
    for oridx in range(num_bins_onregion_radius):
        _A = onregion_acceptance[sk]["gamma"]["point"]["mean"][:, oridx]
        _A_ru = onregion_acceptance[sk]["gamma"]["point"]["relative_uncertainty"][:, oridx]
        _A_au = _A * _A_ru

        A = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A
        )
        A_au = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A_au
        )

        dRdE[:, oridx], dRdE_au[:, oridx] = irf.utils.multiply(
            x=gamma_dKdE, x_au=gamma_dKdE_au, y=A, y_au=A_au,
        )

        T[oridx], T_au[oridx] = integrate_rate_where_known(
            dRdE=dRdE[:, oridx],
            dRdE_au=dRdE_au[:, oridx],
            E_edges=fenergy_bin["edges"],
        )

    json_numpy.write(
        os.path.join(site_gamma_dir, "differential_rate.json"),
        {
            "comment": comment_differential
            + ", "
            + gamma_source["name"]
            + " VS onregion-radius",
            "unit": "s$^{-1} (GeV)$^{-1}$",
            "mean": dRdE,
            "absolute_uncertainty": dRdE_au,
        },
    )
    json_numpy.write(
        os.path.join(site_gamma_dir, "integral_rate.json"),
        {
            "comment": comment_integral
            + ", "
            + gamma_source["name"]
            + " VS onregion-radius",
            "unit": "s$^{-1}$",
            "mean": T,
            "absolute_uncertainty": T_au,
        },
    )

    # cosmic-rays
    # -----------
    for ck in airshower_fluxes[sk]:
        site_particle_dir = os.path.join(site_dir, ck)
        os.makedirs(site_particle_dir, exist_ok=True)

        cosmic_dFdE = airshower_fluxes[sk][ck]["differential_flux"]["values"]
        cosmic_dFdE_au = np.zeros(cosmic_dFdE.shape)

        T = np.zeros(shape=(num_bins_onregion_radius))
        T_au = np.zeros(shape=T.shape)
        dRdE = np.zeros(
            shape=(fenergy_bin["num_bins"], num_bins_onregion_radius)
        )
        dRdE_au = np.zeros(shape=dRdE.shape)
        for oridx in range(num_bins_onregion_radius):
            _Q = onregion_acceptance[sk][ck]["diffuse"]["mean"][:, oridx]
            _Q_ru = onregion_acceptance[sk][ck]["diffuse"]["relative_uncertainty"][:, oridx]
            _Q_au = _Q * _Q_ru

            Q = np.interp(
                x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q,
            )
            Q_au = np.interp(
                x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q_au,
            )

            dRdE[:, oridx], dRdE_au[:, oridx] = irf.utils.multiply(
                x=cosmic_dFdE, x_au=cosmic_dFdE_au, y=Q, y_au=Q_au,
            )

            T[oridx], T_au[oridx] = integrate_rate_where_known(
                dRdE=dRdE[:, oridx],
                dRdE_au=dRdE_au[:, oridx],
                E_edges=fenergy_bin["edges"],
            )

        json_numpy.write(
            os.path.join(site_particle_dir, "differential_rate.json"),
            {
                "comment": comment_differential + " VS onregion-radius",
                "unit": "s$^{-1} (GeV)$^{-1}$",
                "mean": dRdE,
                "absolute_uncertainty": dRdE_au,
            },
        )
        json_numpy.write(
            os.path.join(site_particle_dir, "integral_rate.json"),
            {
                "comment": comment_integral + " VS onregion-radius",
                "unit": "s$^{-1}$",
                "mean": T,
                "absolute_uncertainty": T_au,
            },
        )
