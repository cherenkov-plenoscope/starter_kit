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

ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

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
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)

    # gamma-ray
    # ---------
    os.makedirs(os.path.join(sk_dir, "gamma"), exist_ok=True)

    for ok in ONREGION_TYPES:
        os.makedirs(os.path.join(sk_dir, "gamma", ok), exist_ok=True)

        _A = onregion_acceptance[sk]["gamma"][ok]["point"]["mean"]
        _A_au = onregion_acceptance[sk]["gamma"][ok]["point"]["absolute_uncertainty"]

        A = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A
        )
        A_au = np.interp(
            x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_A_au
        )

        dRdE, dRdE_au = irf.utils.multiply(
            x=gamma_dKdE, x_au=gamma_dKdE_au, y=A, y_au=A_au,
        )

        R, R_au = irf.utils.integrate_rate_where_known(
            dRdE=dRdE,
            dRdE_au=dRdE_au,
            E_edges=fenergy_bin["edges"],
        )

        json_numpy.write(
            os.path.join(sk_dir, "gamma", ok, "differential_rate.json"),
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
            os.path.join(sk_dir, "gamma", ok, "integral_rate.json"),
            {
                "comment": comment_integral
                + ", "
                + gamma_source["name"]
                + " VS onregion-radius",
                "unit": "s$^{-1}$",
                "mean": R,
                "absolute_uncertainty": R_au,
            },
        )

    # cosmic-rays
    # -----------
    for ck in airshower_fluxes[sk]:
        os.makedirs(os.path.join(sk_dir, ck), exist_ok=True)

        cosmic_dFdE = airshower_fluxes[sk][ck]["differential_flux"]["values"]
        cosmic_dFdE_au = np.zeros(cosmic_dFdE.shape)

        for ok in ONREGION_TYPES:
            os.makedirs(os.path.join(sk_dir, ck, ok), exist_ok=True)

            _Q = onregion_acceptance[sk][ck][ok]["diffuse"]["mean"]
            _Q_au = onregion_acceptance[sk][ck][ok]["diffuse"][
                "absolute_uncertainty"
            ]

            Q = np.interp(
                x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q,
            )
            Q_au = np.interp(
                x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q_au,
            )

            dRdE, dRdE_au = irf.utils.multiply(
                x=cosmic_dFdE, x_au=cosmic_dFdE_au, y=Q, y_au=Q_au,
            )

            R, R_au = irf.utils.integrate_rate_where_known(
                dRdE=dRdE,
                dRdE_au=dRdE_au,
                E_edges=fenergy_bin["edges"],
            )

            json_numpy.write(
                os.path.join(sk_dir, ck, ok, "differential_rate.json"),
                {
                    "comment": comment_differential + " VS onregion-radius",
                    "unit": "s$^{-1} (GeV)$^{-1}$",
                    "mean": dRdE,
                    "absolute_uncertainty": dRdE_au,
                },
            )
            json_numpy.write(
                os.path.join(sk_dir, ck, ok, "integral_rate.json"),
                {
                    "comment": comment_integral + " VS onregion-radius",
                    "unit": "s$^{-1}$",
                    "mean": R,
                    "absolute_uncertainty": R_au,
                },
            )
