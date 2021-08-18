#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(PARTICLES)
COSMIC_RAYS.remove("gamma")

iacceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0425_diff_sens_acceptance_interpretation"),
)

airshower_fluxes = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0015_flux_of_airshowers")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fine_energy_bin = energy_binning["interpolation"]
fine_energy_bin_edge_matches = []
for energy in energy_bin["edges"]:
    idx_near = np.argmin(np.abs(fine_energy_bin["edges"] - energy))
    fine_energy_bin_edge_matches.append(idx_near)


gk = "diffuse"

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

for sk in SITES:
    for pk in COSMIC_RAYS:
        for dk in irf.analysis.differential_sensitivity.SCENARIOS:
            sk_pk_dk_dir = os.path.join(pa["out_dir"], sk, pk, dk)
            os.makedirs(sk_pk_dk_dir, exist_ok=True)

            dRdE = np.zeros(shape=(fine_energy_bin["num_bins"], num_onregion_sizes))
            dRdE_au = np.zeros(dRdE.shape)

            R = np.zeros(shape=(energy_bin["num_bins"], num_onregion_sizes))
            R_au = np.zeros(shape=R.shape)

            for ok in range(num_onregion_sizes):
                _iQ = iacceptance[sk][pk][gk][dk]["mean"][:, ok]
                _iQ_au = iacceptance[sk][pk][gk][dk]["absolute_uncertainty"][:, ok]
                iQ = np.interp(
                    x=fine_energy_bin["centers"],
                    xp=energy_bin["centers"],
                    fp=_iQ,
                )
                iQ_au = np.interp(
                    x=fine_energy_bin["centers"],
                    xp=energy_bin["centers"],
                    fp=_iQ_au,
                )
                dFdE = airshower_fluxes[sk][pk]["differential_flux"]["values"]
                dFdE_au = np.zeros(dFdE.shape)

                dRdE[:, ok], dRdE_au[:, ok] = irf.utils.multiply(
                    x=iQ, x_au=iQ_au, y=dFdE, y_au=dFdE_au
                )

                for ee in range(energy_bin["num_bins"]):
                    estart = fine_energy_bin_edge_matches[ee]
                    estop = fine_energy_bin_edge_matches[ee + 1]

                    _dRdE_x_dE = (
                        dRdE[:, ok][estart:estop]
                        * fine_energy_bin["width"][estart:estop]
                    )
                    _dRdE_au_x_dE = (
                        dRdE_au[:, ok][estart:estop]
                        * fine_energy_bin["width"][estart:estop]
                    )

                    R[ee, ok], R_au[ee, ok] = irf.utils.sum(
                        x=_dRdE_x_dE, x_au=_dRdE_au_x_dE
                    )

            json_numpy.write(
                os.path.join(sk_pk_dk_dir, "differential_rate.json"),
                {
                    "comment": "dRdE VS energy VS onregions",
                    "unit": "s$^{-1}$ (GeV)$^{-1}$",
                    "mean": dRdE,
                    "absolute_uncertainty": dRdE_au,
                    "energy_binning": "interpolation",
                },
            )

            json_numpy.write(
                os.path.join(sk_pk_dk_dir, "rate.json"),
                {
                    "comment": "R VS energy VS onregions",
                    "unit": "s$^{-1}$",
                    "mean": R,
                    "absolute_uncertainty": R_au,
                    "energy_binning": "trigger_acceptance_onregion",
                },
            )