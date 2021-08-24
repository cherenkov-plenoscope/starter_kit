#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import json_numpy


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]

num_onregion_sizes = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)

acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fenergy_bin = energy_binning["interpolation"]

for sk in SITES:
    for pk in PARTICLES:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)
        for gk in ["diffuse", "point"]:

            Q = np.zeros((fenergy_bin["num_bins"], num_onregion_sizes))
            Q_au = np.zeros((fenergy_bin["num_bins"], num_onregion_sizes))
            dQaudE = np.zeros((fenergy_bin["num_bins"], num_onregion_sizes))
            for ok in range(num_onregion_sizes):
                print("acceptance", sk, pk, ok)
                _Q = acceptance[sk][pk][gk]["mean"][:, ok]
                _Q_ru = acceptance[sk][pk][gk]["relative_uncertainty"][:, ok]
                _Q_ru[np.isnan(_Q_ru)] = 0.0
                _Q_au = _Q * _Q_ru

                Q[:, ok] = irf.utils.log10interp(
                    x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q,
                )
                Q_au[:, ok] = irf.utils.log10interp(
                    x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q_au,
                )

            json_numpy.write(
                os.path.join(sk_pk_dir, gk + ".json"),
                {
                    "comment": acceptance[sk][pk][gk]["comment"],
                    "mean": Q,
                    "absolute_uncertainty": Q_au,
                    "unit": acceptance[sk][pk][gk]["unit"],
                    "energy_binning_key": "interpolation"
                },
            )
