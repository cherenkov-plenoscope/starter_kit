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
ONREGION_TYPES = sum_config["on_off_measuremnent"]["onregion_types"]

acceptance = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0300_onregion_trigger_acceptance")
)

energy_binning = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)
energy_bin = energy_binning["trigger_acceptance_onregion"]
fenergy_bin = energy_binning["interpolation"]

for sk in SITES:
    for ok in ONREGION_TYPES:
        for pk in PARTICLES:
            sk_ok_pk_dir = os.path.join(pa["out_dir"], sk, ok, pk)
            os.makedirs(sk_ok_pk_dir, exist_ok=True)
            for gk in ["diffuse", "point"]:


                _Q = acceptance[sk][ok][pk][gk]["mean"]
                _Q_au = acceptance[sk][ok][pk][gk]["absolute_uncertainty"]

                Q = irf.utils.log10interp(
                    x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q,
                )
                Q_au = irf.utils.log10interp(
                    x=fenergy_bin["centers"], xp=energy_bin["centers"], fp=_Q_au,
                )

                json_numpy.write(
                    os.path.join(sk_ok_pk_dir, gk + ".json"),
                    {
                        "comment": acceptance[sk][ok][pk][gk]["comment"],
                        "mean": Q,
                        "absolute_uncertainty": Q_au,
                        "unit": acceptance[sk][ok][pk][gk]["unit"],
                        "energy_binning_key": "interpolation"
                    },
                )
