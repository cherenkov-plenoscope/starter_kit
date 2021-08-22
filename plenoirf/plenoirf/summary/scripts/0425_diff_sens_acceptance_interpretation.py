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

# prepare energy confusion
# ------------------------
iEnergy = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0420_diff_sens_energy_interpretation"),
)

# prepare acceptance (after all cuts) in true energy
# --------------------------------------------------
acceptance = json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"], "0300_onregion_trigger_acceptance"
    )
)

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

num_bins_onregion_radius = len(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
ONREGIONS = range(num_bins_onregion_radius)


GEOMETRIES = ["point", "diffuse"]

for sk in SITES:
    for pk in PARTICLES:
        for gk in GEOMETRIES:
            sk_pk_gk_dir = os.path.join(pa["out_dir"], sk, pk, gk)
            os.makedirs(sk_pk_gk_dir, exist_ok=True)

            for dk in irf.analysis.differential_sensitivity.SCENARIOS:
                mm = iEnergy[sk][pk][dk]["counts_normalized_on_ax0"]
                mm_au = iEnergy[sk][pk][dk]["counts_normalized_on_ax0_abs_unc"]
                if "integral_mask" in iEnergy[sk][pk][dk]:
                    _integral_mask = iEnergy[sk][pk][dk]["integral_mask"]
                else:
                    _integral_mask = np.eye(mm.shape[0])

                Q = acceptance[sk][pk][gk]["mean"]
                Q_ru = acceptance[sk][pk][gk]["relative_uncertainty"]
                Q_ru[np.isnan(Q_ru)] = 0.0
                Q_au = Q * Q_ru

                _iQ = np.zeros(shape=(energy_bin["num_bins"], len(ONREGIONS)))
                _iQ_au = np.zeros(_iQ.shape)

                iQ = np.zeros(shape=(energy_bin["num_bins"], len(ONREGIONS)))
                iQ_au = np.zeros(iQ.shape)

                print(sk, pk, gk, dk, _integral_mask)
                for ok in ONREGIONS:
                    for ereco in range(energy_bin["num_bins"]):

                        _integral = np.zeros(energy_bin["num_bins"])
                        _integral_au = np.zeros(energy_bin["num_bins"])
                        for etrue in range(energy_bin["num_bins"]):
                            _integral[etrue], _integral_au[etrue] = irf.utils.multiply(
                                x=mm[etrue, ereco],
                                x_au=mm_au[etrue, ereco],
                                y=Q[etrue, ok],
                                y_au=Q_au[etrue, ok]
                            )
                        _iQ[ereco, ok], _iQ_au[ereco, ok] = irf.utils.sum(
                            x=_integral,
                            x_au=_integral_au
                        )

                    for eout in range(energy_bin["num_bins"]):
                        _integral2 = np.zeros(energy_bin["num_bins"])
                        _integral2_au = np.zeros(energy_bin["num_bins"])
                        for ereco in range(energy_bin["num_bins"]):
                            _integral2[ereco], _integral2_au[ereco] = irf.utils.multiply(
                                x=_integral_mask[eout, ereco],
                                x_au=0.0,
                                y=_iQ[ereco, ok],
                                y_au=_iQ_au[ereco, ok],
                            )
                        iQ[eout, ok], iQ_au[eout, ok] = irf.utils.sum(
                            x=_integral2,
                            x_au=_integral2_au
                        )

                json_numpy.write(
                    os.path.join(sk_pk_gk_dir, dk + ".json"),
                    {
                        "comment": "area/acceptance VS energy VS onregion",
                        "unit": acceptance[sk][pk][gk]["unit"],
                        "mean": iQ,
                        "absolute_uncertainty": iQ_au,
                    },
                )
