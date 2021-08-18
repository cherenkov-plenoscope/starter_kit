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

opj = os.path.join

os.makedirs(pa["out_dir"], exist_ok=True)

SITES = irf_config["config"]["sites"]
PARTICLES = irf_config["config"]["particles"]
COSMIC_RAYS = list(irf_config["config"]["particles"].keys())
COSMIC_RAYS.remove("gamma")

# read energy-migration
# ---------------------
energy_migration = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_energy_estimate_quality"),
)

# write enrgy-interpretation for diff. sens. scenarios
# ----------------------------------------------------
for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)

    for pk in PARTICLES:
        sk_pk_dir = os.path.join(sk_dir, pk)
        os.makedirs(sk_pk_dir, exist_ok=True)

    for dk in irf.analysis.differential_sensitivity.SCENARIOS:
        # split in signal and background
        # ------------------------------
        _c = "confusion_matrix"
        _cbn = "counts_normalized_on_ax0"
        _cbn_au = "counts_normalized_on_ax0_abs_unc"

        s_m = energy_migration[sk]["gamma"][_c][_cbn]
        s_m_au = energy_migration[sk]["gamma"][_c][_cbn_au]

        bg_ms = {}
        bg_ms_au = {}
        for ck in COSMIC_RAYS:
            bg_ms[ck] = energy_migration[sk][ck][_c][_cbn]
            bg_ms_au[ck] = energy_migration[sk][ck][_c][_cbn_au]

        # apply scenarios
        # ---------------
        m = irf.analysis.differential_sensitivity.make_energy_confusion_matrices_for_signal_and_background(
            signal_energy_confusion_matrix=s_m,
            signal_energy_confusion_matrix_abs_unc=s_m_au,
            background_energy_confusion_matrices=bg_ms,
            background_energy_confusion_matrices_abs_unc=bg_ms_au,
            scenario_key=dk,
        )

        # output for each particle
        # ------------------------
        json_numpy.write(
            opj(sk_dir, "gamma", dk + ".json"),
            {
                "ax0_key": energy_migration[sk]["gamma"][_c]["ax0_key"],
                "ax1_key": energy_migration[sk]["gamma"][_c]["ax1_key"],
                _cbn: m["signal_matrix"],
                _cbn_au: m["signal_matrix_abs_unc"],
            },
        )
        for ck in COSMIC_RAYS:
            json_numpy.write(
                opj(sk_dir, ck, dk + ".json"),
                {
                    "ax0_key": energy_migration[sk][ck][_c]["ax0_key"],
                    "ax1_key": energy_migration[sk][ck][_c]["ax1_key"],
                    _cbn: m["background_matrices"][ck],
                    _cbn_au: m["background_matrices_abs_unc"][ck],
                },
            )
