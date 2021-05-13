#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import magnetic_deflection as mdfl
import os

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)


onregion_config = sum_config["on_off_measuremnent"]["onregion"]

NUM_PHOTONS_PIVOT = 1e2

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        event_array = irf.reconstruction.onregion.make_array_from_event_table_for_onregion_estimate(
            event_table=event_table
        )

        eva = event_array
        num_events = eva[spt.IDX].shape[0]

        list_idx = []
        list_angles = []
        for ii in range(num_events):

            (
                _true_cx,
                _true_cy,
            ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
                momentum_x_GeV_per_c=[eva["primary.momentum_x_GeV_per_c"][ii]],
                momentum_y_GeV_per_c=[eva["primary.momentum_y_GeV_per_c"][ii]],
                momentum_z_GeV_per_c=[eva["primary.momentum_z_GeV_per_c"][ii]],
                plenoscope_pointing=irf_config["config"][
                    "plenoscope_pointing"
                ],
            )

            theta_deg = irf.reconstruction.onregion.estimate_required_opening_angle_deg(
                true_cx=_true_cx,
                true_cy=_true_cy,
                reco_cx=eva["reconstructed_trajectory.cx_rad"][ii],
                reco_cy=eva["reconstructed_trajectory.cy_rad"][ii],
                reco_main_axis_azimuth=eva[
                    "reconstructed_trajectory.fuzzy_main_axis_azimuth_rad"
                ][ii],
                reco_num_photons=eva["features.num_photons"][ii],
                reco_core_radius=np.hypot(
                    eva["reconstructed_trajectory.x_m"][ii],
                    eva["reconstructed_trajectory.y_m"][ii],
                ),
                config=onregion_config,
                margin_deg=1e-3,
            )

            list_idx.append(eva[spt.IDX][ii]),
            list_angles.append(theta_deg)

        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "reqired_opening_angle_for_onregion.json"
            ),
            {
                "comment": (
                    "The minimum required opening angle of the onregion to "
                    "contain the true direction of the particle."
                ),
                "unit": "deg",
                spt.IDX: list_idx,
                "opening_angle": list_angles,
            },
        )
