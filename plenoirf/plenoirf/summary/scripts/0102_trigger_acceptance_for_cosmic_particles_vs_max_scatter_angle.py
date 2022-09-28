#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import json_numpy
import magnetic_deflection as mdfl


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]
MAX_SOURCE_ANGLE_DEG = sum_config["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
pointing_azimuth_deg = irf_config["config"]["plenoscope_pointing"][
    "azimuth_deg"
]
pointing_zenith_deg = irf_config["config"]["plenoscope_pointing"]["zenith_deg"]

energy_bin = json_numpy.read(
    os.path.join(pa["summary_dir"], "0005_common_binning", "energy.json")
)["trigger_acceptance_onregion"]

passing_trigger = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)

max_scatter_angles_deg = json_numpy.read(
    os.path.join(
        pa["summary_dir"], "0005_common_binning", "max_scatter_angles_deg.json"
    )
)

for sk in SITES:
    for pk in PARTICLES:
        sk_pk_dir = os.path.join(pa["out_dir"], sk, pk)

        os.makedirs(sk_pk_dir, exist_ok=True)

        shower_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        # diffuse source
        # --------------
        energy_GeV = shower_table["primary"]["energy_GeV"]

        num_grid_cells_above_lose_threshold = shower_table["grid"][
            "num_bins_above_threshold"
        ]
        total_num_grid_cells = shower_table["grid"]["num_bins_thrown"]
        idx_detected = passing_trigger[sk][pk]["idx"]

        mask_passed_trigger = spt.make_mask_of_right_in_left(
            left_indices=shower_table["primary"][spt.IDX],
            right_indices=idx_detected,
        )

        particles_max_scatter_angle_rad = np.deg2rad(
            irf_config["config"]["particles"][pk]["max_scatter_angle_deg"]
        )

        _az_deg = np.rad2deg(shower_table["primary"]["azimuth_rad"])
        _zd_deg = np.rad2deg(shower_table["primary"]["zenith_rad"])

        _mag_az_deg = np.rad2deg(shower_table["primary"]["magnet_azimuth_rad"])
        _mag_zd_deg = np.rad2deg(shower_table["primary"]["magnet_zenith_rad"])

        scatter_deg = mdfl.spherical_coordinates._angle_between_az_zd_deg(
            az1_deg=_az_deg,
            zd1_deg=_zd_deg,
            az2_deg=_mag_az_deg,
            zd2_deg=_mag_zd_deg,
        )

        value = []
        absolute_uncertainty = []
        for ci in range(len(max_scatter_angles_deg[pk])):
            print(
                sk,
                pk,
                "max. scatter {:.3}deg".format(max_scatter_angles_deg[pk][ci]),
            )

            solid_angle_thrown_sr = irf.utils.cone_solid_angle(
                cone_radial_opening_angle_rad=np.deg2rad(
                    max_scatter_angles_deg[pk][ci]
                )
            )

            mask_within_max_scatter_angle = (
                scatter_deg <= max_scatter_angles_deg[pk][ci]
            )

            mask_detected = np.logical_and(
                mask_passed_trigger, mask_within_max_scatter_angle,
            )

            quantity_scatter = (
                shower_table["grid"]["area_thrown_m2"] * solid_angle_thrown_sr
            )

            (
                _q_eff,
                _q_eff_au,
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin["edges"],
                energy_GeV=energy_GeV,
                mask_detected=mask_detected,
                quantity_scatter=quantity_scatter,
                num_grid_cells_above_lose_threshold=(
                    num_grid_cells_above_lose_threshold
                ),
                total_num_grid_cells=total_num_grid_cells,
            )
            value.append(_q_eff)
            absolute_uncertainty.append(_q_eff_au)

        json_numpy.write(
            os.path.join(sk_pk_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source. "
                    "VS max. scatter-angle VS energy-bins"
                ),
                "unit": "m$^{2}$ sr",
                "mean": value,
                "absolute_uncertainty": absolute_uncertainty,
            },
        )
