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

NUM_GRID_BINS = irf_config["grid_geometry"]["num_bins_diameter"] ** 2
MAX_SOURCE_ANGLE_DEG = sum_config["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
MAX_SOURCE_ANGLE = np.deg2rad(MAX_SOURCE_ANGLE_DEG)

pointing_azimuth_deg = irf_config["config"]["plenoscope_pointing"][
    "azimuth_deg"
]
pointing_zenith_deg = irf_config["config"]["plenoscope_pointing"]["zenith_deg"]
analysis_trigger_threshold = sum_config["trigger"]["threshold_pe"]
trigger_modus = sum_config["trigger"]["modus"]

energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance_onregion"]
    + 1,
)
num_bins_energy = sum_config["energy_binning"]["num_bins"][
    "trigger_acceptance_onregion"
]

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

onregion_config = sum_config["on_off_measuremnent"]["onregion"]

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]

SOLID_ANGLE_TO_CONTAIN_SOURCE = np.pi * MAX_SOURCE_ANGLE ** 2.0

cosmic_ray_keys = list(irf_config["config"]["particles"].keys())
cosmic_ray_keys.remove("gamma")


def cut_candidates_for_detection(
    event_table,
    analysis_trigger_threshold,
    trigger_modus,
    max_relative_leakage,
    min_reconstructed_photons,
):
    idx_trigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=event_table["trigger"],
        threshold=analysis_trigger_threshold,
        modus=trigger_modus,
    )
    idx_quality = irf.analysis.cuts.cut_quality(
        feature_table=event_table["features"],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )
    idx_trajectory = event_table["reconstructed_trajectory"][spt.IDX]

    idx_candidates = spt.intersection(
        [idx_trigger, idx_quality, idx_trajectory]
    )

    candidate_event_table = spt.cut_and_sort_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=idx_candidates,
        level_keys=None,
    )

    return candidate_event_table


def make_wighted_mask_wrt_primary_table(
    primary_table, idx_dict_of_weights, default_weight=0.0
):
    num_primaries = primary_table[spt.IDX].shape[0]
    mask = np.zeros(num_primaries)

    for ii in range(num_primaries):
        idx = primary_table[spt.IDX][ii]
        if idx in idx_dict_of_weights:
            mask[ii] = idx_dict_of_weights[idx]
        else:
            mask[ii] = default_weight
    return mask


for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], site_key, particle_key)
        os.makedirs(site_particle_dir, exist_ok=True)

        # SCENARIO: point source
        # ----------------------
        table_diffuse = spt.read(
            path=os.path.join(
                pa["run_dir"],
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
            primary_table=table_diffuse["primary"],
            radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
            azimuth_deg=pointing_azimuth_deg,
            zenith_deg=pointing_zenith_deg,
        )

        # thrown
        table_point = spt.cut_table_on_indices(
            table=table_diffuse,
            structure=irf.table.STRUCTURE,
            common_indices=idx_onregion,
            level_keys=None,
        )

        # detected
        candidate_table_point = cut_candidates_for_detection(
            event_table=table_point,
            analysis_trigger_threshold=analysis_trigger_threshold,
            trigger_modus=trigger_modus,
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )

        candidate_array_point = irf.reconstruction.onregion.make_array_from_event_table_for_onregion_estimate(
            event_table=candidate_table_point
        )
        cap = candidate_array_point
        num_candidate_events = cap[spt.IDX].shape[0]

        Qeff = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        Qunc = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        for oridx in range(num_bins_onregion_radius):
            onregion_config["opening_angle_deg"] = onregion_radii_deg[oridx]

            idx_dict_source_in_onregion = {}
            for ii in range(num_candidate_events):

                _onregion = irf.reconstruction.onregion.estimate_onregion(
                    reco_cx=cap["reconstructed_trajectory.cx_rad"][ii],
                    reco_cy=cap["reconstructed_trajectory.cy_rad"][ii],
                    reco_main_axis_azimuth=cap[
                        "reconstructed_trajectory.fuzzy_main_axis_azimuth_rad"
                    ][ii],
                    reco_num_photons=cap["features.num_photons"][ii],
                    reco_core_radius=np.hypot(
                        cap["reconstructed_trajectory.x_m"][ii],
                        cap["reconstructed_trajectory.y_m"][ii],
                    ),
                    config=onregion_config,
                )

                (
                    _true_cx,
                    _true_cy,
                ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
                    momentum_x_GeV_per_c=[
                        cap["primary.momentum_x_GeV_per_c"][ii]
                    ],
                    momentum_y_GeV_per_c=[
                        cap["primary.momentum_y_GeV_per_c"][ii]
                    ],
                    momentum_z_GeV_per_c=[
                        cap["primary.momentum_z_GeV_per_c"][ii]
                    ],
                    plenoscope_pointing=irf_config["config"][
                        "plenoscope_pointing"
                    ],
                )

                hit = irf.reconstruction.onregion.is_direction_inside(
                    cx=_true_cx, cy=_true_cy, onregion=_onregion
                )

                idx_dict_source_in_onregion[cap[spt.IDX][ii]] = hit

            mask_detected = make_wighted_mask_wrt_primary_table(
                primary_table=table_point["primary"],
                idx_dict_of_weights=idx_dict_source_in_onregion,
            )

            (
                _q_eff,
                _q_unc,
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=table_point["primary"]["energy_GeV"],
                mask_detected=mask_detected,
                quantity_scatter=table_point["grid"]["area_thrown_m2"],
                num_grid_cells_above_lose_threshold=table_point["grid"][
                    "num_bins_above_threshold"
                ],
                total_num_grid_cells=table_point["grid"]["num_bins_thrown"],
            )

            Qeff[:, oridx] = _q_eff
            Qunc[:, oridx] = _q_unc

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "point.json"),
            {
                "comment": (
                    "Effective area "
                    "for a point source, reconstructed in onregion. "
                    "VS energy-bins VS onregion-radii"
                ),
                "unit": "m$^{2}$",
                "mean": Qeff,
                "relative_uncertainty": Qunc,
            },
        )

        # SCENARIO: diffuse source
        # ------------------------

        # thrown
        table_diffuse = table_diffuse

        # detected
        candidate_table_diffuse = cut_candidates_for_detection(
            event_table=table_diffuse,
            analysis_trigger_threshold=analysis_trigger_threshold,
            trigger_modus=trigger_modus,
            max_relative_leakage=max_relative_leakage,
            min_reconstructed_photons=min_reconstructed_photons,
        )

        candidate_array_diffuse = irf.reconstruction.onregion.make_array_from_event_table_for_onregion_estimate(
            event_table=candidate_table_diffuse
        )
        cad = candidate_array_diffuse
        num_candidate_events = cad[spt.IDX].shape[0]

        Qeff = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        Qunc = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))

        for oridx in range(num_bins_onregion_radius):
            onregion_config["opening_angle_deg"] = onregion_radii_deg[oridx]

            idx_dict_probability_for_source_in_onregion = {}
            for ii in range(num_candidate_events):

                _onregion = irf.reconstruction.onregion.estimate_onregion(
                    reco_cx=cad["reconstructed_trajectory.cx_rad"][ii],
                    reco_cy=cad["reconstructed_trajectory.cy_rad"][ii],
                    reco_main_axis_azimuth=cad[
                        "reconstructed_trajectory.fuzzy_main_axis_azimuth_rad"
                    ][ii],
                    reco_num_photons=cad["features.num_photons"][ii],
                    reco_core_radius=np.hypot(
                        cad["reconstructed_trajectory.x_m"][ii],
                        cad["reconstructed_trajectory.y_m"][ii],
                    ),
                    config=onregion_config,
                )

                _probability = (
                    _onregion["ellipse_solid_angle"]
                    / SOLID_ANGLE_TO_CONTAIN_SOURCE
                )

                _optical_axis_to_reconstructed_direction = np.hypot(
                    cad["reconstructed_trajectory.cx_rad"][ii],
                    cad["reconstructed_trajectory.cy_rad"][ii],
                )

                _closest_reconstructed_direction = (
                    _optical_axis_to_reconstructed_direction
                    - _onregion["ellipse_mayor_radius"]
                )

                if _closest_reconstructed_direction > MAX_SOURCE_ANGLE:
                    _probability = 0.0

                idx_dict_probability_for_source_in_onregion[
                    cad[spt.IDX][ii]
                ] = _probability

            mask_probability_for_source_in_onregion = make_wighted_mask_wrt_primary_table(
                primary_table=table_diffuse["primary"],
                idx_dict_of_weights=idx_dict_probability_for_source_in_onregion,
            )

            (
                _q_eff,
                _q_unc,
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=table_diffuse["primary"]["energy_GeV"],
                mask_detected=mask_probability_for_source_in_onregion,
                quantity_scatter=(
                    table_diffuse["grid"]["area_thrown_m2"]
                    * table_diffuse["primary"]["solid_angle_thrown_sr"]
                ),
                num_grid_cells_above_lose_threshold=table_diffuse["grid"][
                    "num_bins_above_threshold"
                ],
                total_num_grid_cells=table_diffuse["grid"]["num_bins_thrown"],
            )

            Qeff[:, oridx] = _q_eff
            Qunc[:, oridx] = _q_unc

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "diffuse.json"),
            {
                "comment": (
                    "Effective acceptance (area x solid angle) "
                    "for a diffuse source, reconstructed in onregion. "
                    "VS energy-bins VS onregion-radii"
                ),
                "unit": "m$^{2}$ sr",
                "mean": Qeff,
                "relative_uncertainty": Qunc,
            },
        )
