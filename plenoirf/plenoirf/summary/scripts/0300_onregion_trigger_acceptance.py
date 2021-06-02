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

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
passing_trajectory_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0059_passing_trajectory_quality")
)

MAX_SOURCE_ANGLE_DEG = sum_config["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
MAX_SOURCE_ANGLE = np.deg2rad(MAX_SOURCE_ANGLE_DEG)
SOLID_ANGLE_TO_CONTAIN_SOURCE = np.pi * MAX_SOURCE_ANGLE ** 2.0

POSSIBLE_ONREGION_POLYGON = irf.reconstruction.onregion.make_circular_polygon(
    radius=MAX_SOURCE_ANGLE,
    num_steps=37
)
pointing_azimuth_deg = irf_config["config"]["plenoscope_pointing"][
    "azimuth_deg"
]
pointing_zenith_deg = irf_config["config"]["plenoscope_pointing"]["zenith_deg"]

energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    sum_config["energy_binning"]["num_bins"]["trigger_acceptance_onregion"]
    + 1,
)
num_bins_energy = sum_config["energy_binning"]["num_bins"][
    "trigger_acceptance_onregion"
]

onregion_config = sum_config["on_off_measuremnent"]["onregion"]

onregion_radii_deg = np.array(
    sum_config["on_off_measuremnent"]["onregion"]["loop_opening_angle_deg"]
)
num_bins_onregion_radius = onregion_radii_deg.shape[0]


cosmic_ray_keys = list(irf_config["config"]["particles"].keys())
cosmic_ray_keys.remove("gamma")


def cut_candidates_for_detection(
    event_table, idx_trajectory_quality, idx_trigger, idx_quality,
):
    idx_self = event_table["primary"][spt.IDX]

    idx_candidates = spt.intersection(
        [idx_self, idx_trigger, idx_quality, idx_trajectory_quality]
    )

    return spt.cut_and_sort_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=idx_candidates,
        level_keys=None,
    )


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


for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        # SCENARIO: point source
        # ----------------------
        diffuse_thrown = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        idx_source_in_possible_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
            primary_table=diffuse_thrown["primary"],
            radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
            azimuth_deg=pointing_azimuth_deg,
            zenith_deg=pointing_zenith_deg,
        )

        # thrown
        point_thrown = spt.cut_table_on_indices(
            table=diffuse_thrown,
            structure=irf.table.STRUCTURE,
            common_indices=idx_source_in_possible_onregion,
            level_keys=None,
        )

        # detected
        point_candidate = cut_candidates_for_detection(
            event_table=point_thrown,
            idx_trajectory_quality=passing_trajectory_quality[sk][pk]["idx"],
            idx_trigger=passing_trigger[sk][pk]["idx"],
            idx_quality=passing_quality[sk][pk]["idx"],
        )

        poicanarr = irf.reconstruction.trajectory_quality.make_rectangular_table(
            event_table=point_candidate,
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        Qeff = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        Qunc = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        for oridx in range(num_bins_onregion_radius):
            onregion_config["opening_angle_deg"] = onregion_radii_deg[oridx]

            idx_dict_source_in_onregion = {}
            for ii in range(poicanarr[spt.IDX].shape[0]):

                _onregion = irf.reconstruction.onregion.estimate_onregion(
                    reco_cx=poicanarr["reconstructed_trajectory/cx_rad"][ii],
                    reco_cy=poicanarr["reconstructed_trajectory/cy_rad"][ii],
                    reco_main_axis_azimuth=poicanarr[
                        "reconstructed_trajectory/fuzzy_main_axis_azimuth_rad"
                    ][ii],
                    reco_num_photons=poicanarr["features/num_photons"][ii],
                    reco_core_radius=np.hypot(
                        poicanarr["reconstructed_trajectory/x_m"][ii],
                        poicanarr["reconstructed_trajectory/y_m"][ii],
                    ),
                    config=onregion_config,
                )

                hit = irf.reconstruction.onregion.is_direction_inside(
                    cx=poicanarr["true_trajectory/cx_rad"][ii],
                    cy=poicanarr["true_trajectory/cy_rad"][ii],
                    onregion=_onregion,
                )

                idx_dict_source_in_onregion[poicanarr[spt.IDX][ii]] = hit

            mask_detected = make_wighted_mask_wrt_primary_table(
                primary_table=point_thrown["primary"],
                idx_dict_of_weights=idx_dict_source_in_onregion,
            )

            (
                _q_eff,
                _q_unc,
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=point_thrown["primary"]["energy_GeV"],
                mask_detected=mask_detected,
                quantity_scatter=point_thrown["grid"]["area_thrown_m2"],
                num_grid_cells_above_lose_threshold=point_thrown["grid"][
                    "num_bins_above_threshold"
                ],
                total_num_grid_cells=point_thrown["grid"]["num_bins_thrown"],
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
        diffuse_thrown = diffuse_thrown

        # detected
        diffuse_candidate = cut_candidates_for_detection(
            event_table=diffuse_thrown,
            idx_trajectory_quality=passing_trajectory_quality[sk][pk]["idx"],
            idx_trigger=passing_trigger[sk][pk]["idx"],
            idx_quality=passing_quality[sk][pk]["idx"],
        )

        difcanarr = irf.reconstruction.trajectory_quality.make_rectangular_table(
            event_table=diffuse_candidate,
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        Qeff = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))
        Qunc = np.zeros(shape=(num_bins_energy, num_bins_onregion_radius))

        for oridx in range(num_bins_onregion_radius):
            onregion_config["opening_angle_deg"] = onregion_radii_deg[oridx]

            idx_dict_probability_for_source_in_onregion = {}
            for ii in range(difcanarr[spt.IDX].shape[0]):

                _onregion = irf.reconstruction.onregion.estimate_onregion(
                    reco_cx=difcanarr["reconstructed_trajectory/cx_rad"][ii],
                    reco_cy=difcanarr["reconstructed_trajectory/cy_rad"][ii],
                    reco_main_axis_azimuth=difcanarr[
                        "reconstructed_trajectory/fuzzy_main_axis_azimuth_rad"
                    ][ii],
                    reco_num_photons=difcanarr["features/num_photons"][ii],
                    reco_core_radius=np.hypot(
                        difcanarr["reconstructed_trajectory/x_m"][ii],
                        difcanarr["reconstructed_trajectory/y_m"][ii],
                    ),
                    config=onregion_config,
                )

                onregion_polygon = irf.reconstruction.onregion.make_polygon(
                    onregion=_onregion,
                    num_steps=37
                )

                overlap_srad = irf.reconstruction.onregion.intersecting_area_of_polygons(
                    a=onregion_polygon,
                    b=POSSIBLE_ONREGION_POLYGON
                )

                probability_to_contain_random_source = (
                    overlap_srad / SOLID_ANGLE_TO_CONTAIN_SOURCE
                )

                idx_dict_probability_for_source_in_onregion[
                    difcanarr[spt.IDX][ii]
                ] = probability_to_contain_random_source

            mask_probability_for_source_in_onregion = make_wighted_mask_wrt_primary_table(
                primary_table=diffuse_thrown["primary"],
                idx_dict_of_weights=idx_dict_probability_for_source_in_onregion,
            )

            (
                _q_eff,
                _q_unc,
            ) = irf.analysis.effective_quantity.effective_quantity_for_grid(
                energy_bin_edges_GeV=energy_bin_edges,
                energy_GeV=diffuse_thrown["primary"]["energy_GeV"],
                mask_detected=mask_probability_for_source_in_onregion,
                quantity_scatter=(
                    diffuse_thrown["grid"]["area_thrown_m2"]
                    * diffuse_thrown["primary"]["solid_angle_thrown_sr"]
                ),
                num_grid_cells_above_lose_threshold=diffuse_thrown["grid"][
                    "num_bins_above_threshold"
                ],
                total_num_grid_cells=diffuse_thrown["grid"]["num_bins_thrown"],
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
