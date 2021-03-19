#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import pandas
import plenopy as pl


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

fig_16_by_9 = sum_config["plot"]["16_by_9"]

num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "point_spread_function"
]
energy_lower_edge = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper_edge = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower_edge, energy_upper_edge, num_energy_bins + 1
)
energy_bin_centers = irf.utils.bin_centers(energy_bin_edges)

theta_square_bin_edges_deg2 = np.linspace(
    0,
    sum_config["point_spread_function"]["theta_square"]["max_angle_deg"] ** 2,
    sum_config["point_spread_function"]["theta_square"]["num_bins"],
)
psf_containment_factor = sum_config["point_spread_function"][
    "containment_factor"
]
pivot_energy_GeV = sum_config["point_spread_function"]["pivot_energy_GeV"]


# READ reconstruction
# ===================
_rec = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0206_fitting_light_field")
)
reconstruction = {}
for sk in _rec:
    reconstruction[sk] = {}
    for pk in _rec[sk]:
        _df = pandas.DataFrame(_rec[sk][pk]["reco"])
        reconstruction[sk][pk] = _df.to_records(index=False)

level_keys = [
    "primary",
    "cherenkovsize",
    "grid",
    "cherenkovpool",
    "cherenkovsizepart",
    "cherenkovpoolpart",
    "core",
    "trigger",
    "pasttrigger",
    "cherenkovclassification",
]

for sk in reconstruction:
    for pk in reconstruction[sk]:

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        event_table = spt.cut_table_on_indices(
            event_table,
            irf.table.STRUCTURE,
            common_indices=reconstruction[sk][pk][spt.IDX],
            level_keys=level_keys,
        )

        t2hist = []
        t2hist_rel_unc = []

        containment_vs_energy = []
        containment_rel_unc_vs_energy = []

        for energy_bin in range(num_energy_bins):

            idx_energy_bin = irf.analysis.cuts.cut_energy_bin(
                primary_table=event_table["primary"],
                lower_energy_edge_GeV=energy_bin_edges[energy_bin],
                upper_energy_edge_GeV=energy_bin_edges[energy_bin + 1],
            )
            common_indices = spt.intersection(
                [idx_energy_bin, reconstruction[sk][pk][spt.IDX]]
            )
            ebin_truth = spt.cut_table_on_indices(
                table=event_table,
                structure=irf.table.STRUCTURE,
                common_indices=common_indices,
                level_keys=level_keys,
            )
            ebin_truth = spt.sort_table_on_common_indices(
                table=ebin_truth, common_indices=common_indices
            )
            ebin_truth_df = spt.make_rectangular_DataFrame(ebin_truth)

            ebin = pandas.merge(
                pandas.DataFrame(reconstruction[sk][pk]),
                ebin_truth_df,
                on=spt.IDX,
            ).to_records(index=False)

            (
                true_cx,
                true_cy,
            ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
                momentum_x_GeV_per_c=ebin["primary.momentum_x_GeV_per_c"],
                momentum_y_GeV_per_c=ebin["primary.momentum_y_GeV_per_c"],
                momentum_z_GeV_per_c=ebin["primary.momentum_z_GeV_per_c"],
                plenoscope_pointing=irf_config["config"][
                    "plenoscope_pointing"
                ],
            )
            true_x = -ebin["core.core_x_m"]
            true_y = -ebin["core.core_y_m"]

            delta_c = np.hypot(ebin["cx"] - true_cx, ebin["cy"] - true_cy)
            delta_c_deg = np.rad2deg(delta_c)

            rrr = irf.analysis.gamma_direction.histogram_point_spread_function(
                delta_c_deg=delta_c_deg,
                theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
                psf_containment_factor=psf_containment_factor,
            )
            t2hist.append(rrr["delta_hist"])
            t2hist_rel_unc.append(rrr["delta_hist_relunc"])
            containment_vs_energy.append(rrr["containment_angle_deg"])
            containment_rel_unc_vs_energy.append(
                rrr["containment_angle_deg_relunc"]
            )

        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "theta_square_histogram_vs_energy.json"
            ),
            {
                "comment": ("Theta-square-histogram VS energy"),
                "energy_bin_edges_GeV": energy_bin_edges,
                "theta_square_bin_edges_deg2": theta_square_bin_edges_deg2,
                "unit": "1",
                "mean": t2hist,
                "relative_uncertainty": t2hist_rel_unc,
            },
        )

        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "containment_angle_vs_energy.json"
            ),
            {
                "comment": ("Containment-angle, true gamma-rays, VS energy"),
                "energy_bin_edges_GeV": energy_bin_edges,
                "unit": "deg",
                "mean": containment_vs_energy,
                "relative_uncertainty": containment_rel_unc_vs_energy,
            },
        )

        fix_onregion_radius_deg = irf.analysis.gamma_direction.estimate_fix_opening_angle_for_onregion(
            energy_bin_centers_GeV=energy_bin_centers,
            point_spread_function_containment_opening_angle_deg=containment_vs_energy,
            pivot_energy_GeV=pivot_energy_GeV,
        )

        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "containment_angle_for_fix_onregion.json"
            ),
            {
                "comment": ("Containment-angle, for the fix onregion"),
                "containment_angle": fix_onregion_radius_deg,
                "unit": "deg",
            },
        )
