#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import pandas
import plenopy as pl

"""
Objective
=========

Quantify the angular resolution of the plenoscope.

Input
-----
- List of reconstructed gamma-ray-directions
- List of true gamma-ray-directions, energy, and more...

Quantities
----------
- Containment theta 68%
- theta parallel component
- theta perpendicular component
- Full width at half maximum

histogram theta2
----------------
- in energy
- in core radius

"""


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

fig_16_by_9 = sum_config["plot"]["16_by_9"]

# energy
# ------
num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "point_spread_function"
]
energy_lower_edge = sum_config["energy_binning"]["lower_edge_GeV"]
energy_upper_edge = sum_config["energy_binning"]["upper_edge_GeV"]
energy_bin_edges = np.geomspace(
    energy_lower_edge, energy_upper_edge, num_energy_bins + 1
)
energy_bin_centers = irf.summary.bin_centers(energy_bin_edges)

num_coarse_energy_bins = np.max([1, num_energy_bins//2])
coarse_energy_bin_edges = np.geomspace(
    energy_lower_edge, energy_upper_edge, num_coarse_energy_bins + 1
)

# core radius bins
# ----------------
num_radius_bins = num_coarse_energy_bins
radius_bin_edges = np.linspace(
    0.0,
    sum_config["point_spread_function"]["core_radius"]["max_radius_m"],
    sum_config["point_spread_function"]["core_radius"]["num_bins"] + 1,
)

# theta square bins
# -----------------
theta_square_bin_edges_deg2 = np.linspace(
    0.0,
    sum_config["point_spread_function"]["theta_square"]["max_angle_deg"] ** 2,
    sum_config["point_spread_function"]["theta_square"]["num_bins"] + 1,
)

psf_containment_factor = sum_config["point_spread_function"][
    "containment_factor"
]
pivot_energy_GeV = sum_config["point_spread_function"]["pivot_energy_GeV"]


# READ reconstruction
# ===================
_rec = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0211_simple_light_field")
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


def make_rectangular_table(
    event_table,
    reconstruction_table,
    plenoscope_pointing
):
    rec_evt_tab = spt.cut_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=reconstruction_table[spt.IDX],
        level_keys=level_keys,
    )
    rec_evt_tab = spt.sort_table_on_common_indices(
        table=rec_evt_tab, common_indices=reconstruction_table[spt.IDX],
    )
    rec_evt_df = spt.make_rectangular_DataFrame(rec_evt_tab)

    et_df = pandas.merge(
        left=pandas.DataFrame(
            {
                spt.IDX: reconstruction_table[spt.IDX],
                "reco_cx": reconstruction_table["cx"],
                "reco_cy": reconstruction_table["cy"],
                "reco_x": reconstruction_table["x"],
                "reco_y": reconstruction_table["y"],
            }
        ),
        right=rec_evt_df,
        on=spt.IDX,
    )

    (
        _true_cx,
        _true_cy,
    ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
        momentum_x_GeV_per_c=et_df["primary.momentum_x_GeV_per_c"],
        momentum_y_GeV_per_c=et_df["primary.momentum_y_GeV_per_c"],
        momentum_z_GeV_per_c=et_df["primary.momentum_z_GeV_per_c"],
        plenoscope_pointing=plenoscope_pointing,
    )
    et_df["true_cx"] = _true_cx
    et_df["true_cy"] = _true_cy
    et_df["true_x"] = -et_df["core.core_x_m"]
    et_df["true_y"] = -et_df["core.core_y_m"]
    et_df["true_r"] = np.hypot(et_df["true_x"], et_df["true_y"])

    # w.r.t. source
    # -------------
    c_para, c_perp = atg.projection.project_light_field_onto_source_image(
        cer_cx_rad=et_df["reco_cx"],
        cer_cy_rad=et_df["reco_cy"],
        cer_x_m=0.0,
        cer_y_m=0.0,
        primary_cx_rad=et_df["true_cx"],
        primary_cy_rad=et_df["true_cy"],
        primary_core_x_m=et_df["true_x"],
        primary_core_y_m=et_df["true_y"],
    )

    et_df["theta_para"] = c_para
    et_df["theta_perp"] = c_perp

    et_df["theta"] = np.hypot(
        et_df["reco_cx"] - et_df["true_cx"],
        et_df["reco_cy"] - et_df["true_cy"]
    )

    return et_df.to_records(index=False)


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

        reconstructed_event_table = make_rectangular_table(
            event_table=event_table,
            reconstruction_table=reconstruction[sk][pk],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        rectab = reconstructed_event_table

        # Point-spread-function VS. Energy
        # --------------------------------

        hist_ene_rad = {
            "comment": ("Theta-square-histogram VS energy VS core-radius"),
            "energy_bin_edges_GeV": energy_bin_edges,
            "core_radius_bin_edges": radius_bin_edges,
            "theta_square_bin_edges_deg2": theta_square_bin_edges_deg2,
            "unit": "1",
        }

        for the in ["theta", "theta_para", "theta_perp"]:
            hist_ene_rad[the] = {}
            hist_ene_rad[the]["mean"] = []
            hist_ene_rad[the]["relative_uncertainty"] = []

            for ene in range(num_energy_bins):
                ene_mask = np.logical_and(
                    rectab["primary.energy_GeV"] >= energy_bin_edges[ene],
                    rectab["primary.energy_GeV"] < energy_bin_edges[ene + 1],
                )

                hist_ene_rad[the]["mean"].append([])
                hist_ene_rad[the]["relative_uncertainty"].append([])

                for rad in range(num_radius_bins):
                    rad_mask = np.logical_and(
                        rectab["primary.energy_GeV"] >= radius_bin_edges[rad],
                        rectab["primary.energy_GeV"] < radius_bin_edges[rad + 1],
                    )

                    ene_rad_mask = np.logical_and(ene_mask, rad_mask)

                    theta_deg = np.rad2deg(rectab[the][ene_rad_mask])
                    theta_deg = np.abs(theta_deg)
                    hi = irf.analysis.gamma_direction.histogram_theta_square(
                        theta_deg=theta_deg,
                        theta_square_bin_edges_deg2=theta_square_bin_edges_deg2,
                    )

                    hist_ene_rad[the]["mean"][ene].append(hi[0])
                    hist_ene_rad[the]["relative_uncertainty"][ene].append(hi[1])

        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "theta_square_histogram_vs_energy_vs_core_radius.json"
            ),
            hist_ene_rad,
        )


        # containment angle 68
        # --------------------
        cont_ene = {
            "comment": ("Containment-angle 68percent VS energy"),
            "energy_bin_edges_GeV": energy_bin_edges,
            "unit": "deg",
        }
        for the in ["theta", "theta_para", "theta_perp"]:
            cont_ene[the] = {"mean": [], "relative_uncertainty": []}
            for ene in range(num_energy_bins):
                ene_mask = np.logical_and(
                    rectab["primary.energy_GeV"] >= energy_bin_edges[ene],
                    rectab["primary.energy_GeV"] < energy_bin_edges[ene + 1],
                )
                theta_deg = np.rad2deg(rectab[the][ene_mask])
                theta_deg = np.abs(theta_deg)
                ca = irf.analysis.gamma_direction.estimate_containment_radius(
                    theta_deg=theta_deg,
                    psf_containment_factor=psf_containment_factor)
                cont_ene[the]["mean"].append(ca[0])
                cont_ene[the]["relative_uncertainty"].append(ca[1])
        irf.json_numpy.write(
            os.path.join(
                site_particle_dir, "containment_angle_vs_energy.json"
            ),
            cont_ene,
        )
