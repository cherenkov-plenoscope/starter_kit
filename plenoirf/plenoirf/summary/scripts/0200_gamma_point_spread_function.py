#!/usr/bin/python
import sys
import numpy as np
import magnetic_deflection as mdfl
import pandas
import plenoirf as irf
import sparse_numeric_table as spt
import os
from os.path import join as opj

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

MAX_SOURCE_ANGLE_DEG = sum_config["gamma_ray_source_direction"][
    "max_angle_relative_to_pointing_deg"
]
pointing_azimuth_deg = irf_config["config"]["plenoscope_pointing"][
    "azimuth_deg"
]
pointing_zenith_deg = irf_config["config"]["plenoscope_pointing"]["zenith_deg"]
trigger_threshold = sum_config["trigger"]["threshold_pe"]
trigger_modus = sum_config["trigger"]["modus"]

num_energy_bins = sum_config["energy_binning"]["num_bins"][
    "point_spread_function"
]
energy_bin_edges = np.geomspace(
    sum_config["energy_binning"]["lower_edge_GeV"],
    sum_config["energy_binning"]["upper_edge_GeV"],
    num_energy_bins + 1,
)
energy_bin_centers = irf.summary.bin_centers(energy_bin_edges)

max_relative_leakage = sum_config["quality"]["max_relative_leakage"]
min_reconstructed_photons = sum_config["quality"]["min_reconstructed_photons"]

theta_square_bin_edges_deg2 = np.linspace(
    0,
    sum_config["point_spread_function"]["theta_square"]["max_angle_deg"] ** 2,
    sum_config["point_spread_function"]["theta_square"]["num_bins"],
)
psf_containment_factor = sum_config["point_spread_function"][
    "containment_factor"
]
pivot_energy_GeV = sum_config["point_spread_function"]["pivot_energy_GeV"]

for site_key in irf_config["config"]["sites"]:
    site_particle_dir = opj(pa["out_dir"], site_key, "gamma")
    os.makedirs(site_particle_dir, exist_ok=True)

    diffuse_gamma_table = spt.read(
        path=opj(
            pa["run_dir"], "event_table", site_key, "gamma", "event_table.tar"
        ),
        structure=irf.table.STRUCTURE,
    )

    idx_passed_trigger = irf.analysis.light_field_trigger_modi.make_indices(
        trigger_table=diffuse_gamma_table["trigger"],
        threshold=trigger_threshold,
        modus=trigger_modus,
    )

    idx_onregion = irf.analysis.cuts.cut_primary_direction_within_angle(
        primary_table=diffuse_gamma_table["primary"],
        radial_angle_deg=MAX_SOURCE_ANGLE_DEG,
        azimuth_deg=pointing_azimuth_deg,
        zenith_deg=pointing_zenith_deg,
    )

    idx_quality = irf.analysis.cuts.cut_quality(
        feature_table=diffuse_gamma_table["features"],
        max_relative_leakage=max_relative_leakage,
        min_reconstructed_photons=min_reconstructed_photons,
    )

    psf_vs_energy = []
    reconstruction_table = []
    for energy_bin in range(num_energy_bins):
        idx_energy_bin = irf.analysis.cuts.cut_energy_bin(
            primary_table=diffuse_gamma_table["primary"],
            lower_energy_edge_GeV=energy_bin_edges[energy_bin],
            upper_energy_edge_GeV=energy_bin_edges[energy_bin + 1],
        )

        idx_gammas = spt.intersection(
            [idx_passed_trigger, idx_onregion, idx_quality, idx_energy_bin,]
        )

        num_airshower = len(idx_gammas)
        gamma_table = spt.cut_table_on_indices(
            table=diffuse_gamma_table,
            structure=irf.table.STRUCTURE,
            common_indices=idx_gammas,
            level_keys=None,
        )
        gamma_table = spt.sort_table_on_common_indices(
            table=gamma_table, common_indices=idx_gammas
        )

        gt = gamma_table
        psf_deg = []
        for evt in range(gamma_table["features"].shape[0]):

            _true_cx, _true_cy = mdfl.discovery._az_zd_to_cx_cy(
                azimuth_deg=np.rad2deg(gt["primary"]["azimuth_rad"][evt]),
                zenith_deg=np.rad2deg(gt["primary"]["zenith_rad"][evt]),
            )

            _rec_cx, _rec_cy = irf.analysis.gamma_direction.estimate(
                light_front_cx=gt["features"]["light_front_cx"][evt],
                light_front_cy=gt["features"]["light_front_cy"][evt],
                image_infinity_cx_mean=gt["features"][
                    "image_infinity_cx_mean"
                ][evt],
                image_infinity_cy_mean=gt["features"][
                    "image_infinity_cy_mean"
                ][evt],
            )

            _delta_cx = _true_cx - _rec_cx
            _delta_cy = _true_cy - _rec_cy
            _delta_cx_deg = np.rad2deg(_delta_cx)
            _delta_cy_deg = np.rad2deg(_delta_cy)
            psf_deg.append([_delta_cx_deg, _delta_cy_deg])

            reconstruction_table.append(
                {
                    spt.IDX: gt["primary"][spt.IDX][evt],
                    "cx": _rec_cx,
                    "cy": _rec_cy,
                    "x": float("nan"),
                    "y": float("nan"),
                }
            )

        psf_vs_energy.append(psf_deg)

    reco_df = pandas.DataFrame(reconstruction_table)
    reco_di = reco_df.to_dict(orient="list")
    irf.json_numpy.write(
        path=opj(site_particle_dir, "reco.json"),
        out_dict=reco_di,
    )

    irf.json_numpy.write(
        opj(site_particle_dir, "point_spread_distribution_vs_energy.json"),
        {
            "comment": (
                "The deviation (delta) between true and "
                "reconstructed source direction for true gamma-rays. "
                "VS energy"
            ),
            "energy_bin_edges_GeV": energy_bin_edges,
            "unit": "deg",
            "deviation": psf_vs_energy,
        },
        indent=None,
    )

    t2hist = []
    t2hist_rel_unc = []

    containment_vs_energy = []
    containment_rel_unc_vs_energy = []

    for energy_bin in range(num_energy_bins):
        psf_deg = np.array(psf_vs_energy[energy_bin])

        rrr = irf.analysis.gamma_direction.histogram_point_spread_function(
            delta_c_deg=np.hypot(psf_deg[:, 0], psf_deg[:, 1]),
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
        opj(site_particle_dir, "theta_square_histogram_vs_energy.json"),
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
        opj(site_particle_dir, "containment_angle_vs_energy.json"),
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
        opj(site_particle_dir, "containment_angle_for_fix_onregion.json"),
        {
            "comment": ("Containment-angle, for the fix onregion"),
            "containment_angle": fix_onregion_radius_deg,
            "unit": "deg",
        },
    )
