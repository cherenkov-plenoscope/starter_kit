#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
import plenopy as pl
import glob
import scipy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

loph_chunk_base_dir = os.path.join(
    pa["summary_dir"], "0068_prepare_loph_passed_trigger_and_quality"
)

# READ light-field-geometry
# =========================
lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

fov_radius_deg = np.rad2deg(
    0.5 * lfg.sensor_plane2imaging_system.max_FoV_diameter
)

fuzzy_binning = pl.fuzzy.direction.EXAMPLE_IMAGE_BINNING
fuzzy_c_bin_edges = np.linspace(
    -fuzzy_binning["radius_deg"],
    fuzzy_binning["radius_deg"],
    fuzzy_binning["num_bins"] + 1
)

fig_16_by_9 = sum_config["plot"]["16_by_9"]


truth_by_index = {}
for sk in irf_config["config"]["sites"]:
    truth_by_index[sk] = {}
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:
        truth_by_index[sk][pk] = {}

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        passed_trigger_idx = np.array(
            _passed_trigger_indices[sk][pk]["passed_trigger"][spt.IDX]
        )

        all_truth = spt.cut_table_on_indices(
            event_table,
            irf.table.STRUCTURE,
            common_indices=passed_trigger_idx,
            level_keys=[
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
            ],
        )
        all_truth = spt.sort_table_on_common_indices(
            table=all_truth, common_indices=passed_trigger_idx
        )
        (
            true_cx,
            true_cy,
        ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
            momentum_x_GeV_per_c=all_truth["primary"]["momentum_x_GeV_per_c"],
            momentum_y_GeV_per_c=all_truth["primary"]["momentum_y_GeV_per_c"],
            momentum_z_GeV_per_c=all_truth["primary"]["momentum_z_GeV_per_c"],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        for ii in range(all_truth["primary"].shape[0]):
            airshower_id = all_truth["primary"][spt.IDX][ii]
            truth_by_index[sk][pk][airshower_id] = {
                "cx": true_cx[ii],
                "cy": true_cy[ii],
                "x": -all_truth["core"]["core_x_m"][ii],
                "y": -all_truth["core"]["core_y_m"][ii],
            }

fuzz_ring_gaussian_kernel = pl.fuzzy.discrete_kernel.gauss1d(num_steps=61)
fuzz_img_gaussian_kernel = pl.fuzzy.discrete_kernel.gauss2d(num_steps=5)

scale = 1.5

fuzzy_model_config = pl.fuzzy.direction.EXAMPLE_MODEL_CONFIG

for sk in irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:

        loph_chunk_paths = glob.glob(
            os.path.join(loph_chunk_base_dir, sk, pk, "chunks", "*.tar")
        )

        run = pl.photon_stream.loph.LopfTarReader(
            loph_chunk_paths[0]
        )

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        for event in run:
            airshower_id, loph = event

            true_cx = truth_by_index[sk][pk][airshower_id]["cx"]
            true_cy = truth_by_index[sk][pk][airshower_id]["cy"]
            true_x = truth_by_index[sk][pk][airshower_id]["x"]
            true_y = truth_by_index[sk][pk][airshower_id]["y"]

            slf = pl.fuzzy.direction.SplitLightField(
                loph_record=loph, light_field_geometry=lfg
            )

            if slf.number_photons < 150:
                continue

            slf_model = pl.fuzzy.direction.estimate_model_from_light_field(
                split_light_field=slf, model_config=fuzzy_model_config
            )
            fuzz_img = pl.fuzzy.direction.make_image_from_model(
                light_field_model=slf_model,
                model_config=fuzzy_model_config,
                image_binning=fuzzy_binning
            )

            smooth_fuzz_img = scipy.signal.convolve2d(
                in1=fuzz_img,
                in2=fuzz_img_gaussian_kernel,
                mode="same"
            )

            reco_cx_deg, reco_cy_deg = pl.fuzzy.direction.argmax_image_cx_cy_deg(
                image=smooth_fuzz_img, image_binning=fuzzy_binning,
            )
            med_cx_deg = np.rad2deg(slf.median_cx)
            med_cy_deg = np.rad2deg(slf.median_cy)
            azimuth_main_axis = np.arctan2(
                (reco_cy_deg - med_cy_deg),
                (reco_cx_deg - med_cx_deg),
            )

            ring = pl.fuzzy.direction.project_image_onto_ring(
                image=smooth_fuzz_img,
                image_binning=fuzzy_binning,
                ring_cx_deg=med_cx_deg,
                ring_cy_deg=med_cy_deg,
                ring_radius_deg=1.5
            )
            smooth_ring = pl.fuzzy.direction.circular_convolve1d(
                in1=ring, in2=fuzz_ring_gaussian_kernel
            )
            smooth_ring /= np.max(smooth_ring)

            azi_maxima = pl.fuzzy.direction.circular_argmaxima(smooth_ring)
            azi_maxima_weights = [smooth_ring[mm] for mm in azi_maxima]

            azimuth_main2_axis = np.deg2rad(np.argmax(smooth_ring))

            fuzzy_result = {
                "median_cx_deg": med_cx_deg,
                "median_cy_deg": med_cy_deg,
                "main_axis_azimuth_deg": np.rad2deg(azimuth_main2_axis),
                "reco_cx_deg": reco_cx_deg,
                "reco_cy_deg": reco_cy_deg,
            }

            fig_ring = plt.figure(figsize=(16 / scale, 9 / scale), dpi=100 * scale)
            ax_ring = fig_ring.add_axes([0.1, 0.1, 0.8, 0.8])
            ax_ring.plot(smooth_ring, "k")
            for yy in range(len(azi_maxima)):
                ax_ring.plot(azi_maxima[yy], azi_maxima_weights[yy], "or")
            ax_ring.set_xlim([0, 360])
            ax_ring.set_ylim([0.0, 1.0])
            ax_ring.set_xlabel("main-axis-azimuth / deg")
            ax_ring.set_ylabel("probability density / deg$^{-1}$")
            ax_ring.spines["top"].set_color("none")
            ax_ring.spines["right"].set_color("none")
            ax_ring.spines["bottom"].set_color("none")
            ax_ring.spines["left"].set_color("none")
            ax_ring.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
            path_ring = os.path.join(
                pa["out_dir"], sk, pk, "{:09d}_ring.jpg".format(airshower_id),
            )
            fig_ring.savefig(path_ring)
            plt.close(fig_ring)



            fig = plt.figure(figsize=(16 / scale, 9 / scale), dpi=100 * scale)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            for pax in range(slf.number_paxel):
                ax.plot(
                    np.rad2deg(slf.image_sequences[pax][:, 0]),
                    np.rad2deg(slf.image_sequences[pax][:, 1]),
                    "xb",
                    alpha=0.03,
                )
            ax.pcolor(
                fuzzy_c_bin_edges,
                fuzzy_c_bin_edges,
                smooth_fuzz_img,
                cmap="Reds"
            )
            phi = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(
                fov_radius_deg * np.cos(phi), fov_radius_deg * np.sin(phi), "k"
            )
            ax.plot(
                [med_cx_deg, med_cx_deg + 100*np.cos(azimuth_main2_axis)],
                [med_cy_deg, med_cy_deg + 100*np.sin(azimuth_main2_axis)],
                ":b"
            )
            ax.plot(reco_cx_deg, reco_cy_deg, "og")
            ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "xk")

            info_str = "reco. Cherenkov: {: 4d}p.e.".format(
                loph["photons"]["channels"].shape[0],
            )
            info_str += "\n"
            info_str += "core x {: 5.1f}m, y {: 5.1f}m, r {: 5.1f}m".format(
                true_x, true_y, np.hypot(true_x, true_y)
            )

            ax.set_title(info_str)

            ax.set_xlim([-1.05*fov_radius_deg, 1.05*fov_radius_deg])
            ax.set_ylim([-1.05*fov_radius_deg, 1.05*fov_radius_deg])
            ax.set_aspect("equal")
            ax.set_xlabel("cx / deg")
            ax.set_ylabel("cy / deg")
            ax.spines["top"].set_color("none")
            ax.spines["right"].set_color("none")
            ax.spines["bottom"].set_color("none")
            ax.spines["left"].set_color("none")
            ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

            path = os.path.join(
                pa["out_dir"], sk, pk, "{:09d}.jpg".format(airshower_id),
            )

            fig.savefig(path)
            plt.close(fig)
