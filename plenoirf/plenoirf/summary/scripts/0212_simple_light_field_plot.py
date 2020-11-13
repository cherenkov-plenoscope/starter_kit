#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import pandas
import plenopy as pl
import glob

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

ib = atg.model.IMAGE_BINNING
c_bin_edges = np.linspace(
    -ib["radius_deg"], ib["radius_deg"], ib["num_bins"] + 1
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

        reco = []
        for event in run:
            airshower_id, loph = event

            true_cx = truth_by_index[sk][pk][airshower_id]["cx"]
            true_cy = truth_by_index[sk][pk][airshower_id]["cy"]
            true_x = truth_by_index[sk][pk][airshower_id]["x"]
            true_y = truth_by_index[sk][pk][airshower_id]["y"]

            slf = atg.model.SplitLightField(
                loph_record=loph, light_field_geometry=lfg
            )
            img = atg.model.make_image(
                split_light_field=slf,
                model_config=atg.model.MODEL_CONFIG,
                image_binning=ib
            )
            reco_cx_deg, reco_cy_deg = atg.model.argmax_image_cx_cy_deg(
                image=img, image_binning=ib)

            scale = 1.5
            fig = plt.figure(figsize=(16 / scale, 9 / scale), dpi=100 * scale)
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            for pax in range(slf.number_paxel):
                ax.plot(
                    np.rad2deg(slf.image_sequences[pax][:, 0]),
                    np.rad2deg(slf.image_sequences[pax][:, 1]),
                    "xb",
                    alpha=0.03,
                )

            ax.pcolor(c_bin_edges, c_bin_edges, img, cmap="Reds")
            phi = np.linspace(0, 2 * np.pi, 1000)
            ax.plot(
                fov_radius_deg * np.cos(phi), fov_radius_deg * np.sin(phi), "k"
            )

            ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "xk")

            ax.plot(reco_cx_deg, reco_cy_deg, "og")

            info_str = "reco. Cherenkov: {: 4d}p.e.".format(
                loph["photons"]["channels"].shape[0],
            )
            info_str += "\n"
            info_str += "core x {: 5.1f}m, y {: 5.1f}m, r {: 5.1f}m".format(
                true_x, true_y, np.hypot(true_x, true_y)
            )

            ax.set_title(info_str)

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
