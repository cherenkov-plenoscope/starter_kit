#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import airshower_template_generator as atg
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
    fuzzy_binning["num_bins"] + 1,
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

kernel_3x3 = pl.fuzzy.discrete_kernel.gauss2d(num_steps=3)

fuzzy_model_config = pl.fuzzy.direction.EXAMPLE_MODEL_CONFIG

long_fit_cfg = {
    "c_para": {
        "start": np.deg2rad(-4.0),
        "stop": np.deg2rad(4.0),
        "num_supports": 128
    },
    "r_para": {
        "start": -640,
        "stop": 640,
        "num_supports": 96,
        "num_bins_scan_radius": 5,
    },
    "c_perp_width": np.deg2rad(0.05)
}

def read_shower_maximum_object_distance(
    site_key,
    particle_key,
    key="image_smallest_ellipse_object_distance"
):
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
    features = event_table["features"]
    out = {}
    for ii in range(features.shape[0]):
        out[features[spt.IDX][ii]] = features[key][ii]
    return out


class CoreRadiusFinder:
    def __init__(
        self,
        main_axis_azimuth,
        median_cx,
        median_cy,
        cx,
        cy,
        x,
        y
    ):
        self.main_axis_azimuth = main_axis_azimuth
        self.cx = cx
        self.cy = cy
        self.x = x
        self.y = y
        self.median_cx = median_cx
        self.median_cy = median_cy

    def _source_direction_cx_cy(self, c_para):
        source_cx = self.median_cx + np.cos(self.main_axis_azimuth) * c_para
        source_cy = self.median_cy + np.sin(self.main_axis_azimuth) * c_para
        return source_cx, source_cy

    def _core_position_x_y(self, r_para):
        core_x = 0.0 + np.cos(self.main_axis_azimuth) * r_para
        core_y = 0.0 + np.sin(self.main_axis_azimuth) * r_para
        return core_x, core_y

    def project_light_field_on_para_perp(self, c_para, r_para):
        source_cx, source_cy = self._source_direction_cx_cy(c_para=c_para)
        core_x, core_y = self._core_position_x_y(r_para=r_para)

        WRT_DOWNWARDS = -1.0
        c_para, c_perp = atg.projection.project_light_field_onto_source_image(
            cer_cx_rad=WRT_DOWNWARDS * self.cx,
            cer_cy_rad=WRT_DOWNWARDS * self.cy,
            cer_x_m=self.x,
            cer_y_m=self.y,
            primary_cx_rad=WRT_DOWNWARDS * source_cx,
            primary_cy_rad=WRT_DOWNWARDS * source_cy,
            primary_core_x_m=core_x,
            primary_core_y_m=core_y,
        )
        return c_para, c_perp

    def response(self, c_para, r_para, cer_perp_distance_threshold):
        cer_c_para, cer_c_perp = self.project_light_field_on_para_perp(
            c_para,
            r_para
        )

        num = len(cer_c_perp)

        l_trans_max = atg.model.lorentz_transversal(
            c_deg=0.0, peak_deg=0.0, width_deg=cer_perp_distance_threshold
        )
        l_trans = atg.model.lorentz_transversal(
            c_deg=cer_c_perp, peak_deg=0.0, width_deg=cer_perp_distance_threshold
        )
        l_trans /= l_trans_max

        perp_weight = np.sum(l_trans) / num

        return perp_weight


def angle_between(v1, v2):

    def unit_vector(vector):
        return vector / np.linalg.norm(vector)

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def squarespace(start, stop, num):
    sqrt_space = np.linspace(
        np.sign(start) * np.sqrt(np.abs(start)),
        np.sign(stop) * np.sqrt(np.abs(stop)),
        num
    )
    signs = np.sign(sqrt_space)
    square_space = sqrt_space ** 2
    square_space *= signs
    return square_space


def matching_core_radius(c_para, epsilon, m):
    rrr = c_para - 0.5 * np.pi + epsilon
    out =  m * (np.cos(epsilon) + np.sin(epsilon) * np.tan(rrr) )
    return -1.0 * out


for sk in irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:

        reco_obj = read_shower_maximum_object_distance(
            site_key=sk,
            particle_key=pk
        )

        loph_chunk_paths = glob.glob(
            os.path.join(loph_chunk_base_dir, sk, pk, "chunks", "*.tar")
        )

        run = pl.photon_stream.loph.LopfTarReader(loph_chunk_paths[0])

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        for event in run:
            airshower_id, loph_record = event

            true_cx = truth_by_index[sk][pk][airshower_id]["cx"]
            true_cy = truth_by_index[sk][pk][airshower_id]["cy"]
            true_x = truth_by_index[sk][pk][airshower_id]["x"]
            true_y = truth_by_index[sk][pk][airshower_id]["y"]

            slf = pl.fuzzy.direction.SplitLightField(
                loph_record=loph_record, light_field_geometry=lfg
            )

            if slf.number_photons < 150:
                continue

            slf_model = pl.fuzzy.direction.estimate_model_from_light_field(
                split_light_field=slf, model_config=fuzzy_model_config
            )
            fuzz_img = pl.fuzzy.direction.make_image_from_model(
                light_field_model=slf_model,
                model_config=fuzzy_model_config,
                image_binning=fuzzy_binning,
            )

            smooth_fuzz_img = scipy.signal.convolve2d(
                in1=fuzz_img, in2=fuzz_img_gaussian_kernel, mode="same"
            )

            (
                reco_cx_deg,
                reco_cy_deg,
            ) = pl.fuzzy.direction.argmax_image_cx_cy_deg(
                image=smooth_fuzz_img, image_binning=fuzzy_binning,
            )
            med_cx_deg = np.rad2deg(slf.median_cx)
            med_cy_deg = np.rad2deg(slf.median_cy)

            azimuth_ring = pl.fuzzy.direction.project_image_onto_ring(
                image=smooth_fuzz_img,
                image_binning=fuzzy_binning,
                ring_cx_deg=med_cx_deg,
                ring_cy_deg=med_cy_deg,
                ring_radius_deg=1.5,
            )
            smooth_azimuth_ring = pl.fuzzy.direction.circular_convolve1d(
                in1=azimuth_ring, in2=fuzz_ring_gaussian_kernel
            )
            smooth_azimuth_ring /= np.max(smooth_azimuth_ring)

            azi_maxima = pl.fuzzy.direction.circular_argmaxima(smooth_azimuth_ring)
            azi_maxima_weights = [smooth_azimuth_ring[mm] for mm in azi_maxima]
            main_axis_azimuth = np.deg2rad(np.argmax(smooth_azimuth_ring))

            fuzzy_result = {
                "median_cx_deg": med_cx_deg,
                "median_cy_deg": med_cy_deg,
                "main_axis_azimuth_deg": np.rad2deg(main_axis_azimuth),
                "reco_cx_deg": reco_cx_deg,
                "reco_cy_deg": reco_cy_deg,
            }

            main_axis_azimuth_offsets_deg = np.linspace(-5, 5, 21)

            longi_fits = []
            for num_longi_fit_iterations in range(21):

                longi_fit = {}
                longi_fit["main_axis_azimuth_deg"] = (
                    fuzzy_result["main_axis_azimuth_deg"] +
                    main_axis_azimuth_offsets_deg[num_longi_fit_iterations]
                )
                longi_fit["median_cx_deg"] = fuzzy_result["median_cx_deg"]
                longi_fit["median_cy_deg"] = fuzzy_result["median_cy_deg"]
                # longitudinal fit
                # ================

                lixel_ids = loph_record["photons"]["channels"]
                crf = CoreRadiusFinder(
                    main_axis_azimuth=np.deg2rad(longi_fit["main_axis_azimuth_deg"]),
                    median_cx=np.deg2rad(longi_fit["median_cx_deg"]),
                    median_cy=np.deg2rad(longi_fit["median_cy_deg"]),
                    cx=lfg.cx_mean[lixel_ids],
                    cy=lfg.cy_mean[lixel_ids],
                    x=lfg.x_mean[lixel_ids],
                    y=lfg.y_mean[lixel_ids],
                )

                c_para_supports = squarespace(
                    start=long_fit_cfg["c_para"]["start"],
                    stop=long_fit_cfg["c_para"]["stop"],
                    num=long_fit_cfg["c_para"]["num_supports"],
                )

                r_para_supports = squarespace(
                    start=long_fit_cfg["r_para"]["start"],
                    stop=long_fit_cfg["r_para"]["stop"],
                    num=long_fit_cfg["r_para"]["num_supports"],
                )

                # mask c_para r_para
                # ------------------

                shower_max_z = reco_obj[airshower_id]
                shower_median_direction_z = np.sqrt(
                    1.0 -
                    slf.median_cx ** 2 -
                    slf.median_cy ** 2
                )
                distance_aperture_center_to_shower_maximum = (
                    shower_max_z / shower_median_direction_z
                )

                shower_median_direction = [
                    slf.median_cx,
                    slf.median_cy,
                    shower_median_direction_z
                ]

                core_axis_direction = [
                    np.cos(main_axis_azimuth),
                    np.sin(main_axis_azimuth),
                    0.0
                ]

                epsilon = angle_between(
                    shower_median_direction,
                    core_axis_direction
                )

                c_para_r_para_mask = np.zeros(
                    shape=(
                        long_fit_cfg["c_para"]["num_supports"],
                        long_fit_cfg["r_para"]["num_supports"]
                    ),
                    dtype=np.int
                )

                for cbin, c_para in enumerate(c_para_supports):
                    matching_r_para = matching_core_radius(
                        c_para=c_para,
                        epsilon=epsilon,
                        m=distance_aperture_center_to_shower_maximum
                    )

                    closest_r_para_bin = np.argmin(
                        np.abs(r_para_supports - matching_r_para)
                    )

                    if (
                        closest_r_para_bin > 0 and
                        closest_r_para_bin < (long_fit_cfg["r_para"]["num_supports"] - 1)
                    ):
                        rbin_range = np.arange(
                            closest_r_para_bin - long_fit_cfg["r_para"]["num_bins_scan_radius"],
                            closest_r_para_bin + long_fit_cfg["r_para"]["num_bins_scan_radius"]
                        )

                        for rbin in rbin_range:
                            if rbin >= 0 and rbin < long_fit_cfg["r_para"]["num_supports"]:
                                c_para_r_para_mask[cbin, rbin] = 1


                response = np.zeros(
                    shape=(
                        long_fit_cfg["c_para"]["num_supports"],
                        long_fit_cfg["r_para"]["num_supports"]
                    )
                )
                for cbin, c_para in enumerate(c_para_supports):
                    for rbin, r_para in enumerate(r_para_supports):

                        if c_para_r_para_mask[cbin, rbin]:
                            response[cbin, rbin] = crf.response(
                                c_para=c_para,
                                r_para=r_para,
                                cer_perp_distance_threshold=long_fit_cfg["c_perp_width"],
                            )

                argmax_c_para, argmax_r_para = pl.fuzzy.direction.argmax2d(
                    response
                )
                max_c_para = c_para_supports[argmax_c_para]
                max_r_para = r_para_supports[argmax_r_para]
                max_response = response[argmax_c_para, argmax_r_para]

                reco_fit_cx_deg = (
                    longi_fit["median_cx_deg"] +
                    np.cos(np.deg2rad(longi_fit["main_axis_azimuth_deg"])) *
                    np.rad2deg(max_c_para)
                )
                reco_fit_cy_deg = (
                    longi_fit["median_cy_deg"] +
                    np.sin(np.deg2rad(longi_fit["main_axis_azimuth_deg"])) *
                    np.rad2deg(max_c_para)
                )

                reco_fit_x = (
                    np.cos(np.deg2rad(longi_fit["main_axis_azimuth_deg"])) *
                    max_r_para
                )

                reco_fit_y = (
                    np.sin(np.deg2rad(longi_fit["main_axis_azimuth_deg"])) *
                    max_r_para
                )

                longi_fit["c_para"] = float(max_c_para)
                longi_fit["r_para"] = float(max_r_para)
                longi_fit["response"] = float(max_response)

                longi_fit["cx_deg"] = float(reco_fit_cx_deg)
                longi_fit["cy_deg"] = float(reco_fit_cy_deg)
                longi_fit["x_m"] = float(reco_fit_x)
                longi_fit["y_m"] = float(reco_fit_y)

                longi_fits.append(longi_fit)

                # end longitudinal fit
                # --------------------

                #####

                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                inte = np.log10(-1.0 * (response - np.max(response)))
                ax.pcolor(
                    np.rad2deg(c_para_supports),
                    r_para_supports,
                    inte.T,
                    cmap="Blues",
                )
                ax.plot(np.rad2deg(max_c_para), max_r_para, "og")

                ax.set_xlabel("c_para / deg")
                ax.set_ylabel("r_para / m")
                ax.spines["top"].set_color("none")
                ax.spines["right"].set_color("none")
                ax.spines["bottom"].set_color("none")
                ax.spines["left"].set_color("none")
                ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

                path = os.path.join(
                    pa["out_dir"], sk, pk, "{:09d}_{:03d}_resp.jpg".format(
                        airshower_id,
                        num_longi_fit_iterations
                    )
                )

                fig.savefig(path)
                plt.close(fig)

                fig_ring = irf.summary.figure.figure(fig_16_by_9)
                ax_ring = fig_ring.add_axes([0.1, 0.1, 0.8, 0.8])
                ax_ring.plot(smooth_azimuth_ring, "k")
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

                # -------------

                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.075, 0.1, 0.4, 0.8])
                ax_core = fig.add_axes([0.575, 0.1, 0.4, 0.8])
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
                    cmap="Reds",
                )
                phi = np.linspace(0, 2 * np.pi, 1000)
                ax.plot(
                    fov_radius_deg * np.cos(phi), fov_radius_deg * np.sin(phi), "k"
                )
                ax.plot(
                    [med_cx_deg, med_cx_deg + 100 * np.cos(np.deg2rad(longi_fit["main_axis_azimuth_deg"]))],
                    [med_cy_deg, med_cy_deg + 100 * np.sin(np.deg2rad(longi_fit["main_axis_azimuth_deg"]))],
                    ":b",
                )
                ax.plot(reco_cx_deg, reco_cy_deg, "og")

                ax.plot(reco_fit_cx_deg, reco_fit_cy_deg, "oc")

                ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "xk")

                info_str = "reco. Cherenkov: {: 4d}p.e.\n response: {:.3f}".format(
                    loph_record["photons"]["channels"].shape[0],
                    longi_fit["response"],
                )

                ax.set_title(info_str)

                ax.set_xlim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_ylim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_aspect("equal")
                ax.set_xlabel("cx / deg")
                ax.set_ylabel("cy / deg")
                ax.spines["top"].set_color("none")
                ax.spines["right"].set_color("none")
                ax.spines["bottom"].set_color("none")
                ax.spines["left"].set_color("none")
                ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

                ax_core.plot(reco_fit_x, reco_fit_y, "oc")
                ax_core.plot([0, reco_fit_x], [0, reco_fit_y], "c", alpha=0.5)

                ax_core.plot(true_x, true_y, "xk")
                ax_core.plot([0, true_x], [0, true_y], "k", alpha=0.5)

                ax_core.set_xlim([-640, 640])
                ax_core.set_ylim([-640, 640])
                ax_core.set_aspect("equal")
                ax_core.set_xlabel("x / m")
                ax_core.set_ylabel("y / m")
                ax_core.spines["top"].set_color("none")
                ax_core.spines["right"].set_color("none")
                ax_core.spines["bottom"].set_color("none")
                ax_core.spines["left"].set_color("none")
                ax_core.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)

                path = os.path.join(
                    pa["out_dir"], sk, pk, "{:09d}_{:03d}.jpg".format(
                        airshower_id,
                        num_longi_fit_iterations
                    ),
                )

                fig.savefig(path)
                plt.close(fig)
