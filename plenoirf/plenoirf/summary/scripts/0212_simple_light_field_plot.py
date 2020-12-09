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
from iminuit import Minuit

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

user_fuzzy_config = {
    "image": {
        "radius_deg": fov_radius_deg + 1.0,
        "num_bins": 128,
        "smoothing_kernel_width_deg": 0.3125,
    },
    "azimuth_ring": {
        "num_bins": 360,
        "radius_deg": 1.,
        "smoothing_kernel_width_deg": 41.0,
    },
    "ellipse_model": {
        "min_num_photons": 3,
    }
}

long_fit_user_config = {
    "c_para": {
        "start_deg": -4.0,
        "stop_deg": 4.0,
        "num_supports": 128,
    },
    "r_para": {
        "start_m": -640,
        "stop_m": 640.0,
        "num_supports": 96,
    },
    "scan": {
        "num_bins_radius": 2,
    },
    "shower_model": {
        "c_perp_width_deg": 0.1,
    },
}

def _fuzzy_compile_user_config(user_config):
    uc = user_config
    cfg = {}

    uimg = uc["image"]
    img = {}
    img["radius"] = np.deg2rad(uimg["radius_deg"])
    img["num_bins"] = uimg["num_bins"]
    img["c_bin_edges"] = np.linspace(
        -img["radius"],
        +img["radius"],
        img["num_bins"] + 1,
    )
    img["c_bin_centers"] = irf.summary.bin_centers(
        bin_edges=img["c_bin_edges"]
    )
    _image_bins_per_rad = img["num_bins"] / (2.0 * img["radius"])
    img["smoothing_kernel_width"] = np.deg2rad(
        uimg["smoothing_kernel_width_deg"]
    )
    img["smoothing_kernel"] = pl.fuzzy.discrete_kernel.gauss2d(
        num_steps=int(
            np.round(
                img["smoothing_kernel_width"] * _image_bins_per_rad
            )
        )
    )
    cfg["image"] = img

    uazr = uc["azimuth_ring"]
    azr = {}
    azr["num_bins"] = uazr["num_bins"]
    azr["bin_edges"] = np.linspace(
        0.0,
        2.0 * np.pi,
        azr["num_bins"],
        endpoint=False
    )
    azr["radius"] = np.deg2rad(uazr["radius_deg"])
    _ring_bins_per_rad = azr["num_bins"] / (2.0 * np.pi)
    azr["smoothing_kernel_width"] = np.deg2rad(
        uazr["smoothing_kernel_width_deg"]
    )
    azr["smoothing_kernel"] = pl.fuzzy.discrete_kernel.gauss1d(
        num_steps=int(
            np.round(
                _ring_bins_per_rad *
                azr["smoothing_kernel_width"]
            )
        )
    )
    cfg["azimuth_ring"] = azr
    cfg["ellipse_model"] = dict(uc["ellipse_model"])
    return cfg


fuzzy_config = _fuzzy_compile_user_config(user_config=user_fuzzy_config)


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


def _compile_user_config(user_config):
    uc = user_config
    cfg = {}
    cfg["c_para"] = {}
    cfg["c_para"]["start"] = np.deg2rad(uc["c_para"]["start_deg"])
    cfg["c_para"]["stop"] = np.deg2rad(uc["c_para"]["stop_deg"])
    cfg["c_para"]["num_supports"] = uc["c_para"]["num_supports"]
    cfg["c_para"]["supports"] = squarespace(
        start=cfg["c_para"]["start"],
        stop=cfg["c_para"]["stop"],
        num=cfg["c_para"]["num_supports"],
    )
    cfg["r_para"] = {}
    cfg["r_para"]["start"] = uc["r_para"]["start_m"]
    cfg["r_para"]["stop"] = uc["r_para"]["stop_m"]
    cfg["r_para"]["num_supports"] = uc["r_para"]["num_supports"]
    cfg["r_para"]["supports"] = squarespace(
        start=cfg["r_para"]["start"],
        stop=cfg["r_para"]["stop"],
        num=cfg["r_para"]["num_supports"],
    )
    cfg["scan"] = dict(uc["scan"])
    cfg["shower_model"] = {}
    cfg["shower_model"]["c_perp_width"] = np.deg2rad(
        uc["shower_model"]["c_perp_width_deg"]
    )
    return cfg


long_fit_cfg = _compile_user_config(long_fit_user_config)


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
                "energy_GeV": all_truth["primary"]["energy_GeV"][ii],
            }


def my_axes_look(ax):
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_color("none")
    ax.spines["left"].set_color("none")
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    return ax


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
        main_axis_support_cx,
        main_axis_support_cy,
        light_field_cx,
        light_field_cy,
        light_field_x,
        light_field_y
    ):
        self.main_axis_azimuth = main_axis_azimuth
        self.cx = light_field_cx
        self.cy = light_field_cy
        self.x = light_field_x
        self.y = light_field_y
        self.main_axis_support_cx = main_axis_support_cx
        self.main_axis_support_cy = main_axis_support_cy

    def _source_direction_cx_cy(self, c_para):
        source_cx = self.main_axis_support_cx + np.cos(self.main_axis_azimuth) * c_para
        source_cy = self.main_axis_support_cy + np.sin(self.main_axis_azimuth) * c_para
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


def matching_core_radius(c_para, epsilon, m):
    rrr = c_para - 0.5 * np.pi + epsilon
    out =  m * (np.cos(epsilon) + np.sin(epsilon) * np.tan(rrr) )
    return -1.0 * out

PLOT_RING = False
PLOT_OVERVIEW = True


def estimate_main_axis_to_core_using_fuzzy_method(
    split_light_field,
    model_config,
    image_binning,
    image_smoothing_kernel,
    ring_binning,
    ring_smoothing_kernel,
):
    median_cx = split_light_field.median_cx
    median_cy = split_light_field.median_cy

    # make fuzzy image
    # ----------------
    # A probability-density for the shower's main-axis and primary particle's
    # direction.

    slf_model = pl.fuzzy.direction.estimate_model_from_light_field(
        split_light_field=split_light_field, model_config=model_config
    )
    fuzzy_image = pl.fuzzy.direction.make_image_from_model(
        split_light_field_model=slf_model,
        image_binning=image_binning,
    )
    fuzzy_image_smooth = scipy.signal.convolve2d(
        in1=fuzzy_image, in2=image_smoothing_kernel, mode="same"
    )
    reco_cx, reco_cy = pl.fuzzy.direction.argmax_image_cx_cy(
        image=fuzzy_image_smooth, image_binning=image_binning,
    )

    median_cx_std = np.std([a["median_cx"] for a in slf_model])
    median_cy_std = np.std([a["median_cy"] for a in slf_model])

    # make ring to find main-axis
    # ---------------------------

    azimuth_ring = pl.fuzzy.direction.project_image_onto_ring(
        image=fuzzy_image_smooth,
        image_binning=image_binning,
        ring_cx=median_cx,
        ring_cy=median_cy,
        ring_radius=ring_binning["radius"],
        ring_binning=ring_binning,
    )
    azimuth_ring_smooth = pl.fuzzy.direction.circular_convolve1d(
        in1=azimuth_ring, in2=ring_smoothing_kernel
    )
    azimuth_ring_smooth /= np.max(azimuth_ring_smooth)

    # analyse ring to find main-axis
    # ------------------------------

    # maximum
    main_axis_azimuth = ring_binning["bin_edges"][
        np.argmax(azimuth_ring_smooth)
    ]

    # relative uncertainty
    _unc = np.mean(azimuth_ring_smooth)
    main_axis_azimuth_uncertainty = _unc ** 2.0

    result = {}
    result["main_axis_support_cx"] = median_cx
    result["main_axis_support_cy"] = median_cy
    result["main_axis_support_uncertainty"] = np.hypot(
        median_cx_std, median_cy_std
    )
    result["main_axis_azimuth"] = float(main_axis_azimuth)
    result["main_axis_azimuth_uncertainty"] = main_axis_azimuth_uncertainty
    result["reco_cx"] = reco_cx
    result["reco_cy"] = reco_cy

    debug = {}
    debug["split_light_field_model"] = slf_model
    debug["fuzzy_image"] = fuzzy_image
    debug["fuzzy_image_smooth"] = fuzzy_image_smooth
    debug["azimuth_ring"] = azimuth_ring
    debug["azimuth_ring_smooth"] = azimuth_ring_smooth

    return result, debug


def add_axes_fuzzy_debug(ax, ring_binning, fuzzy_result, fuzzy_debug):
    azi = fuzzy_result["main_axis_azimuth"]
    ax.plot(
        np.rad2deg(ring_binning["bin_edges"]),
        fuzzy_debug["azimuth_ring_smooth"],
        "k"
    )
    ax.plot(np.rad2deg(azi), 1.0, "or")

    unc = 0.5 * fuzzy_result["main_axis_azimuth_uncertainty"]
    ax.plot(
        np.rad2deg([azi - unc, azi + unc]),
        [0.5, 0.5],
        "-r"
    )

    ax.set_xlim([0, 360])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("main-axis-azimuth / deg")
    ax.set_ylabel("probability density / 1")


def estimate_core_radius_using_shower_model(
    main_axis_support_cx,
    main_axis_support_cy,
    main_axis_azimuth,
    light_field_cx,
    light_field_cy,
    light_field_x,
    light_field_y,
    shower_maximum_cx,
    shower_maximum_cy,
    shower_maximum_object_distance,
    config,
):
    core_radius_finder = CoreRadiusFinder(
        main_axis_azimuth=main_axis_azimuth,
        main_axis_support_cx=main_axis_support_cx,
        main_axis_support_cy=main_axis_support_cy,
        light_field_cx=light_field_cx,
        light_field_cy=light_field_cy,
        light_field_x=light_field_x,
        light_field_y=light_field_y,
    )

    # mask c_para r_para
    # ------------------

    shower_median_direction_z = np.sqrt(
        1.0 -
        shower_maximum_cx ** 2 -
        shower_maximum_cy ** 2
    )
    distance_aperture_center_to_shower_maximum = (
        shower_maximum_object_distance / shower_median_direction_z
    )

    shower_maximum_direction = [
        shower_maximum_cx,
        shower_maximum_cy,
        shower_median_direction_z
    ]

    core_axis_direction = [
        np.cos(main_axis_azimuth),
        np.sin(main_axis_azimuth),
        0.0
    ]

    epsilon = angle_between(shower_maximum_direction, core_axis_direction)

    c_para_r_para_mask = np.zeros(
        shape=(
            config["c_para"]["num_supports"],
            config["r_para"]["num_supports"]
        ),
        dtype=np.int
    )

    for cbin, c_para in enumerate(config["c_para"]["supports"]):
        matching_r_para = matching_core_radius(
            c_para=c_para,
            epsilon=epsilon,
            m=distance_aperture_center_to_shower_maximum
        )

        closest_r_para_bin = np.argmin(
            np.abs(config["r_para"]["supports"] - matching_r_para)
        )

        if (
            closest_r_para_bin > 0 and
            closest_r_para_bin < (config["r_para"]["num_supports"] - 1)
        ):
            if config["scan"]["num_bins_radius"] == 0:
                rbin_range = [closest_r_para_bin]
            else:
                rbin_range = np.arange(
                    closest_r_para_bin - config["scan"]["num_bins_radius"],
                    closest_r_para_bin + config["scan"]["num_bins_radius"]
                )

            for rbin in rbin_range:
                if rbin >= 0 and rbin < config["r_para"]["num_supports"]:
                    c_para_r_para_mask[cbin, rbin] = 1

    # populate c_para r_para
    # ----------------------
    c_para_r_para_response = np.zeros(
        shape=(
            config["c_para"]["num_supports"],
            config["r_para"]["num_supports"]
        )
    )
    for cbin, c_para in enumerate(config["c_para"]["supports"]):
        for rbin, r_para in enumerate(config["r_para"]["supports"]):

            if c_para_r_para_mask[cbin, rbin]:
                c_para_r_para_response[cbin, rbin] = core_radius_finder.response(
                    c_para=c_para,
                    r_para=r_para,
                    cer_perp_distance_threshold=config["shower_model"]["c_perp_width"],
                )

    # find highest response in c_para r_para
    # --------------------------------------
    argmax_c_para, argmax_r_para = pl.fuzzy.direction.argmax2d(
        c_para_r_para_response
    )
    max_c_para = config["c_para"]["supports"][argmax_c_para]
    max_r_para = config["r_para"]["supports"][argmax_r_para]
    max_response = c_para_r_para_response[argmax_c_para, argmax_r_para]

    # store finding
    # -------------

    reco_cx = main_axis_support_cx + np.cos(main_axis_azimuth) * max_c_para
    reco_cy = main_axis_support_cy + np.sin(main_axis_azimuth) * max_c_para
    reco_x = np.cos(main_axis_azimuth) * max_r_para
    reco_y = np.sin(main_axis_azimuth) * max_r_para

    result = {}

    result["c_main_axis_parallel"] = float(max_c_para)
    result["r_main_axis_parallel"] = float(max_r_para)
    result["shower_model_response"] = float(max_response)

    result["primary_particle_cx"] = float(reco_cx)
    result["primary_particle_cy"] = float(reco_cy)
    result["primary_particle_x"] = float(reco_x)
    result["primary_particle_y"] = float(reco_y)

    debug = {}
    debug["c_para_r_para_mask"] = c_para_r_para_mask
    debug["c_para_r_para_response"] = c_para_r_para_response
    debug["shower_maximum_direction"] = shower_maximum_direction
    debug["core_axis_direction"] = core_axis_direction
    debug["epsilon"] = epsilon

    return result, debug



class MainAxisToCoreFinder:
    def __init__(
        self,

        light_field_cx,
        light_field_cy,
        light_field_x,
        light_field_y,

        shower_maximum_cx,
        shower_maximum_cy,
        shower_maximum_object_distance,
        config,
    ):
        self.config = config
        self.shower_maximum_cx = shower_maximum_cx
        self.shower_maximum_cy = shower_maximum_cy
        self.shower_maximum_object_distance = shower_maximum_object_distance
        self.light_field_cx = light_field_cx
        self.light_field_cy = light_field_cy
        self.light_field_x = light_field_x
        self.light_field_y = light_field_y
        self.final_result = None

    def _support(self, main_axis_azimuth, main_axis_support_perp_offset):
        perp_azimuth_rad = main_axis_azimuth + 0.5 * np.pi
        offset_rad = main_axis_support_perp_offset
        cx = self.shower_maximum_cx + offset_rad * np.cos(perp_azimuth_rad)
        cy = self.shower_maximum_cy + offset_rad * np.sin(perp_azimuth_rad)
        return cx, cy

    def evaluate_shower_model(
        self,
        main_axis_azimuth,
        main_axis_support_perp_offset
    ):
        main_axis_support_cx, main_axis_support_cy = self._support(
            main_axis_azimuth=main_axis_azimuth,
            main_axis_support_perp_offset=main_axis_support_perp_offset,
        )

        result, _ = estimate_core_radius_using_shower_model(
            main_axis_support_cx=main_axis_support_cx,
            main_axis_support_cy=main_axis_support_cy,
            main_axis_azimuth=main_axis_azimuth,
            light_field_cx=self.light_field_cx,
            light_field_cy=self.light_field_cy,
            light_field_x=self.light_field_x,
            light_field_y=self.light_field_y,
            shower_maximum_cx=self.shower_maximum_cx,
            shower_maximum_cy=self.shower_maximum_cy,
            shower_maximum_object_distance=self.shower_maximum_object_distance,
            config=self.config,
        )
        self.final_result = result

        self.final_result["main_axis_azimuth_deg"] = np.rad2deg(
            main_axis_azimuth
        )
        self.final_result["main_axis_support_cx_deg"] = np.rad2deg(
            main_axis_support_cx
        )
        self.final_result["main_axis_support_cy_deg"] = np.rad2deg(
            main_axis_support_cy
        )
        info = "sup. off, {:6f}, cx {:.6f}, cy {:.6f}, azi. {:.4f}, resp. {:.5f}".format(
            np.rad2deg(main_axis_support_perp_offset),
            self.final_result["main_axis_support_cx_deg"],
            self.final_result["main_axis_support_cy_deg"],
            self.final_result["main_axis_azimuth_deg"],
            self.final_result["shower_model_response"],
        )
        return 1.0 - self.final_result["shower_model_response"]


for sk in irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:

        reco_obj = read_shower_maximum_object_distance(
            site_key=sk,
            particle_key=pk
        )

        loph_chunk_paths = glob.glob(
            os.path.join(loph_chunk_base_dir, sk, pk, "chunks", "*.tar")
        )

        run = pl.photon_stream.loph.LopfTarReader(loph_chunk_paths[1])

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        for event in run:
            airshower_id, loph_record = event

            truth = dict(truth_by_index[sk][pk][airshower_id])

            split_light_field = pl.fuzzy.direction.SplitLightField(
                loph_record=loph_record, light_field_geometry=lfg
            )

            if split_light_field.number_photons < 120:
                continue

            fuzzy_result, fuzzy_debug = estimate_main_axis_to_core_using_fuzzy_method(
                split_light_field=split_light_field,
                model_config=fuzzy_config["ellipse_model"],
                image_binning=fuzzy_config["image"],
                image_smoothing_kernel=fuzzy_config["image"]["smoothing_kernel"],
                ring_binning=fuzzy_config["azimuth_ring"],
                ring_smoothing_kernel=fuzzy_config["azimuth_ring"]["smoothing_kernel"],
            )

            if PLOT_RING:
                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax = my_axes_look(ax=ax)
                add_axes_fuzzy_debug(
                    ax=ax,
                    ring_binning=fuzzy_config["azimuth_ring"],
                    fuzzy_result=fuzzy_result,
                    fuzzy_debug=fuzzy_debug,
                )
                path = os.path.join(
                    pa["out_dir"], sk, pk, "{:09d}_ring.jpg".format(
                        airshower_id,
                    )
                )
                fig.savefig(path)
                plt.close(fig)

            # Minuit
            lixel_ids = loph_record["photons"]["channels"]

            main_axis_to_core_finder = MainAxisToCoreFinder(
                light_field_cx=lfg.cx_mean[lixel_ids],
                light_field_cy=lfg.cy_mean[lixel_ids],
                light_field_x=lfg.x_mean[lixel_ids],
                light_field_y=lfg.y_mean[lixel_ids],
                shower_maximum_cx=split_light_field.median_cx,
                shower_maximum_cy=split_light_field.median_cy,
                shower_maximum_object_distance=reco_obj[airshower_id],
                config=long_fit_cfg,
            )

            minimizer = Minuit(
                fcn=main_axis_to_core_finder.evaluate_shower_model,

                main_axis_azimuth=fuzzy_result["main_axis_azimuth"],
                error_main_axis_azimuth=fuzzy_result["main_axis_azimuth_uncertainty"],
                limit_main_axis_azimuth=(
                    fuzzy_result["main_axis_azimuth"] - 2.0*np.pi,
                    fuzzy_result["main_axis_azimuth"] + 2.0*np.pi
                ),

                main_axis_support_perp_offset=0.0,
                error_main_axis_support_perp_offset=fuzzy_result["main_axis_support_uncertainty"],
                limit_main_axis_support_perp_offset=(
                    -5.0 * fuzzy_result["main_axis_support_uncertainty"],
                    5.0 * fuzzy_result["main_axis_support_uncertainty"]
                ),

                print_level=0,
                errordef=Minuit.LEAST_SQUARES,
            )
            minimizer.migrad()

            min_res = {
                "main_axis_azimuth_deg": np.rad2deg(
                    minimizer.values["main_axis_azimuth"]
                ),
                "main_axis_support_perp_offset_deg": np.rad2deg(
                    minimizer.values["main_axis_support_perp_offset"]
                ),
            }


            # true response
            # -------------

            true_main_axis_azimuth = np.pi + np.arctan2(truth["y"], truth["x"])
            true_r_para = np.hypot(truth["x"], truth["y"]) * np.sign(true_main_axis_azimuth - np.pi)
            true_c_para = np.hypot(
                split_light_field.median_cx - truth["cx"],
                split_light_field.median_cy - truth["cy"]
            )

            truth_core_radius_finder = CoreRadiusFinder(
                main_axis_azimuth=true_main_axis_azimuth,
                main_axis_support_cx=split_light_field.median_cx,
                main_axis_support_cy=split_light_field.median_cy,
                light_field_cx=lfg.cx_mean[lixel_ids],
                light_field_cy=lfg.cy_mean[lixel_ids],
                light_field_x=lfg.x_mean[lixel_ids],
                light_field_y=lfg.y_mean[lixel_ids],
            )

            true_response = truth_core_radius_finder.response(
                c_para=true_c_para,
                r_para=true_r_para,
                cer_perp_distance_threshold=long_fit_cfg["shower_model"]["c_perp_width"],
            )

            fit2 = main_axis_to_core_finder.final_result

            if PLOT_OVERVIEW:
                fit2_cx_deg = np.rad2deg(fit2["primary_particle_cx"])
                fit2_cy_deg = np.rad2deg(fit2["primary_particle_cy"])
                fit2_x = fit2["primary_particle_x"]
                fit2_y = fit2["primary_particle_y"]

                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.075, 0.1, 0.4, 0.8])
                ax_core = fig.add_axes([0.575, 0.1, 0.4, 0.8])
                for pax in range(split_light_field.number_paxel):
                    ax.plot(
                        np.rad2deg(split_light_field.image_sequences[pax][:, 0]),
                        np.rad2deg(split_light_field.image_sequences[pax][:, 1]),
                        "xb",
                        alpha=0.03,
                    )
                ax.pcolor(
                    np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                    np.rad2deg(fuzzy_config["image"]["c_bin_edges"]),
                    fuzzy_debug["fuzzy_image_smooth"],
                    cmap="Reds",
                )
                phi = np.linspace(0, 2 * np.pi, 1000)
                ax.plot(
                    fov_radius_deg * np.cos(phi), fov_radius_deg * np.sin(phi), "k"
                )
                ax.plot(
                    [
                        fit2["main_axis_support_cx_deg"],
                        fit2["main_axis_support_cx_deg"] + 100 * np.cos(np.deg2rad(fit2["main_axis_azimuth_deg"]))
                    ],
                    [
                        fit2["main_axis_support_cy_deg"],
                        fit2["main_axis_support_cy_deg"] + 100 * np.sin(np.deg2rad(fit2["main_axis_azimuth_deg"]))
                    ],
                    ":c",
                )

                ax.plot(
                    np.rad2deg(fuzzy_result["reco_cx"]),
                    np.rad2deg(fuzzy_result["reco_cy"]),
                    "og"
                )
                ax.plot(fit2_cx_deg, fit2_cy_deg, "oc")
                ax.plot(np.rad2deg(truth["cx"]), np.rad2deg(truth["cy"]), "xk")

                info_str = "Energy: {: .1f}GeV, reco. Cherenkov: {: 4d}p.e.\n response of shower-model: {:.4f} ({:.4f})".format(
                    truth["energy_GeV"],
                    loph_record["photons"]["channels"].shape[0],
                    fit2["shower_model_response"],
                    true_response,
                )

                ax.set_title(info_str)

                ax.set_xlim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_ylim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_aspect("equal")
                ax.set_xlabel("cx / deg")
                ax.set_ylabel("cy / deg")
                ax = my_axes_look(ax=ax)

                ax_core.plot(fit2_x, fit2_y, "oc")
                ax_core.plot([0, fit2_x], [0, fit2_y], "c", alpha=0.5)

                ax_core.plot(truth["x"], truth["y"], "xk")
                ax_core.plot([0, truth["x"]], [0, truth["y"]], "k", alpha=0.5)

                ax_core.set_xlim([-640, 640])
                ax_core.set_ylim([-640, 640])
                ax_core.set_aspect("equal")
                ax_core.set_xlabel("x / m")
                ax_core.set_ylabel("y / m")
                ax_core = my_axes_look(ax=ax_core)
                path = os.path.join(
                    pa["out_dir"], sk, pk, "{:09d}.jpg".format(
                        airshower_id,
                    ),
                )

                fig.savefig(path)
                plt.close(fig)
