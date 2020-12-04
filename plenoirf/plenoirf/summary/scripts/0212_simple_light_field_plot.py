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

fuzzy_ring_gaussian_kernel = pl.fuzzy.discrete_kernel.gauss1d(num_steps=41)
fuzzy_image_gaussian_kernel = pl.fuzzy.discrete_kernel.gauss2d(num_steps=5)
fuzzy_model_config = pl.fuzzy.direction.EXAMPLE_MODEL_CONFIG
fuzzy_image_binning = {
    "radius_deg": fov_radius_deg + 1.0,
    "num_bins": 128,
}
fuzzy_image_c_bin_edges = np.linspace(
    -fuzzy_image_binning["radius_deg"],
    fuzzy_image_binning["radius_deg"],
    fuzzy_image_binning["num_bins"] + 1,
)
fuzzy_ring_radius_deg = 1.5

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

PLOT_RING = True
PLOT_C_PARA_R_PARA_RESPONSE = False
PLOT_OVERVIEW = True


def estimate_main_axis_to_core_using_fuzzy_method(
    split_light_field,
    fuzzy_model_config,
    fuzzy_image_binning,
    fuzzy_image_gaussian_kernel,
    fuzzy_ring_gaussian_kernel,
    fuzzy_ring_radius_deg=1.5,
):
    median_cx_deg = np.rad2deg(split_light_field.median_cx)
    median_cy_deg = np.rad2deg(split_light_field.median_cy)

    # make fuzzy image
    # ----------------
    # A probability-density for the shower's main-axis and primary particle's
    # direction.

    slf_model = pl.fuzzy.direction.estimate_model_from_light_field(
        split_light_field=split_light_field, model_config=fuzzy_model_config
    )
    fuzzy_image = pl.fuzzy.direction.make_image_from_model(
        split_light_field_model=slf_model,
        model_config=fuzzy_model_config,
        image_binning=fuzzy_image_binning,
    )
    fuzzy_image_smooth = scipy.signal.convolve2d(
        in1=fuzzy_image, in2=fuzzy_image_gaussian_kernel, mode="same"
    )
    reco_cx_deg, reco_cy_deg = pl.fuzzy.direction.argmax_image_cx_cy_deg(
        image=fuzzy_image_smooth, image_binning=fuzzy_image_binning,
    )

    median_cx_std_deg = np.rad2deg(np.std([a["median_cx"] for a in slf_model]))
    median_cy_std_deg = np.rad2deg(np.std([a["median_cy"] for a in slf_model]))

    # make ring to find main-axis
    # ---------------------------

    azimuth_ring = pl.fuzzy.direction.project_image_onto_ring(
        image=fuzzy_image_smooth,
        image_binning=fuzzy_image_binning,
        ring_cx_deg=median_cx_deg,
        ring_cy_deg=median_cy_deg,
        ring_radius_deg=fuzzy_ring_radius_deg,
    )
    azimuth_ring_smooth = pl.fuzzy.direction.circular_convolve1d(
        in1=azimuth_ring, in2=fuzzy_ring_gaussian_kernel
    )
    azimuth_ring_smooth /= np.max(azimuth_ring_smooth)

    # analyse ring to find main-axis
    # ------------------------------

    # maximum
    main_axis_azimuth_deg = np.argmax(azimuth_ring_smooth)

    # relative uncertainty
    _unc = np.mean(azimuth_ring_smooth)
    main_axis_azimuth_uncertainty_deg = 360.0 * _unc ** 2.0

    result = {}
    result["main_axis_support_cx_deg"] = median_cx_deg
    result["main_axis_support_cy_deg"] = median_cy_deg
    result["main_axis_support_uncertainty_deg"] = np.hypot(
        median_cx_std_deg, median_cy_std_deg
    )
    result["main_axis_azimuth_deg"] = float(main_axis_azimuth_deg)
    result["main_axis_azimuth_uncertainty_deg"] = main_axis_azimuth_uncertainty_deg
    result["reco_cx_deg"] = reco_cx_deg
    result["reco_cy_deg"] = reco_cy_deg

    debug = {}
    debug["split_light_field_model"] = slf_model
    debug["fuzzy_image"] = fuzzy_image
    debug["fuzzy_image_smooth"] = fuzzy_image_smooth
    debug["azimuth_ring"] = azimuth_ring
    debug["azimuth_ring_smooth"] = azimuth_ring_smooth

    return result, debug


def add_axes_fuzzy_debug(ax, fuzzy_result, fuzzy_debug):
    azi_deg = fuzzy_result["main_axis_azimuth_deg"]
    ax.plot(fuzzy_debug["azimuth_ring_smooth"], "k")
    ax.plot(azi_deg, 1.0, "or")

    unc_deg = 0.5 * fuzzy_result["main_axis_azimuth_uncertainty_deg"]
    ax.plot([azi_deg - unc_deg, azi_deg + unc_deg], [0.5, 0.5], "-r")

    ax.set_xlim([0, 360])
    ax.set_ylim([0.0, 1.0])
    ax.set_xlabel("main-axis-azimuth / deg")
    ax.set_ylabel("probability density / deg$^{-1}$")


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
        median_cx=main_axis_support_cx,
        median_cy=main_axis_support_cy,
        cx=light_field_cx,
        cy=light_field_cy,
        x=light_field_x,
        y=light_field_y,
    )

    c_para_supports = squarespace(
        start=config["c_para"]["start"],
        stop=config["c_para"]["stop"],
        num=config["c_para"]["num_supports"],
    )

    r_para_supports = squarespace(
        start=config["r_para"]["start"],
        stop=config["r_para"]["stop"],
        num=config["r_para"]["num_supports"],
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
            closest_r_para_bin < (config["r_para"]["num_supports"] - 1)
        ):
            rbin_range = np.arange(
                closest_r_para_bin - config["r_para"]["num_bins_scan_radius"],
                closest_r_para_bin + config["r_para"]["num_bins_scan_radius"]
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
    for cbin, c_para in enumerate(c_para_supports):
        for rbin, r_para in enumerate(r_para_supports):

            if c_para_r_para_mask[cbin, rbin]:
                c_para_r_para_response[cbin, rbin] = core_radius_finder.response(
                    c_para=c_para,
                    r_para=r_para,
                    cer_perp_distance_threshold=config["c_perp_width"],
                )

    # find highest response in c_para r_para
    # --------------------------------------

    argmax_c_para, argmax_r_para = pl.fuzzy.direction.argmax2d(
        c_para_r_para_response
    )
    max_c_para = c_para_supports[argmax_c_para]
    max_r_para = r_para_supports[argmax_r_para]
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
    debug["c_para_supports"] = c_para_supports
    debug["r_para_supports"] = r_para_supports
    debug["c_para_r_para_mask"] = c_para_r_para_mask
    debug["c_para_r_para_response"] = c_para_r_para_response
    debug["shower_maximum_direction"] = shower_maximum_direction
    debug["core_axis_direction"] = core_axis_direction
    debug["epsilon"] = epsilon

    return result, debug



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

            truth = dict(truth_by_index[sk][pk][airshower_id])

            split_light_field = pl.fuzzy.direction.SplitLightField(
                loph_record=loph_record, light_field_geometry=lfg
            )

            if split_light_field.number_photons < 120:
                continue

            fuzzy_result, fuzzy_debug = estimate_main_axis_to_core_using_fuzzy_method(
                split_light_field=split_light_field,
                fuzzy_model_config=fuzzy_model_config,
                fuzzy_image_binning=fuzzy_image_binning,
                fuzzy_image_gaussian_kernel=fuzzy_image_gaussian_kernel,
                fuzzy_ring_gaussian_kernel=fuzzy_ring_gaussian_kernel,
                fuzzy_ring_radius_deg=fuzzy_ring_radius_deg,
            )

            print(fuzzy_result)

            if PLOT_RING:
                fig = irf.summary.figure.figure(fig_16_by_9)
                ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                ax = my_axes_look(ax=ax)
                add_axes_fuzzy_debug(
                    ax=ax,
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

            num_azi_scan = 11
            num_sup_scan = 5
            azi_range_deg = 0.5 * fuzzy_result["main_axis_azimuth_uncertainty_deg"]

            main_axis_azimuth_offsets_deg = np.linspace(
                -azi_range_deg,
                azi_range_deg,
                num_azi_scan,
            )

            sup_range_deg = 2.0 * fuzzy_result["main_axis_support_uncertainty_deg"]
            main_axis_perp_supports_deg = np.linspace(
                -sup_range_deg,
                sup_range_deg,
                num_sup_scan
            )

            print(
                "azi_range_deg: ", azi_range_deg,
                "sup_range_deg: ", sup_range_deg
            )

            longi_fits = []
            for i_azi in range(num_azi_scan):
                longi_fits.append([])
                for i_sup in range(num_sup_scan):
                    longi_fits[i_azi].append(None)

            for i_azi in range(num_azi_scan):
                for i_sup in range(num_sup_scan):

                    fit = {}
                    fit["main_axis_azimuth_deg"] = (
                        fuzzy_result["main_axis_azimuth_deg"] +
                        main_axis_azimuth_offsets_deg[i_azi]
                    )
                    fit["main_axis_support_cx_deg"] = (
                        fuzzy_result["main_axis_support_cx_deg"]
                         + main_axis_perp_supports_deg[i_sup] *
                         np.cos(np.deg2rad(90.0 + fit["main_axis_azimuth_deg"]))
                    )
                    fit["main_axis_support_cy_deg"] = (
                        fuzzy_result["main_axis_support_cy_deg"]
                         + main_axis_perp_supports_deg[i_sup] *
                         np.sin(np.deg2rad(90.0 + fit["main_axis_azimuth_deg"]))
                    )

                    lixel_ids = loph_record["photons"]["channels"]

                    lresult, ldebug = estimate_core_radius_using_shower_model(
                        main_axis_support_cx=np.deg2rad(fit["main_axis_support_cx_deg"]),
                        main_axis_support_cy=np.deg2rad(fit["main_axis_support_cy_deg"]),
                        main_axis_azimuth=np.deg2rad(fit["main_axis_azimuth_deg"]),
                        light_field_cx=lfg.cx_mean[lixel_ids],
                        light_field_cy=lfg.cy_mean[lixel_ids],
                        light_field_x=lfg.x_mean[lixel_ids],
                        light_field_y=lfg.y_mean[lixel_ids],
                        shower_maximum_cx=split_light_field.median_cx,
                        shower_maximum_cy=split_light_field.median_cy,
                        shower_maximum_object_distance=reco_obj[airshower_id],
                        config=long_fit_cfg,
                    )

                    for key in lresult:
                        fit[key] = lresult[key]

                    longi_fits[i_azi][i_sup] = fit

                    # end longitudinal fit
                    # --------------------

                    #####

                    if PLOT_C_PARA_R_PARA_RESPONSE:
                        fig = irf.summary.figure.figure(fig_16_by_9)
                        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
                        inte = np.log10(
                            -1.0 * (
                                ldebug["c_para_r_para_response"] - np.max(
                                    ldebug["c_para_r_para_response"]
                                )
                            )
                        )
                        ax.pcolor(
                            np.rad2deg(ldebug["c_para_supports"]),
                            ldebug["r_para_supports"],
                            inte.T,
                            cmap="Blues",
                        )
                        ax.plot(
                            np.rad2deg(lresult["c_main_axis_parallel"]),
                            lresult["r_main_axis_parallel"],
                            "og"
                        )
                        ax.set_xlabel("c main-axis parallel / deg")
                        ax.set_ylabel("r main-axis parallel / m")
                        ax = my_axes_look(ax=ax)
                        path = os.path.join(
                            pa["out_dir"], sk, pk, "{:09d}_{:03d}_{:03d}_resp.jpg".format(
                                airshower_id,
                                i_azi,
                                i_sup
                            )
                        )
                        fig.savefig(path)
                        plt.close(fig)

                        # -------------

                    print(i_azi, i_sup)


            fit = None
            best_response = 0.0
            for i_azi in range(num_azi_scan):
                for i_sup in range(num_sup_scan):
                    lo_fi = longi_fits[i_azi][i_sup]
                    if lo_fi["shower_model_response"] > best_response:
                        best_response = lo_fi["shower_model_response"]
                        fit = dict(lo_fi)


            if PLOT_OVERVIEW:
                reco_fit_cx_deg = np.rad2deg(fit["primary_particle_cx"])
                reco_fit_cy_deg = np.rad2deg(fit["primary_particle_cy"])
                reco_fit_x = fit["primary_particle_x"]
                reco_fit_y = fit["primary_particle_y"]

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
                    fuzzy_image_c_bin_edges,
                    fuzzy_image_c_bin_edges,
                    fuzzy_debug["fuzzy_image_smooth"],
                    cmap="Reds",
                )
                phi = np.linspace(0, 2 * np.pi, 1000)
                ax.plot(
                    fov_radius_deg * np.cos(phi), fov_radius_deg * np.sin(phi), "k"
                )
                ax.plot(
                    [
                        fit["main_axis_support_cx_deg"],
                        fit["main_axis_support_cx_deg"] + 100 * np.cos(np.deg2rad(fit["main_axis_azimuth_deg"]))
                    ],
                    [
                        fit["main_axis_support_cy_deg"],
                        fit["main_axis_support_cy_deg"] + 100 * np.sin(np.deg2rad(fit["main_axis_azimuth_deg"]))
                    ],
                    ":b",
                )
                ax.plot(
                    fuzzy_result["reco_cx_deg"],
                    fuzzy_result["reco_cy_deg"],
                    "og"
                )
                ax.plot(reco_fit_cx_deg, reco_fit_cy_deg, "oc")

                ax.plot(np.rad2deg(truth["cx"]), np.rad2deg(truth["cy"]), "xk")

                info_str = "Energy: {: .1f}GeV, reco. Cherenkov: {: 4d}p.e.\n response of shower-model: {:.4f}".format(
                    truth["energy_GeV"],
                    loph_record["photons"]["channels"].shape[0],
                    fit["shower_model_response"],
                )

                ax.set_title(info_str)

                ax.set_xlim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_ylim([-1.05 * fov_radius_deg, 1.05 * fov_radius_deg])
                ax.set_aspect("equal")
                ax.set_xlabel("cx / deg")
                ax.set_ylabel("cy / deg")
                ax = my_axes_look(ax=ax)
                ax_core.plot(reco_fit_x, reco_fit_y, "oc")
                ax_core.plot([0, reco_fit_x], [0, reco_fit_y], "c", alpha=0.5)

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
