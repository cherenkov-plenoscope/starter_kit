#!/usr/bin/python
import os
import plenoirf
import numpy as np
import solid_angle_utils
import plenoptics
import json_numpy
import sebastians_matplotlib_addons as sebplt
import argparse
import pandas

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argparser = argparse.ArgumentParser()
argparser.add_argument("--work_dir", type=str)
argparser.add_argument("--out_dir", type=str)


args = argparser.parse_args()

work_dir = args.work_dir
out_dir = args.out_dir

os.makedirs(out_dir, exist_ok=True)

config = json_numpy.read_tree(os.path.join(work_dir, "config"))

INSTRUMENTS = plenoptics.analysis.guide_stars.list_instruments_observing_guide_stars(
    config=config
)
GUIDE_STAR_KEYS = plenoptics.analysis.guide_stars.list_guide_star_keys(
    config=config
)


max_instrument_fov_half_angle_deg = 0.0
min_num_valid_stars = float("inf")
psf = {}

for instrument_key in INSTRUMENTS:
    psf[instrument_key] = []
    instruments_sensor_key = config["instruments"][instrument_key]["sensor"]
    instrument_fov_half_angle_deg = (
        0.5 * config["sensors"][instruments_sensor_key]["max_FoV_diameter_deg"]
    )

    max_instrument_fov_half_angle_deg = np.max(
        [max_instrument_fov_half_angle_deg, instrument_fov_half_angle_deg]
    )

    image_responses = json_numpy.read(
        os.path.join(work_dir, "analysis", instrument_key, "star.json")
    )

    ll = []
    num_valid_stars = 0
    for star_key in image_responses:
        if star_key in GUIDE_STAR_KEYS:
            continue
        num_valid_stars += 1
        image_response = image_responses[star_key]
        image_response["image"]["angle80"]

        rec = {}
        rec["angle80_rad"] = image_response["image"]["angle80"]
        if np.isnan(rec["angle80_rad"]):
            continue
        rec["cx_deg"] = image_response["image"]["binning"]["image"]["center"][
            "cx_deg"
        ]
        rec["cy_deg"] = image_response["image"]["binning"]["image"]["center"][
            "cy_deg"
        ]
        rec["cc_deg"] = np.hypot(rec["cx_deg"], rec["cy_deg"])
        if rec["cc_deg"] > instrument_fov_half_angle_deg:
            continue
        ll.append(rec)
        psf[instrument_key] = pandas.DataFrame(ll).to_records(index=False)

    min_num_valid_stars = np.min([min_num_valid_stars, num_valid_stars])

num_oa_bins = int(np.sqrt(min_num_valid_stars))
if num_oa_bins < 3:
    num_oa_bins = 3

oa_bin_edges_deg = (
    np.sqrt(np.linspace(0, 1, num_oa_bins + 1))
    * max_instrument_fov_half_angle_deg
)

psf_vs_oa_stats = {}
psf_vs_oa = {}
for instrument_key in INSTRUMENTS:

    st = [[] for i in range(num_oa_bins)]

    for oa in range(len(psf[instrument_key])):
        cc_deg = psf[instrument_key]["cc_deg"][oa]
        b = np.digitize(x=cc_deg, bins=oa_bin_edges_deg) - 1
        if b >= 0 and b < num_oa_bins:
            _angle80_rad = psf[instrument_key]["angle80_rad"][oa]
            if not np.isnan(_angle80_rad):
                st[b].append(_angle80_rad)

    psf_vs_oa_stats[instrument_key] = st

    psf_vs_oa[instrument_key] = {
        "mean": np.zeros(num_oa_bins),
        "median": np.zeros(num_oa_bins),
        "std": np.zeros(num_oa_bins),
        "median_spread_68": np.zeros(num_oa_bins),
        "num": np.zeros(num_oa_bins),
    }
    for b in range(num_oa_bins):
        psf_vs_oa[instrument_key]["mean"][b] = np.mean(
            psf_vs_oa_stats[instrument_key][b]
        )
        psf_vs_oa[instrument_key]["median"][b] = np.median(
            psf_vs_oa_stats[instrument_key][b]
        )
        psf_vs_oa[instrument_key]["std"][b] = np.std(
            psf_vs_oa_stats[instrument_key][b]
        )
        psf_vs_oa[instrument_key]["median_spread_68"][
            b
        ] = plenoptics.analysis.statistical_estimators.median_spread(
            a=psf_vs_oa_stats[instrument_key][b], containment=0.68,
        )
        psf_vs_oa[instrument_key]["num"][b] = len(
            psf_vs_oa_stats[instrument_key][b]
        )


ylabel_name = r"solid angle containing 80%"
label_sep = r"$\,/\,$"


same_scenario_different_sensors = {}
for mirror_def_key in config["mirror_deformations"]:
    for sensor_tra_key in config["sensor_transformations"]:
        sstt_key = (mirror_def_key, sensor_tra_key)
        same_scenario_different_sensors[sstt_key] = []

        for instrument_key in INSTRUMENTS:
            is_mirror_def = (
                config["instruments"][instrument_key]["mirror_deformation"]
                == mirror_def_key
            )
            is_sensor_tra = (
                config["instruments"][instrument_key]["sensor_transformation"]
                == sensor_tra_key
            )
            if is_mirror_def and is_sensor_tra:
                same_scenario_different_sensors[sstt_key].append(
                    instrument_key
                )


PLOTS = []
for instrument_key in INSTRUMENTS:
    PLOTS.append(
        {
            "instruments": [instrument_key],
            "filename": "instrument_{:s}".format(instrument_key),
        }
    )


for scenario_key in same_scenario_different_sensors:
    if len(same_scenario_different_sensors[scenario_key]):
        mirror_def_key, sensor_tra_key = scenario_key
        PLOTS.append(
            {
                "instruments": same_scenario_different_sensors[scenario_key],
                "filename": "mirror_deformation_{:s}_sensor_transformation_{:s}".format(
                    mirror_def_key, sensor_tra_key
                ),
            }
        )

SOLID_ANGLE_80_SR_START = 1e-6
SOLID_ANGLE_80_SR_STOP = 100e-6
SOLID_ANGLE_SCALE = 1e6

ylim_usr = SOLID_ANGLE_SCALE * np.array(
    [SOLID_ANGLE_80_SR_START, SOLID_ANGLE_80_SR_STOP]
)

for PLOT in PLOTS:
    fig = sebplt.figure(style={"rows": 1440, "cols": 1280, "fontsize": 1})
    ax_usr = sebplt.add_axes(fig, [0.12, 0.08, 0.77, 0.9])
    sebplt.ax_add_grid(ax=ax_usr, add_minor=True)
    ax_deg2 = ax_usr.twinx()
    ax_deg2.spines["top"].set_visible(False)

    ax_usr.set_ylim(ylim_usr)
    ax_usr.set_ylabel(ylabel_name + label_sep + r"$\mu$sr")

    SOLID_ANGLE_80_DEG2_START = solid_angle_utils.sr2squaredeg(
        ylim_usr[0] / SOLID_ANGLE_SCALE
    )
    SOLID_ANGLE_80_DEG2_STOP = solid_angle_utils.sr2squaredeg(
        ylim_usr[1] / SOLID_ANGLE_SCALE
    )
    ax_deg2.set_ylim(
        np.array([SOLID_ANGLE_80_DEG2_START, SOLID_ANGLE_80_DEG2_STOP])
    )
    ax_deg2.set_ylabel(r"(1$^{\circ}$)$^2$")

    for instrument_key in PLOT["instruments"]:

        if "diag9" in instrument_key:
            linestyle = "-"
            label = "P-61"
        elif "diag3" in instrument_key:
            linestyle = "--"
            label = "P-7"
        elif "diag1" in instrument_key:
            linestyle = ":"
            label = "P-1"
        else:
            linestyle = "-."
            label = None

        oa_rad = psf_vs_oa[instrument_key]["median"]
        oa_std_rad = psf_vs_oa[instrument_key]["median_spread_68"]

        sa_usr = SOLID_ANGLE_SCALE * solid_angle_utils.cone.solid_angle(
            half_angle_rad=oa_rad
        )
        sa_upper_usr = SOLID_ANGLE_SCALE * solid_angle_utils.cone.solid_angle(
            half_angle_rad=oa_rad + oa_std_rad
        )
        sa_lower_usr = SOLID_ANGLE_SCALE * solid_angle_utils.cone.solid_angle(
            half_angle_rad=oa_rad - oa_std_rad
        )

        sebplt.ax_add_histogram(
            ax=ax_usr,
            bin_edges=oa_bin_edges_deg ** 2,
            bincounts=sa_usr,
            linestyle=linestyle,
            linecolor="k",
            linealpha=1.0,
            bincounts_upper=sa_upper_usr,
            bincounts_lower=sa_lower_usr,
            face_color="k",
            face_alpha=0.1,
            label=label,
            draw_bin_walls=True,
        )

    ax_usr.set_xlim([0.0, oa_bin_edges_deg[-1] ** 2])
    legend = ax_usr.legend(loc="upper left")
    xt_deg2 = np.array(ax_usr.get_xticks())
    xticklabels = []
    for xx_deg2 in xt_deg2:
        if xx_deg2 >= 0.0:
            xtl = r"{:.2f}".format(np.sqrt(xx_deg2)) + r"$^{2}$"
        else:
            xtl = r""
        xticklabels.append(xtl)
    ax_usr.set_xticklabels(xticklabels)

    ax_usr.set_xlabel(
        r"(angle off the mirror's optical axis)$^{2}\,/\,(1^{\circ{}})^{2}$"
    )
    ax_usr.semilogy()
    ax_deg2.semilogy()
    fig_filename = "{:s}.jpg".format(PLOT["filename"])
    fig.savefig(os.path.join(out_dir, fig_filename))
    sebplt.close(fig)


average_angle80_rad = {}
for instrument_key in INSTRUMENTS:
    off_axis_weight = np.pi * psf[instrument_key]["cc_deg"] ** 2
    off_axis_weight /= np.sum(off_axis_weight)
    average_angle80_rad[instrument_key] = np.average(
        psf[instrument_key]["angle80_rad"], weights=off_axis_weight,
    )

# export average
# --------------

out_average_angle80_rad = {}
for instrument_key in INSTRUMENTS:
    ha_rad = average_angle80_rad[instrument_key]

    sa_sr = solid_angle_utils.cone.solid_angle(half_angle_rad=ha_rad)
    sa_deg2 = solid_angle_utils.sr2squaredeg(sa_sr)

    out_average_angle80_rad[instrument_key] = {
        "half_angle": {"deg": np.rad2deg(ha_rad), "rad": ha_rad},
        "solid_angle": {"sr": sa_sr, "deg2": sa_deg2},
    }

with open(os.path.join(out_dir, "average_containment80.txt"), "wt") as f:
    f.write(json_numpy.dumps(out_average_angle80_rad, indent=4))
