#!/usr/bin/python
import os
import plenoirf
import numpy as np
import aberration_demo as abe
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

INSTRUMENTS = abe.full.plots.impact_of_deformations.guide_stars.list_instruments_observing_guide_stars(
    config=config
)
GUIDE_STAR_KEYS = abe.full.plots.impact_of_deformations.guide_stars.list_guide_star_keys(
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
        rec["cx_deg"] = image_response["image"]["binning"]["image"]["center"][
            "cx_deg"
        ]
        rec["cy_deg"] = image_response["image"]["binning"]["image"]["center"][
            "cy_deg"
        ]
        rec["cc_deg"] = np.hypot(rec["cx_deg"], rec["cy_deg"])

        if rec["cc_deg"] <= instrument_fov_half_angle_deg:
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
            st[b].append(psf[instrument_key]["angle80_rad"][oa])

    psf_vs_oa_stats[instrument_key] = st

    psf_vs_oa[instrument_key] = {
        "mean": np.zeros(num_oa_bins),
        "std": np.zeros(num_oa_bins),
        "num": np.zeros(num_oa_bins),
    }
    for b in range(num_oa_bins):
        psf_vs_oa[instrument_key]["mean"][b] = np.mean(
            psf_vs_oa_stats[instrument_key][b]
        )
        psf_vs_oa[instrument_key]["std"][b] = np.std(
            psf_vs_oa_stats[instrument_key][b]
        )
        psf_vs_oa[instrument_key]["num"][b] = len(
            psf_vs_oa_stats[instrument_key][b]
        )


SOLID_ANGLE_80_SR_START = 0
SOLID_ANGLE_80_SR_STOP = 20e-6
SOLID_ANGLE_SCALE = 1e6

ylabel_name = r"solid angle containing 80%"
label_sep = r"$\,/\,$"

for instrument_key in INSTRUMENTS:

    fig = sebplt.figure(style={"rows": 720, "cols": 1280, "fontsize": 1})
    ax_usr = sebplt.add_axes(fig, [0.15, 0.2, 0.725, 0.75])
    ax_deg2 = ax_usr.twinx()
    ax_deg2.spines["top"].set_visible(False)

    ax_usr.set_ylim(
        SOLID_ANGLE_SCALE
        * np.array([SOLID_ANGLE_80_SR_START, SOLID_ANGLE_80_SR_STOP])
    )
    ax_usr.set_ylabel(ylabel_name + label_sep + r"$\mu$sr")

    solid_angle_80_deg2_start = plenoirf.utils.sr2squaredeg(
        SOLID_ANGLE_80_SR_START
    )
    solid_angle_80_deg2_stop = plenoirf.utils.sr2squaredeg(
        SOLID_ANGLE_80_SR_STOP
    )
    ax_deg2.set_ylim(
        np.array([solid_angle_80_deg2_start, solid_angle_80_deg2_stop])
    )
    ax_deg2.set_ylabel(r"(1$^{\circ}$)$^2$")

    oa_rad = psf_vs_oa[instrument_key]["mean"]
    oa_std_rad = psf_vs_oa[instrument_key]["std"]

    sa_usr = SOLID_ANGLE_SCALE * plenoirf.utils.cone_solid_angle(
        cone_radial_opening_angle_rad=oa_rad
    )
    sa_upper_usr = SOLID_ANGLE_SCALE * plenoirf.utils.cone_solid_angle(
        cone_radial_opening_angle_rad=oa_rad + oa_std_rad
    )
    sa_lower_usr = SOLID_ANGLE_SCALE * plenoirf.utils.cone_solid_angle(
        cone_radial_opening_angle_rad=oa_rad - oa_std_rad
    )

    sebplt.ax_add_histogram(
        ax=ax_usr,
        bin_edges=oa_bin_edges_deg ** 2,
        bincounts=sa_usr,
        linestyle="-",
        linecolor="k",
        linealpha=1.0,
        bincounts_upper=sa_upper_usr,
        bincounts_lower=sa_lower_usr,
        face_color="k",
        face_alpha=0.1,
        label=None,
        draw_bin_walls=True,
    )

    xt_deg2 = ax_usr.get_xticks()
    ax_usr.set_xticklabels(
        [r"{:.2f}".format(np.sqrt(xx)) + r"$^{2}$" for xx in xt_deg2]
    )

    ax_usr.set_xlabel(r"(offaxis angle)$^{2}\,/\,(1^{\circ{}})^{2}$")
    fig_filename = "instrument_{:s}.jpg".format(instrument_key)
    fig.savefig(os.path.join(out_dir, fig_filename))
    sebplt.close(fig)


"""

# plot psf80 vs off-axes
# ----------------------
NUM_PAXEL_STYLE = {
    "paxel000001": {
        "color": "gray",
        "linestyle": "-",
        "alpha": 1,
        "marker": "P",
    },
    "paxel000003": {
        "color": "gray",
        "linestyle": "-",
        "alpha": 0.3,
        "marker": "s",
    },
    "paxel000009": {
        "color": "black",
        "linestyle": "-",
        "alpha": 1,
        "marker": "o",
    },
}
SOLID_ANGLE_SCALE = 1e6

solid_angle_80_sr_start = 0
solid_angle_80_sr_stop = 20e-6

fig = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
ax = sebplt.add_axes(fig=fig, span=[0.15, 0.2, 0.72, 0.75],)
ax_deg2 = ax.twinx()
ax_deg2.spines["top"].set_visible(False)


ylabel_name = r"solid angle containing 80%"
label_sep = r"$\,/\,$"
ax.set_ylim(
    SOLID_ANGLE_SCALE
    * np.array([solid_angle_80_sr_start, solid_angle_80_sr_stop])
)
# ax.semilogy()
ax.set_ylabel(ylabel_name + label_sep + r"$\mu$sr")

solid_angle_80_deg2_start = plenoirf.utils.sr2squaredeg(
    solid_angle_80_sr_start
)
solid_angle_80_deg2_stop = plenoirf.utils.sr2squaredeg(solid_angle_80_sr_stop)
ax_deg2.set_ylim(
    np.array([solid_angle_80_deg2_start, solid_angle_80_deg2_stop])
)
# ax_deg2.semilogy()
ax_deg2.set_ylabel(r"(1$^{\circ}$)$^2$")

sebplt.ax_add_grid(ax=ax, add_minor=True)

average_angle80_in_fov = {}
for isens, pkey in enumerate(coll):

    offaxis_angles_deg = config["sources"]["off_axis_angles_deg"]
    angles80_rad = np.zeros(len(offaxis_angles_deg))
    for iang, akey in enumerate(coll[pkey]):
        angles80_rad[iang] = coll[pkey][akey]["image"]["angle80"]

    cone80_solid_angle_sr = np.zeros(len(angles80_rad))
    for iang in range(len(angles80_rad)):
        cone80_solid_angle_sr[iang] = plenoirf.utils.cone_solid_angle(
            cone_radial_opening_angle_rad=angles80_rad[iang]
        )

    off_axis_weight = np.pi * offaxis_angles_deg ** 2
    off_axis_weight /= np.sum(off_axis_weight)
    average_angle80_in_fov[pkey] = np.average(
        angles80_rad, weights=off_axis_weight,
    )

    ax.plot(
        offaxis_angles_deg,
        cone80_solid_angle_sr * SOLID_ANGLE_SCALE,
        color=NUM_PAXEL_STYLE[pkey]["color"],
        linestyle=NUM_PAXEL_STYLE[pkey]["linestyle"],
        alpha=NUM_PAXEL_STYLE[pkey]["alpha"],
    )

    for iang in range(len(angles80_rad)):
        if iang in OFFAXIS_ANGLE_IDXS:
            markersize = 8
        else:
            markersize = 3
        marker = NUM_PAXEL_STYLE[pkey]["marker"]
        ax.plot(
            offaxis_angles_deg[iang],
            cone80_solid_angle_sr[iang] * SOLID_ANGLE_SCALE,
            color=NUM_PAXEL_STYLE[pkey]["color"],
            alpha=NUM_PAXEL_STYLE[pkey]["alpha"],
            marker=marker,
            markersize=markersize,
            linewidth=0,
        )

ax.set_xlabel(r"angle off the mirror's optical axis$\,/\,1^{\circ}$")

fig.savefig(os.path.join(out_dir, "psf_vs_num_paxel_vs_off_axis.jpg"))
sebplt.close(fig)


with open(
    os.path.join(out_dir, "psf_vs_num_paxel_vs_off_axis.txt"), "wt"
) as f:
    f.write("{:>20s},".format("offaxis/deg"))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format(pkey))
    f.write("\n")

    # half angle
    # ==========
    f.write("{:>20s},".format(""))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format("half-angle-80/deg"))
    f.write("\n")

    for iang, akey in enumerate(coll[pkey]):
        offax_deg = config["sources"]["off_axis_angles_deg"][iang]
        f.write("{: 20.2},".format(offax_deg))
        for isens, pkey in enumerate(coll):

            half_angle80_rad = coll[pkey][akey]["image"]["angle80"]
            half_angle80_deg = np.rad2deg(half_angle80_rad)
            f.write("{: 20.2},".format(half_angle80_deg))

        f.write("\n")

    f.write("\n")

    # solid angle
    # ===========
    f.write("{:>20s},".format(""))
    for isens, pkey in enumerate(coll):
        f.write("{:>20s},".format("solid-angle-80/usr"))
    f.write("\n")

    for iang, akey in enumerate(coll[pkey]):
        offax_deg = config["sources"]["off_axis_angles_deg"][iang]
        f.write("{: 20.2},".format(offax_deg))
        for isens, pkey in enumerate(coll):

            half_angle80_rad = coll[pkey][akey]["image"]["angle80"]
            solid_angle_80_sr = plenoirf.utils.cone_solid_angle(
                cone_radial_opening_angle_rad=half_angle80_rad
            )
            f.write("{: 20.2},".format(SOLID_ANGLE_SCALE * solid_angle_80_sr))

        f.write("\n")

    f.write("\n")

    # average
    # =======
    f.write("{:>20s},".format(""))
    for pkey in coll:
        f.write("{:>20s},".format("avg-solid-80/usr"))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        avg_solid_angle_usr = (
            SOLID_ANGLE_SCALE
            * plenoirf.utils.cone_solid_angle(
                cone_radial_opening_angle_rad=average_angle80_in_fov[pkey]
            )
        )
        f.write("{: 20.2},".format(avg_solid_angle_usr))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        f.write("{:>20s},".format("avg-angle-80/deg"))
    f.write("\n")

    f.write("{:>20s},".format(""))
    for pkey in coll:
        average_angle80_in_fov_deg = np.rad2deg(average_angle80_in_fov[pkey])
        f.write("{: 20.2},".format(average_angle80_in_fov_deg))
    f.write("\n")
"""
