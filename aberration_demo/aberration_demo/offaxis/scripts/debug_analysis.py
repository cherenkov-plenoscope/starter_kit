import corsika_primary as cpw
import os
import plenoirf
import numpy as np
import plenopy
import scipy
from scipy import spatial
from scipy import stats
import aberration_demo as abe
import json_numpy
import sebastians_matplotlib_addons as sebplt
import sys

argv = sys.argv

# --------
# work_dir = "aberration_demo_2022-03-09"
# mkey = "sphere_monolith"
# npax = 1
# ofa = 1

assert len(argv) >= 5
if argv[1] == "-i":
    si = 1
    assert len(argv) == 6
else:
    si = 0
    assert len(argv) == 5

work_dir = argv[1 + si]
mkey = argv[2 + si]
npax = int(argv[3 + si])
ofa = int(argv[4 + si])

OBJECT_DISTANCE = 1e6
CONTAINMENT_PERCENTILE = 80

config = abe.read_config(work_dir=work_dir)
prng = np.random.Generator(np.random.PCG64(config["seed"]))

pkey = abe.PAXEL_FMT.format(npax)
akey = abe.ANGLE_FMT.format(ofa)

analysis_dir = os.path.join(work_dir, "analysis")
os.makedirs(analysis_dir, exist_ok=True)

mdir = os.path.join(analysis_dir, mkey)
os.makedirs(mdir, exist_ok=True)

pdir = os.path.join(mdir, pkey)
os.makedirs(pdir, exist_ok=True)

adir = os.path.join(pdir, akey)
os.makedirs(adir, exist_ok=True)

summary_path = os.path.join(adir, "summary.json")

if not os.path.exists(summary_path):
    print("read light_field_geometry")
    light_field_geometry = abe.LightFieldGeometry(
        path=os.path.join(
            work_dir, "geometries", mkey, pkey, akey, "light_field_geometry",
        ),
        off_axis_angle_deg=config["sources"]["off_axis_angles_deg"][ofa],
    )

    print("read event")
    event = plenopy.Event(
        path=os.path.join(work_dir, "responses", mkey, pkey, akey, "1",),
        light_field_geometry=light_field_geometry,
    )

    """
    print("histogram_arrival_times")
    traw, tbinedges = abe.analysis.histogram_arrival_times(
        raw_sensor_response=event.raw_sensor_response,
        time_delays_to_be_subtracted=light_field_geometry.time_delay_image_mean,
        time_delay_std=light_field_geometry.time_delay_std,
        prng=prng,
        num_sub_samples=5,
    )
    """

    print("calibrate_plenoscope_response")
    calibrated_response = abe.analysis.calibrate_plenoscope_response(
        light_field_geometry=light_field_geometry,
        event=event,
        object_distance=OBJECT_DISTANCE,
    )

    cres = calibrated_response

    print("encirclement2d")
    psf_cx, psf_cy, psf_angle80 = abe.analysis.encirclement2d(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        prng=prng,
        percentile=CONTAINMENT_PERCENTILE,
        num_sub_samples=1,
    )

    thisbinning = dict(config["binning"])
    thisbinning["image"]["center"]["cx_deg"] = config["sources"][
        "off_axis_angles_deg"
    ][ofa][0]
    thisbinning["image"]["center"]["cy_deg"] = config["sources"][
        "off_axis_angles_deg"
    ][ofa][1]
    thisimg_bin_edges = abe.analysis.binning_image_bin_edges(
        binning=thisbinning
    )

    print("histogram2d_std")
    imgraw = abe.analysis.histogram2d_std(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        bins=thisimg_bin_edges,
        prng=prng,
        num_sub_samples=1000,
    )[0]

    print("time encirclement1d")
    time_80_start, time_80_stop = abe.analysis.encirclement1d(
        x=cres["time"]["bin_centers"],
        f=cres["time"]["weights"],
        percentile=CONTAINMENT_PERCENTILE,
    )
    print("time full_width_half_maximum")
    (time_fwhm_start, time_fwhm_stop,) = abe.analysis.full_width_half_maximum(
        x=cres["time"]["bin_centers"], f=cres["time"]["weights"],
    )

    print("time export")
    out = {}
    out["statistics"] = {}
    out["statistics"]["image_beams"] = {}
    out["statistics"]["image_beams"][
        "total"
    ] = light_field_geometry.number_lixel
    out["statistics"]["image_beams"]["valid"] = np.sum(
        cres["image_beams"]["valid"]
    )
    out["statistics"]["photons"] = {}
    out["statistics"]["photons"][
        "total"
    ] = event.raw_sensor_response.number_photons
    out["statistics"]["photons"]["valid"] = np.sum(
        cres["image_beams"]["weights"]
    )

    out["time"] = cres["time"]
    out["time"]["fwhm"] = {}
    out["time"]["fwhm"]["start"] = time_fwhm_start
    out["time"]["fwhm"]["stop"] = time_fwhm_stop
    out["time"]["containment80"] = {}
    out["time"]["containment80"]["start"] = time_80_start
    out["time"]["containment80"]["stop"] = time_80_stop

    out["image"] = {}
    out["image"]["angle80"] = psf_angle80
    out["image"]["binning"] = thisbinning
    out["image"]["raw"] = imgraw

    with open(summary_path, "wt") as f:
        f.write(json_numpy.dumps(out))

with open(summary_path, "rt") as f:
    txt = f.read()
    out = json_numpy.loads(txt)


bin_edges_cx, bin_edges_cy = abe.analysis.binning_image_bin_edges(
    binning=out["image"]["binning"]
)


fig = sebplt.figure(sebplt.FIGURE_4_3)
ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
ax_cb = sebplt.add_axes(fig=fig, span=[0.85, 0.1, 0.02, 0.8])
img_raw_norm = out["image"]["raw"] / np.max(out["image"]["raw"])
cmap_psf = ax.pcolormesh(
    np.rad2deg(bin_edges_cx),
    np.rad2deg(bin_edges_cy),
    np.transpose(img_raw_norm),
    cmap="Greys",
    norm=sebplt.plt_colors.PowerNorm(gamma=0.33),
)
sebplt.plt.colorbar(cmap_psf, cax=ax_cb, extend="max")

ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.33)
sebplt.ax_add_circle(
    ax=ax,
    x=out["image"]["binning"]["image"]["center"]["cx_deg"],
    y=out["image"]["binning"]["image"]["center"]["cy_deg"],
    r=np.rad2deg(out["image"]["angle80"]),
    linewidth=0.5,
    linestyle="-",
    color="r",
    alpha=1,
    num_steps=360,
)
ax.set_aspect("equal")
ax.set_xlabel(r"$c_x$ / $1^\circ{}$")
ax.set_ylabel(r"$c_y$ / $1^\circ{}$")
fig.savefig(os.path.join(adir, "img.jpg"))
sebplt.close(fig)


fig = sebplt.figure(sebplt.FIGURE_1_1)
ax = sebplt.add_axes(fig, [0.1, 0.1, 0.8, 0.8])
sebplt.ax_add_histogram(
    ax=ax,
    bin_edges=out["time"]["bin_edges"],
    bincounts=out["time"]["weights"] / np.max(out["time"]["weights"]),
    face_color="k",
    face_alpha=0.1,
    draw_bin_walls=True,
)
ax.plot(
    [out["time"]["fwhm"]["start"], out["time"]["fwhm"]["stop"]],
    [0.5, 0.5],
    "r-",
)
ax.semilogy()
ax.set_xlabel("time / s")

fig.savefig(os.path.join(adir, "img_time.jpg"))
sebplt.close(fig)
