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

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argv = sys.argv
if argv[0] == "ipython" and argv[1] == "-i":
    argv.pop(1)

work_dir = argv[1]
out_dir = os.path.join(work_dir, "figures", "curves")
os.makedirs(out_dir, exist_ok=True)

config = abe.read_config(work_dir=work_dir)
coll = abe.read_analysis(work_dir=work_dir)

# summary plot of poin-spread-functions
# -------------------------------------

OFFAXIS_ANGLE_IDXS = [0, 4, 8]
OFF_AXIS_ANGLE_LABEL = r"off-axis-angle / 1$^\circ$"
GRID_ANGLE_DEG = 0.2

MIRROR_COLORS = {
    "parabola_segmented": "black",
    "davies_cotton": "gray",
}

PAXEL_STYLE = {
    abe.PAXEL_FMT.format(1): "o-",
    abe.PAXEL_FMT.format(3): "o--",
    abe.PAXEL_FMT.format(9): "o:",
}


fig_psf_sum = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
ax_psf_sum = sebplt.add_axes(
    fig=fig_psf_sum,
    span=[0.15, 0.15, 0.8, 0.8],
    style={"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": True},
)
for mkey in MIRROR_COLORS:
    for isens, pkey in enumerate(coll[mkey]):
        cxs_deg = []
        theta80_rad = []
        for iofa, akey in enumerate(coll[mkey][pkey]):
            cxs_deg.append(config["sources"]["off_axis_angles_deg"][iofa])
            theta80_rad.append(coll[mkey][pkey][akey]["image"]["angle80"])
        theta80_rad = np.array(theta80_rad)

        omega80_sr = plenoirf.utils.cone_solid_angle(
            cone_radial_opening_angle_rad=theta80_rad,
        )

        omega80_deg2 = plenoirf.utils.sr2squaredeg(solid_angle_sr=omega80_sr)

        ax_psf_sum.plot(
            cxs_deg,
            omega80_deg2,
            PAXEL_STYLE[pkey],
            color=MIRROR_COLORS[mkey],
        )
ax_psf_sum.set_xlabel(OFF_AXIS_ANGLE_LABEL)
ax_psf_sum.set_ylabel(r"solid angle 80% / (1$^\circ$)$^{2}$")
fig_psf_sum.savefig(os.path.join(out_dir, "psf_overview.jpg"))
sebplt.close(fig_psf_sum)


fig_tsf_sum = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
ax_tsf_sum = sebplt.add_axes(
    fig=fig_tsf_sum,
    span=[0.15, 0.15, 0.8, 0.8],
    style={"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": True},
)
for mkey in MIRROR_COLORS:
    for isens, pkey in enumerate(coll[mkey]):
        cxs_deg = []
        time80_ns = []
        for iofa, akey in enumerate(coll[mkey][pkey]):
            cxs_deg.append(config["sources"]["off_axis_angles_deg"][iofa])
            t80start = coll[mkey][pkey][akey]["time"]["containment80"]["start"]
            t80stop = coll[mkey][pkey][akey]["time"]["containment80"]["stop"]

            time80_ns.append(1e9 * (t80stop - t80start))

        ax_tsf_sum.plot(
            cxs_deg, time80_ns, PAXEL_STYLE[pkey], color=MIRROR_COLORS[mkey]
        )
ax_tsf_sum.set_xlabel(OFF_AXIS_ANGLE_LABEL)
ax_tsf_sum.set_ylabel(r"duration 80% / ns")
fig_tsf_sum.savefig(os.path.join(out_dir, "tsf_overview.jpg"))
sebplt.close(fig_tsf_sum)
