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

config = abe.offaxis.read_config(work_dir=work_dir)
coll = abe.offaxis.read_analysis(work_dir=work_dir)

# summary plot of poin-spread-functions
# -------------------------------------

OFFAXIS_ANGLE_IDXS = [0, 4, 8]
OFF_AXIS_ANGLE_LABEL = r"off-axis-angle / 1$^\circ$"
GRID_ANGLE_DEG = 0.2

MIRROR_COLORS = {
    "parabola_segmented": "black",
    "sphere_monolith": "gray",
    # "davies_cotton": "gray",
}

PAXEL_STYLE = {
    abe.offaxis.PAXEL_FMT.format(1): "o-",
    abe.offaxis.PAXEL_FMT.format(3): "o--",
    abe.offaxis.PAXEL_FMT.format(9): "o:",
}


psf_representations = {
    "solid_angle": r"spread's solid angle 80% / (1$^\circ$)$^{2}$",
    "radial_angle": r"spread's radial angle 80% / 1$^\circ$",
}

for rkey in ["solid_angle", "radial_angle"]:
    fig = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
    ax = sebplt.add_axes(
        fig=fig,
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

            omega80_deg2 = plenoirf.utils.sr2squaredeg(
                solid_angle_sr=omega80_sr
            )

            if rkey == "solid_angle":
                values = omega80_deg2
            else:
                values = np.sqrt(omega80_deg2 / np.pi)

            ax.plot(
                cxs_deg, values, PAXEL_STYLE[pkey], color=MIRROR_COLORS[mkey],
            )
    ax.set_xlabel(OFF_AXIS_ANGLE_LABEL)
    ax.set_ylabel(psf_representations[rkey])
    fig.savefig(os.path.join(out_dir, "psf_overview_{:s}.jpg".format(rkey)))
    sebplt.close(fig)


fig = sebplt.figure(style={"rows": 640, "cols": 1280, "fontsize": 1})
ax = sebplt.add_axes(
    fig=fig,
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

        ax.plot(
            cxs_deg, time80_ns, PAXEL_STYLE[pkey], color=MIRROR_COLORS[mkey]
        )
ax.set_xlabel(OFF_AXIS_ANGLE_LABEL)
ax.set_ylabel(r"duration 80% / ns")
fig.savefig(os.path.join(out_dir, "tsf_overview.jpg"))
sebplt.close(fig)
