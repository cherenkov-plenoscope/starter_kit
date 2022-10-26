import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as sebplt
import sys
import aberration_demo as abe
import plenoirf

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argv = sys.argv
if argv[0] == "ipython" and argv[1] == "-i":
    argv.pop(1)

work_dir = argv[1]
out_dir = os.path.join(work_dir, "figures", "abc")
os.makedirs(out_dir, exist_ok=True)

config = abe.read_config(work_dir=work_dir)
coll = abe.read_analysis(work_dir=work_dir)

offaxis_angles_deg = config["sources"]["off_axis_angles_deg"]
max_offaxis_angle_deg = np.max(offaxis_angles_deg)

MSTYLES = {
    "sphere_monolith": {"linestyle": "-"},
    "davies_cotton": {"linestyle": ":"},
    "parabola_segmented": {"linestyle": "--"},
}
PAXSTYLES = {
    1: {"alpha": 0.25},
    3: {"alpha": 0.5},
    9: {"alpha": 1.0},
}

fig = sebplt.figure(sebplt.FIGURE_4_3)
ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
max_psf80s_deg = 0.0
for mkey in config["mirror"]["keys"]:
    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = abe.PAXEL_FMT.format(npax)
        psf80s_deg = np.zeros(len(config["sources"]["off_axis_angles_deg"]))
        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            akey = abe.ANGLE_FMT.format(ofa)
            if akey not in coll[mkey][pkey]:
                continue
            psf80s_deg[ofa] = np.rad2deg(
                coll[mkey][pkey][akey]["image"]["angle80"]
            )
        ax.plot(
            offaxis_angles_deg,
            psf80s_deg,
            linestyle=MSTYLES[mkey]["linestyle"],
            alpha=PAXSTYLES[npax]["alpha"],
            color="k",
        )
        max_psf80s_deg = np.max([np.max(psf80s_deg), max_psf80s_deg])
ax.set_ylim([0.0, 1.2 * max_psf80s_deg])
ax.set_xlim([0.0, 1.2 * max_offaxis_angle_deg])
ax.set_xlabel(r"off axis / $1^\circ{}$")
ax.set_ylabel(r"psf80 / $1^\circ{}$")
fig.savefig(os.path.join(out_dir, "psf.jpg"))
sebplt.close(fig)


fig = sebplt.figure(sebplt.FIGURE_4_3)
ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
max_t80 = 0.0
for mkey in config["mirror"]["keys"]:
    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        pkey = abe.PAXEL_FMT.format(npax)

        t80s = np.zeros(len(config["sources"]["off_axis_angles_deg"]))
        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            akey = abe.ANGLE_FMT.format(ofa)
            if akey not in coll[mkey][pkey]:
                continue

            t_stop = coll[mkey][pkey][akey]["time"]["containment80"]["stop"]
            t_start = coll[mkey][pkey][akey]["time"]["containment80"]["start"]
            t80s[ofa] = t_stop - t_start
        ax.plot(
            offaxis_angles_deg,
            t80s,
            linestyle=MSTYLES[mkey]["linestyle"],
            alpha=PAXSTYLES[npax]["alpha"],
            color="k",
        )
        max_t80 = np.max([np.max(t80s), max_t80])
ax.set_ylim([0.0, 1.2 * max_t80])
ax.set_xlim([0.0, 1.2 * max_offaxis_angle_deg])
ax.set_xlabel(r"off axis / $1^\circ{}$")
ax.set_ylabel(r"time / s")
fig.savefig(os.path.join(out_dir, "time.jpg"))
sebplt.close(fig)
