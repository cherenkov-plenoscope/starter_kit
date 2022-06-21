import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as sebplt
import sys

work_dir = sys.argv[1]

with open(os.path.join(work_dir, "config.json"), "rt") as f:
    config = json_numpy.loads(f.read())

plot_dir = os.path.join(work_dir, "plot")

with open(os.path.join(plot_dir, "summary.json"), "rt") as f:
    coll = json_numpy.loads(f.read())

offaxis_angles_deg = np.linalg.norm(
    np.array(config["sources"]["off_axis_angles_deg"]), axis=1
)
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
        paxkey = "paxel{:d}".format(npax)
        psf80s_deg = np.zeros(len(config["sources"]["off_axis_angles_deg"]))
        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            ofakey = "{:03d}".format(ofa)
            psf80s_deg[ofa] = np.rad2deg(
                coll[mkey][paxkey][ofakey]["image"]["angle80"]
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
fig.savefig(os.path.join(plot_dir, "psf.jpg"))
sebplt.close(fig)


fig = sebplt.figure(sebplt.FIGURE_4_3)
ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
max_t80 = 0.0
for mkey in config["mirror"]["keys"]:
    for npax in config["sensor"]["num_paxel_on_diagonal"]:
        paxkey = "paxel{:d}".format(npax)

        t80s = np.zeros(len(config["sources"]["off_axis_angles_deg"]))
        for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
            ofakey = "{:03d}".format(ofa)
            t_stop = coll[mkey][paxkey][ofakey]["time"]["containment80"][
                "stop"
            ]
            t_start = coll[mkey][paxkey][ofakey]["time"]["containment80"][
                "start"
            ]
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
fig.savefig(os.path.join(plot_dir, "time.jpg"))
sebplt.close(fig)
