import os
import plenoirf
import numpy as np
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
out_dir = os.path.join(work_dir, "figures", "details")
os.makedirs(out_dir, exist_ok=True)

config = abe.read_config(work_dir=work_dir)
coll = abe.read_analysis(work_dir=work_dir)

for mkey in coll:
    for pkey in coll[mkey]:
        for akey in coll[mkey][pkey]:

            tcoll = coll[mkey][pkey][akey]
            scenario_key = mkey + "_" + pkey + "_" + akey

            bin_edges_cx, bin_edges_cy = abe.analysis.binning_image_bin_edges(
                binning=tcoll["image"]["binning"]
            )

            img_path = os.path.join(
                out_dir, "image_" + scenario_key + ".jpg"
            )
            if not os.path.exists(img_path):
                fig = sebplt.figure(sebplt.FIGURE_4_3)
                ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
                ax_cb = sebplt.add_axes(
                    fig=fig, span=[0.85, 0.1, 0.02, 0.8]
                )
                img_raw_norm = tcoll["image"]["raw"] / np.max(
                    tcoll["image"]["raw"]
                )
                cmap_psf = ax.pcolormesh(
                    np.rad2deg(bin_edges_cx),
                    np.rad2deg(bin_edges_cy),
                    np.transpose(img_raw_norm),
                    cmap="Greys",
                    norm=sebplt.plt_colors.PowerNorm(gamma=0.33),
                )
                sebplt.plt.colorbar(cmap_psf, cax=ax_cb, extend="max")

                ax.grid(
                    color="k", linestyle="-", linewidth=0.66, alpha=0.33
                )
                sebplt.ax_add_circle(
                    ax=ax,
                    x=tcoll["image"]["binning"]["image"]["center"][
                        "cx_deg"
                    ],
                    y=tcoll["image"]["binning"]["image"]["center"][
                        "cy_deg"
                    ],
                    r=np.rad2deg(tcoll["image"]["angle80"]),
                    linewidth=0.5,
                    linestyle="-",
                    color="r",
                    alpha=1,
                    num_steps=360,
                )
                ax.set_aspect("equal")
                ax.set_xlabel(r"$c_x$ / $1^\circ{}$")
                ax.set_ylabel(r"$c_y$ / $1^\circ{}$")
                fig.savefig(img_path)
                sebplt.close(fig)

            time_path = os.path.join(
                out_dir, "time_" + scenario_key + ".jpg"
            )
            if not os.path.exists(time_path):
                fig = sebplt.figure(sebplt.FIGURE_1_1)
                ax = sebplt.add_axes(fig, [0.1, 0.1, 0.8, 0.8])
                sebplt.ax_add_histogram(
                    ax=ax,
                    bin_edges=tcoll["time"]["bin_edges"],
                    bincounts=tcoll["time"]["weights"]
                    / np.max(tcoll["time"]["weights"]),
                    face_color="k",
                    face_alpha=0.1,
                    draw_bin_walls=True,
                )
                ax.plot(
                    [
                        tcoll["time"]["fwhm"]["start"],
                        tcoll["time"]["fwhm"]["stop"],
                    ],
                    [0.5, 0.5],
                    "r-",
                )
                ax.semilogy()
                ax.set_xlabel("time / s")
                fig.savefig(time_path)
                sebplt.close(fig)