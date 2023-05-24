#!/usr/bin/python
import os
import plenoirf
import numpy as np
import plenoptics as abe
import sebastians_matplotlib_addons as sebplt
import argparse

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
CMAPS = abe.plot.CMAPS

for cmap_key in CMAPS:
    cmap_dir = os.path.join(out_dir, cmap_key)
    os.makedirs(cmap_dir, exist_ok=True)

    cmap = abe.plot.init_cmap(
        vmin=0.0, vmax=1.0, key=cmap_key, gamma=CMAPS[cmap_key]["gamma"]
    )

    fig = sebplt.figure(style={"rows": 120, "cols": 1280, "fontsize": 1})
    ax = sebplt.add_axes(fig, [0.1, 0.8, 0.8, 0.15])
    ax.text(0.5, -4.7, r"intensity$\,/\,$1")
    sebplt.plt.colorbar(cmap, cax=ax, extend="max", orientation="horizontal")
    fig_cmap_filename = "cmap_{:s}.jpg".format(cmap_key)
    fig.savefig(os.path.join(cmap_dir, fig_cmap_filename))
    sebplt.close(fig)
