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
import binning_utils

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

argv = sys.argv
if argv[0] == "ipython" and argv[1] == "-i":
    argv.pop(1)

work_dir = argv[1]
out_dir = os.path.join(work_dir, "figures", "deformation_polygon")
os.makedirs(out_dir, exist_ok=True)

config = abe.deformations.read_config(work_dir=work_dir)

deformation_polynom = config["deformation_polynom"]

N = 512
R = config["mirror"]["outer_radius"]

x_bin_edges = np.linspace(-R, R, N+1)
y_bin_edges = np.linspace(-R, R, N+1)
x_bin_centers = binning_utils.centers(x_bin_edges)
y_bin_centers = binning_utils.centers(y_bin_edges)

zs = np.zeros(shape=(N, N))

VMINMAX_M = 0.1

for ix in range(N):
    for iy in range(N):
        x = x_bin_centers[ix]
        y = y_bin_centers[iy]
        inside = abe.deformations.parabola_segmented.is_inside_hexagon(
            position=[x, y, 0],
            hexagon_inner_radius=config["mirror"]["inner_radius"],
        )
        if inside:
            z = abe.deformations.parabola_segmented.z_deformation(
                x=x,
                y=y,
                deformation_polynom=deformation_polynom,
            )
        else:
            z = 0.0
        zs[ix, iy] = z

assert np.all(np.abs(zs) < VMINMAX_M)

fig = sebplt.figure(style={"rows": 1000, "cols": 1280, "fontsize": 1.1})
ax = sebplt.add_axes(fig, [0.05, 0.15, 0.8, 0.8])
cax = sebplt.add_axes(fig, [0.8, 0.15, 0.03, 0.8])
cmap = ax.pcolormesh(
    x_bin_edges,
    y_bin_edges,
    np.transpose(zs)*1e3,
    cmap="seismic",
    vmax=VMINMAX_M*1e3,
    vmin=-VMINMAX_M*1e3,
)
ax.set_aspect("equal")
sebplt.ax_add_grid(ax)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
cbar.set_label("z / mm")
fig.savefig(os.path.join(out_dir, "deformation_polygon.jpg"))
sebplt.close(fig)
