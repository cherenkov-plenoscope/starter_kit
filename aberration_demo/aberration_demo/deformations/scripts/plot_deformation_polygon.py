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

demfap = abe.deformations.deformation_map.read(
    path=abe.deformations.deformation_map.EXAMPLE_DEFORMATION_MAP_PATH
)

demfap_zeor = abe.deformations.deformation_map.init_zero(
    mirror_diameter_m=(
        demfap["pixel_bin"]["limits"][1] - demfap["pixel_bin"]["limits"][0]
    ),
)

facets = abe.deformations.parabola_segmented.make_facets(
    mirror_config=config["mirror"], deformation=demfap,
)

N = 512
R_inner = 71 / 2
R_outer = 2 / np.sqrt(3) * R_inner
f = config["mirror"]["focal_length"]
fihr = config["mirror"]["facet_inner_hex_radius"]

x_bin_edges = np.linspace(-R_outer, R_outer, N + 1)
y_bin_edges = np.linspace(-R_outer, R_outer, N + 1)
x_bin_centers = binning_utils.centers(x_bin_edges)
y_bin_centers = binning_utils.centers(y_bin_edges)

zs = np.zeros(shape=(N, N))
an = np.zeros(shape=(N, N))

STEP5MM = 0.005

for ix in range(N):
    for iy in range(N):
        x = x_bin_centers[ix]
        y = y_bin_centers[iy]
        inside = abe.deformations.parabola_segmented.is_inside_hexagon(
            position=[x, y, 0], hexagon_inner_radius=R_inner,
        )
        if inside:
            z = abe.deformations.deformation_map.evaluate(
                deformation_map=demfap, x_m=x, y_m=y,
            )

            actual_surface_normal = abe.deformations.parabola_segmented.surface_normal(
                x=x, y=y, focal_length=f, deformation=demfap, delta=0.5 * fihr,
            )

            targeted_surface_normal = abe.deformations.parabola_segmented.surface_normal(
                x=x,
                y=y,
                focal_length=f,
                deformation=demfap_zeor,
                delta=0.5 * fihr,
            )

            aa = abe.deformations.parabola_segmented.angle_between(
                actual_surface_normal, targeted_surface_normal
            )

            if np.isnan(aa):
                print("----------------------")
                print("actual_surface_normal", actual_surface_normal)
                print("targeted_surface_normal", targeted_surface_normal)
                print(x, y, z, np.rad2deg(aa))
                aa = 0.0

        else:
            z = 0.0
            aa = 0.0

        an[ix, iy] = aa
        zs[ix, iy] = z

VMINMAX_M = np.max(np.abs(zs))
VMINMAX_M = STEP5MM * np.ceil(VMINMAX_M / STEP5MM)

fig = sebplt.figure(style={"rows": 1200, "cols": 1440, "fontsize": 1.1})
ax = sebplt.add_axes(fig, [0.075, 0.125, 0.8, 0.8])
cax = sebplt.add_axes(fig, [0.85, 0.125, 0.03, 0.8])
cmap = ax.pcolormesh(
    x_bin_edges,
    y_bin_edges,
    np.transpose(zs) * 1e3,
    cmap="seismic",
    vmax=VMINMAX_M * 1e3,
    vmin=-VMINMAX_M * 1e3,
)

for facet in facets:
    fx = facet["pos"][0]
    fy = facet["pos"][1]
    fr = facet["outer_radius"]
    # ax.plot(fx, fy, "k,")

    ffxx = []
    ffyy = []
    for fphi in np.linspace(0, np.pi * 2, 7):
        ffx = fx + np.cos(fphi) * fr
        ffy = fy + np.sin(fphi) * fr
        ffxx.append(ffx)
        ffyy.append(ffy)
    ax.plot(ffxx, ffyy, "k-", linewidth=0.1)

ax.set_aspect("equal")
ax.set_xlim([-R_outer, R_outer])
ax.set_ylim([-R_outer, R_outer])
sebplt.ax_add_grid(ax)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
cbar.set_label("z / mm")
fig.savefig(os.path.join(out_dir, "deformation_map.jpg"))
sebplt.close(fig)


fig = sebplt.figure(style={"rows": 1200, "cols": 1440, "fontsize": 1.1})
ax = sebplt.add_axes(fig, [0.075, 0.125, 0.8, 0.8])
cax = sebplt.add_axes(fig, [0.85, 0.125, 0.03, 0.8])
cmap = ax.pcolormesh(
    x_bin_edges,
    y_bin_edges,
    np.transpose(np.rad2deg(an)),
    cmap="viridis",
    vmax=np.rad2deg(np.max(an)),
    vmin=1e-2,
    norm=sebplt.matplotlib.colors.LogNorm(),
)
ax.set_aspect("equal")
ax.set_xlim([-R_outer, R_outer])
ax.set_ylim([-R_outer, R_outer])
sebplt.ax_add_grid(ax)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
cbar.set_label("angle / deg")
fig.savefig(os.path.join(out_dir, "deformation_map_normal.jpg"))
sebplt.close(fig)
