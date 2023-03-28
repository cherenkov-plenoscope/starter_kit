import os
import plenoirf
import numpy as np
import scipy
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
plot_dir = os.path.join(work_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

config = abe.utils.read_json(path=os.path.join(work_dir, "config.json"))

demfap = abe.deformations.deformation_map.init_from_mirror_and_deformation_configs(
    mirror_config=config["mirror"],
    mirror_deformation_config=config["mirror_deformation"],
)
demfap_zeor = abe.deformations.deformation_map.init_from_mirror_and_deformation_configs(
    mirror_config=config["mirror"],
    mirror_deformation_config=config["mirror_deformation"],
    amplitude_scaleing=0.0,
)
facets = abe.deformations.parabola_segmented.make_facets(
    mirror_config=config["mirror"], mirror_deformation=demfap,
)

STEP_Z_M = 0.005
STEP_ANGLE_DEG = 0.1

N = 64
R_hex_outer = config["mirror"]["max_outer_aperture_radius"]
R_hex_inner = R_hex_outer * np.sqrt(3) / 2

x_bin_edges = np.linspace(-R_hex_outer, R_hex_outer, N + 1)
y_bin_edges = np.linspace(-R_hex_outer, R_hex_outer, N + 1)
x_bin_centers = binning_utils.centers(x_bin_edges)
y_bin_centers = binning_utils.centers(y_bin_edges)

deformation_z_m = np.zeros(shape=(N, N))
misalignment_angle_rad = np.zeros(shape=(N, N))


for ix in range(N):
    for iy in range(N):
        x = x_bin_centers[ix]
        y = y_bin_centers[iy]
        inside = abe.deformations.parabola_segmented.is_inside_hexagon(
            position=[x, y, 0], hexagon_inner_radius=R_hex_inner,
        )
        if inside:
            z = abe.deformations.deformation_map.evaluate(
                deformation_map=demfap, x_m=x, y_m=y,
            )

            actual_surface_normal = abe.deformations.parabola_segmented.mirror_surface_normal(
                x=x,
                y=y,
                focal_length=config["mirror"]["focal_length"],
                deformation=demfap,
                delta=0.5 * config["mirror"]["facet_inner_hex_radius"],
            )

            targeted_surface_normal = abe.deformations.parabola_segmented.mirror_surface_normal(
                x=x,
                y=y,
                focal_length=config["mirror"]["focal_length"],
                deformation=demfap_zeor,
                delta=0.5 * config["mirror"]["facet_inner_hex_radius"],
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

        misalignment_angle_rad[ix, iy] = aa
        deformation_z_m[ix, iy] = z

ZMINMAX_M = np.max(np.abs(deformation_z_m))
ZMINMAX_M = STEP_Z_M * np.ceil(ZMINMAX_M / STEP_Z_M)

fig = sebplt.figure(style={"rows": 1200, "cols": 1440, "fontsize": 1.1})
ax = sebplt.add_axes(fig, [0.075, 0.125, 0.8, 0.8])
cax = sebplt.add_axes(fig, [0.85, 0.125, 0.03, 0.8])
cmap = ax.pcolormesh(
    x_bin_edges,
    y_bin_edges,
    np.transpose(deformation_z_m) * 1e3,
    cmap="seismic",
    vmax=ZMINMAX_M * 1e3,
    vmin=-ZMINMAX_M * 1e3,
)

"""
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
"""
ax.set_aspect("equal")
ax.set_xlim([-R_hex_outer, R_hex_outer])
ax.set_ylim([-R_hex_outer, R_hex_outer])
sebplt.ax_add_grid(ax)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
cbar.set_label("z / mm")
fig.savefig(os.path.join(plot_dir, "deformation_map.jpg"))
sebplt.close(fig)


misalignment_angle_deg = np.rad2deg(misalignment_angle_rad)

AMINMAX_DEG = np.max(np.abs(misalignment_angle_deg))
AMINMAX_DEG = STEP_ANGLE_DEG * np.ceil(AMINMAX_DEG / STEP_ANGLE_DEG)


fig = sebplt.figure(style={"rows": 1200, "cols": 1440, "fontsize": 1.1})
ax = sebplt.add_axes(fig, [0.075, 0.125, 0.8, 0.8])
cax = sebplt.add_axes(fig, [0.85, 0.125, 0.03, 0.8])
cmap = ax.pcolormesh(
    x_bin_edges,
    y_bin_edges,
    np.transpose(misalignment_angle_deg),
    cmap="viridis",
    vmax=AMINMAX_DEG,
    vmin=0.0,
)
ax.set_aspect("equal")
ax.set_xlim([-R_hex_outer, R_hex_outer])
ax.set_ylim([-R_hex_outer, R_hex_outer])
sebplt.ax_add_grid(ax)
ax.set_xlabel("x / m")
ax.set_ylabel("y / m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
cbar.set_label("misalignment / 1$^\circ$")
fig.savefig(os.path.join(plot_dir, "deformation_map_normal.jpg"))
sebplt.close(fig)
