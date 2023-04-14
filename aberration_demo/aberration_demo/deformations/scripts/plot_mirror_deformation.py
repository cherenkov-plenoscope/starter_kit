import os
import plenoirf
import plenopy
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

figstyle = {"rows": 1200, "cols": 1440, "fontsize": 1.4}
ax_span = [0.075, 0.125, 0.8, 0.8]
cax_span = [0.85, 0.125, 0.03, 0.8]
axstyle = {"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}

config = abe.utils.read_json(path=os.path.join(work_dir, "config.json"))

demfap_zeor = abe.deformations.deformation_map.init_from_mirror_and_deformation_configs(
    mirror_dimensions=config["mirror"]["dimensions"],
    mirror_deformation=config["mirror"]["deformation"],
    amplitude_scaleing=0.0,
)

demfap = abe.deformations.deformation_map.init_from_mirror_and_deformation_configs(
    mirror_dimensions=config["mirror"]["dimensions"],
    mirror_deformation=config["mirror"]["deformation"],
)

facets = abe.deformations.parabola_segmented.make_facets(
    mirror_dimensions=config["mirror"]["dimensions"],
    mirror_deformation_map=demfap,
)

STEP_Z_M = 0.005
STEP_ANGLE_DEG = 0.1

N = 128
R_hex_outer = config["mirror"]["dimensions"]["max_outer_aperture_radius"]
R_hex_inner = R_hex_outer * np.sqrt(3) / 2

facets_x_m = []
facets_y_m = []
facets_z_m = []
facets_a_deg = []
for facet in facets:
    x = facet["pos"][0]
    y = facet["pos"][1]
    facets_x_m.append(x)
    facets_y_m.append(y)
    z = abe.deformations.deformation_map.evaluate(
        deformation_map=demfap, x_m=x, y_m=y,
    )
    facets_z_m.append(z)

    actual_surface_normal = abe.deformations.parabola_segmented.mirror_surface_normal(
        x=x,
        y=y,
        focal_length=config["mirror"]["dimensions"]["focal_length"],
        mirror_deformation_map=demfap,
        delta=0.5 * config["mirror"]["dimensions"]["facet_inner_hex_radius"],
    )

    targeted_surface_normal = abe.deformations.parabola_segmented.mirror_surface_normal(
        x=x,
        y=y,
        focal_length=config["mirror"]["dimensions"]["focal_length"],
        mirror_deformation_map=demfap_zeor,
        delta=0.5 * config["mirror"]["dimensions"]["facet_inner_hex_radius"],
    )

    aa_rad = abe.deformations.parabola_segmented.angle_between(
        actual_surface_normal, targeted_surface_normal
    )
    aa_deg = np.rad2deg(aa_rad)
    if np.isnan(aa_deg):
        print("----------------------")
        print("actual_surface_normal", actual_surface_normal)
        print("targeted_surface_normal", targeted_surface_normal)
        print(x, y, z, aa_deg)
        aa_deg = 0.0
    facets_a_deg.append(aa_deg)

facets_x_m = np.array(facets_x_m)
facets_y_m = np.array(facets_y_m)
facets_z_m = np.array(facets_z_m)
facets_a_deg = np.array(facets_a_deg)

ZMINMAX_M = np.max(np.abs(facets_z_m))
ZMINMAX_M = STEP_Z_M * np.ceil(ZMINMAX_M / STEP_Z_M)

AMINMAX_DEG = np.max(np.abs(facets_a_deg))
AMINMAX_DEG = STEP_ANGLE_DEG * np.ceil(AMINMAX_DEG / STEP_ANGLE_DEG)

# deformation in z
# ----------------
fig = sebplt.figure(style=figstyle)
ax = sebplt.add_axes(fig, ax_span, style=axstyle)
cax = sebplt.add_axes(fig, cax_span)
cmap = plenopy.plot.image.add2ax(
    ax=ax,
    I=facets_z_m * 1e3,
    px=facets_x_m,
    py=facets_y_m,
    colormap="seismic",
    hexrotation=30,
    vmin=-ZMINMAX_M * 1e3,
    vmax=ZMINMAX_M * 1e3,
    colorbar=False,
    norm=None,
)
axlabel = r"deformation in $z\,/\,$mm"
ax.set_title(axlabel)
ax.set_aspect("equal")
ax.set_xlim([-R_hex_outer, R_hex_outer])
ax.set_ylim([-R_hex_outer, R_hex_outer])
ax.set_xlabel(r"$x\,/\,$m")
ax.set_ylabel(r"$y\,/\,$m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
# cbar.set_label(axlabel)
fig.savefig(os.path.join(plot_dir, "mirror_deformation_z_only_facets.jpg"))
sebplt.close(fig)

# deformation angle
# -----------------
fig = sebplt.figure(style=figstyle)
ax = sebplt.add_axes(fig, ax_span, style=axstyle)
cax = sebplt.add_axes(fig, cax_span)
cmap = plenopy.plot.image.add2ax(
    ax=ax,
    I=facets_a_deg,
    px=facets_x_m,
    py=facets_y_m,
    colormap="inferno",
    hexrotation=30,
    vmin=0.0,
    vmax=AMINMAX_DEG,
    colorbar=False,
    norm=None,
)
axlabel = r"$\vert$ misalignment $\vert\,/\,1^\circ$"
ax.set_title(axlabel)
ax.set_aspect("equal")
ax.set_xlim([-R_hex_outer, R_hex_outer])
ax.set_ylim([-R_hex_outer, R_hex_outer])
ax.set_xlabel(r"$x\,/\,$m")
ax.set_ylabel(r"$y\,/\,$m")
cbar = sebplt.plt.colorbar(cmap, cax=cax)
# cbar.set_label(axlabel)
fig.savefig(os.path.join(plot_dir, "mirror_deformation_angle_only_facets.jpg"))
sebplt.close(fig)
