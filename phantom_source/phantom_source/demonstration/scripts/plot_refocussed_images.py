#!/usr/bin/python
import numpy as np
import plenopy
import plenoirf
import phantom_source
import json_numpy
import binning_utils
import sebastians_matplotlib_addons as sebplt
import argparse

argparser = argparse.ArgumentParser(
    prog="plot_refocussed_images",
    description=(
        "Plot a stack of refocussed images from the "
        "observation of a phantom-source"
    ),
)

argparser.add_argument(
    "work_dir",
    metavar="WORK_DIR",
    type=str,
    help="Path to the work_dir of the phantom_source.demonstration.",
)

args = argparser.parse_args()

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

work_dir = args.work_dir
plot_dir = os.path.join(work_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

config = json_numpy.read(os.path.join(work_dir, "config.json"))
prng = np.random.Generator(np.random.MT19937(seed=config["seed"]))

light_field_geometry = plenopy.LightFieldGeometry(
    config["light_field_geometry_path"]
)
max_FoV_diameter_deg = np.rad2deg(
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
)

phantom_source_light_field = phantom_source.demonstration.read_light_field(
    path=os.path.join(work_dir, "phantom_source_light_field")
)

image_edge_ticks_deg = np.linspace(-3, 3, 7)
image_edge_bin = binning_utils.Binning(
    bin_edges=np.deg2rad(np.linspace(-3.5, 3.5, int(3 * (7 / 0.067)))),
)
image_bins = [image_edge_bin["edges"], image_edge_bin["edges"]]

# from 1210_demonstrate_resolution_of_depth
# -----------------------------------------
systematic_reco_over_true = 1.0169723853658978

TRUE_DEPTH = phantom_source.demonstration.TRUE_DEPTH
object_distances = np.array([TRUE_DEPTH[k] for k in TRUE_DEPTH])
reco_object_distances = systematic_reco_over_true * object_distances


CMAPS = {
    "magma_r": {"gamma": 1.0, "linecolor": "black",},
}


images_dir = os.path.join(work_dir, "images")
os.makedirs(images_dir, exist_ok=True)

img_vmax = 0.0
for obj_idx in range(len(reco_object_distances)):
    reco_object_distance = reco_object_distances[obj_idx]

    image_path = os.path.join(images_dir, "{:06d}.float32".format(obj_idx))

    if os.path.exists(image_path):
        img = phantom_source.demonstration.read_image(path=image_path)
    else:
        img = phantom_source.demonstration.compute_image(
            light_field_geometry=light_field_geometry,
            light_field=phantom_source_light_field,
            object_distance=reco_object_distance,
            bins=image_bins,
            prng=prng,
        )
        phantom_source.demonstration.write_image(path=image_path, image=img)

    img_vmax = np.max([img_vmax, np.max(img)])

for cmapkey in CMAPS:
    for obj_idx in range(len(object_distances)):
        image_path = os.path.join(images_dir, "{:06d}.float32".format(obj_idx))
        img = phantom_source.demonstration.read_image(path=image_path)

        fig_filename = "{:06d}_{:s}.jpg".format(obj_idx, cmapkey)
        fig_path = os.path.join(plot_dir, fig_filename)

        fig = sebplt.figure(
            style={"rows": 1280, "cols": 1280, "fontsize": 1.0}
        )
        ax = sebplt.add_axes(fig=fig, span=[0.0, 0.0, 1, 1],)
        ax.set_aspect("equal")
        cmap = ax.pcolormesh(
            np.rad2deg(image_edge_bin["edges"]),
            np.rad2deg(image_edge_bin["edges"]),
            np.transpose(img) / img_vmax,
            cmap=cmapkey,
            norm=sebplt.plt_colors.PowerNorm(
                gamma=CMAPS[cmapkey]["gamma"], vmin=0.0, vmax=1.0,
            ),
        )
        sebplt.ax_add_circle(
            ax=ax,
            x=0.0,
            y=0.0,
            r=0.5 * max_FoV_diameter_deg,
            linewidth=0.66,
            linestyle="-",
            color=CMAPS[cmapkey]["linecolor"],
            alpha=0.1,
            num_steps=360 * 5,
        )
        ax.set_xlim(np.rad2deg(image_edge_bin["limits"]))
        ax.set_ylim(np.rad2deg(image_edge_bin["limits"]))
        sebplt.ax_add_grid_with_explicit_ticks(
            ax=ax,
            xticks=image_edge_ticks_deg,
            yticks=image_edge_ticks_deg,
            alpha=0.1,
        )
        fig.savefig(fig_path)
        sebplt.close(fig)

    # colormap
    # --------

    fig_cmap = sebplt.figure(style={"rows": 120, "cols": 1280, "fontsize": 1})
    ax_cmap = sebplt.add_axes(fig_cmap, [0.1, 0.8, 0.8, 0.15])
    ax_cmap.text(0.5, -4.7, "intensity / 1")
    sebplt.plt.colorbar(
        cmap, cax=ax_cmap, extend="max", orientation="horizontal"
    )
    fig_cmap_filename = "cmap_{:s}.jpg".format(cmapkey)
    fig_cmap.savefig(os.path.join(plot_dir, fig_cmap_filename))
    sebplt.close(fig_cmap)


phantom_source_meshes_img = json_numpy.read(
    os.path.join(work_dir, "phantom_source_meshes_img.json")
)

fig_filename = "phantom_source_meshes.jpg"
fig_path = os.path.join(plot_dir, fig_filename)

fig = sebplt.figure(style={"rows": 1280, "cols": 1280, "fontsize": 1.0})
ax = sebplt.add_axes(fig=fig, span=[0.0, 0.0, 1, 1],)
for mesh in phantom_source_meshes_img:
    phantom_source.plot.ax_add_mesh(ax=ax, mesh=mesh, color="k")
ax.set_aspect("equal")
sebplt.ax_add_circle(
    ax=ax,
    x=0.0,
    y=0.0,
    r=0.5 * max_FoV_diameter_deg,
    linewidth=0.66,
    linestyle="-",
    color=CMAPS[cmapkey]["linecolor"],
    alpha=0.1,
    num_steps=360 * 5,
)
ax.set_xlim(np.rad2deg(image_edge_bin["limits"]))
ax.set_ylim(np.rad2deg(image_edge_bin["limits"]))
sebplt.ax_add_grid_with_explicit_ticks(
    ax=ax, xticks=image_edge_ticks_deg, yticks=image_edge_ticks_deg, alpha=0.1
)
fig.savefig(fig_path)
sebplt.close(fig)


fig_filename = "phantom_source_meshes_3d.jpg"
fig_path = os.path.join(plot_dir, fig_filename)

fig = sebplt.figure(style={"rows": 2000, "cols": 1280, "fontsize": 1.0})
ax = sebplt.add_axes(fig=fig, span=[0.0, 0.0, 1, 1], style=sebplt.AXES_BLANK)

UU = 0.1
the = np.deg2rad(50)

for imesh, mesh in enumerate(phantom_source_meshes_img):
    zz = 5.3 * imesh
    projection = np.array(
        [
            [np.cos(the), -np.sin(the) * (1.0 / UU), 0],
            [np.sin(the), np.cos(the) * UU, zz],
            [0, 0, 1],
        ]
    )
    depth_m = object_distances[imesh]
    depth_km = 1e-3 * depth_m
    ax.text(
        x=5 * 0.5 * max_FoV_diameter_deg,
        y=zz - 2.5,
        s="{: 2.1f}".format(depth_km) + r"$\,$" + "km",
        fontsize=12 * 1.5,
    )
    sebplt.pseudo3d.ax_add_grid(
        ax=ax,
        projection=projection,
        x_bin_edges=image_edge_ticks_deg,
        y_bin_edges=image_edge_ticks_deg,
        alpha=0.33,
        linewidth=0.2,
        color="k",
        linestyle="-",
    )
    sebplt.pseudo3d.ax_add_circle(
        ax=ax,
        projection=projection,
        x=0.0,
        y=0.0,
        r=0.5 * max_FoV_diameter_deg,
        alpha=0.33,
        linewidth=0.2,
        color="k",
        linestyle="-",
    )
    sebplt.pseudo3d.ax_add_mesh(
        ax=ax, projection=projection, mesh=mesh, color="k", linestyle="-",
    )

fig.savefig(fig_path)
sebplt.close(fig)
