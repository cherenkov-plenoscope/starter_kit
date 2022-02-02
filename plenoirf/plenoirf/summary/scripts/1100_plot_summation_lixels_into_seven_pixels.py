#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import os
import json_numpy
import sebastians_matplotlib_addons as seb
from matplotlib.collections import PolyCollection

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

object_distances = [21e3, 29e3, 999e3]
central_seven_pixel_ids = [4221, 4124, 4222, 4220, 4125, 4317, 4318]
colors = ["k", "g", "b", "r", "c", "m", "orange"]

edgecolors = "none"
linewidths = None

pixel_spacing_rad = (
    light_field_geometry.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat
)
image_outer_radius_rad = 0.5 * (
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
    - pixel_spacing_rad
)

image_geometry = pl.trigger.geometry.init_trigger_image_geometry(
    image_outer_radius_rad=image_outer_radius_rad,
    pixel_spacing_rad=pixel_spacing_rad,
    pixel_radius_rad=pixel_spacing_rad / 2,
    max_number_nearest_lixel_in_pixel=7,
)

for obj, object_distance in enumerate(object_distances):
    if obj == 0:
        fig = seb.figure(style=seb.FIGURE_1_1)
        ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])
    else:
        fig = seb.figure(style=seb.FIGURE_1_1)
        ax = seb.add_axes(fig=fig, span=[0.175, 0.15, 0.75, 0.8])

    cpath = os.path.join(
        pa["out_dir"], "lixel_to_pixel_{:06d}.json".format(obj)
    )

    if not os.path.exists(cpath):
        lixel_to_pixel = pl.trigger.geometry.estimate_projection_of_light_field_to_image(
            light_field_geometry=light_field_geometry,
            object_distance=object_distance,
            image_pixel_cx_rad=image_geometry["pixel_cx_rad"],
            image_pixel_cy_rad=image_geometry["pixel_cy_rad"],
            image_pixel_radius_rad=image_geometry["pixel_radius_rad"],
            max_number_nearest_lixel_in_pixel=image_geometry[
                "max_number_nearest_lixel_in_pixel"
            ],
        )

        json_numpy.write(path=cpath, out_dict=lixel_to_pixel, indent=None)
    else:
        lixel_to_pixel = json_numpy.read(path=cpath)

    colored_lixels = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    for i, pixel_id in enumerate(central_seven_pixel_ids):

        valid_polygons = []
        additional_colored_lixels = np.zeros(
            light_field_geometry.number_lixel, dtype=np.bool
        )
        for j, poly in enumerate(light_field_geometry.lixel_polygons):
            if pixel_id in lixel_to_pixel[j]:
                valid_polygons.append(poly)
                additional_colored_lixels[j] = True

        coll = PolyCollection(
            valid_polygons,
            facecolors=[colors[i] for _ in range(len(valid_polygons))],
            edgecolors=edgecolors,
            linewidths=linewidths,
        )
        ax.add_collection(coll)

        colored_lixels += additional_colored_lixels

    not_colored = np.invert(colored_lixels)
    not_colored_polygons = []
    for j, poly in enumerate(light_field_geometry.lixel_polygons):
        if not_colored[j]:
            not_colored_polygons.append(poly)

    coll = PolyCollection(
        not_colored_polygons,
        facecolors=["w" for _ in range(len(not_colored_polygons))],
        edgecolors="k",
        linewidths=linewidths,
    )
    ax.add_collection(coll)

    ax.set_aspect("equal")
    if obj == 0:
        ax.set_xlabel("photo-sensor-plane-x/m")
        ax.set_ylabel("photo-sensor-plane-y/m")
    else:
        ax.get_xaxis().set_visible(False)
    ax.set_xlim([-0.35, 0.35])
    ax.set_ylim([-0.35, 0.35])

    if obj == 0:
        ax2 = fig.add_axes([0.66, 0.1, 0.33, 1])
    else:
        ax2 = fig.add_axes([0.66, 0.0, 0.33, 1])

    ax2.set_aspect("equal")
    ax2.set_axis_off()
    irf.summary.figure.add_aperture_plane_to_ax(ax=ax2)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 3.95])
    t = object_distance / 1e3 / 20
    irf.summary.figure.add_rays_to_ax(ax=ax2, object_distance=t, linewidth=0.5)
    ax2.text(
        x=0.1,
        y=2 * t,
        s="{:0.0f}km".format(object_distance / 1e3),
        fontsize=12,
    )
    if obj + 1 == len(object_distances):
        ax2.text(x=0.1, y=3.7, s="infinity", fontsize=12)

    fig.savefig(
        os.path.join(
            pa["out_dir"],
            "refocus_lixel_summation_7_{obj:d}.png".format(obj=obj),
        )
    )
    seb.close("all")
