import plenopy as pl
import lixel_summation_tools as lst
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.collections import PolyCollection
import os

out_dir = os.path.join('examples', 'refocus_lixel_summation_seven_pixels')
light_field_geometry = pl.LightFieldGeometry(
    os.path.join('.', 'run', 'light_field_calibration'))

object_distances = [21e3, 29e3, 999e3]
central_seven_pixel_ids = [4221, 4124, 4222, 4220, 4125, 4317, 4318]
colors = ['k', 'g', 'b', 'r', 'c', 'm', 'orange']

w = 8
h = 4
dpi = 128
edgecolors = 'none'
linewidths = None

for obj, object_distance in enumerate(object_distances):
    if obj == 0:
        fig = plt.figure(figsize=(w, h*1.1), dpi=dpi)
        ax = fig.add_axes([0, .1, .66, .9])
    else:
        fig = plt.figure(figsize=(w, h), dpi=dpi)
        ax = fig.add_axes([0, 0, .66, 1])

    colored_lixels = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    for i, pixel_id in enumerate(central_seven_pixel_ids):
        mask = lst.make_summation_mask_for_pixel(
            pixel_id=pixel_id,
            object_distance=object_distance,
            light_field_geometry=light_field_geometry)

        valid_polygons = []
        for j, poly in enumerate(light_field_geometry.lixel_polygons):
            if mask[j]:
                valid_polygons.append(poly)

        coll = PolyCollection(
            valid_polygons,
            facecolors=[colors[i] for _ in range(len(valid_polygons))],
            edgecolors=edgecolors,
            linewidths=linewidths,)
        ax.add_collection(coll)
        colored_lixels += mask.astype(np.bool)

    not_colored = np.invert(colored_lixels)
    not_colored_polygons = []
    for j, poly in enumerate(light_field_geometry.lixel_polygons):
        if not_colored[j]:
            not_colored_polygons.append(poly)

    coll = PolyCollection(
        not_colored_polygons,
        facecolors=['w' for _ in range(len(not_colored_polygons))],
        edgecolors='k',
        linewidths=linewidths,)
    ax.add_collection(coll)

    ax.set_aspect('equal')
    if obj == 0:
        ax.set_xlabel('photo-sensor-plane-x/m')
        ax.set_ylabel('photo-sensor-plane-y/m')
    else:
        ax.get_xaxis().set_visible(False)
    ax.set_xlim([-.35, .35])
    ax.set_ylim([-.35, .35])

    if obj == 0:
        ax2 = fig.add_axes([0.66, .1, .33, 1])
    else:
        ax2 = fig.add_axes([0.66, 0., .33, 1])

    ax2.set_aspect('equal')
    ax2.set_axis_off()
    lst.add_aperture_plane_to_ax(ax=ax2)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 3.95])
    t = object_distance/1e3/20
    lst.add_rays_to_ax(ax=ax2, object_distance=t, linewidth=.5)
    ax2.text(
        x=0.1,
        y=2*t,
        s='{:0.0f}km'.format(object_distance/1e3),
        fontsize=12)
    if obj+1 == len(object_distances):
        ax2.text(x=0.1, y=3.7, s='infinity', fontsize=12)

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(
        os.path.join(
            out_dir,
            'refocus_lixel_summation_7_{obj:d}.png'.format(obj=obj)),
        dpi=dpi)
    plt.close('all')
