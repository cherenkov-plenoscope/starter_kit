import plenopy as pl
import lixel_summation_tools as lst
import matplotlib.pyplot as plt

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

object_distances = [7.5e3, 11e3, 15e3, 21e3, 29e3, 999e3]
pixel_id = light_field_geometry.number_pixel//2


w = 8
h = 4
dpi = 128
for obj, object_distance in enumerate(object_distances):
    if obj == 0:
        fig = plt.figure(figsize=(w, h*1.1), dpi=dpi)
    else:
        fig = plt.figure(figsize=(w, h), dpi=dpi)

    mask = lst.make_summation_mask_for_pixel(
        pixel_id=pixel_id,
        object_distance=object_distance,
        light_field_geometry=light_field_geometry)

    if obj == 0:
        ax = fig.add_axes([0, .1, .66, .9])
    else:
        ax = fig.add_axes([0, 0, .66, 1])
    # [x0, y0, width, height]

    pl.plot.light_field_geometry.colored_lixels(
        lss=light_field_geometry,
        I=mask,
        ax=ax,
        cmap='binary',
        edgecolors='k')

    if obj == 0:
        ax.set_xlabel('photo-sensor-plane-x/m')
        ax.set_ylabel('photo-sensor-plane-y/m')
    else:
        ax.get_xaxis().set_visible(False)
    ax.set_xlim([-.35, .35])
    ax.set_ylim([-.35, .35])


    if obj == 0:
        ax2 = fig.add_axes([0.66, 0.1, .33, .9])
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

    plt.savefig('refocus_lixel_summation_{obj:d}.png'.format(obj=obj), dpi=dpi)
    plt.close('all')
