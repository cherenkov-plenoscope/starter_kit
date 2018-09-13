import plenopy as pl
import matplotlib.pyplot as plt

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

object_distances = [7.5e3, 11e3, 15e3, 21e3, 29e3, 999e3]
pixel_id = light_field_geometry.number_pixel//2


def make_summation_mask_for_pixel(
    pixel_id,
    object_distance,
    light_field_geometry,
):
    number_nearest_neighbor_pixels = 1
    c_epsilon = 2*np.deg2rad(0.11)
    image_rays = pl.image.ImageRays(light_field_geometry)
    cx, cy = image_rays.cx_cy_in_object_distance(object_distance)
    cxy = np.vstack((cx, cy)).T
    distances, pixel_indicies = image_rays.pixel_pos_tree.query(
        x=cxy,
        k=number_nearest_neighbor_pixels)
    lixel_summation = [[] for i in range(light_field_geometry.number_lixel)]
    for lix in range(light_field_geometry.number_lixel):
        if distances[lix] <= c_epsilon:
            lixel_summation[pixel_indicies[lix]].append(lix)

    lixel_summation = pl.trigger.lixel_summation_to_sparse_matrix(
        lixel_summation=lixel_summation,
        number_lixel=light_field_geometry.number_lixel,
        number_pixel=light_field_geometry.number_pixel)

    mask = lixel_summation[pixel_id]
    mask = mask.todense()
    mask = 1.0 * mask
    z = np.zeros(light_field_geometry.number_lixel)
    for i in range(light_field_geometry.number_lixel):
        z[i] = mask.T[i]
    mask = z
    return mask


def add_aperture_plane_to_ax(ax, color='k'):
    c = color
    ax.plot([-1, 1], [0, 0], color=c)
    N = 25
    s = 1/N
    x_starts = np.linspace(-1, 1, N) - s
    x_ends = np.linspace(-1, 1, N)
    for i in range(N):
        ax.plot([x_starts[i], x_ends[i]], [-s, 0], color=c)


def add_rays_to_ax(ax, object_distance, color='k', linewidth=1):
    c = color
    N = 4
    x_starts = np.linspace(-0.9, 0.9, N)
    y_starts = np.zeros(N)

    x_ends = -x_starts*100
    y_ends = 2*object_distance*np.ones(N)*100

    for i in range(N):
        ax.plot(
            [x_starts[i], x_ends[i]],
            [y_starts[i], y_ends[i]],
            color=c,
            linewidth=linewidth)

w = 8
h = 4
dpi = 128
for obj, object_distance in enumerate(object_distances):
    if obj == 0:
        fig = plt.figure(figsize=(w, h*1.1), dpi=dpi)
    else:
        fig = plt.figure(figsize=(w, h), dpi=dpi)

    mask = make_summation_mask_for_pixel(
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
    add_aperture_plane_to_ax(ax=ax2)
    ax2.set_xlim([-1, 1])
    ax2.set_ylim([-0.05, 3.95])
    t = object_distance/1e3/20
    add_rays_to_ax(ax=ax2, object_distance=t, linewidth=.5)
    ax2.text(
        x=0.1,
        y=2*t,
        s='{:0.0f}km'.format(object_distance/1e3),
        fontsize=12)
    if obj+1 == len(object_distances):
        ax2.text(x=0.1, y=3.7, s='infinity', fontsize=12)

    plt.savefig('refocus_lixel_summation_{obj:d}.png'.format(obj=obj), dpi=dpi)
    plt.close('all')
