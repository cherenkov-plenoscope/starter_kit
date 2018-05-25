import plenopy as pl
import matplotlib.pyplot as plt

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

object_distances = [10e3, 18e3, 32e3, 999e3]
pixel_id = light_field_geometry.number_pixel//2

def make_summation_mask_for_pixel(
    pixel_id,
    object_distance,
    light_field_geometry,
):
    number_nearest_neighbor_pixels = 1
    c_epsilon = 2*np.deg2rad(0.07)
    image_rays =  pl.image.ImageRays(light_field_geometry)
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
        ax.plot([x_starts[i], x_ends[i]], [y_starts[i], y_ends[i]],
            color=c,
            linewidth=linewidth)

num_panels = len(object_distances)

panel_h = 1
panel_w = 1

tele_h = panel_h
tele_w = 0.5

space_h = 0.1
space_w = 0.1
top_margin = 0.0
bottom_margin = 0.2
l_margin = 0.2
r_margin = 0.1

fig_h = top_margin + num_panels*(panel_h + space_h) + bottom_margin
fig_w = l_margin + panel_w + space_w + tele_w + r_margin

fsize = pl.plot.FigureSize(
    relative_width=fig_w,
    relative_hight=fig_h,
    pixel_rows=640*fig_h,
    dpi=200)

fig = plt.figure(figsize=(fsize.width, fsize.hight), dpi=fsize.dpi)

for obj, object_distance in enumerate(object_distances):

    mask = make_summation_mask_for_pixel(
        pixel_id=pixel_id,
        object_distance=object_distance,
        light_field_geometry=light_field_geometry)

    ax = fig.add_axes(
        (l_margin/fig_w, (obj*(panel_h+space_h) + bottom_margin)/fig_h,
        panel_w/fig_w, panel_h/fig_h)
    )
    """
        (
            (l_anchor+space_w+ruler_w)/fig_w, # pos rel w
            b_anchor/fig_h, # pos rel h
            panel_w/fig_w, # rel w
            panel_h/fig_h)) # rel h
    """

    pl.plot.light_field_geometry.colored_lixels(
        lss=light_field_geometry,
        I=mask,
        ax=ax,
        cmap='binary',
        edgecolors='k')

    #x = np.linspace(0,1,100)
    #ax.plot(x, x**2,)

    if obj == 0:
        ax.set_xlabel('photo-sensor-plane-x/m')
        ax.set_ylabel('photo-sensor-plane-y/m')
    else:
        ax.get_xaxis().set_visible(False)
    ax.set_xlim([-.35,.35])
    ax.set_ylim([-.35,.35])


    ax2 = fig.add_axes(
        ((l_margin + panel_w + space_w)/fig_w, (obj*(panel_h+space_h) + bottom_margin)/fig_h,
        tele_w/fig_w, tele_h/fig_h)
    )
    ax2.set_aspect('equal')
    ax2.set_axis_off()
    add_aperture_plane_to_ax(ax=ax2)
    ax2.set_xlim([-1,1])
    ax2.set_ylim([-0.05,3.95])
    t = object_distance/1e3/20
    add_rays_to_ax(ax=ax2, object_distance=t, linewidth=.5)
    ax2.text(x=0.1, y=2*t, s='{:0.0f}km'.format(object_distance/1e3), fontsize=12)
    if obj+1 == len(object_distances):
        ax2.text(x=0.1, y=3.7, s='infinity', fontsize=12)


plt.savefig('pixel_patch_{obj:d}.jpg'.format(obj=obj), dpi=fsize.dpi)
plt.close('all')