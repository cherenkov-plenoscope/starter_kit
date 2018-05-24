import plenopy as pl
import matplotlib.pyplot as plt

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

object_distances = [999e3, 35e3, 23e3, 15e3, 10e3]
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

fig_h = 4
fig_w = 1

fsize = pl.plot.FigureSize(
    relative_width=fig_w,
    relative_hight=fig_h,
    pixel_rows=1280,
    dpi=400)

panel_h = 1
panel_w = 1

fig = plt.figure(figsize=(fsize.width, fsize.hight), dpi=fsize.dpi)

for obj, object_distance in enumerate(object_distances):

    mask = make_summation_mask_for_pixel(
        pixel_id=pixel_id,
        object_distance=object_distance,
        light_field_geometry=light_field_geometry)

    ax = fig.add_axes(
        (
            (l_anchor+space_w+ruler_w)/fig_w,
            b_anchor/fig_h,
            panel_w/fig_w,
            panel_h/fig_h))

    pl.plot.light_field_geometry.colored_lixels(
        lss=light_field_geometry,
        I=mask,
        ax=ax,
        cmap='binary',
        edgecolors='k')
    ax.set_xlabel('photo-sensor-plane-x/m')
    ax.set_ylabel('photo-sensor-plane-y/m')
    ax.set_xlim([-.35,.35])
    ax.set_ylim([-.35,.35])
    plt.savefig('pixel_patch_{obj:d}.jpg'.format(obj=obj))
    plt.close('all')