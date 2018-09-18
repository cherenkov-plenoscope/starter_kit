import plenopy as pl
import lixel_summation_tools as lst
import matplotlib.pyplot as plt
import scipy

light_field_geometry = pl.LightFieldGeometry('./run/light_field_calibration/')

object_distance = 999e3

c_radii = np.deg2rad(np.linspace(0, 3, 6))
c_direction = np.array([1, 1])
c_direction_norm = np.linalg.norm(c_direction)
cxys = []
for c_radius in c_radii:
    cxys.append(c_direction/c_direction_norm*c_radius)
cxys = np.array(cxys)

pixel_ids = []
for cxy in cxys:
    pixel_ids .append(
        light_field_geometry.pixel_pos_tree.query(cxy)[1])


lixel_cx_cy_tree = scipy.spatial.cKDTree(
    np.array([  light_field_geometry.cx_mean,
                light_field_geometry.cy_mean]).T)


pixel_radius = np.deg2rad(0.06667)*0.6
masks = []
for cxy in cxys:
    proto_mask = lixel_cx_cy_tree.query(cxy, k=2000)
    mask = []
    for j, lixel_distance in enumerate(proto_mask[0]):
        if lixel_distance <= pixel_radius:
            mask.append(proto_mask[1][j])
    masks.append(np.array(mask))




figure_radius = 0.35
w = 6
h = 6
dpi = 128
"""
for i, pixel_id in enumerate(pixel_ids):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    lixel_ids = masks[i]
    x_center = np.mean(light_field_geometry.lixel_positions_x[lixel_ids])
    y_center = np.mean(light_field_geometry.lixel_positions_y[lixel_ids])

    ax = fig.add_axes([0.12, 0.12, 0.85, 0.85])
        # [x0, y0, width, height]

    mask = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
    mask[lixel_ids] = True

    pl.plot.light_field_geometry.colored_lixels(
        lss=light_field_geometry,
        I=mask,
        ax=ax,
        cmap='binary',
        edgecolors='k')

    ax.set_xlabel('photo-sensor-plane-x/m')
    ax.set_ylabel('photo-sensor-plane-y/m')

    ax.set_xlim([x_center - figure_radius, x_center + figure_radius])
    ax.set_ylim([y_center - figure_radius, y_center + figure_radius])

    plt.savefig(
        'aberration_pixel_{pixel:0d}_{direction:0d}mdeg.png'.format(
            pixel=pixel_id,
            direction=int(1000*np.rad2deg(c_radii[i]))),
        dpi=dpi)
    plt.close('all')
"""
# all spreads overview
# --------------------

w = 6
h = 6
dpi = 128*2
fig = plt.figure(figsize=(w, h), dpi=dpi)
ax = fig.add_axes([0.12, 0.12, 0.85, 0.85])
# [x0, y0, width, height]

mask = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
for lixel_ids in masks:
    mask[lixel_ids] = True

pl.plot.light_field_geometry.colored_lixels(
    lss=light_field_geometry,
    I=mask,
    ax=ax,
    cmap='binary',
    edgecolors='lightgrey',
    linewidths=(0.02,))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('photo-sensor-plane-x/m')
ax.set_ylabel('photo-sensor-plane-y/m')

plt.savefig(
    'aberration_overview.jpg',
    dpi=dpi)
plt.close('all')
