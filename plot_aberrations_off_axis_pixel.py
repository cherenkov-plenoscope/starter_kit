import os
import plenopy as pl
import lixel_summation_tools as lst
import matplotlib.pyplot as plt
import scipy

out_dir = os.path.join('.', 'examples', 'aberrations_off_axis_pixel')
os.makedirs(out_dir, exist_ok=True)

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
    np.array([light_field_geometry.cx_mean, light_field_geometry.cy_mean]).T)


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
w = 5
h = 5
dpi = 128

for i, pixel_id in enumerate(pixel_ids):
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    lixel_ids = masks[i]
    x_center = np.mean(light_field_geometry.lixel_positions_x[lixel_ids])
    y_center = np.mean(light_field_geometry.lixel_positions_y[lixel_ids])

    ax = fig.add_axes([0.13, 0.13, 0.87, 0.87])
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
        os.path.join(
            out_dir,
            'aberration_pixel_{pixel:0d}_{direction:0d}mdeg.png'.format(
                pixel=pixel_id,
                direction=int(1000*np.rad2deg(c_radii[i])))),
        dpi=dpi)
    plt.close('all')

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
    edgecolors='grey',
    linewidths=(0.02,))

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel('photo-sensor-plane-x/m')
ax.set_ylabel('photo-sensor-plane-y/m')

for i, mask in enumerate(masks):
    lixel_ids = mask
    x_center = np.mean(light_field_geometry.lixel_positions_x[lixel_ids])
    y_center = np.mean(light_field_geometry.lixel_positions_y[lixel_ids])
    Ax = x_center - figure_radius
    Ay = y_center - figure_radius
    Bx = x_center + figure_radius
    By = y_center - figure_radius

    Cx = x_center + figure_radius
    Cy = y_center + figure_radius
    Dx = x_center - figure_radius
    Dy = y_center + figure_radius

    ax.plot([Ax, Bx], [Ay, By], 'k', linewidth=0.5)
    ax.plot([Bx, Cx], [By, Cy], 'k', linewidth=0.5)
    ax.plot([Cx, Dx], [Cy, Dy], 'k', linewidth=0.5)
    ax.plot([Dx, Ax], [Dy, Ay], 'k', linewidth=0.5)

plt.savefig(
    os.path.join(out_dir, 'aberration_overview.jpg'),
    dpi=dpi)
plt.close('all')


with open(os.path.join(out_dir, 'aberration_overview.txt'), 'wt') as fout:
    for i in range(len(c_radii)):
        lixel_ids = masks[i]
        x_center = np.mean(light_field_geometry.lixel_positions_x[lixel_ids])
        y_center = np.mean(light_field_geometry.lixel_positions_y[lixel_ids])
        fout.write(
            "{:d} & {:.2f} & {:.2f} & {:.1f} & {:.2f} & {:.2f}\\\\\n".format(
                pixel_ids[i],
                np.rad2deg(cxys[i, 0]),
                np.rad2deg(cxys[i, 1]),
                np.rad2deg(c_radii[i]),
                x_center,
                y_center
            )
        )
