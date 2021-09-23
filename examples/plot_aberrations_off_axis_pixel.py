import os
import sys
import plenopy as pl
import scipy

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")

assert len(sys.argv) == 3, "Require path to light_field_geometry and out_dir"
light_field_geometry_path = sys.argv[1]
out_dir = sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(light_field_geometry_path)

object_distance = 999e3

c_radii = np.deg2rad(np.linspace(0, 3, 6))
c_direction = np.array([1, 1])
c_direction_norm = np.linalg.norm(c_direction)
cxys = []
for c_radius in c_radii:
    cxys.append(c_direction / c_direction_norm * c_radius)
cxys = np.array(cxys)
c_names = [chr(c) for c in range(ord("A"), ord("A") + len(c_radii))]

pixel_ids = []
for cxy in cxys:
    pixel_ids.append(light_field_geometry.pixel_pos_tree.query(cxy)[1])


lixel_cx_cy_tree = scipy.spatial.cKDTree(
    np.array([light_field_geometry.cx_mean, light_field_geometry.cy_mean]).T
)


pixel_radius = np.deg2rad(0.06667) * 0.6
masks = []
for cxy in cxys:
    proto_mask = lixel_cx_cy_tree.query(cxy, k=2000)
    mask = []
    for j, lixel_distance in enumerate(proto_mask[0]):
        if lixel_distance <= pixel_radius:
            mask.append(proto_mask[1][j])
    masks.append(np.array(mask))

figure_radius = 0.35
dpi = 320
for i, pixel_id in enumerate(pixel_ids):
    fig = plt.figure(figsize=(960 / dpi, 960 / dpi), dpi=dpi)
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
        cmap="binary",
        edgecolors="grey",
        linewidths=0.67,
    )

    # ax.set_xlabel('sensor-plane-x/m')
    # ax.set_ylabel('sensor-plane-y/m')

    ax.text(
        s=c_names[i], x=0.1, y=0.8, fontsize=48, transform=ax.transAxes,
    )

    ax.set_xlim([x_center - figure_radius, x_center + figure_radius])
    ax.set_ylim([y_center - figure_radius, y_center + figure_radius])

    plt.savefig(
        os.path.join(
            out_dir,
            "aberration_pixel_{pixel:0d}_{direction:0d}mdeg.jpg".format(
                pixel=pixel_id, direction=int(1000 * np.rad2deg(c_radii[i]))
            ),
        ),
        dpi=dpi,
    )
    plt.close("all")

# all spreads overview
# --------------------

fig = plt.figure(figsize=(960 / dpi, 960 / dpi), dpi=dpi)
ax = fig.add_axes([0.16, 0.16, 0.82, 0.82])

mask = np.zeros(light_field_geometry.number_lixel, dtype=np.bool)
for lixel_ids in masks:
    mask[lixel_ids] = True

pl.plot.light_field_geometry.colored_lixels(
    lss=light_field_geometry,
    I=mask,
    ax=ax,
    cmap="binary",
    edgecolors="grey",
    linewidths=(0.02,),
)

ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.set_xlabel("sensor-plane-x/m")
ax.set_ylabel("sensor-plane-y/m")

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

    ax.plot([Ax, Bx], [Ay, By], "k", linewidth=0.5)
    ax.plot([Bx, Cx], [By, Cy], "k", linewidth=0.5)
    ax.plot([Cx, Dx], [Cy, Dy], "k", linewidth=0.5)
    ax.plot([Dx, Ax], [Dy, Ay], "k", linewidth=0.5)
    ax.text(
        s=c_names[i],
        x=x_center - 1.0 * figure_radius,
        y=y_center + 1.25 * figure_radius,
    )

plt.savefig(os.path.join(out_dir, "aberration_overview.jpg"), dpi=dpi)
plt.close("all")


with open(os.path.join(out_dir, "aberration_overview.txt"), "wt") as fout:
    for i in range(len(c_radii)):
        lixel_ids = masks[i]
        x_center = np.mean(light_field_geometry.lixel_positions_x[lixel_ids])
        y_center = np.mean(light_field_geometry.lixel_positions_y[lixel_ids])
        fout.write(
            "{:s} & {:d} & {:.2f} & {:.2f}\\\\\n".format(
                c_names[i],
                pixel_ids[i],
                np.rad2deg(c_radii[i]),
                np.hypot(x_center, y_center),
            )
        )
