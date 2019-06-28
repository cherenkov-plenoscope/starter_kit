import gev_limits as gli
import numpy as np
from scipy import signal

gauss2d = (1./16.)*np.array([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]])

lut = gli.lookup.LookUpTable('__gev_limits_gamma/run.lut/')

cx = np.deg2rad(1.731)
cy = np.deg2rad(0.9997)
delta_c = np.deg2rad(.1)

core_x = -100
core_y = 0
delta_core = 35

num_ph = 80
deltal_num_ph = 25

NUM_PIXEL_ROI = 12
NUM_TIMESLICES_COMPRESSION = 4
NUM_TIMESLICES = 2

mask_c = np.hypot(lut.particle_cx - cx, lut.particle_cy - cy) < delta_c
mask_num_ph = np.abs(lut.num_photons - num_ph) < deltal_num_ph

mask_core = np.hypot(lut.particle_x - core_x, lut.particle_y - core_y) < delta_core

mask = mask_c*mask_num_ph*mask_core

lfs = []
for event in np.arange(lut.num_events)[mask]:
    lfs.append(lut._raw_light_field_sequence(event))
lfs = np.concatenate(lfs)

lfs = lut._raw_light_field_sequence(104)

ROI_CX = int(np.round(np.mean(lfs[:, 0])))
ROI_CY = int(np.round(np.mean(lfs[:, 1])))
ROI_T = int(np.round(np.median(lfs[:, 4])))
cx_bin_edges = np.arange(ROI_CX - NUM_PIXEL_ROI, ROI_CX + NUM_PIXEL_ROI + 1)
cy_bin_edges = np.arange(ROI_CY - NUM_PIXEL_ROI, ROI_CY + NUM_PIXEL_ROI + 1)
t_bin_edges = np.arange(
    ROI_T - NUM_TIMESLICES*NUM_TIMESLICES_COMPRESSION,
    ROI_T + NUM_TIMESLICES*NUM_TIMESLICES_COMPRESSION + NUM_TIMESLICES_COMPRESSION,
    NUM_TIMESLICES_COMPRESSION)

imgs = []
cimgs = []
vmax = 0
for pax_ix in range(lut.plenoscope.num_paxel_on_diagonal):
    ximgs = []
    xcimgs = []
    for pax_iy in range(lut.plenoscope.num_paxel_on_diagonal):
        paxel_mask = (lfs[:, 2] == pax_ix)*(lfs[:, 3] == pax_iy)
        img = np.histogram2d(
            lfs[paxel_mask, 0],
            lfs[paxel_mask, 1],
            bins=(cx_bin_edges, cy_bin_edges))[0]
        cimg = signal.convolve2d(img, gauss2d, mode="same")
        xcimgs.append(cimg)
        ximgs.append(img)
        if np.max(cimg) > vmax:
            vmax = np.max(cimg)
    imgs.append(ximgs)
    cimgs.append(xcimgs)


fig = plt.figure(
        figsize=(16, 16),
        dpi=200)
SUB_FIG_WIDTH = 1/lut.plenoscope.num_paxel_on_diagonal
for pax_ix in range(lut.plenoscope.num_paxel_on_diagonal):
    for pax_iy in range(lut.plenoscope.num_paxel_on_diagonal):
        ax = fig.add_axes([
            SUB_FIG_WIDTH*pax_ix,
            SUB_FIG_WIDTH*pax_iy,
            SUB_FIG_WIDTH*.98,
            SUB_FIG_WIDTH*.98])
        ax.set_axis_off()
        cimg = cimgs[pax_ix][pax_iy]
        ax.pcolor(cimg, vmax=vmax, cmap='inferno')
plt.savefig('integral_image.png')
plt.close("all")
print("num events", mask.sum())

FOCAL_LENGTH = 106.5
PIXEL_RAD = np.mean(np.gradient(lut.plenoscope.cx))
assert PIXEL_RAD == np.mean(np.gradient(lut.plenoscope.cy))
FOV_RADIUS_RAD = np.deg2rad(lut.plenoscope.field_of_view_radius_deg)

NUM_REFS = 7
final_object_distance = 5e3
final_sensor_distance = 1/(1/FOCAL_LENGTH - 1/final_object_distance)
sensor_distances = np.linspace(FOCAL_LENGTH, final_sensor_distance, NUM_REFS)
object_distances = 1/(1/FOCAL_LENGTH - 1/sensor_distances)

cxs = -lut.plenoscope.cx[lfs[:, 0]]
cys = -lut.plenoscope.cy[lfs[:, 1]]
xs = lut.plenoscope.x[lfs[:, 2]]
ys = lut.plenoscope.y[lfs[:, 3]]

focal_x = -np.tan(cxs)*FOCAL_LENGTH
focal_y = -np.tan(cys)*FOCAL_LENGTH
num_rays = cxs.shape[0]
aperture_pos = np.array([xs, ys, np.zeros(num_rays)]).T
focal_pos = np.array([focal_x, focal_y, FOCAL_LENGTH*np.ones(num_rays)]).T
img_ray_dir = focal_pos - aperture_pos
img_ray_dir_lengths = np.sqrt(np.sum(img_ray_dir**2, axis=1))
img_ray_dir = img_ray_dir/img_ray_dir_lengths[:, np.newaxis]

ref_imgs = []
ref_cimgs = []
for sensor_distance in sensor_distances:
    alpha = sensor_distance/img_ray_dir[:, 2]
    sens_x = xs + img_ray_dir[:, 0]*alpha
    sens_y = ys + img_ray_dir[:, 1]*alpha
    sens_cxs = -np.arctan(sens_x/FOCAL_LENGTH)
    sens_cys = -np.arctan(sens_y/FOCAL_LENGTH)
    """
    img = np.histogram2d(
        (sens_cxs + FOV_RADIUS_RAD)/PIXEL_RAD,
        (sens_cys + FOV_RADIUS_RAD)/PIXEL_RAD,
        bins=(cx_bin_edges, cy_bin_edges))[0]
    """
    img = np.histogram2d(
        -sens_cxs,
        -sens_cys,
        bins=(
            lut.plenoscope.cx_bin_edges,
            lut.plenoscope.cy_bin_edges))[0]
    cimg = signal.convolve2d(img, gauss2d, mode="same")
    ref_imgs.append(img)
    ref_cimgs.append(cimg)
ref_imgs = np.array(ref_imgs)
ref_cimgs = np.array(ref_cimgs)
ref_vmax = np.max(ref_imgs)
ref_cvmax = np.max(ref_cimgs)

max_density_layer = np.argmax(np.max(np.max(ref_cimgs,axis=1),axis=1))
print(object_distances[max_density_layer])

fig = plt.figure(
        figsize=(8, 8*NUM_REFS),
        dpi=200)
SUB_FIG_HEIGHT = 1/NUM_REFS
for obj, sensor_distance in enumerate(sensor_distances):
    ax = fig.add_axes([
        0,
        SUB_FIG_HEIGHT*obj,
        1*.98,
        SUB_FIG_HEIGHT*.98])
    ax.set_axis_off()
    ax.pcolor(ref_cimgs[obj, :, :], vmax=ref_cvmax, cmap='inferno')
plt.savefig('refocus_stack.png')
plt.close('all')

# FULL IMAGE
fig = plt.figure(
        figsize=(8, 8),
        dpi=200)
ax = fig.add_axes([
    0,
    0,
    1*.98,
    1*.98])
ax.set_axis_off()
img = np.histogram2d(
    lfs[:, 0],
    lfs[:, 1],
    bins=(cx_bin_edges, cy_bin_edges))[0]
cimg = signal.convolve2d(img, gauss2d, mode="same")
ax.pcolor(cimg, cmap='inferno')
plt.savefig('full_image.png')
plt.close('all')