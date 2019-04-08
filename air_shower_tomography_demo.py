#! /usr/bin/env python
import docopt
import os
from os.path import join
from subprocess import call
import plenopy as pl
import corsika_wrapper as cw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import gridspec

out_dir = join('examples', 'air_shower_tomography_demo')
os.makedirs(out_dir, exist_ok=True)

steering_card = cw.read_steering_card(
    join(
        'resources',
        'acp',
        '71m',
        'high_energy_example_gamma_corsika_steering_card.txt'
    )
)

if not os.path.exists(join(out_dir, 'gamma.evtio')):
    cw.corsika(
        steering_card=steering_card,
        output_path=join(out_dir, 'gamma.evtio'),
        save_stdout=True
    )

if not os.path.exists(join(out_dir, 'gamma.acp')):
    call([
        join('build', 'merlict', 'merlict-plenoscope-propagation'),
        '--lixel', join('resources', 'acp', '71m', 'light_field_calibration'),
        '--input', join(out_dir, 'gamma.evtio'),
        '--config', join('resources', 'acp', 'mct_propagation_config.xml'),
        '--output', join(out_dir, 'gamma.acp'),
        '--random_seed', '0',
        '--all_truth'
    ])

print('load event')
run = pl.Run(join(out_dir, 'gamma.acp'))
# event = run[1]

x_bin_edges = np.linspace(-.6e3, .6e3, 192+1)
y_bin_edges = np.linspace(-.6e3, .6e3, 192+1)
z_bin_edges = np.linspace(0, 15e3, 192+1)

# visualize 3D reconstruction power
# rec = pl.tomography.narrow_angle.Reconstruction(event)

print('baselines_in_voxels')
voxel_number_of_baselines_3D = pl.tomography.baselines_in_voxels(
    light_field_geometry=run.light_field_geometry,
    x_bin_edges=x_bin_edges,
    y_bin_edges=y_bin_edges,
    z_bin_edges=z_bin_edges,
)

print('save baselines_in_voxels image slices')

xz_plane_slice_voxel_number_of_baselines_3D = voxel_number_of_baselines_3D[
    :, 96-1:96+2, :
].sum(axis=1)/3

"""
fig = plt.figure()
im = plt.imshow(
    np.rot90(xz_plane_slice_voxel_number_of_baselines_3D, k=-1),
    extent=[
        x_bin_edges[0],
        x_bin_edges[-1],
        z_bin_edges[0]/1e3,
        z_bin_edges[-1]/1e3
    ],
    aspect=1e3/1,#(x_bin_edges[-1]-x_bin_edges[0]) / (z_bin_edges[-1]-z_bin_edges[0]),
    cmap='viridis',
    origin="lower",
    interpolation='None',
    norm=colors.PowerNorm(gamma=1./2.)
)
fig.colorbar(
    im,
    orientation="vertical",
    pad=0.2,
    label='number of stereo baselines in 3D volume cell'
)
plt.xlabel('x/m, y=0')
plt.ylabel('object distance above aperture/km')
plt.show()
"""
s = 4.0

xz_ratio = 3.0

dpi = 150
l_margin = .175*s
r_margin = .05*s
t_margin = .05*s
b_margin = .175*s

img_x_extent = x_bin_edges[-1]-x_bin_edges[0]
img_z_extent = z_bin_edges[-1]-z_bin_edges[0]

img_w = 1.0*s
img_h = img_w*img_z_extent/(img_x_extent*xz_ratio)

space_h = 0.15*s

colbar_h = .05*s
colbar_w = img_w

fig_w = (
    l_margin +
    img_w +
    r_margin
)
fig_h = (
    t_margin +
    img_h +
    space_h +
    colbar_h +
    b_margin
)
fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)


l_img_anchor = l_margin
b_img_anchor = fig_h - t_margin - img_h
ax_img = fig.add_axes(
    (
        l_img_anchor/fig_w,
        b_img_anchor/fig_h,
        img_w/fig_w,
        img_h/fig_h
    )
)
l_col_anchor = l_img_anchor
b_col_anchor = fig_h - t_margin - img_h - space_h - colbar_h
ax_colbar = fig.add_axes(
    (
        l_col_anchor/fig_w,
        b_col_anchor/fig_h,
        colbar_w/fig_w,
        colbar_h/fig_h
    )
)

# R = run.light_field_geometry.expected_aperture_radius_of_imaging_system
# ax_img.plot([-R,+R],[10e-3, 10e-3], 'r')
# ax_img.plot([-R,-R],[-R/1e3, R/1e3], 'r')
# ax_img.plot([+R,+R],[-R/1e3, R/1e3], 'r')

ax_img.set_xlabel('x/m')
ax_img.set_ylabel('object distance above aperture/km')
im = ax_img.imshow(
    np.rot90(xz_plane_slice_voxel_number_of_baselines_3D, k=-1),
    extent=[
        x_bin_edges[0],
        x_bin_edges[-1],
        z_bin_edges[0]/1e3,
        z_bin_edges[-1]/1e3
    ],
    aspect=1e3/xz_ratio,
    cmap='viridis',
    origin="lower",
    interpolation='None',
    norm=colors.PowerNorm(gamma=1./2.)
)

cbar = plt.colorbar(
    im,
    orientation="horizontal",
    label='number of stereo baselines in 3D volume cell',
    cax=ax_colbar
)
cbar.ax.set_xticklabels(cbar.ax.get_xticklabels(), rotation='vertical')

plt.savefig(join(out_dir, 'number_of_baselines_71m_ACP.png'), dpi=dpi)
