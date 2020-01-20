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

out_dir = join('examples', 'tomography')
os.makedirs(out_dir, exist_ok=True)

num_bins = 200

num_x_bins = num_bins
num_y_bins = num_bins
num_z_bins = num_bins
xy_radius = 1.e3

x_bin_edges = np.linspace(-xy_radius, xy_radius, num_x_bins+1)
y_bin_edges = np.linspace(-xy_radius, xy_radius, num_y_bins+1)
z_bin_edges = np.linspace(0, 20e3, num_z_bins+1)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

cache_path = os.path.join(out_dir, 'number_of_baselines_3D.npy')

if not os.path.exists(cache_path):
    print('rays')
    rays = pl.tomography.Rays.from_light_field_geometry(
        light_field_geometry)

    print('system_matrix')
    system_matrix = pl.tomography.narrow_angle.deconvolution.make_cached_tomographic_system_matrix(
        supports=-rays.support,
        directions=rays.direction,
        x_bin_edges=x_bin_edges,
        y_bin_edges=y_bin_edges,
        z_bin_edges=z_bin_edges,)

    print('find baselines')
    n_paxels_in_voxel = np.zeros(num_x_bins*num_y_bins*num_z_bins)
    for paxel in range(light_field_geometry.number_paxel):
        print(paxel, ' of ', light_field_geometry.number_paxel)
        rays_in_this_paxel = np.zeros(
            light_field_geometry.number_paxel,
            dtype=np.bool)
        rays_in_this_paxel[paxel] = True
        rays_in_this_paxel = np.tile(
            rays_in_this_paxel,
            light_field_geometry.number_pixel)

        paxel_system_matrix_integral = system_matrix[
            :,
            rays_in_this_paxel].sum(axis=1)
        paxel_system_matrix_integral = np.array(
            paxel_system_matrix_integral
        ).reshape((paxel_system_matrix_integral.shape[0],))
        n_paxels_in_voxel += paxel_system_matrix_integral > 0

    number_baselines_in_voxel = (
        n_paxels_in_voxel**2 -
        n_paxels_in_voxel)/2.0

    voxel_number_of_baselines_3D = number_baselines_in_voxel.reshape(
        (num_x_bins, num_y_bins, num_z_bins,),
        order='C')
    print('save baselines_in_voxels image slices')
    np.save(
        file=cache_path,
        arr=voxel_number_of_baselines_3D)
else:
    voxel_number_of_baselines_3D = np.load(cache_path)

xz = voxel_number_of_baselines_3D[:, 96-1:96+2, :].sum(axis=1)/3

cfgs = [
    {
        'path': 'beamer',
        'rows': 1080,
        'cols': 1920,
        'dpi': 300,
        'ax': [.1, .12, .73, .81],
        'cax': [0.87, .1, .03, .85]
    },
    {
        'path': 'print',
        'rows': 1920,
        'cols': 1920,
        'dpi': 300,
        'ax': [.1, .1, .73, .86],
        'cax': [0.87, .1, .03, .85]
    },
]

bin_volume = (
    np.gradient(x_bin_edges[:])[0] *
    np.gradient(y_bin_edges[:])[0] *
    z_bin_edges[1])

for cfg in cfgs:
    fig = plt.figure(
        figsize=(
            cfg['cols']/cfg['dpi'],
            cfg['rows']/cfg['dpi']),
        dpi=cfg['dpi'])
    ax = fig.add_axes(cfg['ax'])
    col = ax.pcolor(
        x_bin_edges,
        z_bin_edges*1e-3,
        xz.T,
        cmap='Greys',
        norm=colors.PowerNorm(gamma=1./2.))
    ax.set_xlabel('x / m')
    ax.set_ylabel('object-distance / km')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)

    cax = fig.add_axes(cfg['cax'])
    cbar = plt.colorbar(
        col,
        orientation="vertical",
        label='stereo-baselines / volume-cell$^{-1}$ ' +
        '{:.0f}'.format(bin_volume) +
        r'$^{-1}$ m$^{-3}$',
        cax=cax)

    fig.savefig(
        os.path.join(
            out_dir,
            'number_of_baselines_{:s}.png'.format(
                cfg['path'])))
    plt.close('all')
