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

if not os.path.exists(join(out_dir,'gamma.evtio')):
    cw.corsika(    
        steering_card=steering_card, 
        output_path=join(out_dir,'gamma.evtio'), 
        save_stdout=True
    )

if not os.path.exists(join(out_dir,'gamma.acp')):  
    call([
        join('build','mctracer','mctPlenoscopePropagation'),
        '--lixel', join('resources','acp','71m','light_field_calibration'),
        '--input', join(out_dir,'gamma.evtio'),
        '--config', join('resources','acp','mct_propagation_config.xml'),
        '--output', join(out_dir,'gamma.acp'),
        '--random_seed', '0',
        '--all_truth'
    ])

print('load event')
run = pl.Run(join(out_dir,'gamma.acp'))
event = run[1]

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

fig = plt.figure()
im = plt.imshow( 
    np.rot90(voxel_number_of_baselines_3D[:,96-1:96+2,:].sum(axis=1)/3, k=-1), 
    extent=[x_bin_edges[0], x_bin_edges[-1], z_bin_edges[0]/1e3, z_bin_edges[-1]/1e3], 
    aspect=1e3/3,#(x_bin_edges[-1]-x_bin_edges[0]) / (z_bin_edges[-1]-z_bin_edges[0]),
    cmap='viridis',
    origin="lower",
    interpolation='None',
    norm=colors.PowerNorm(gamma=1./2.)
)
fig.colorbar(im, orientation="vertical", pad=0.2, label='number of baselines')
plt.xlabel('x/m, y=0')
plt.ylabel('object distance/km')
plt.show()


