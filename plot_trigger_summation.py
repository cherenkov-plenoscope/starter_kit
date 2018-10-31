import plenopy as pl
import lixel_summation_tools as lst
import matplotlib.pyplot as plt
import os

out_dir = os.path.join('.', 'examples', 'trigger_patch_summation')
os.makedirs(out_dir, exist_ok=True)

lfg = pl.LightFieldGeometry(
    os.path.join('.', 'run', 'light_field_calibration'))

object_distances = [7.5e3, 15e3, 22.5e3, 999e3]

trigger_patch_matrices = pl.trigger.prepare_refocus_sum_trigger(
    light_field_geometry=lfg,
    object_distances=object_distances)

trigger_patches = [
    {'id': 1337, 'xlim': [-3.97, -2.97], 'ylim': [+2.4, 3.4]},
    {'id': 4221, 'xlim': [-.5, .5], 'ylim': [-.5, .5]},
]

w = 8
h = 4
dpi = 128

for trigger_patch in range(len(trigger_patches)):
    trigger_patch_id = trigger_patches[trigger_patch]['id']
    for object_distance_id in range(4):

        mask = trigger_patch_matrices[
            'lixel_summations'][object_distance_id][trigger_patch_id]
        mask = mask.todense()
        mask = 1.0 * mask
        z = np.zeros(lfg.number_lixel)
        for i in range(lfg.number_lixel):
            z[i] = mask.T[i]
        mask = z

        fig = plt.figure(figsize=(w, h*1.1), dpi=dpi)
        ax = fig.add_axes([0, .1, .66, .9])
        pl.plot.light_field_geometry.colored_lixels(
            lss=lfg,
            I=mask,
            ax=ax,
            cmap='binary',
            edgecolors='k')
        ax.set_xlabel('photo-sensor-plane-x/m')
        ax.set_ylabel('photo-sensor-plane-y/m')
        ax.set_xlim(trigger_patches[trigger_patch]['xlim'])
        ax.set_ylim(trigger_patches[trigger_patch]['ylim'])
        ax2 = fig.add_axes([0.66, 0., .33, 1])
        ax2.set_aspect('equal')
        ax2.set_axis_off()
        lst.add_aperture_plane_to_ax(ax=ax2)
        ax2.set_xlim([-1, 1])
        ax2.set_ylim([-0.05, 3.95])
        t = object_distances[object_distance_id]/1e3/20
        lst.add_rays_to_ax(
            ax=ax2,
            object_distance=t,
            linewidth=.5)
        ax2.text(
            x=0.1,
            y=2*t,
            s='{:0.1f}km'.format(object_distances[object_distance_id]/1e3),
            fontsize=12)
        if object_distance_id+1 == len(object_distances):
            ax2.text(x=0.1, y=3.7, s='infinity', fontsize=12)
        plt.savefig(
            os.path.join(
                out_dir,
                'trigger_patch_{id:d}_mask_obj_{obj:d}.jpg'.format(
                    id=trigger_patch_id,
                    obj=object_distance_id)),
            dpi=dpi)
        plt.close('all')