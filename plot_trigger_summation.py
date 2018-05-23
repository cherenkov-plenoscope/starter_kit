import plenopy as pl
import matplotlib.pyplot as plt

lfg = pl.LightFieldGeometry('./run/light_field_calibration/')

trigger_patch_matrices = pl.trigger.prepare_refocus_sum_trigger(
    light_field_geometry=lfg,
    object_distances=[10e3, 15e3, 20e3, 999e3])

trigger_patches = [
    {'id': 1337, 'xlim': [-3.97, -2.97], 'ylim': [+2.4, 3.4]},
    {'id': 4221, 'xlim': [-.5, .5], 'ylim': [-.5, .5]},
]

fsize = pl.plot.FigureSize(
    relative_width=1,
    relative_hight=1,
    pixel_rows=1000,
    dpi=200)

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

        fig = plt.figure(figsize=(fsize.width, fsize.hight), dpi=fsize.dpi)
        ax = plt.gca()
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
        plt.savefig(
            'trigger_patch_{id:d}_mask_obj_{obj:d}.png'.format(
                id=trigger_patch_id,
                obj=object_distance_id),
            dpi=fsize.dpi)
        plt.close('all')