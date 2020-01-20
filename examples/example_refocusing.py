import plenopy as pl
import json
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import glob

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# interesting events:
# event in run | run number | event number | particle type
# 006163_000719_000016_01
# 006678_000792_000037_01
# 006912_000826_000015_01


light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

run = pl.Run(
    os.path.join('run', 'irf', 'gamma', 'past_trigger'),
    light_field_geometry=light_field_geometry)

out_dir = os.path.join('examples', 'classic_view')

# Plot example events
#  ------------------
object_distances = np.logspace(
    np.log10(3e3),
    np.log10(25e3),
    9)
roi_r = np.deg2rad(0.5)
pixel_fov = 0.5*np.deg2rad(.0667)

event_nums = [6106, 6163, 6678, 6912, 6554]

events_yet = glob.glob(os.path.join(out_dir, '*.txt'))
event_nums_yet = np.sort([int(os.path.basename(e)[0:6]) for e in events_yet])

for evt_num in event_nums_yet:
    event = run[evt_num]
    try:
        id_str = '{trigger:06d}_{run:06d}_{event:06d}_{prmpar:02d}'.format(
            trigger=evt_num,
            run=event.simulation_truth.event.corsika_run_header.number,
            event=event.simulation_truth.event.corsika_event_header.number,
            prmpar=event.simulation_truth.event.corsika_event_header.primary_particle_id)

        roi = pl.classify.center_for_region_of_interest(event)
        photons = pl.classify.RawPhotons.from_event(event)

        cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
            roi=roi,
            photons=photons,
            object_distances=object_distances,
            number_refocuses=8)

        img_refocused = []
        vmin_refocused = 0
        vmax_refocused = []
        for i, object_distance in enumerate(object_distances):
            img = pl.plot.refocus.refocus_images(
                light_field_geometry=light_field_geometry,
                photon_lixel_ids=cherenkov_photons.lixel_ids,
                object_distances=[object_distance],
            )[0]
            img_refocused.append(img)
            vmax_refocused.append(np.max(img.intensity))
        vmax_refocused = np.max(vmax_refocused)

        roi_xlim = np.rad2deg(
            [roi['cx_center_roi'] - roi_r, roi['cx_center_roi'] + roi_r])
        roi_ylim = np.rad2deg(
            [roi['cy_center_roi'] - roi_r, roi['cy_center_roi'] + roi_r])

        for i, object_distance in enumerate(object_distances):
            fig = plt.figure(figsize=(6, 6), dpi=320)
            ax = fig.add_axes((0., 0., 1, 1))
            pl.plot.image.add_pixel_image_to_ax(
                img_refocused[i],
                ax,
                vmin=vmin_refocused,
                vmax=vmax_refocused,
                colorbar=False,
                colormap='inferno')
            ax.text(
                x=roi_xlim[0] + 0.05,
                y=roi_ylim[0] + 0.05,
                s='${:.1f}\,$km'.format(object_distance/1e3),
                fontdict={'family': 'serif',
                    'color':  'white',
                    'weight': 'normal',
                    'size': 24,})
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(roi_xlim)
            ax.set_ylim(roi_ylim)
            fig.savefig(
                os.path.join(out_dir, id_str+'_refocused_{:d}.jpg'.format(i)))
            plt.close('all')

        true_cherenkov_photons = (
            event.simulation_truth.detector.pulse_origins >= 0)

        true_emission_altitudes = (event.simulation_truth.
            air_shower_photon_bunches.emission_height[
                event.simulation_truth.detector.pulse_origins[
                    true_cherenkov_photons]])

        obj_bin_edges = np.linspace(0e3, 20e3, ((40 - 0) + 1))

        true_emission_positions_all = (
            pl.tomography.simulation_truth.
                emission_positions_of_photon_bunches(
                    event.simulation_truth.air_shower_photon_bunches))[
                        'emission_positions']

        true_emission_positions = true_emission_positions_all[
            event.simulation_truth.detector.pulse_origins[
                true_cherenkov_photons]]

        xy_bin_edges = np.linspace(-125, 125, 61)

        fig = plt.figure(figsize=(6, 9), dpi=320)
        ax1 = fig.add_axes((0.1, 0.13, 0.4, 0.85))
        ax2 = fig.add_axes((0.575, 0.13, 0.4, 0.85))
        xz_hist = np.histogram2d(
            true_emission_positions[:,0],
            true_emission_positions[:,2]/1e3,
            bins=(xy_bin_edges, obj_bin_edges/1e3),
        )[0]
        yz_hist = np.histogram2d(
            true_emission_positions[:,1],
            true_emission_positions[:,2]/1e3,
            bins=(xy_bin_edges, obj_bin_edges/1e3),
        )[0]
        vmax = np.max([np.max(xz_hist), np.max(yz_hist)])
        im1 = ax1.imshow(xz_hist.T,
            cmap='Greys',
            norm=colors.PowerNorm(
                gamma=1, vmin=xz_hist.min(), vmax=xz_hist.max()),
            origin='lower',
            extent=(
                xy_bin_edges.min(),
                xy_bin_edges.max(),
                obj_bin_edges.min()/1e3,
                obj_bin_edges.max()/1e3
            ),
            vmin=0, vmax=vmax)
        ax1.set_aspect(40)
        ax1.spines['right'].set_visible(False)
        ax1.spines['top'].set_visible(False)
        ax1.set_xlabel('$x$/m')
        ax1.set_ylabel('object-distance $g$/km')
        im2 = ax2.imshow(yz_hist.T,
            cmap='Greys',
            norm=colors.PowerNorm(
                gamma=1, vmin=xz_hist.min(), vmax=xz_hist.max()),
            origin='lower',
            extent=(
                xy_bin_edges.min(),
                xy_bin_edges.max(),
                obj_bin_edges.min()/1e3,
                obj_bin_edges.max()/1e3
            ),
            vmin=0, vmax=vmax)
        ax2.set_aspect(40)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.set_xlabel('$y$/m')
        ax2.yaxis.set_ticklabels([])
        ax_colorbar = fig.add_axes([0.1, 0.04, 0.875, 0.02])
        fig.colorbar(im1, cax=ax_colorbar, orientation='horizontal')
        fig.savefig(
            os.path.join(out_dir, id_str+'_true_emissions_x_y_z.jpg'))
        plt.close('all')

    except KeyboardInterrupt:
        raise
    except:
        raise
