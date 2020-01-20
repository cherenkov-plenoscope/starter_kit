import plenopy as pl
import json
import os
import matplotlib.pyplot as plt

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

# Set up macro-paxels
# -------------------
R = light_field_geometry.expected_aperture_radius_of_imaging_system
number_macro_paxel = 7
macro_paxel_radius = R/3
macro_paxel_x = [0.0]
macro_paxel_y = [0.0]
for i, phi in enumerate(
    np.linspace(0, 2*np.pi, number_macro_paxel - 1, endpoint=False)):
    macro_paxel_x.append((R - macro_paxel_radius)*np.cos(phi))
    macro_paxel_y.append((R - macro_paxel_radius)*np.sin(phi))

# Plot the macro-paxels on the aperture
# -------------------------------------
fig = plt.figure(figsize=(6, 6), dpi=320)
ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
circ = plt.Circle((0, 0), radius=R, color='k', fill=False)
ax.add_patch(circ)
for i in range(number_macro_paxel):
    circ = plt.Circle(
        (macro_paxel_x[i], macro_paxel_y[i]),
        radius=macro_paxel_radius,
        color='k',
        fill=False)
    ax.add_patch(circ)
    ax.text(
        x=macro_paxel_x[i],
        y=macro_paxel_y[i],
        s=str(i),
        fontdict={'family': 'serif',
            'color':  'black',
            'weight': 'normal',
            'size': 16,})
ax.set_xlim(-R*1.03, R*1.03)
ax.set_ylim(-R*1.03, R*1.03)
ax.set_xlabel('$x$/m')
ax.set_ylabel('$y$/m')
plt.savefig(os.path.join(out_dir, 'aperture_segmentation_seve_telescopes.png'))
plt.close('all')

# Plot example events
#  ------------------
object_distance = 10.e3
roi_r = np.deg2rad(0.5)
pixel_fov = 0.5*np.deg2rad(.0667)

for evt_num, event in enumerate(run):
    try:
        id_str = '{trigger:06d}_{run:06d}_{event:06d}_{prmpar:02d}'.format(
            trigger=evt_num,
            run=event.simulation_truth.event.corsika_run_header.number,
            event=event.simulation_truth.event.corsika_event_header.number,
            prmpar=event.simulation_truth.event.corsika_event_header.primary_particle_id)

        print(evt_num)
        if event.simulation_truth.event.corsika_event_header.total_energy_GeV < 100:
            continue

        roi = pl.classify.center_for_region_of_interest(event)
        photons = pl.classify.RawPhotons.from_event(event)

        cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
            roi=roi,
            photons=photons)

        if cherenkov_photons.photon_ids.shape[0] < 10000:
            continue

        info_path = os.path.join(out_dir, id_str+'.txt')
        if os.path.exists(info_path):
            continue
        with open(info_path, 'wt') as fout:
            fout.write(event.simulation_truth.event.short_event_info())

        pax_photons = []
        for i in range(number_macro_paxel):
            d_off = np.hypot(
                cherenkov_photons.x - pax_x[i],
                cherenkov_photons.y - pax_y[i])
            pax_photons.append(cherenkov_photons.cut(d_off <= r))

        img_segmented = []
        vmin_segmented = 0
        vmax_segmented = []
        for i in range(number_macro_paxel):
            img = pl.plot.refocus.refocus_images(
                light_field_geometry=light_field_geometry,
                photon_lixel_ids=pax_photons[i].lixel_ids,
                object_distances=[object_distance],
            )[0]
            img_segmented.append(img)
            vmax_segmented.append(np.max(img.intensity))
        vmax_segmented = np.max(vmax_segmented)

        roi_xlim = np.rad2deg(
            [roi['cx_center_roi'] - roi_r, roi['cx_center_roi'] + roi_r])
        roi_ylim = np.rad2deg(
            [roi['cy_center_roi'] - roi_r, roi['cy_center_roi'] + roi_r])

        fig = plt.figure(figsize=(6, 6), dpi=320)
        ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
        img_combined = pl.plot.refocus.refocus_images(
            light_field_geometry=light_field_geometry,
            photon_lixel_ids=cherenkov_photons.lixel_ids,
            object_distances=[object_distance],
        )[0]
        vmin = np.min(img_combined.intensity)
        vmax = np.max(img_combined.intensity)
        pl.plot.image.add_pixel_image_to_ax(
            img_combined,
            ax,
            vmin=vmin,
            vmax=vmax,
            colorbar=False,
            colormap='inferno')
        ax.plot(
            [roi_xlim[0], roi_xlim[1], roi_xlim[1], roi_xlim[0], roi_xlim[0]],
            [roi_ylim[0], roi_ylim[0], roi_ylim[1], roi_ylim[1], roi_ylim[0]],
            'gray')
        ax.set_xlabel('$c_x$/deg')
        ax.set_ylabel('$c_y$/deg')
        fig.savefig(os.path.join(out_dir, id_str+'_combined.jpg'))

        fig = plt.figure(figsize=(6, 6), dpi=320)
        ax_width = 0.28
        for i in range(number_macro_paxel):
            ax_x = macro_paxel_x[i]/(6*macro_paxel_radius) + 0.5 - ax_width/2
            ax_y = macro_paxel_y[i]/(6*macro_paxel_radius) + 0.5 - ax_width/2

            ax = fig.add_axes((ax_x, ax_y, ax_width, ax_width))
            pl.plot.image.add_pixel_image_to_ax(
                img_segmented[i],
                ax,
                vmin=vmin_segmented,
                vmax=vmax_segmented,
                colorbar=False,
                colormap='inferno')
            ax.text(
                x=roi_xlim[0] + 0.05,
                y=roi_ylim[0] + 0.05,
                s=str(i),
                fontdict={'family': 'serif',
                    'color':  'white',
                    'weight': 'normal',
                    'size': 16,})
            ax.axis('off')
            ax.set_aspect('equal')
            ax.set_xlim(roi_xlim)
            ax.set_ylim(roi_ylim)
        fig.savefig(os.path.join(out_dir, id_str+'_segmented.jpg'))


        plt.close('all')
    except KeyboardInterrupt:
        raise
    except:
        raise