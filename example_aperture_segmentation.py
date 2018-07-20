import plenopy as pl
import json
import os
import matplotlib.pyplot as plt


light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

run = pl.Run(
    os.path.join('run', 'irf', 'electron', 'past_trigger'),
    light_field_geometry=light_field_geometry)

number_macro_paxel = 7
r = light_field_geometry.expected_aperture_radius_of_imaging_system/3
pax_x = [0.0]
pax_y = [0.0]
pax_r = r * np.ones(number_macro_paxel)

object_distance = 10.e3

for i, phi in enumerate(
    np.linspace(0, 2*np.pi, number_macro_paxel - 1, endpoint=False)):
    pax_x.append(2*r*np.cos(phi))
    pax_y.append(2*r*np.sin(phi))

roi_r = np.deg2rad(0.5)
pixel_fov = 0.5*np.deg2rad(.0667)

for evt_num, event in enumerate(run):

    print(evt_num)
    #if event.simulation_truth.event.corsika_event_header.total_energy_GeV < 25:
    #    continue

    roi = pl.classify.center_for_region_of_interest(event)
    photons = pl.classify.RawPhotons.from_event(event)

    cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
        roi=roi,
        photons=photons)

    #if cherenkov_photons.photon_ids.shape[0] < 1000:
    #    continue

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
    fig.savefig(
        os.path.join(
            'examples', 'classic_view', '{:d}_combined.jpg'.format(evt_num)))


    fig = plt.figure(figsize=(6, 6), dpi=320)
    ax_width = 0.25
    for i in range(number_macro_paxel):
        ax_x = pax_x[i]/(6*r) + 0.5 - ax_width/2
        ax_y = pax_y[i]/(6*r) + 0.5 - ax_width/2

        ax = fig.add_axes((ax_x, ax_y, ax_width, ax_width))
        pl.plot.image.add_pixel_image_to_ax(
            img_segmented[i],
            ax,
            vmin=vmin_segmented,
            vmax=vmax_segmented,
            colorbar=False,
            colormap='inferno')

        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(np.rad2deg([roi['cx_center_roi'] - roi_r, roi['cx_center_roi'] + roi_r]))
        ax.set_ylim(np.rad2deg([roi['cy_center_roi'] - roi_r, roi['cy_center_roi'] + roi_r]))
    fig.savefig(
        os.path.join(
            'examples', 'classic_view', '{:d}_segmented.jpg'.format(evt_num)))


    plt.close('all')