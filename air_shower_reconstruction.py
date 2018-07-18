from skimage.measure import LineModelND, ransac
from mpl_toolkits.mplot3d import Axes3D
import plenopy as pl
import json
import os
import scipy

light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

run = pl.Run(
    os.path.join('run', 'irf', 'electron', 'past_trigger'),
    light_field_geometry=light_field_geometry)

sum_tpr = 0
sum_ppv = 0
number_events = 0

for event in run:
    primary_momentum = event.simulation_truth.event.corsika_event_header.momentum()
    primary_direction = primary_momentum/np.linalg.norm(primary_momentum)

    core_cx = - primary_direction[0]
    core_cy = - primary_direction[1]
    core_x = event.simulation_truth.event.corsika_event_header.core_position_x_meter()
    core_y = event.simulation_truth.event.corsika_event_header.core_position_y_meter()
    energy = event.simulation_truth.event.corsika_event_header.total_energy_GeV

    if energy < 15:
        continue

    roi = pl.classify.center_for_region_of_interest(event)
    photons = pl.classify.RawPhotons.from_event(event)

    cherenkov_photons = pl.classify.cherenkov_photons_in_roi_in_image(
        roi=roi,
        photons=photons)

    if cherenkov_photons.photon_ids.shape[0] < 1000:
        continue

    b = pl.classify.benchmark(
        pulse_origins=event.simulation_truth.detector.pulse_origins,
        photon_ids_cherenkov=cherenkov_photons.photon_ids)

    tpr = b['number_true_positives']/(
        b['number_true_positives'] + b['number_false_negatives'])
    ppv = b['number_true_positives']/(
        b['number_true_positives'] + b['number_false_positives'])

    sum_tpr += tpr
    sum_ppv += ppv
    number_events += 1

    #print(
    #    'tpr:', sum_tpr/number_events,
    #    'ppv:', sum_ppv/number_events)

    if (
            photons.photon_ids.shape[0] -
            b['number_true_negatives'] -
            b['number_true_positives'] -
            b['number_false_positives'] -
            b['number_false_negatives']
        ) != 0:
        break

    trigger_offset = np.sqrt(
        (roi['cx_center_roi'] - core_cx)**2 +
        (roi['cy_center_roi'] - core_cy)**2)

    flash_offset = np.sqrt(
        (np.mean(cherenkov_photons.cx) - core_cx)**2 +
        (np.mean(cherenkov_photons.cy) - core_cy)**2)

    B, inlier = pl.tools.ransac_3d_plane.fit(
        xyz_point_cloud=np.c_[
            cherenkov_photons.x,
            cherenkov_photons.y,
            cherenkov_photons.t_pap*3e8],
        max_number_itarations=500,
        min_number_points_for_plane_fit=10,
        max_orthogonal_distance_of_inlier=0.025,)
    c_pap_time = np.array([B[0], B[1], B[2]])
    if c_pap_time[2] > 0:
        c_pap_time *= -1
    c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
    pap_time_offset = np.sqrt(
        (c_pap_time[0] - core_cx)**2 +
        (c_pap_time[1] - core_cy)**2)

    print(
        '{o:.2f}deg, {f:.2f}deg, {p:.2f}deg, {e:.2f}GeV'.format(
            o=np.rad2deg(trigger_offset),
            f=np.rad2deg(flash_offset),
            p=np.rad2deg(pap_time_offset),
            e=energy))


    number_macro_paxel = 7
    r = light_field_geometry.expected_aperture_radius_of_imaging_system/3
    pax_x = [0.0]
    pax_y = [0.0]
    pax_r = r * np.ones(number_macro_paxel)
    pax_photons = []
    for i, phi in enumerate(
        np.linspace(0, 2*np.pi, number_macro_paxel - 1, endpoint=False)):
        pax_x.append(2*r*np.cos(phi))
        pax_y.append(2*r*np.sin(phi))
    for i in range(number_macro_paxel):
        d_off = np.hypot(
            cherenkov_photons.x - pax_x[i],
            cherenkov_photons.y - pax_y[i])
        mask = d_off <= r
        pax_photons.append(cherenkov_photons.cut(mask))

    roi_r = np.deg2rad(0.5)
    pixel_fov = 0.5*np.deg2rad(.0667)

    fig, axes = plt.subplots(number_macro_paxel + 1)
    for i in range(number_macro_paxel):
        axes[i].hist2d(
            np.rad2deg(pax_photons[i].cx_cy_in_object_distance(7.5e3)[0]),
            np.rad2deg(pax_photons[i].cx_cy_in_object_distance(7.5e3)[1]),
            bins=(
                np.rad2deg(np.linspace(
                    c_pap_time[0] - roi_r,
                    c_pap_time[0] + roi_r,
                    int(2*roi_r/pixel_fov))),
                np.rad2deg(np.linspace(
                    c_pap_time[1] - roi_r,
                    c_pap_time[1] + roi_r,
                    int(2*roi_r/pixel_fov)))
            ),
        )
        axes[i].set_aspect('equal')
    i = i + 1
    axes[i].hist2d(
        np.rad2deg(cherenkov_photons.cx_cy_in_object_distance(17.5e3)[0]),
        np.rad2deg(cherenkov_photons.cx_cy_in_object_distance(17.5e3)[1]),
        bins=(
            np.rad2deg(np.linspace(
                c_pap_time[0] - roi_r,
                c_pap_time[0] + roi_r,
                int(2*roi_r/pixel_fov))),
            np.rad2deg(np.linspace(
                c_pap_time[1] - roi_r,
                c_pap_time[1] + roi_r,
                int(2*roi_r/pixel_fov)))
        ),
    )
    axes[i].set_aspect('equal')

    plt.show()


    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #ax.set_xlim(roi['cx_center_roi'] - c_radius, roi['cx_center_roi'] + c_radius)
    #ax.set_ylim(roi['cy_center_roi'] - c_radius, roi['cy_center_roi'] + c_radius)
    #ax.set_zlim(start_time, end_time)
    ax.set_xlabel('x/rad')
    ax.set_ylabel('y/rad')
    ax.set_zlabel('t/s')
    ax.scatter(
        cherenkov_photons.cx_cy_in_object_distance(roi['object_distance'])[0],
        cherenkov_photons.cx_cy_in_object_distance(roi['object_distance'])[1],
        cherenkov_photons.t_img,
        lw=0,
        alpha=0.075,
        s=55.,
        c='b'
    )
    plt.show()
    """
