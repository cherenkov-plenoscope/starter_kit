from skimage.measure import LineModelND, ransac
from mpl_toolkits.mplot3d import Axes3D
import plenopy as pl
import json
import os
import scipy
import shutil


def plot_photon_cloud(
    roi,
    photons,
    mask
):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(roi['cx_center_roi'] - c_radius, roi['cx_center_roi'] + c_radius)
    ax.set_ylim(roi['cy_center_roi'] - c_radius, roi['cy_center_roi'] + c_radius)
    ax.set_zlim(start_time, end_time)
    ax.set_xlabel('x/rad')
    ax.set_ylabel('y/rad')
    ax.set_zlabel('t/s')
    ax.scatter(
        photons['cx'][mask],
        photons['cy'][mask],
        photons['t_img'][mask],
        lw=0,
        alpha=0.075,
        s=55.,
        c='b'
    )
    plt.show()


run = pl.Run(
    '/home/sebastian/Desktop/phd/starter_kit/run/irf/proton/past_trigger',
    light_field_geometry=pl.LightFieldGeometry(
        '/home/sebastian/Desktop/phd/starter_kit/run/light_field_calibration'))

image_rays = pl.image.ImageRays(run.light_field_geometry)


t_radius = 5e-9
c_radius = np.deg2rad(0.3)

sum_tpr = 0
sum_ppv = 0
number_events = 0

for event in run:
    # print(event, event.simulation_truth.detector.number_air_shower_pulses())

    roi = pl.photon_classification.center_for_region_of_interest(event)
    ph = pl.photon_classification.RawPhotons.from_event(event)

    #print(np.rad2deg(roi['cx_center_roi']), np.rad2deg(roi['cy_center_roi']))
    start_time = roi['time_center_roi'] - t_radius
    end_time = roi['time_center_roi'] + t_radius


    ph_mask_time = (
        (ph.t_img >= start_time) &
        (ph.t_img < end_time))

    ph_cx, ph_cy = ph.cx_cy_in_object_distance(roi['object_distance'])
    ph_c_distance_square = (
        (roi['cx_center_roi'] - ph_cx)**2 +
        (roi['cy_center_roi'] - ph_cy)**2)
    c_radius_square = c_radius**2
    ph_mask_c = ph_c_distance_square <= c_radius_square

    ph_mask_roi = ph_mask_time & ph_mask_c

    ph_roi = ph.cut(ph_mask_roi)

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(roi['cx_center_roi'] - c_radius, roi['cx_center_roi'] + c_radius)
    ax.set_ylim(roi['cy_center_roi'] - c_radius, roi['cy_center_roi'] + c_radius)
    ax.set_zlim(start_time, end_time)
    ax.set_xlabel('x/rad')
    ax.set_ylabel('y/rad')
    ax.set_zlabel('t/s')
    ax.scatter(
        ph_roi.cx_cy_in_object_distance(roi['object_distance'])[0],
        ph_roi.cx_cy_in_object_distance(roi['object_distance'])[1],
        ph_roi.t_img,
        lw=0,
        alpha=0.075,
        s=55.,
        c='b'
    )
    plt.show()
    """

    photon_labels = pl.photon_classification.cluster_air_shower_photons_based_on_density(
        cx_cy_arrival_time_point_cloud=np.c_[
            ph_roi.cx_cy_in_object_distance(roi['object_distance'])[0],
            ph_roi.cx_cy_in_object_distance(roi['object_distance'])[1],
            ph_roi.t_img,
        ],
        epsilon_cx_cy_radius=np.deg2rad(0.075),
        min_number_photons=20,
        deg_over_s=0.375e9
    )
    cherenkov_mask = photon_labels >= 0

    ph_cherenkov = ph_roi.cut(cherenkov_mask)

    """
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    #ax.set_zlim(start_time, end_time)
    ax.set_xlabel('x/m')
    ax.set_ylabel('y/m')
    ax.set_zlabel('z/m')
    ax.scatter(
        ph_cherenkov.x,
        ph_cherenkov.y,
        ph_cherenkov.t_pap*3e8,
        lw=0,
        alpha=0.075,
        s=55.,
        c='b'
    )
    plt.show()
    """
    pulse_origins = event.simulation_truth.detector.pulse_origins

    classified_cherenkov = ph_cherenkov.photon_ids
    classified_nsb = np.setdiff1d(
        np.arange(ph.photon_ids.shape[0]),
        classified_cherenkov)

    if classified_cherenkov.sum() == 0:
        print('no cluster')
        continue

    is_cherenkov = pulse_origins >= 0
    is_nsb = pulse_origins < 0

    # is Cherenkov AND classified as Cherenkov
    tp = is_cherenkov[classified_cherenkov].sum()

    # is Cherenkov AND classified as NSB
    fp = is_cherenkov[classified_nsb].sum()

    # is NSB AND classified as Cherenkov
    fn = is_nsb[classified_cherenkov].sum()

    # is NSB AND classified as NSB
    tn = is_nsb[classified_nsb].sum()

    tpr = tp/(tp + fn)
    ppv = tp/(tp + fp)

    sum_tpr += tpr
    sum_ppv += ppv
    number_events += 1

    print(
        'tpr:', sum_tpr/number_events,
        'ppv:', sum_ppv/number_events)

    if (ph.photon_ids.shape[0] - tn - tp - fp - fn) != 0:
        break
