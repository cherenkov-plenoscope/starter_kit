from skimage.measure import LineModelND, ransac
from mpl_toolkits.mplot3d import Axes3D
import plenopy as pl
import json
import os
import scipy
import shutil


run = pl.Run(
    '/home/sebastian/Desktop/phd/starter_kit/run/irf/electron/past_trigger',
    light_field_geometry=pl.LightFieldGeometry(
        '/home/sebastian/Desktop/phd/starter_kit/run/light_field_calibration'))

time_radius_roi = 5e-9
c_radius = np.deg2rad(0.3)
epsilon_cx_cy_radius = np.deg2rad(0.075)
min_number_photons = 20
deg_over_s = 0.375e9

sum_tpr = 0
sum_ppv = 0
number_events = 0

for event in run:
    roi = pl.classify.center_for_region_of_interest(event)
    ph = pl.classify.RawPhotons.from_event(event)

    ph_cherenkov = pl.classify.cherenkov_photons_in_roi_in_image(
        roi=roi,
        photons=ph)

    b = pl.classify.benchmark(
        pulse_origins=event.simulation_truth.detector.pulse_origins,
        photon_ids_cherenkov=ph_cherenkov.photon_ids)

    tpr = b['number_true_positives']/(
        b['number_true_positives'] + b['number_false_negatives'])
    ppv = b['number_true_positives']/(
        b['number_true_positives'] + b['number_false_positives'])

    sum_tpr += tpr
    sum_ppv += ppv
    number_events += 1

    print(
        'tpr:', sum_tpr/number_events,
        'ppv:', sum_ppv/number_events)

    if (
            ph.photon_ids.shape[0] -
            b['number_true_negatives'] -
            b['number_true_positives'] -
            b['number_false_positives'] -
            b['number_false_negatives']
        ) != 0:
        break



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
