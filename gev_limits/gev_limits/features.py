import numpy as np
from collections import namedtuple
import plenopy as pl
from . import light_field
import matplotlib.pyplot as plt


HillasEllipse = namedtuple(
    'HillasEllipse', [
    'cx_mean',
    'cy_mean',
    'cx_major',
    'cy_major',
    'cx_minor',
    'cy_minor',
    'std_major',
    'std_minor'])


def hillas_ellipse(cxs, cys):
    cx_mean = np.mean(cxs)
    cy_mean = np.mean(cys)

    cov_matrix = np.cov(np.c_[cxs, cys].T)
    eigen_vals, eigen_vecs= np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_vals)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0
    major_axis = eigen_vecs[:, major_idx]
    major_std = np.sqrt(eigen_vals[major_idx])
    minor_axis = eigen_vecs[:, minor_idx]
    minor_std = np.sqrt(eigen_vals[minor_idx])
    return HillasEllipse(
        cx_mean=cx_mean,
        cy_mean=cy_mean,
        cx_major=major_axis[0],
        cy_major=major_axis[1],
        cx_minor=minor_axis[0],
        cy_minor=minor_axis[1],
        std_major=major_std,
        std_minor=minor_std)


def light_front_surface_normal(xs, ys, zs):
    B, inlier = pl.tools.ransac_3d_plane.fit(
        xyz_point_cloud=np.c_[xs, ys, zs],
        max_number_itarations=100,
        min_number_points_for_plane_fit=10,
        max_orthogonal_distance_of_inlier=0.025,)
    c_pap_time = np.array([B[0], B[1], B[2]])
    if c_pap_time[2] > 0:
        c_pap_time *= -1
    c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
    return c_pap_time


def angle_between(v1, v2):
    v1_u = v1/np.linalg.norm(v1)
    v2_u = v2/np.linalg.norm(v2)
    return np.arccos(np.dot(v1_u, v2_u))


def extract(lut):
    NUM_REFOCUS_SLICES = 12
    OBJECT_DISTANCES = np.geomspace(5e3, 100e3, NUM_REFOCUS_SLICES)

    features = []
    for event_idx in range(lut.num_events):
        print(event_idx)
        feat = {}
        # simulation truth
        # ----------------
        feat['particle_id'] = lut.particle_id[event_idx]
        feat['particle_energy'] = lut.particle_energy[event_idx]
        feat['particle_cx'] = lut.particle_cx[event_idx]
        feat['particle_cy'] = lut.particle_cy[event_idx]
        feat['particle_x'] = lut.particle_x[event_idx]
        feat['particle_y'] = lut.particle_y[event_idx]
        feat['particle_height_first_interaction'] = lut.particle_height_first_interaction[event_idx]

        # from the light-field-sequence
        # -----------------------------
        lfs = lut._raw_light_field_sequence(event_idx)

        feat['num_photons'] = lfs.shape[0]

        image_photons = light_field.get_image_photons(
            lfs=lfs,
            plenoscope=lut.plenoscope)
        tpap = lut.plenoscope.t[lfs[:, 4]]


        xs, ys, zs = light_field.get_light_front(
            lfs=lfs,
            plenoscope=lut.plenoscope)
        light_front_normal = light_front_surface_normal(
            xs=xs,
            ys=ys,
            zs=zs)
        feat['light_front_cx'] = light_front_normal[0]
        feat['light_front_cy'] = light_front_normal[1]


        hillas = []
        for obj in OBJECT_DISTANCES:
            cxs, cys = light_field.get_refocused_cx_cy(
                image_photons=image_photons,
                object_distance=obj)
            timg = tpap -1.0*light_field.add_to_tpap_to_get_timg(
                cx=cxs,
                cy=cys,
                x=image_photons.x,
                y=image_photons.y)

            timg = np.round(timg/.5e-9)*.5e-9

            eli = hillas_ellipse(cxs=cxs, cys=cys)

            time_grad_cx = np.polyfit(x=cxs, y=timg, deg=1)[0]
            time_grad_cy = np.polyfit(x=cys, y=timg, deg=1)[0]

            e = [
                eli.cx_mean,
                eli.cy_mean,
                eli.cx_major,
                eli.cy_major,
                eli.cx_minor,
                eli.cy_minor,
                eli.std_major,
                eli.std_minor,
                np.hypot(eli.std_major, eli.std_minor),
                time_grad_cx,
                time_grad_cy
            ]

            hillas.append(e)
        hillas = np.array(hillas)

        max_dense_obj = np.argmin(hillas[:, 8])
        if max_dense_obj == 0 or max_dense_obj == NUM_REFOCUS_SLICES - 1:
            continue

        feat['object_distance_for_smallest_hillas_ellipse'] = OBJECT_DISTANCES[max_dense_obj]
        feat['std_smallest_hillas_ellipse'] = hillas[max_dense_obj, 8]

        feat['hillas_mean_cx'] = hillas[max_dense_obj, 0]
        feat['hillas_mean_cy'] = hillas[max_dense_obj, 1]
        feat['hillas_major_cx'] = hillas[max_dense_obj, 2]
        feat['hillas_major_cy'] = hillas[max_dense_obj, 3]
        feat['hillas_major_std'] = hillas[max_dense_obj, 6]
        feat['hillas_minor_std'] = hillas[max_dense_obj, 7]
        feat['time_grad_cx'] = hillas[max_dense_obj, 9]
        feat['time_grad_cy'] = hillas[max_dense_obj, 10]

        print(feat['time_grad_cx'], feat['time_grad_cy'])
        ellipseity = feat['hillas_major_std']/feat['hillas_minor_std']

        hillas_cx_mean = hillas[max_dense_obj, 0]
        hillas_cy_mean = hillas[max_dense_obj, 1]
        hillas_cx_major = hillas[max_dense_obj, 2]
        hillas_cy_major = hillas[max_dense_obj, 3]
        hillas_start_cx = hillas_cx_mean - 0.01*hillas_cx_major*ellipseity
        hillas_start_cy = hillas_cy_mean - 0.01*hillas_cy_major*ellipseity
        hillas_stop_cx = hillas_cx_mean + 0.01*hillas_cx_major*ellipseity
        hillas_stop_cy = hillas_cy_mean + 0.01*hillas_cy_major*ellipseity

        start_cx = hillas[0, 0]
        start_cy = hillas[0, 1]
        end_cx = hillas[-1, 0]
        end_cy = hillas[-1, 1]
        del_cx = start_cx - end_cx
        del_cy = start_cy - end_cy
        rec_cx = start_cx + 10*del_cx
        rec_cy = start_cy + 10*del_cy

        fig = plt.figure(figsize=(4, 4), dpi=100)
        ax = fig.add_axes([.1,.1,.9,.9])
        ax.plot(-np.rad2deg(hillas[:, 0]), -np.rad2deg(hillas[:, 1]))
        ax.plot(-np.rad2deg(hillas[:, 0]), -np.rad2deg(hillas[:, 1]), 'x')

        ax.plot(
            np.rad2deg(feat['light_front_cx']),
            np.rad2deg(feat['light_front_cy']),
            'xk',
            alpha=0.2)

        ax.plot(
            [-np.rad2deg(start_cx), -np.rad2deg(rec_cx)],
            [-np.rad2deg(start_cy), -np.rad2deg(rec_cy)], 'g', alpha=0.2)

        ax.plot(
            [-np.rad2deg(hillas_start_cx), -np.rad2deg(hillas_stop_cx)],
            [-np.rad2deg(hillas_start_cy), -np.rad2deg(hillas_stop_cy)],
            'k',
            alpha=0.2)

        ax.plot(
            [-np.rad2deg(hillas_cx_mean), -np.rad2deg(rec_cx)],
            [-np.rad2deg(hillas_cy_mean), -np.rad2deg(rec_cy)], 'g', alpha=0.2)

        ax.plot(
            -np.rad2deg(hillas_cx_mean) + feat['time_grad_cx']*1e6,
            -np.rad2deg(hillas_cy_mean) + feat['time_grad_cy']*1e6, 'gx',)

        ax.plot(np.rad2deg(feat['particle_cx']), np.rad2deg(feat['particle_cy']), 'or')

        ax.plot(feat['particle_x']/150*3.25, feat['particle_y']/150*3.25, 'ob')

        ax.set_aspect('equal')
        ax.set_xlim(-3.25, 3.25)
        ax.set_ylim(-3.25, 3.25)
        plt.show()

        print(hillas[:, 8], max_dense_obj)


        features.append(feat)
    return features