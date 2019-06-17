from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
import json
import plenopy as pl
import corsika_wrapper as cw
import simpleio as sio
import os
import subprocess as sp
import lookup

np.random.seed = 1

SPEED_OF_LIGHT = 3e8
PLOT = False

PhotonObservables = namedtuple(
    'PhotonObservables',
    ['x', 'y', 'cx', 'cy', 'relative_arrival_times'])


LightFieldGeometry = namedtuple(
    'LightFieldGeometry',
    [
        'x',
        'y',
        'cx',
        'cy',
        't',
        'x_bin_edges',
        'y_bin_edges',
        'cx_bin_edges',
        'cy_bin_edges',
        't_bin_edges'])


def init_LightFieldGeometry(
    aperture_radius=35.5,
    field_of_view_radius_deg=3.25,
    paxel_on_diagonal=8,
    pixel_on_diagonal=int(np.round(6.5/0.0667)),
    time_radius=25e-9,
    num_time_slices=100,
):
    cxy_bin_edges = np.linspace(
        -np.deg2rad(field_of_view_radius_deg),
        +np.deg2rad(field_of_view_radius_deg),
        pixel_on_diagonal + 1)
    cxy_bin_centers = (cxy_bin_edges[0: -1] + cxy_bin_edges[1:])/2

    xy_bin_edges = np.linspace(
        -aperture_radius,
        +aperture_radius,
        paxel_on_diagonal + 1)
    xy_bin_centers = (xy_bin_edges[0: -1] + xy_bin_edges[1:])/2

    t_bin_edges = np.linspace(
        -time_radius,
        time_radius,
        num_time_slices + 1)
    t_bin_centers = (t_bin_edges[0: -1] + t_bin_edges[1:])/2

    return LightFieldGeometry(
        cx=cxy_bin_centers,
        cx_bin_edges=cxy_bin_edges,
        cy=cxy_bin_centers,
        cy_bin_edges=cxy_bin_edges,
        x=xy_bin_centers,
        x_bin_edges=xy_bin_edges,
        y=xy_bin_centers,
        y_bin_edges=xy_bin_edges,
        t=t_bin_centers,
        t_bin_edges=t_bin_edges,)


def photons_to_light_field_sequence(ph, lfg):
    cx_idx = np.digitize(ph.cx, bins=lfg.cx_bin_edges)
    cx_valid = (cx_idx > 0)*(cx_idx < lfg.cx_bin_edges.shape[0])
    cy_idx = np.digitize(ph.cy, bins=lfg.cy_bin_edges)
    cy_valid = (cy_idx > 0)*(cy_idx < lfg.cy_bin_edges.shape[0])
    x_idx = np.digitize(ph.x, bins=lfg.x_bin_edges)
    x_valid = (x_idx > 0)*(x_idx < lfg.x_bin_edges.shape[0])
    y_idx = np.digitize(ph.y, bins=lfg.y_bin_edges)
    y_valid = (y_idx > 0)*(y_idx < lfg.y_bin_edges.shape[0])
    t_idx = np.digitize(ph.relative_arrival_times, bins=lfg.t_bin_edges)
    t_valid = (t_idx > 0)*(t_idx < lfg.t_bin_edges.shape[0])
    valid = cx_valid * cy_valid * x_valid * y_valid * t_valid
    photons_column_wise = np.array(
        [cx_idx[valid], cy_idx[valid], x_idx[valid], y_idx[valid], t_idx[valid]],
        dtype=np.uint8)
    photons_row_wise = photons_column_wise.T
    return photons_row_wise.flatten()


def argmax2d(img):
    arg1d = np.argmax(img)
    idx0 = arg1d//img.shape[0]
    idx1 = arg1d - idx0*img.shape[0]
    return int(idx0), idx1


def add_ring(ax, radius, cx=0, cy=0):
    phis = np.linspace(0, 2*np.pi, 1024)
    xs = cx + radius*np.cos(phis)
    ys = cy + radius*np.sin(phis)
    ax.plot(xs, ys, '-w')


def pandas_DataFrame_from_jsonl(path):
    r = []
    with open(path, 'rt') as fin:
        for line in fin:
            r.append(json.loads(line))
    return pd.DataFrame.from_dict(r)


def PhotonObservables_from_bunches(cherenkov_photon_bunches):
    cpb = cherenkov_photon_bunches
    relative_arrival_times = cpb.arrival_time_since_first_interaction
    relative_arrival_times -= np.median(relative_arrival_times)

    num_photons = cpb.x.shape[0]
    probs = np.random.uniform(size=num_photons)
    passed = cpb.probability_to_reach_observation_level > probs

    return PhotonObservables(
        x=cpb.x[passed],
        y=cpb.y[passed],
        cx=cpb.cx[passed],
        cy=cpb.cy[passed],
        relative_arrival_times=relative_arrival_times[passed])


def cut_PhotonObservables(photons, mask):
    return PhotonObservables(
        x=photons.x[mask],
        y=photons.y[mask],
        cx=photons.cx[mask],
        cy=photons.cy[mask],
        relative_arrival_times=photons.relative_arrival_times[mask])


def surface_normal_of_light_front(photons_x, photons_y, photons_z):
    B, inlier = pl.tools.ransac_3d_plane.fit(
        xyz_point_cloud=np.c_[photons_x, photons_y, photons_z],
        max_number_itarations=500,
        min_number_points_for_plane_fit=10,
        max_orthogonal_distance_of_inlier=0.025,)
    c_pap_time = np.array([B[0], B[1], B[2]])
    if c_pap_time[2] > 0:
        c_pap_time *= -1
    c_pap_time = c_pap_time/np.linalg.norm(c_pap_time)
    return float(c_pap_time[0]), float(c_pap_time[1])


def integration_width_for_one_sigma(hist, bin_edges):
    one_sigma = 0.68
    integral = np.cumsum(hist/np.sum(hist))
    bin_centers = (bin_edges[0:-1] + bin_edges[1:])/2
    x = np.linspace(
        np.min(bin_centers),
        np.max(bin_centers),
        100*bin_centers.shape[0])
    f = np.interp(x=x, fp=integral, xp=bin_centers)
    return x[np.argmin(np.abs(f - one_sigma))]


def mean_in_90percentile(a):
    num_samples = len(a)
    asort = np.sort(a)
    start = int(np.floor(num_samples*0.05))
    stop = int(np.floor(num_samples*0.95))
    return np.mean(a[start: stop])


def refocus_cx_cy(
    photons_cx,
    photons_cy,
    photons_x,
    photons_y,
    object_distance=10e3,
    focal_length=1.):
    x_focal_plane = -np.tan(photons_cx)*focal_length
    y_focal_plane = -np.tan(photons_cy)*focal_length
    z_focal_plane = focal_length

    x_aperture_to_focal_dir = x_focal_plane - photons_x
    y_aperture_to_focal_dir = y_focal_plane - photons_y
    z_aperture_to_focal_dir = z_focal_plane - 0.0

    aperture_to_focal_norm = np.sqrt(
        x_aperture_to_focal_dir**2 +
        y_aperture_to_focal_dir**2 +
        z_aperture_to_focal_dir**2)

    x_aperture_to_focal_dir /= aperture_to_focal_norm
    y_aperture_to_focal_dir /= aperture_to_focal_norm
    z_aperture_to_focal_dir /= aperture_to_focal_norm

    sensor_plane_distance = 1./(1./focal_length - 1./object_distance)

    alpha = sensor_plane_distance/z_aperture_to_focal_dir

    x_sensor_plane = photons_x + alpha*x_aperture_to_focal_dir
    y_sensor_plane = photons_y + alpha*y_aperture_to_focal_dir

    cx_sensor_plane = -np.arctan(x_sensor_plane/sensor_plane_distance)
    cy_sensor_plane = -np.arctan(y_sensor_plane/sensor_plane_distance)
    return cx_sensor_plane, cy_sensor_plane


c_bin_edges = np.linspace(
    -np.deg2rad(3.5),
    np.deg2rad(3.5),
    np.int64(np.floor(7/0.0667)))

aperture_bin_edges = np.linspace(-36, 36, 10)

object_distances = np.logspace(
    np.log10(5e3),
    np.log10(35e3),
    7)


MIRROR_REFLECTIVITY = 0.8
PHOTO_DETECTION_EFFICIENCY = 0.25

sites = {
    "gamsberg": {
        "observation_level_altitude_asl": 2347.0,
        "atmosphere_model": 10,
        "earth_magnetic_field_x_muT": 12.5,
        "earth_magnetic_field_z_muT": -25.9,
    },
}
"""
    "paranal": {
        "observation_level_altitude_asl": 5000.0,
        "atmosphere_model": 26,
        "earth_magnetic_field_x_muT": 20.815,
        "earth_magnetic_field_z_muT": -11.366,
    },
    "paranal_magnetic-off": {
        "observation_level_altitude_asl": 5000.0,
        "atmosphere_model": 26,
        "earth_magnetic_field_x_muT": 1e-9,
        "earth_magnetic_field_z_muT": 1e-9,
    },
}
"""

particles = {
    'gamma': {
        "prmpar": 1,
        "E_start": 0.8,
        "E_stop": 1.6,
        "max_theta": 2.
    }
}  #, 'electron', 'proton']

out_dir = './limits/run'
os.makedirs(out_dir, exist_ok=True)

# make steering-cards
# -------------------
for site in sites:
    for particle in particles:
        steering_card_path = "{:s}_{:s}.txt".format(site, particle)
        steering_card_path = os.path.join(out_dir, steering_card_path)
        card = '' \
        'RUNNR 1\n' \
        'EVTNR 1\n' \
        'NSHOW 2560\n' \
        'PRMPAR  {prmpar:d}\n' \
        'ESLOPE -1.0\n' \
        'ERANGE {E_start:.2e} {E_stop:.2e}\n' \
        'THETAP 0.  {max_theta:.2f}\n' \
        'PHIP 0.  360.\n' \
        'SEED 1 0 0\n' \
        'SEED 2 0 0\n' \
        'SEED 3 0 0\n' \
        'SEED 4 0 0\n' \
        'OBSLEV {obslev:.2e}\n' \
        'FIXCHI 0.\n' \
        'MAGNET {Bx:.2e} {Bz:.2e}\n' \
        'ELMFLG T T\n' \
        'MAXPRT 1\n' \
        'PAROUT F F\n' \
        'TELESCOPE 0. 0. 0. 3.600e+03\n' \
        'ATMOSPHERE {atmosphere:d} T\n' \
        'CWAVLG 250 700\n' \
        'CSCAT 1 1.500e+04 0.0\n' \
        'CERQEF F T F\n' \
        'CERSIZ 1\n' \
        'CERFIL F\n' \
        'TELFIL "."\n' \
        'TSTART T\n' \
        ''.format(
            prmpar=particles[particle]["prmpar"],
            E_start=particles[particle]["E_start"],
            E_stop=particles[particle]["E_stop"],
            max_theta=particles[particle]["max_theta"],
            Bx=sites[site]["earth_magnetic_field_x_muT"],
            Bz=sites[site]["earth_magnetic_field_z_muT"],
            atmosphere=sites[site]["atmosphere_model"],
            obslev=sites[site]["observation_level_altitude_asl"]*1e2,)
        with open(steering_card_path, 'wt') as f:
            f.write(card)


# make air-showers
# ----------------
for site in sites:
    for particle in particles:
        evtio_run_path = "{:s}_{:s}.evtio".format(site, particle)
        evtio_run_path = os.path.join(out_dir, evtio_run_path)
        if not os.path.exists(evtio_run_path):
            steering_card_path = "{:s}_{:s}.txt".format(site, particle)
            steering_card_path = os.path.join(out_dir, steering_card_path)
            steering_card = cw.read_steering_card(steering_card_path)
            cw.corsika(
                steering_card=steering_card,
                output_path=evtio_run_path,
                save_stdout=True)

        simpleio_run_path = "{:s}_{:s}.sio".format(site, particle)
        simpleio_run_path = os.path.join(out_dir, simpleio_run_path)
        if not os.path.exists(simpleio_run_path):
            sp.call([
                './build/merlict/merlict-eventio-converter',
                '-i', evtio_run_path,
                '-o', simpleio_run_path])


lfg = init_LightFieldGeometry()
# extract features
# ----------------
for site in sites:
    for particle in particles:
        simpleio_run_path = "{:s}_{:s}.sio".format(site, particle)
        simpleio_run_path = os.path.join(out_dir, simpleio_run_path)
        run = sio.SimpleIoRun(simpleio_run_path)

        feature_path = "{:s}_{:s}.jsonl".format(site, particle)
        feature_path = os.path.join(out_dir, feature_path)
        if os.path.exists(feature_path):
            continue
        fout = open(feature_path, 'wt')
        print(feature_path)

        lookup_dir = "{:s}_{:s}.lookup".format(site, particle)
        lookup_dir = os.path.join(out_dir, lookup_dir)
        lua = lookup.LookUpAppender(lookup_dir)

        fs = []
        for eidx in range(run.number_events):
            event = run[eidx]
            photons = PhotonObservables_from_bunches(
                event.cherenkov_photon_bunches)

            fov_radius = np.deg2rad(3.25)
            photons_c = np.hypot(photons.cx, photons.cy)
            in_fov = photons_c < fov_radius

            photons = cut_PhotonObservables(
                photons=photons,
                mask=in_fov)

            num_photons_ground = len(photons.x)
            dices = np.random.uniform(size=num_photons_ground)
            detected = dices < MIRROR_REFLECTIVITY*PHOTO_DETECTION_EFFICIENCY
            photons = cut_PhotonObservables(
                photons=photons,
                mask=detected)

            # event truth
            # -----------
            features = {}
            features['event'] = int(event.header.raw[2 - 1])
            features['run'] = int(event.header.raw[44 - 1])
            features['particle_id'] = int(event.header.primary_particle_id)
            features['particle_energy'] = float(event.header.total_energy_GeV)
            particle_momentum = event.header.momentum()
            particle_direction = particle_momentum/np.linalg.norm(
                particle_momentum)
            features['particle_cx'] = float(particle_direction[0])
            features['particle_cy'] = float(particle_direction[1])
            features['particle_x'] = float(event.header.core_position_x_meter())
            features['particle_y'] = float(event.header.core_position_y_meter())
            features['particle_height_first_interaction'] = float(
                np.abs(event.header.raw[7 - 1]*1e-2))

            num_photons = photons.x.shape[0]
            features['num_photons'] = float(num_photons)

            print(particle, eidx, num_photons)

            features['trigger'] = int(0)
            # simple trigger
            # --------------
            if num_photons > 50:
                features['trigger'] = int(1)

                lfs_idx = photons_to_light_field_sequence(ph=photons, lfg=lfg)
                lua.append_event(
                    light_field_sequence_uint8=lfs_idx,
                    features=features)
                # extract event-features
                # ----------------------

                # point-cloud
                # -----------
                lf_cx, lf_cy = surface_normal_of_light_front(
                    photons_x=photons.x,
                    photons_y=photons.y,
                    photons_z=SPEED_OF_LIGHT*photons.relative_arrival_times)
                features['light_front_cx'] = lf_cx
                features['light_front_cy'] = lf_cy


                # image
                # -----
                features['median_cx'] = float(np.median(photons.cx))
                features['median_cy'] = float(np.median(photons.cy))

                # refocus images
                # --------------
                imgs = []
                for obj in object_distances:
                    cx_refocus, cy_refocus = refocus_cx_cy(
                        photons_cx=photons.cx,
                        photons_cy=photons.cy,
                        photons_x=photons.x,
                        photons_y=photons.y,
                        object_distance=obj)
                    img = np.histogram2d(
                        cx_refocus,
                        cy_refocus,
                        bins=c_bin_edges)[0]
                    imgs.append(img)
                imgs = np.array(imgs)
                max_on_slices = np.max(imgs, axis=(1, 2))
                densest_slice = np.argmax(max_on_slices)
                img_highest_density = imgs[densest_slice, :, :]
                cxidx, cyidx = argmax2d(img_highest_density)
                features['refocus_max_cx'] = c_bin_edges[cxidx],
                features['refocus_max_cy'] = c_bin_edges[cyidx],

                if PLOT:
                    vmax = np.max(imgs)
                    vmin = np.min(imgs)
                    for i in range(len(object_distances)):
                        fig = plt.figure(figsize=(6, 6), dpi=60)
                        ax = fig.add_axes((0., 0., 1, 1))
                        ax.pcolor(
                            c_bin_edges,
                            c_bin_edges,
                            imgs[i, :, :].T,
                            vmin=vmin,
                            vmax=vmax)
                        ax.plot(
                            features['particle_cx'],
                            features['particle_cy'],
                            'rx')
                        ax.plot(
                            features['light_front_cx'],
                            features['light_front_cy'],
                            'ro')
                        ax.plot(
                            features['median_cx'],
                            features['median_cy'],
                            'go')
                        ax.plot(
                            features['refocus_max_cx'],
                            features['refocus_max_cy'],
                            'gx')
                        add_ring(ax, np.deg2rad(1))
                        add_ring(ax, np.deg2rad(2))
                        add_ring(ax, np.deg2rad(3))
                        fig_path = "{:s}_{:s}_{:d}_ref{:d}.png".format(
                            site,
                            particle,
                            eidx,
                            i)
                        fig_path = os.path.join(out_dir, fig_path)
                        fig.savefig(fig_path)
                        plt.close('all')

                    """
                    img_densities = []
                    for img in imgs:
                        has_photons = img > 0
                        img_densities.append(
                            mean_in_90percentile(img[has_photons])/np.sum(img[has_photons])
                        )
                    img_densities = np.array(img_densities)
                    dense_obj = np.argmax(img_densities)
                    features['cx_median_dense'] = float(cx_medians[dense_obj])
                    features['cy_median_dense'] = float(cy_medians[dense_obj])

                    if PLOT:
                        plt.Figure()
                        plt.plot(object_distance, img_densities)
                        plt.semilogx()
                        plt.savefig('{:s}_{:d}.png'.format(particle, eidx))
                        plt.close('all')
                    """


                # aperture intensity
                # -------------------

                aperture_intensity = np.histogram2d(
                    photons.x,
                    photons.y,
                    bins=aperture_bin_edges)[0]

                if PLOT:
                    fig = plt.figure(figsize=(6, 6), dpi=30)
                    ax = fig.add_axes((0., 0., 1, 1))
                    ax.pcolor(
                        aperture_bin_edges,
                        aperture_bin_edges,
                        aperture_intensity)
                    add_ring(ax, 36)
                    fig_path = "{:s}_{:s}_{:d}_aperture.png".format(
                        site,
                        particle,
                        eidx)
                    fig_path = os.path.join(out_dir, fig_path)
                    fig.savefig(fig_path)
                    plt.close('all')


            fs.append(features)
            fout.write(json.dumps(features)+'\n')
        fout.close()


# histogram features
# ------------------
for site in sites:
    for particle in particles:
        feature_path = "{:s}_{:s}.jsonl".format(site, particle)
        feature_path = os.path.join(out_dir, feature_path)
        r = pandas_DataFrame_from_jsonl(feature_path)
        passed_trigger = r.trigger == 1
        r = r[passed_trigger]

        hist_dir = "{:s}_{:s}_histograms".format(site, particle)
        hist_dir = os.path.join(out_dir, hist_dir)
        os.makedirs(hist_dir, exist_ok=True)

        cx_residuum = r.particle_cx - r.light_front_cx
        cy_residuum = r.particle_cy - r.light_front_cy
        c_residuum = np.hypot(cx_residuum, cy_residuum)

        num_bins = np.int64(np.sqrt(c_residuum.shape[0]))
        theta_bin_edges = np.linspace(
            np.deg2rad(0.),
            np.deg2rad(2.),
            num_bins)

        theta_bins = np.histogram(
            c_residuum,
            bins=theta_bin_edges)[0]

        one_sigma_radius = integration_width_for_one_sigma(
            hist=theta_bins,
            bin_edges=theta_bin_edges)

        fig = plt.figure(figsize=(8, 4.5), dpi=100)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        h = ax.hist(
            np.rad2deg(c_residuum),
            bins=np.rad2deg(theta_bin_edges),
            fc='gray',
            ec='none')
        ax.axvline(np.rad2deg(one_sigma_radius), color='k')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlim([0., 2.])
        ax.set_xlabel('Residual incident-direction / deg')
        ax.set_ylabel('Number of events / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig_path = "{:s}_{:s}_c_residuum.png".format(site, particle)
        fig_path = os.path.join(hist_dir, fig_path)
        fig.savefig(fig_path)
        plt.close('all')

        num_photons_bin_edges = np.linspace(0, 1000, num_bins)
        fig = plt.figure(figsize=(8, 4.5), dpi=100)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        h = ax.hist(
            r.num_photons,
            bins=num_photons_bin_edges,
            fc='gray',
            ec='none')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('Number photons / 1')
        ax.set_ylabel('Number of events / 1')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig_path = "{:s}_{:s}_num_photons.png".format(site, particle)
        fig_path = os.path.join(hist_dir, fig_path)
        fig.savefig(fig_path)
        plt.close('all')

        fig = plt.figure(figsize=(8, 4.5), dpi=100)
        ax = fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.hist2d(
            np.rad2deg(r.light_front_cx),
            np.rad2deg(r.light_front_cy),
            bins=np.linspace(-3,3,80))
        ax.set_aspect('equal')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.set_xlabel('cx / deg')
        ax.set_ylabel('cy / deg')
        ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
        fig_path = "{:s}_{:s}_cxcy_light_front.png".format(site, particle)
        fig_path = os.path.join(hist_dir, fig_path)
        fig.savefig(fig_path)
        plt.close('all')
