import numpy as np
import tempfile
import os
import gev_limits as gli
import sun_grid_engine_map as sge
import shutil
import pandas as pd
from multiprocessing import pool

out_dir = os.path.abspath('.')

gamma_particle = {
    'prmpar': 1,
    'max_zenith_angle_deg': 4.25,
    "energy":             [0.23, 0.8, 3.0, 35,   81,   432,],#  1000],
    "max_scatter_radius": [150,  150, 460, 1100, 1235, 1410,]# 1660]
}

electron_particle = {
    'prmpar': 3,
    'max_zenith_angle_deg': 6.5,
    "energy":             [0.23, 1.0,  10,  100,],#  1000],
    "max_scatter_radius": [150,  150,  500, 1100,],# 2600]
}

proton_particle = {
    'prmpar': 14,
    'max_zenith_angle_deg': 6.5,
    "energy":             [5.0, 25, 250,],# 1000],
    "max_scatter_radius": [200, 350, 700,],# 1250]
}

portal_instrument = {
    'aperture_radius': 35.5,
    'num_paxel_on_diagonal': 8,
    'field_of_view_radius_deg': 3.25,
    'num_pixel_on_diagonal': int(np.round(6.5/0.0667)),
    'time_radius': 25e-9,
    'num_time_slices': 100,
    'relative_arrival_times_std': 1e-9,
    'mirror_reflectivity': 0.85,
    'photo_detection_efficiency': 0.3,
}

TRIGGER_THRESHOLD = 90
NSB_RATE_PIXEL = 25e6*(5e-9)*427*(1/7)

gamsberg_site = {
    'atmosphere': 10,
    'observation_level_altitude_asl': 2347.0,
    'earth_magnetic_field_x_muT': 12.5,
    'earth_magnetic_field_z_muT': -25.9,
}

paranal_site = {
    'atmosphere': 26,
    'observation_level_altitude_asl': 5000.0,
    'earth_magnetic_field_x_muT': 20.815,
    'earth_magnetic_field_z_muT': -11.366,
}

NUM_RUNS = 4
NUM_EVENTS_IN_RUN = 20

MERLICT_PATH = os.path.abspath('./build/merlict/merlict-eventio-converter')
assert os.path.exists(MERLICT_PATH)

particles = {
    "gamma": gamma_particle,
    "electron": electron_particle,
    "proton": proton_particle
}

sites = {
    "paranal": paranal_site,
    "gamsberg": gamsberg_site
}

for site in sites:
    for particle in particles:
        site_dir = os.path.join(out_dir ,'__gev_limits_{:s}'.format(site))
        lut_path = os.path.join(site_dir, '{:s}.lut'.format(particle))
        if os.path.exists(lut_path):
            continue

        map_and_reduce_dir = os.path.join(
            site_dir, 'map_and_reduce_{:s}'.format(particle))
        os.makedirs(map_and_reduce_dir)

        __site = sites[site].copy()
        if particle in ['electron', 'proton']:
            __site['earth_magnetic_field_x_muT'] = 1e-9
            __site['earth_magnetic_field_z_muT'] = 1e-9

        jobs = gli.map_and_reduce.make_jobs(
            map_and_reduce_dir=map_and_reduce_dir,
            random_seed=1,
            num_runs=NUM_RUNS,
            max_num_events_in_run=NUM_EVENTS_IN_RUN,
            eventio_converter_path=MERLICT_PATH,
            instrument=portal_instrument,
            particle=particles[particle],
            site=__site,
            nsb_rate_pixel=NSB_RATE_PIXEL,
            trigger_threshold=TRIGGER_THRESHOLD)

        #for job in jobs:
        #    gli.map_and_reduce.run_job(job)
        tpool = pool.ThreadPool(4)
        results = tpool.map(gli.map_and_reduce.run_job, jobs)
        #sge.map(gli.map_and_reduce.run_job, jobs)

        lut_paths = []
        for job in jobs:
            lut_paths.append(job['out_path'])
        gli.lookup.concatenate(lut_paths, lut_path)

        lut = gli.lookup.LookUpTable(lut_path)
        shutil.rmtree(map_and_reduce_dir)


# thrown
# ------
for site in sites:
    for particle in particles:
        site_dir = os.path.join(out_dir ,'__gev_limits_{:s}'.format(site))
        lut_path = os.path.join(site_dir, '{:s}.lut'.format(particle))
        if os.path.exists(lut_path+'.thrown.msg'):
            continue
        thrown = gli.thrown_structure.read_events_thrown(
            os.path.join(lut_path, 'thrown.float32'))
        df = pd.DataFrame(thrown)
        df.to_msgpack(lut_path+'.thrown.msg')


def plot_features(light_field_sequence, plenoscope, event_features, i, path):
    ef = event_features
    os.makedirs(path, exist_ok=True)

    aperture_histogram = make_aperture_histogram(
        lfs=lfs,
        plenoscope=plenoscope)

    fig = plt.figure(figsize=(16, 8), dpi=50)
    ax = fig.add_axes([(0.1)/2, 0.1, (.8)/2, .8])
    ax.pcolor(
        plenoscope.x_bin_edges,
        plenoscope.y_bin_edges,
        aperture_histogram.T,
        cmap='inferno')
    ax.plot(
        ef['aperture_paxel_intensity_max_x'],
        ef['aperture_paxel_intensity_max_y'],
        'xr')
    axt = fig.add_axes([(1)/2, 0, (1)/2, 1])
    axt.set_axis_off()
    axt.text(
        0.1,
        0.85,
        'aperture_intensity_peakness: {:.2f}'.format(
            ef['aperture_intensity_peakness']))
    axt.text(
        0.1,
        0.8,
        r'aperture_intensity_slope_x: {:.2f}p.e.m$^{{-1}}$'.format(
            ef['aperture_intensity_slope_x']))
    axt.text(
        0.1,
        0.75,
        r'aperture_intensity_slope_y: {:.2f}p.e.m$^{{-1}}$'.format(
            ef['aperture_intensity_slope_y']))
    plt.savefig(os.path.join(path, '{:06d}_aperture.png'.format(i)))
    plt.close('all')


def argmax2d(X):
    return np.unravel_index(X.argmax(), X.shape)


def make_aperture_histogram(lfs, plenoscope):
    return np.histogram2d(
        lut.plenoscope.x[lfs[:, 2]],
        lut.plenoscope.y[lfs[:, 3]],
        bins=[plenoscope.x_bin_edges, plenoscope.y_bin_edges])[0]


def make_aperture_x_histogram(lfs, plenoscope):
    return np.histogram(
        lut.plenoscope.x[lfs[:, 2]],
        bins=plenoscope.x_bin_edges)[0]


def make_aperture_y_histogram(lfs, plenoscope):
    return np.histogram(
        lut.plenoscope.y[lfs[:, 3]],
        bins=plenoscope.y_bin_edges)[0]


def estimate_cxy_bin_edges(lfs, plenoscope, num_pixel_roi=12):
    cxs = plenoscope.cx[lfs[:, 0]]
    cys = plenoscope.cx[lfs[:, 1]]

    roi_cx = np.mean(cxs)
    roi_cy = np.mean(cys)

    pixel_diameter = 2*(
        np.deg2rad(plenoscope.field_of_view_radius_deg)/
        plenoscope.num_pixel_on_diagonal)

    cx_bin_edges = np.arange(
        roi_cx - num_pixel_roi*pixel_diameter,
        roi_cx + (1+num_pixel_roi)*pixel_diameter,
        pixel_diameter)

    cy_bin_edges = np.arange(
        roi_cy - num_pixel_roi*pixel_diameter,
        roi_cy + (1+num_pixel_roi)*pixel_diameter,
        pixel_diameter)

    return cx_bin_edges, cy_bin_edges


NUM_REFS = 32
NUM_PIXEL_ROI = 12
OBJECT_DISTANCES = np.geomspace(1e3, 60e3, NUM_REFS)


for site in sites:
    for particle in particles:
        site_dir = os.path.join(out_dir ,'__gev_limits_{:s}'.format(site))
        lut_path = os.path.join(site_dir, '{:s}.lut'.format(particle))
        if os.path.exists(lut_path+'.features.msg'):
            continue

        lut = gli.lookup.LookUpTable(lut_path)

        features_site_particle = []
        for event_idx in range(lut.num_events):
            if event_idx > 5000:
                break

            event_features = {}

            event_features['run'] = lut.run[event_idx]
            event_features['event'] = lut.event[event_idx]

            lfs = lut._raw_light_field_sequence(event_idx)

            event_features['num_photons'] = lfs.shape[0]

            c_radial = np.hypot(
                lut.plenoscope.cx[lfs[:, 0]],
                lut.plenoscope.cy[lfs[:, 1]])

            event_features['num_photons_on_edge_field_of_view'] = np.sum(
                c_radial > 0.9*np.deg2rad(
                    lut.plenoscope.field_of_view_radius_deg))

            # aperture
            # --------
            aperture_histogram = make_aperture_histogram(
                lfs=lfs,
                plenoscope=lut.plenoscope)

            event_features['aperture_intensity_peakness'] = np.std(
                aperture_histogram.flatten())/np.mean(
                aperture_histogram.flatten())

            aperture_x_histogram = make_aperture_x_histogram(
                lfs=lfs,
                plenoscope=lut.plenoscope)
            event_features['aperture_intensity_slope_x'] = np.polyfit(
                x=lut.plenoscope.x,
                y=aperture_x_histogram,
                deg=1)[0]
            aperture_y_histogram = make_aperture_y_histogram(
                lfs=lfs,
                plenoscope=lut.plenoscope)
            event_features['aperture_intensity_slope_y'] = np.polyfit(
                x=lut.plenoscope.y,
                y=aperture_y_histogram,
                deg=1)[0]

            event_features['aperture_paxel_intensity_max_x'] = lut.plenoscope.x[
                argmax2d(aperture_histogram)[0]]
            event_features['aperture_paxel_intensity_max_y'] = lut.plenoscope.x[
                argmax2d(aperture_histogram)[1]]

            # light-front
            # -----------
            """
            lf_xs, lf_ys, lf_zs = gli.light_field.get_light_front(
                lfs=lfs,
                plenoscope=lut.plenoscope)
            light_front_normal = gli.features.light_front_surface_normal(
                xs=lf_xs,
                ys=lf_ys,
                zs=lf_zs)
            event_features['light_front_cx'] = light_front_normal[0]
            event_features['light_front_cy'] = light_front_normal[1]
            """
            # refocus-stack
            # -------------

            image_photons = gli.light_field.get_image_photons(
                lfs=lfs,
                plenoscope=lut.plenoscope)

            tpap = lut.plenoscope.t[lfs[:, 4]]
            hillas = []
            for obj in OBJECT_DISTANCES:
                cxs, cys = gli.light_field.get_refocused_cx_cy(
                    image_photons=image_photons,
                    object_distance=obj)
                timg = tpap -1.0*gli.light_field.add_to_tpap_to_get_timg(
                    cx=cxs,
                    cy=cys,
                    x=image_photons.x,
                    y=image_photons.y)

                timg = np.round(timg/.5e-9)*.5e-9

                eli = gli.features.hillas_ellipse(cxs=cxs, cys=cys)

                time_grad_cx = np.polyfit(x=cxs, y=timg, deg=1)[0]
                time_grad_cy = np.polyfit(x=cys, y=timg, deg=1)[0]

                ellipse_area = np.pi*eli.std_major*eli.std_minor
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
                    time_grad_cy,
                    ellipse_area
                ]
                hillas.append(e)
            hillas = np.array(hillas)

            ellipse_areas = hillas[:, 11]

            idx_min_area = np.argmin(ellipse_areas)
            if (
                idx_min_area == 0 or
                idx_min_area == OBJECT_DISTANCES.shape[0] - 1
            ):
                continue

            event_features['hillas_smallest_ellipse_solid_angle'] = ellipse_areas[idx_min_area]
            event_features['hillas_smallest_ellipse_object_distance'] = OBJECT_DISTANCES[
                idx_min_area]

            event_features['hillas_smallest_ellipse_cx'] = hillas[idx_min_area, 0]
            event_features['hillas_smallest_ellipse_cy'] = hillas[idx_min_area, 1]

            """
            cx_bin_edges, cy_bin_edges = estimate_cxy_bin_edges(
                lfs=lfs,
                plenoscope=lut.plenoscope,
                num_pixel_roi=NUM_PIXEL_ROI)

            refocus_stack = gli.light_field.get_refocus_stack(
                image_photons=image_photons,
                object_distances=OBJECT_DISTANCES,
                cx_bin_edges=cx_bin_edges,
                cy_bin_edges=cy_bin_edges)
            refocus_stack = gli.light_field.smear_out_refocus_stack(
                refocus_stack)




            vmax = np.max(refocus_stack)
            fig = plt.figure(figsize=(8, 8*NUM_REFS), dpi=50)
            SUB_FIG_HEIGHT = 1/NUM_REFS
            for obj, object_distance in enumerate(OBJECT_DISTANCES):
                axi = fig.add_axes([
                    0,
                    SUB_FIG_HEIGHT*obj,
                    1*.98,
                    SUB_FIG_HEIGHT*.98])
                axi.set_axis_off()
                axi.pcolor(
                    refocus_stack[obj, :, :],
                    vmax=vmax,
                    cmap='inferno')
                #axi.plot(0.5, 0.5, "xw")
                axi.text(
                    0.1, 0.1,
                    "{:.1f}km".format(1e-3*object_distance), color="white")
            plt.savefig(os.path.join(lut_path,
                '{:06d}_refocus_stack.png'.format(event_idx)))
            plt.close('all')
            """
            print(event_idx)
            """
            plot_features(
                event_features=event_features,
                light_field_sequence=lfs,
                plenoscope=lut.plenoscope,
                path=os.path.join(lut_path),
                i=event_idx)
            """
            features_site_particle.append(event_features)
        df = pd.DataFrame(features_site_particle)
        df.to_msgpack(lut_path+'.features.msg')


# learn
# -----
import sklearn
from sklearn import neural_network


def norm_hillas_smallest_ellipse_object_distance(obj):
    return np.log10(obj) - 3

def norm_hillas_smallest_ellipse_solid_angle(sa):
    return np.log10(sa) + 7


def add_hist(ax, bin_edges, bincounts, linestyle, color, alpha):
    assert bin_edges.shape[0] == bincounts.shape[0] + 1
    for i, bincount in enumerate(bincounts):
        ax.plot(
            [bin_edges[i], bin_edges[i + 1]],
            [bincount, bincount],
            linestyle)
        ax.fill_between(
            x=[bin_edges[i], bin_edges[i + 1]],
            y1=[bincount, bincount],
            color=color,
            alpha=alpha,
            edgecolor='none')


for site in sites:
    site_dir = os.path.join(out_dir ,'__gev_limits_{:s}'.format(site))

    gammas = pd.read_msgpack(os.path.join(site_dir, 'gamma.lut.features.msg'))
    electrons = pd.read_msgpack(os.path.join(site_dir, 'electron.lut.features.msg'))
    protons = pd.read_msgpack(os.path.join(site_dir, 'proton.lut.features.msg'))

    num_bins = int(np.sqrt(gammas.shape[0]))

    #-------------------------------------------------
    hsesa_bin_edges = np.geomspace(
        1e-7,
        1e-3,
        num_bins)

    hsesa_gamma = np.histogram(
        gammas.hillas_smallest_ellipse_solid_angle,
        bins=hsesa_bin_edges)[0]
    hsesa_proton = np.histogram(
        protons.hillas_smallest_ellipse_solid_angle,
        bins=hsesa_bin_edges)[0]

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1,.15,.85,.8])
    add_hist(
        ax=ax,
        bin_edges=hsesa_bin_edges,
        bincounts=hsesa_gamma,
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=hsesa_bin_edges,
        bincounts=hsesa_proton,
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(0.05, 0.95, 'gamma', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.9, 'proton', color='red', transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel(r'solid-angle of smallest hillas-ellipse in refocus-image-stack/sr')
    ax.set_ylabel(r'number events/1')
    plt.savefig('{:s}_{:s}.png'.format(site, 'hillas_smallest_ellipse_solid_angle'))
    plt.close('all')

    #-------------------------------------------------
    hseod_bin_edges = np.geomspace(
        1e3,
        50e3,
        num_bins/2.5)

    hseod_gamma = np.histogram(
        gammas.hillas_smallest_ellipse_object_distance,
        bins=hseod_bin_edges)[0]
    hseod_proton = np.histogram(
        protons.hillas_smallest_ellipse_object_distance,
        bins=hseod_bin_edges)[0]

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1,.15,.85,.8])
    add_hist(
        ax=ax,
        bin_edges=hseod_bin_edges,
        bincounts=hseod_gamma,
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=hseod_bin_edges,
        bincounts=hseod_proton,
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(0.05, 0.95, 'gamma', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.9, 'proton', color='red', transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel(r'object-distance of smallest hillas-ellipse in refocus-image-stack/m')
    ax.set_ylabel(r'number events/1')
    plt.savefig('{:s}_{:s}.png'.format(site, 'hillas_smallest_ellipse_object_distance'))
    plt.close('all')

    #-------------------------------------------------
    aip_bin_edges = np.geomspace(
        0.5,
        3,
        num_bins)

    aip_gamma = np.histogram(
        gammas.aperture_intensity_peakness,
        bins=aip_bin_edges)[0]
    aip_proton = np.histogram(
        protons.aperture_intensity_peakness,
        bins=aip_bin_edges)[0]

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1,.15,.85,.8])
    add_hist(
        ax=ax,
        bin_edges=aip_bin_edges,
        bincounts=aip_gamma,
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=aip_bin_edges,
        bincounts=aip_proton,
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(0.05, 0.95, 'gamma', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.9, 'proton', color='red', transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel(r'aperture intensity peakness/1')
    ax.text(0.05, -0.13, 'evenly', color='k', transform=ax.transAxes)
    ax.text(0.95, -0.13, 'peaking', color='k', transform=ax.transAxes)
    ax.set_ylabel(r'number events/1')
    plt.savefig('{:s}_{:s}.png'.format(site, 'aperture_intensity_peakness'))
    plt.close('all')


    #-------------------------------------------------
    sb_bin_edges = np.geomspace(
        1e4,
        1e9,
        num_bins)

    sb_gamma = np.histogram(
        gammas.num_photons/gammas.hillas_smallest_ellipse_solid_angle,
        bins=sb_bin_edges)[0]
    sb_proton = np.histogram(
        protons.num_photons/protons.hillas_smallest_ellipse_solid_angle,
        bins=sb_bin_edges)[0]

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1,.15,.85,.8])
    add_hist(
        ax=ax,
        bin_edges=sb_bin_edges,
        bincounts=sb_gamma,
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=sb_bin_edges,
        bincounts=sb_proton,
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(0.05, 0.95, 'gamma', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.9, 'proton', color='red', transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel(r'solid angle brightness / p.e. sr$^{-1}$')
    ax.set_ylabel(r'number events/1')
    plt.savefig('{:s}_{:s}.png'.format(site, 'solid_angle_brightness'))
    plt.close('all')

    #-------------------------------------------------
    np_bin_edges = np.geomspace(
        10,
        1e4,
        num_bins)

    np_gamma = np.histogram(
        gammas.num_photons,
        bins=np_bin_edges)[0]
    np_proton = np.histogram(
        protons.num_photons,
        bins=np_bin_edges)[0]

    fig = plt.figure(figsize=(8, 4), dpi=100)
    ax = fig.add_axes([.1,.15,.85,.8])
    add_hist(
        ax=ax,
        bin_edges=np_bin_edges,
        bincounts=np_gamma,
        linestyle='k-',
        color='blue',
        alpha=0.5)
    add_hist(
        ax=ax,
        bin_edges=np_bin_edges,
        bincounts=np_proton,
        linestyle='k-',
        color='red',
        alpha=0.5)
    ax.text(0.05, 0.95, 'gamma', color='blue', transform=ax.transAxes)
    ax.text(0.05, 0.9, 'proton', color='red', transform=ax.transAxes)
    ax.loglog()
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_xlabel(r'photo electrons / p.e.')
    ax.set_ylabel(r'number events/1')
    plt.savefig('{:s}_{:s}.png'.format(site, 'num_photons'))
    plt.close('all')
    #-------------------------------------------------

    X_gamma = np.array([
        norm_hillas_smallest_ellipse_object_distance(
            gammas.hillas_smallest_ellipse_object_distance),
        norm_hillas_smallest_ellipse_solid_angle(
            gammas.hillas_smallest_ellipse_solid_angle),
        gammas.aperture_intensity_peakness
    ]).T
    y_gamma = np.ones(gammas.shape[0])

    X_proton = np.array([
        norm_hillas_smallest_ellipse_object_distance(
            protons.hillas_smallest_ellipse_object_distance),
        norm_hillas_smallest_ellipse_solid_angle(
            protons.hillas_smallest_ellipse_solid_angle),
        protons.aperture_intensity_peakness
    ]).T
    y_proton = 0*np.ones(protons.shape[0])

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        np.concatenate([X_gamma, X_proton]),
        np.concatenate([y_gamma, y_proton]),
        test_size=0.25,
        random_state=27)

    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    clf = sklearn.neural_network.MLPRegressor(
        solver='lbfgs',
        alpha=1e-3,
        hidden_layer_sizes=(5, 5, 5),
        random_state=1,
        verbose=True,
        max_iter=1000)

    clf.fit(x_train, y_train)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true=np.logical_not(y_test),
        y_score=clf.predict(x_test))

    auc = sklearn.metrics.roc_auc_score(
        y_true=y_test,
        y_score=clf.predict(x_test))

    fig = plt.figure(figsize=(4, 4), dpi=100)
    ax = fig.add_axes([.2,.2,.72,.72])
    ax.plot(tpr, fpr, 'k')
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    ax.set_title('area under curve {:.2f}'.format(auc))
    ax.set_xlabel('false positive rate / 1\nproton acceptance')
    ax.set_ylabel('true positive rate / 1\ngamma-ray acceptance')
    ax.semilogx()
    plt.savefig('{:s}_{:s}.png'.format(site, 'roc'))
    plt.close('all')
