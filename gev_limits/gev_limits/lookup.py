import gzip
import numpy as np
import os
import json
from . import job_structure
from . import thrown_structure
from . import light_field
from . import features
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from scipy import spatial


KEYS = [
    # index
    ('run', np.uint32),
    ('event', np.uint32),

    # simulation-truth
    ('particle_id', np.uint32),
    ('particle_energy', np.float32),
    ('particle_cx', np.float32),
    ('particle_cy', np.float32),
    ('particle_y', np.float32),
    ('particle_x', np.float32),
    ('particle_height_first_interaction', np.float32),

    # air-shower-features
    ('num_photons', np.float32),
    ('image_cx_median', np.float32),
    ('image_cy_median', np.float32),
]

SORTED_KEYS = [
    ('num_photons', np.float32),
    ('image_cx_median', np.float32),
    ('image_cy_median', np.float32),
]

LFS_PATH = 'light_field_sequences'
LFS_LENGTH_PATH = 'light_field_sequences_lengths'
JOB_PATH = 'job.jsonl'
THROWN_PATH = 'thrown.float32'


class LookUpAppender():
    def __init__ (
        self,
        path,
        random_seed,
        num_events,
        eventio_converter_path,
        instrument,
        particle,
        site,
        trigger_threshold
    ):
        self.path = path
        os.makedirs(path, exist_ok=True)

        job = {
            'random_seed': random_seed,
            'num_events': num_events,
            'eventio_converter_path': eventio_converter_path,
            'instrument': instrument,
            'particle': particle,
            'site': site,
            'trigger_threshold': trigger_threshold}
        job_path = os.path.join(path, JOB_PATH)
        with open(job_path, 'wt') as fout:
            fout.write(json.dumps(job))

        self.lfs_path = os.path.join(path, LFS_PATH)
        self.lfs_length_path = os.path.join(path, LFS_LENGTH_PATH)
        self.f_lfs = open(self.lfs_path, 'ba')
        self.f_lfs_length = open(self.lfs_length_path, 'ba')
        for i in range(len(KEYS)):
            name = KEYS[i][0]
            setattr(self, name, open(os.path.join(path, name), 'ba'))

    def append_event(
        self,
        light_field_sequence_uint8,
        features,
    ):
        lfs_gzip_bytes = gzip.compress(light_field_sequence_uint8.tobytes())
        length_lfs_gzip = np.uint32(len(lfs_gzip_bytes))
        self.f_lfs_length.write(length_lfs_gzip.tobytes())
        self.f_lfs.write(lfs_gzip_bytes)
        for i in range(len(KEYS)):
            name = KEYS[i][0]
            f = getattr(self, name)
            value_byte = np.array([features[name]], dtype=KEYS[i][1])[0]
            f.write(value_byte.tobytes())

    def __del__(self):
        self.f_lfs.close()
        self.f_lfs_length.close()
        for i in range(len(KEYS)):
            name = KEYS[i][0]
            getattr(self, name).close()


class LookUpTable():
    def __init__(self, path):
        self._job = read_job(os.path.join(path, JOB_PATH))
        instrument = self._job['instrument']
        self.plenoscope = light_field.init_Plenoscope(
            aperture_radius=instrument['aperture_radius'],
            num_paxel_on_diagonal=instrument['num_paxel_on_diagonal'],
            field_of_view_radius_deg=instrument['field_of_view_radius_deg'],
            num_pixel_on_diagonal=instrument['num_pixel_on_diagonal'],
            time_radius=instrument['time_radius'],
            num_time_slices=instrument['num_time_slices'])

        for i in range(len(KEYS)):
            name = KEYS[i][0]
            with open(os.path.join(path, name), 'rb') as fi:
                setattr(
                    self,
                    name,
                    np.frombuffer(fi.read(), dtype=KEYS[i][1]))
        self.num_events = len(self.particle_energy)
        for i in range(len(KEYS)):
            name = KEYS[i][0]
            assert len(getattr(self, name)) == self.num_events
        with open(os.path.join(path, LFS_LENGTH_PATH), 'rb') as fi:
            self.light_field_sequences_lengths = np.frombuffer(
                fi.read(),
                dtype=np.uint32)
        with open(os.path.join(path, LFS_PATH), 'rb') as fi:
            self._raw_light_field_sequences = []
            for length in self.light_field_sequences_lengths:
                self._raw_light_field_sequences.append(fi.read(length))
        self.init_sorted_keys()

    def init_sorted_keys(self):
        for i in range(len(SORTED_KEYS)):
            name = SORTED_KEYS[i][0]
            values = getattr(self, name)
            aso = np.argsort(values).astype(np.uint32)
            setattr(self, 'argsort_'+name, aso)
            setattr(self, 'sorted_'+name, values[aso].astype(SORTED_KEYS[i][1]))

    def idx_for_number_photons_within(self, min_val, max_val):
        return self._idx_for_attribute_within('num_photons', min_val, max_val)

    def _idx_for_attribute_within(self, key, min_val, max_val):
        sorted_values = getattr(self, 'sorted_'+key)
        argsort_values = getattr(self, 'argsort_'+key)
        ll = np.searchsorted(sorted_values, min_val)
        ul = np.searchsorted(sorted_values, max_val)
        return argsort_values[np.arange(ll, ul)]

    def _raw_light_field_sequence(self, index):
        flat = np.frombuffer(
            gzip.decompress(self._raw_light_field_sequences[index]),
            dtype=np.uint8)
        num_photons = flat.shape[0]//5
        return flat.reshape((num_photons, 5))

    def light_field_sequence(self, index):
        raw_indexed_lfs = self._raw_light_field_sequence(index)
        lfg = self.plenoscope
        lfs = raw_indexed_lfs
        return np.array([
            lfg.cx[lfs[:, 0]],
            lfg.cy[lfs[:, 1]],
            lfg.x[lfs[:, 2]],
            lfg.y[lfs[:, 3]],
            lfg.t[lfs[:, 4]]
        ]).T

def concatenate(lut_paths, out_path):
    os.makedirs(out_path, exist_ok=True)
    for lut_path in lut_paths:

        with open(os.path.join(lut_path, JOB_PATH), "rt") as fi:
            with open(os.path.join(out_path, JOB_PATH), "at") as fo:
                for line in fi:
                    fo.write(line + '\n')

        for i in range(len(KEYS)):
            name = KEYS[i][0]
            with open(os.path.join(lut_path, name), "rb") as fi:
                with open(os.path.join(out_path, name), "ab") as fo:
                    fo.write(fi.read())

        with open(os.path.join(lut_path, LFS_LENGTH_PATH), "rb") as fi:
            with open(os.path.join(out_path, LFS_LENGTH_PATH), "ab") as fo:
                fo.write(fi.read())

        with open(os.path.join(lut_path, LFS_PATH), "rb") as fi:
            with open(os.path.join(out_path, LFS_PATH), "ab") as fo:
                fo.write(fi.read())

        with open(os.path.join(lut_path, THROWN_PATH), "rb") as fi:
            with open(os.path.join(out_path, THROWN_PATH), "ab") as fo:
                fo.write(fi.read())

    assert_valid_jobs(
        job_path=os.path.join(out_path, JOB_PATH))



def assert_valid_jobs(job_path):
    jobs = []
    with open(job_path, "rt") as fi:
        for line in fi:
            jobs.append(json.loads(line))

    for job in jobs:
        for key in job['instrument']:
            assert key in job_structure.INSTRUMENT_KEYS
        for key in job['particle']:
            assert key in job_structure.PARTICLE_KEYS
        for key in job['site']:
            assert key in job_structure.SITE_KEYS

    j0 = jobs[0]
    for jN in jobs:
        assert j0['trigger_threshold'] == jN['trigger_threshold']
        i0 = j0['instrument']
        iN = jN['instrument']
        for key in job_structure.INSTRUMENT_KEYS:
            assert i0[key] == iN[key]

        p0 = j0['particle']
        pN = jN['particle']
        for key in job_structure.PARTICLE_KEYS:
            assert p0[key] == pN[key]

        s0 = j0['site']
        sN = jN['site']
        for key in job_structure.SITE_KEYS:
            assert s0[key] == sN[key]

def read_job(path):
    with open(path, 'rt') as fin:
        job = json.loads(fin.readline())
    return {
        'instrument': job['instrument'],
        'particle': job['particle'],
        'site': job['site']}


def residual_particle_direction(lut, i0, i1):
    return np.hypot(
        np.abs(lut.particle_cx[i0] - lut.particle_cx[i1]),
        np.abs(lut.particle_cy[i0] - lut.particle_cy[i1]))


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


def self_similarity_simple(
    lut,
    max_source_zenith_deg=2.,
    i_start=1,
    i_end=10000,
    num_photons_rel_tolerance=0.25,
    cxy_tolerance=3,
    refocus_stack_tolerance=0.5,
    min_num_similar_events=10,
):
    NUM_REFS = 12
    NUM_PIXEL_ROI = 12
    OBJECT_DISTANCES = np.geomspace(5e3, 100e3, NUM_REFS)

    residuals = []
    for i in np.arange(i_start ,i_end):

        if (
            np.hypot(lut.particle_cx[i], lut.particle_cy[i]) >
            np.deg2rad(max_source_zenith_deg)
        ):
            continue

        lfs_i = lut._raw_light_field_sequence(i)

        ROI_CX = int(np.round(np.mean(lfs_i[:, 0])))
        ROI_CY = int(np.round(np.mean(lfs_i[:, 1])))

        cx_bin_edges_idx = np.arange(
            ROI_CX - NUM_PIXEL_ROI,
            ROI_CX + NUM_PIXEL_ROI + 1)
        cx_bin_edges = lut.plenoscope.cx_bin_edges[cx_bin_edges_idx]
        cy_bin_edges_idx = np.arange(
            ROI_CY - NUM_PIXEL_ROI,
            ROI_CY + NUM_PIXEL_ROI + 1)
        cy_bin_edges = lut.plenoscope.cy_bin_edges[cy_bin_edges_idx]

        image_photons_i = light_field.get_image_photons(
            lfs=lfs_i,
            plenoscope=lut.plenoscope)

        refocus_stack_i = light_field.get_refocus_stack(
            image_photons=image_photons_i,
            object_distances=OBJECT_DISTANCES,
            cx_bin_edges=cx_bin_edges,
            cy_bin_edges=cy_bin_edges)

        max_dense_obj_idx_i = np.argmax(
            np.max(
                np.max(
                    refocus_stack_i,
                    axis=1),
                axis=1))

        hillas_i = []
        for obj in OBJECT_DISTANCES:
            cxs, cys = light_field.get_refocused_cx_cy(
                image_photons=image_photons_i,
                object_distance=obj)
            eli_i = features.hillas_ellipse(cxs=cxs, cys=cys)
            hillas_i.append(eli_i)

        # light-front
        # -----------
        lf_xs, lf_ys, lf_zs = light_field.get_light_front(
            lfs=lfs_i,
            plenoscope=lut.plenoscope)
        light_front_normal_i = features.light_front_surface_normal(
            xs=lf_xs,
            ys=lf_ys,
            zs=lf_zs)

        refocus_stack_i = light_field.smear_out_refocus_stack(refocus_stack_i)

        refocus_stack_i_norm = np.linalg.norm(refocus_stack_i)

        # mask matches
        #--------------
        mask = np.ones(lut.num_events, dtype=np.bool)
        mask[i] = False

        # mask direction
        #----------------
        delta_image_cx_median = (
            lut.image_cx_median[i] - lut.image_cx_median)
        delta_image_cy_median = (
            lut.image_cy_median[i] - lut.image_cy_median)
        delta_image_c_median = np.hypot(
            delta_image_cx_median,
            delta_image_cy_median)
        image_mask = delta_image_c_median < cxy_tolerance

        # mask num photons
        #------------------
        num_photons_i = lut.num_photons[i]
        size_mask = (np.abs(num_photons_i - lut.num_photons) >
                num_photons_rel_tolerance*num_photons_i)

        mask = mask*image_mask*size_mask

        truths = []
        for j in np.arange(lut.num_events)[mask]:
            lfs_j = lut._raw_light_field_sequence(j)

            image_photons_j = light_field.get_image_photons(
                lfs=lfs_j,
                plenoscope=lut.plenoscope)

            refocus_stack_j = light_field.get_refocus_stack(
                image_photons=image_photons_j,
                object_distances=OBJECT_DISTANCES,
                cx_bin_edges=cx_bin_edges,
                cy_bin_edges=cy_bin_edges)

            refocus_stack_j = light_field.smear_out_refocus_stack(
                refocus_stack_j)

            hillas_j = []
            for obj in OBJECT_DISTANCES:
                cxs, cys = light_field.get_refocused_cx_cy(
                    image_photons=image_photons_j,
                    object_distance=obj)
                eli_j = features.hillas_ellipse(cxs=cxs, cys=cys)
                hillas_j.append(eli_j)

            diff = np.linalg.norm(
                refocus_stack_i.flatten() -
                refocus_stack_j.flatten())

            diff = diff/refocus_stack_i_norm

            if diff > refocus_stack_tolerance:
                continue

            r = np.hypot(
                np.abs(lut.particle_cx[i] - lut.particle_cx[j]),
                np.abs(lut.particle_cy[i] - lut.particle_cy[j]))
            r_deg = np.rad2deg(r)

            obj_i = max_dense_obj_idx_i

            ellipse_ratio_i = hillas_i[obj_i].std_major/hillas_i[obj_i].std_minor
            ellipse_ratio_j = hillas_j[obj_i].std_major/hillas_j[obj_i].std_minor

            if np.abs(ellipse_ratio_i - ellipse_ratio_j) > .5:
                continue

            theta_i = np.arctan2(
                    hillas_i[obj_i].cy_major,
                    hillas_i[obj_i].cx_major)
            theta_j = np.arctan2(
                    hillas_j[obj_i].cy_major,
                    hillas_j[obj_i].cx_major)

            theta_mod_i = np.mod(theta_i, np.pi)
            theta_mod_j = np.mod(theta_j, np.pi)
            delta_theta_ij = np.abs(theta_mod_i - theta_mod_j)
            if delta_theta_ij > np.deg2rad(30):
                continue

            # light-front
            # -----------
            lf_xs, lf_ys, lf_zs = light_field.get_light_front(
                lfs=lfs_j,
                plenoscope=lut.plenoscope)
            light_front_normal_j = features.light_front_surface_normal(
                xs=lf_xs,
                ys=lf_ys,
                zs=lf_zs)

            delta_light_front_ij = features.angle_between(
                light_front_normal_i,
                light_front_normal_j)

            if delta_light_front_ij > np.deg2rad(0.1):
                continue

            print(
                i,
                j,
                '{:.2f}diff'.format(diff),
                '{:.2f}deg'.format(r_deg),
                '{:.2f}eli'.format(ellipse_ratio_i),
                'eli {:.2f}deg'.format(np.rad2deg(delta_theta_ij)),
                'lf {:.2f}deg'.format(np.rad2deg(delta_light_front_ij))
                )

            # match
            truths.append([
                lut.particle_cx[j],
                lut.particle_cy[j],
                lut.particle_x[j],
                lut.particle_y[j],
                lut.particle_height_first_interaction[j],
                lut.particle_energy[j],
                diff
            ])
            """
            vmax_i = np.max(refocus_stack_i)
            vmax_j = np.max(refocus_stack_j)
            fig = plt.figure(
            figsize=(8*2, 8*NUM_REFS), dpi=50)
            SUB_FIG_HEIGHT = 1/NUM_REFS
            for obj, object_distance in enumerate(OBJECT_DISTANCES):
                axi = fig.add_axes([
                    0,
                    SUB_FIG_HEIGHT*obj,
                    .5*.98,
                    SUB_FIG_HEIGHT*.98])
                axi.set_axis_off()
                axi.pcolor(
                    refocus_stack_i[obj, :, :],
                    vmax=vmax_i,
                    cmap='inferno')
                if obj == 0:
                    axi.text(NUM_PIXEL_ROI, NUM_PIXEL_ROI,"{:.1f}".format(diff))

                axj = fig.add_axes([
                    .5,
                    SUB_FIG_HEIGHT*obj,
                    .5*.98,
                    SUB_FIG_HEIGHT*.98])
                axj.set_axis_off()
                axj.pcolor(
                    refocus_stack_j[obj, :, :],
                    vmax=vmax_j,
                    cmap='inferno')
            plt.savefig('refocus_stack_{:d}_{:d}.png'.format(i, j))
            plt.close('all')
            """

        truths = np.array(truths)

        if truths.shape[0] >= min_num_similar_events:
            fig = plt.figure(
            figsize=(8, 8), dpi=50)
            ax = fig.add_axes([.1,.1,.9,.9])
            ax.plot(
                np.rad2deg(lut.particle_cx[i]),
                np.rad2deg(lut.particle_cy[i]),
                'xr')
            ax.plot(
                np.rad2deg(truths[:, 0]),
                np.rad2deg(truths[:, 1]),
                'xk')
            plt.savefig('residuals_{:d}.png'.format(i))
            plt.close('all')

            fig = plt.figure(
            figsize=(8, 8), dpi=50)
            ax = fig.add_axes([.1,.1,.9,.9])
            ax.plot(
                lut.particle_x[i],
                lut.particle_y[i],
                'xr')
            ax.plot(
                truths[:, 2],
                truths[:, 3],
                'xk')
            plt.savefig('residuals_core_{:d}.png'.format(i))
            plt.close('all')

            fig = plt.figure(
            figsize=(8, 8), dpi=50)
            ax = fig.add_axes([.1,.1,.9,.9])
            ax.hist(
                truths[:, 4],
                bins=np.linspace(5e3, 40e3, 10))
            ax.axvline(lut.particle_height_first_interaction[i], color='r')
            ax.axvline(np.median(truths[:, 4]),  color='k')
            plt.savefig('residuals_first_interaction_{:d}.png'.format(i))
            plt.close('all')

            fig = plt.figure(
            figsize=(8, 8), dpi=50)
            ax = fig.add_axes([.1,.1,.9,.9])
            ax.hist(
                truths[:, 5],
                bins=np.linspace(0.8, 1.6, 10))
            ax.axvline(lut.particle_energy[i],  color='r')
            ax.axvline(np.median(truths[:, 5]),  color='k')
            plt.savefig('residuals_energy_{:d}.png'.format(i))
            plt.close('all')

            cx_reconstructed_mean = np.mean(truths[:, 0])
            cy_reconstructed_mean = np.mean(truths[:, 1])
            c_std = np.hypot(
                np.std(truths[:, 0]), np.std(truths[:, 1]))
            c_std_deg = np.rad2deg(c_std)

            r = np.hypot(
                np.abs(lut.particle_cx[i] - cx_reconstructed_mean),
                np.abs(lut.particle_cy[i] - cy_reconstructed_mean))
            r_deg = np.rad2deg(r)
            residuals.append(r_deg)

            rhist, rbinedges = np.histogram(
                residuals,
                bins=np.linspace(0, 2, 20))

            r_sigma = integration_width_for_one_sigma(rhist, rbinedges)

            print(
                i,
                '{:.2f}+-{:.2f} deg, {:d}    {:.2f} deg68'.format(
                    r_deg,
                    c_std_deg,
                    np.sum(mask),
                    r_sigma))
            print(rhist)
        else:
            print(i, 'no match')



def plot_refocus_stack(
    lut,
    max_source_zenith_deg=2.,
):
    NUM_REFS = 12
    NUM_PIXEL_ROI = 12
    OBJECT_DISTANCES = np.geomspace(2.5e3, 55e3, NUM_REFS)

    residuals = []
    for i in np.arange(lut.num_events):

        if (
            np.hypot(lut.particle_cx[i], lut.particle_cy[i]) >
            np.deg2rad(max_source_zenith_deg)
        ):
            continue

        lfs_i = lut._raw_light_field_sequence(i)

        ROI_CX = int(np.round(np.mean(lfs_i[:, 0])))
        ROI_CY = int(np.round(np.mean(lfs_i[:, 1])))

        cx_bin_edges_idx = np.arange(
            ROI_CX - NUM_PIXEL_ROI,
            ROI_CX + NUM_PIXEL_ROI + 1)
        if (
            np.min(cx_bin_edges_idx) < 0 or
            np.max(cx_bin_edges_idx) > lut.plenoscope.num_pixel_on_diagonal):
            continue
        cx_bin_edges = lut.plenoscope.cx_bin_edges[cx_bin_edges_idx]
        cy_bin_edges_idx = np.arange(
            ROI_CY - NUM_PIXEL_ROI,
            ROI_CY + NUM_PIXEL_ROI + 1)
        if (
            np.min(cy_bin_edges_idx) < 0 or
            np.max(cy_bin_edges_idx) > lut.plenoscope.num_pixel_on_diagonal):
            continue
        cy_bin_edges = lut.plenoscope.cy_bin_edges[cy_bin_edges_idx]

        image_photons_i = light_field.get_image_photons(
            lfs=lfs_i,
            plenoscope=lut.plenoscope)

        refocus_stack_i = light_field.get_refocus_stack(
            image_photons=image_photons_i,
            object_distances=OBJECT_DISTANCES,
            cx_bin_edges=cx_bin_edges,
            cy_bin_edges=cy_bin_edges)

        max_dense_obj_idx_i = np.argmax(
            np.max(
                np.max(
                    refocus_stack_i,
                    axis=1),
                axis=1))

        hillas_i = []
        for obj in OBJECT_DISTANCES:
            cxs, cys = light_field.get_refocused_cx_cy(
                image_photons=image_photons_i,
                object_distance=obj)
            eli_i = features.hillas_ellipse(cxs=cxs, cys=cys)
            hillas_i.append(eli_i)

        # light-front
        # -----------
        lf_xs, lf_ys, lf_zs = light_field.get_light_front(
            lfs=lfs_i,
            plenoscope=lut.plenoscope)
        light_front_normal_i = features.light_front_surface_normal(
            xs=lf_xs,
            ys=lf_ys,
            zs=lf_zs)

        refocus_stack_i = light_field.smear_out_refocus_stack(refocus_stack_i)

        print(OBJECT_DISTANCES)
        vmax_i = np.max(refocus_stack_i)
        fig = plt.figure(
        figsize=(8, 8*NUM_REFS), dpi=50)
        SUB_FIG_HEIGHT = 1/NUM_REFS
        for obj, object_distance in enumerate(OBJECT_DISTANCES):
            axi = fig.add_axes([
                0,
                SUB_FIG_HEIGHT*obj,
                1*.98,
                SUB_FIG_HEIGHT*.98])
            axi.set_axis_off()
            axi.pcolor(
                refocus_stack_i[obj, :, :],
                vmax=vmax_i,
                cmap='inferno')
            #axi.plot(0.5, 0.5, "xw")
            axi.text(0.1, 0.1, "{:.1f}km".format(1e-3*object_distance), color="white")
        plt.savefig('refocus_stack_{:d}.png'.format(i))
        plt.close('all')

