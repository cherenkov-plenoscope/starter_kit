import gzip
import numpy as np
import os
import json
from . import job_structure
from . import thrown_structure
from . import light_field
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift


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

    def _light_field_sequence(self, index):
        raw_lfs = self._raw_light_field_sequence(index)
        return light_field.light_field_sequence_to_photons(
            light_field_sequence=raw_lfs,
            plenoscope=self.plenoscope)

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


def self_similarity(
    lut,
    num_photons_rel_tolerance=0.25,
    cxy_tolerance = 3,
    sim_img_tolerance=0.75,
    sim_ape_tolerance=0.4,
    sim_lf_tolerance=0.45,
    max_source_zenith_deg=2.,
    min_num_similar_events=10,
):
    residuals = []
    for i in range(lut.num_events):
        if (
            np.hypot(lut.particle_cx[i], lut.particle_cy[i]) >
            np.deg2rad(max_source_zenith_deg)
        ):
            continue

        lfs0 = lut._raw_light_field_sequence(i)

        sim_particle_cxs = []
        sim_particle_cys = []
        sim_weights = []
        for j in range(lut.num_events):
            if i == j:
                continue
            if (
                np.abs(lut.num_photons[i] - lut.num_photons[j]) >
                num_photons_rel_tolerance*lut.num_photons[i]
            ):
                continue

            delta_image_cx_median = (
                lut.image_cx_median[i] - lut.image_cx_median[j])
            delta_image_cy_median = (
                lut.image_cy_median[i] - lut.image_cy_median[j])
            delta_image_c_median = np.hypot(
                delta_image_cx_median,
                delta_image_cy_median)
            if delta_image_c_median > cxy_tolerance:
                continue

            rij_deg = np.rad2deg(np.hypot(
                np.abs(lut.particle_cx[i] - lut.particle_cx[j]),
                np.abs(lut.particle_cy[i] - lut.particle_cy[j])))

            lfs1 = lut._raw_light_field_sequence(j)


            sim_ape = light_field.similarity_aperture(lfs0, lfs1, lut.plenoscope)
            if sim_ape < sim_ape_tolerance:
                continue

            sim_img = light_field.similarity_image(lfs0, lfs1, lut.plenoscope)
            if sim_img < sim_img_tolerance:
                continue

            print(
                i,
                j,
                '{:.2f}deg'.format(rij_deg),
                '{:.2f}img'.format(sim_img),
                '{:.2f}ap'.format(sim_ape))


            sim_particle_cxs.append(lut.particle_cx[j])
            sim_particle_cys.append(lut.particle_cy[j])
            sim_weights.append(sim_img)

        sim_weights = np.array(sim_weights)
        num_matches = len(sim_particle_cxs)
        if num_matches > min_num_similar_events:
            cx_reconstructed_mean = np.average(
                sim_particle_cxs, weights=sim_weights)
            cy_reconstructed_mean = np.average(
                sim_particle_cys, weights=sim_weights)
            c_std = np.hypot(
                np.std(sim_particle_cxs), np.std(sim_particle_cys))
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
            # Mean shift
            bandwidth = np.deg2rad(.5)
            clustering = MeanShift(bandwidth=bandwidth, cluster_all=False)
            clustering.fit(np.array([sim_particle_cxs ,sim_particle_cys]).T)
            num_in_cluster = np.sum(clustering.labels_ >= 0)

            cluster_mask = clustering.labels_ > -1
            cluster_keys = np.sort(list(set(clustering.labels_[cluster_mask])))
            cluster_counts = []
            for key in cluster_keys:
                cluster_counts.append(np.sum(clustering.labels_ == key))
            largest_cluster = cluster_keys[np.argmax(cluster_counts)]
            best_pos = clustering.cluster_centers_[largest_cluster]

            print(
                i,
                '{:.2f}+-{:.2f} deg, {:d} ({:d})num,    {:.2f} deg68'.format(
                    r_deg,
                    c_std_deg,
                    num_matches,
                    num_in_cluster,
                    r_sigma))
            plt.figure()
            plt.plot(
                np.rad2deg(sim_particle_cxs),
                np.rad2deg(sim_particle_cys),
                'kx')
            plt.plot(
                np.rad2deg(cx_reconstructed_mean),
                np.rad2deg(cy_reconstructed_mean),
                'ro')
            plt.plot(
                np.rad2deg(lut.particle_cx[i]),
                np.rad2deg(lut.particle_cy[i]),
                'bo')
            plt.plot(
                np.rad2deg(best_pos[0]),
                np.rad2deg(best_pos[1]),
                'gx')
            plt.show()
        else:
            print(i, 'no match')
