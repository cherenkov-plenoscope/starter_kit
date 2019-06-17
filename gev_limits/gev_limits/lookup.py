import gzip
import numpy as np
import os
import json
from . import job_structure
from . import thrown_structure
from . import light_field


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
]

SORTED_KEYS = [
    ('num_photons', np.float32),
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