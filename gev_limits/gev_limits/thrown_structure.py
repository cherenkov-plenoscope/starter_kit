import numpy as np

THROWN_KEYS = [
    'run',
    'event',
    'particle_id',
    'particle_energy',
    'particle_cx',
    'particle_cy',
    'particle_x',
    'particle_y',
    'particle_height_first_interaction',
    'num_photons',
    'trigger',
    'max_scatter_radius',
    'trigger_response',
]


def features_to_array_float32(features):
    row = []
    for key in THROWN_KEYS:
        row.append(np.float32(features[key]))
    return np.array(row, dtype=np.float32)


def read_events_thrown(path):
    with open(path, 'rb') as fin:
        raw = np.frombuffer(fin.read(), dtype=np.float32)
    num_keys = len(THROWN_KEYS)
    raw = raw.reshape((len(raw)//num_keys, num_keys))
    events_thrown = {}
    for idx, key in enumerate(THROWN_KEYS):
        events_thrown[key] = raw[:, idx].copy()
    return events_thrown

