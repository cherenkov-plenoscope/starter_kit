import numpy as np
from collections import namedtuple


EventSummary = namedtuple(
    'EventSummary',
    [
        "run_num",
        "event_num",
        "particle_id",
        "particle_energy",
        "particle_momentum_px",
        "particle_momentum_py",
        "particle_momentum_pz",
        "particle_cx",
        "particle_cy",
        "particle_zenith_theta",
        "particle_azimuth_phi",
        "particle_core_x",
        "particle_core_y",
        "num_photons",
        "height_observation_level",
        "xs_median",
        "xs_mean",
        "xs_std",
        "ys_median",
        "ys_mean",
        "ys_std",
        "cxs_median",
        "cxs_mean",
        "cxs_std",
        "cys_median",
        "cys_mean",
        "cys_std",
    ])


NUM_FLOATS_IN_EVENTSUMMARY = 25

run_num = 0
event_num = 1
particle_id = 2
particle_energy = 3
particle_momentum_px = 4
particle_momentum_py = 5
particle_momentum_pz = 6
particle_zenith_theta = 7
particle_azimuth_phi = 8
particle_core_x = 9
particle_core_y = 10
num_photons = 11
height_observation_level = 12
xs_median = 13
xs_mean = 14
xs_std = 15
ys_median = 16
ys_mean = 17
ys_std = 18
cxs_median = 19
cxs_mean = 20
cxs_std = 21
cys_median = 22
cys_mean = 23
cys_std = 24


def read_event_summary_block(path):
    raw = np.fromfile(path, dtype=np.float32)
    num_events = raw.shape[0]//NUM_FLOATS_IN_EVENTSUMMARY
    return raw.reshape((num_events, NUM_FLOATS_IN_EVENTSUMMARY))


def read_EventSummary(path, min_num_photons=100):
    block = read_event_summary_block(path)
    valid = block[:, num_photons] >= min_num_photons

    momentum = np.array([
        block[valid, particle_momentum_px],
        block[valid, particle_momentum_py],
        block[valid, particle_momentum_pz],
    ]).T

    momentum_norm = np.sqrt(np.sum(momentum**2, axis=1))

    directions = momentum/momentum_norm[:, np.newaxis]

    s = EventSummary(
        run_num=block[valid, run_num],
        event_num=block[valid, event_num],
        particle_id=block[valid, particle_id],
        particle_energy=block[valid, particle_energy],
        particle_momentum_px=block[valid, particle_momentum_px],
        particle_momentum_py=block[valid, particle_momentum_py],
        particle_momentum_pz=block[valid, particle_momentum_pz],

        particle_cx=directions[:, 0],
        particle_cy=directions[:, 1],

        particle_zenith_theta=block[valid, particle_zenith_theta],
        particle_azimuth_phi=block[valid, particle_azimuth_phi],
        particle_core_x=block[valid, particle_core_x],
        particle_core_y=block[valid, particle_core_y],
        num_photons=block[valid, num_photons],
        height_observation_level=block[valid, height_observation_level],
        xs_median=block[valid, xs_median],
        xs_mean=block[valid, xs_mean],
        xs_std=block[valid, xs_std],
        ys_median=block[valid, ys_median],
        ys_mean=block[valid, ys_mean],
        ys_std=block[valid, ys_std],
        cxs_median=block[valid, cxs_median],
        cxs_mean=block[valid, cxs_mean],
        cxs_std=block[valid, cxs_std],
        cys_median=block[valid, cys_median],
        cys_mean=block[valid, cys_mean],
        cys_std=block[valid, cys_std],
    )
    return s
