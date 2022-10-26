import corsika_primary as cpw
import numpy as np
import os


def write_photon_bunches(
    cx, cy, size, path, prng, aperture_radius, BUFFER_SIZE=10000
):
    """
    Draw parallel and isochor corsika-bunches and write them into a
    corsika like EventTape.

    Parameters
    ----------
    path : str
        Path to write Event-Tape to.
    size : int
        Number of bunches
    """
    assert size >= 0
    tmp_path = path + ".tmp"
    with cpw.event_tape.EventTapeWriter(path=tmp_path) as run:
        runh = np.zeros(273, dtype=np.float32)
        runh[cpw.I.RUNH.MARKER] = cpw.I.RUNH.MARKER_FLOAT32
        runh[cpw.I.RUNH.RUN_NUMBER] = 1
        runh[cpw.I.RUNH.NUM_EVENTS] = 1

        evth = np.zeros(273, dtype=np.float32)
        evth[cpw.I.EVTH.MARKER] = cpw.I.EVTH.MARKER_FLOAT32
        evth[cpw.I.EVTH.EVENT_NUMBER] = 1
        evth[cpw.I.EVTH.PARTICLE_ID] = 1
        evth[cpw.I.EVTH.TOTAL_ENERGY_GEV] = 1.0
        evth[cpw.I.EVTH.RUN_NUMBER] = runh[cpw.I.RUNH.RUN_NUMBER]
        evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1

        run.write_runh(runh)
        run.write_evth(evth)

        size_written = 0
        while size_written < size:
            block_size = BUFFER_SIZE
            if block_size + size_written > size:
                block_size = size - size_written
            size_written += block_size

            bunches = cpw.calibration_light_source.draw_parallel_and_isochor_bunches(
                cx=-1.0 * cx,
                cy=-1.0 * cy,
                aperture_radius=aperture_radius,
                wavelength=433e-9,
                size=block_size,
                prng=prng,
                speed_of_light=299792458,
            )
            run.write_bunches(bunches)
    os.rename(tmp_path, path)