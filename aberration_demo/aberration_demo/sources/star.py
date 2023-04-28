import corsika_primary
import numpy as np
import json_numpy
import plenoirf
import plenopy
import os
import tempfile
from .. import utils

EXAMPLE_STAR_CONFIG = {
    "type": "star",
    "cx_deg": 0.0,
    "cy_deg": 1.0,
    "areal_photon_density_per_m2": 20,
    "seed": 122,
}


def make_response_to_star(
    star_config, light_field_geometry_path, merlict_config,
):
    instgeom = utils.get_instrument_geometry_from_light_field_geometry(
        light_field_geometry_path=light_field_geometry_path
    )
    illum_radius = (
        1.5 * instgeom["expected_imaging_system_max_aperture_radius"]
    )
    illum_area = np.pi * illum_radius ** 2
    num_photons = int(
        np.round(star_config["areal_photon_density_per_m2"] * illum_area)
    )

    prng = np.random.Generator(np.random.PCG64(star_config["seed"]))

    with tempfile.TemporaryDirectory(
        prefix="plenoscope-aberration-demo_"
    ) as tmp_dir:
        star_light_path = os.path.join(tmp_dir, "star_light.tar")

        write_photon_bunches(
            cx=np.deg2rad(star_config["cx_deg"]),
            cy=np.deg2rad(star_config["cy_deg"]),
            size=num_photons,
            path=star_light_path,
            prng=prng,
            aperture_radius=illum_radius,
            BUFFER_SIZE=10000,
        )

        run_path = os.path.join(tmp_dir, "run")

        merlict_plenoscope_propagator_config_path = os.path.join(
            tmp_dir, "merlict_propagation_config.json"
        )
        json_numpy.write(
            merlict_plenoscope_propagator_config_path,
            merlict_config["merlict_propagation_config"],
        )

        plenoirf.production.merlict.plenoscope_propagator(
            corsika_run_path=star_light_path,
            output_path=run_path,
            light_field_geometry_path=light_field_geometry_path,
            merlict_plenoscope_propagator_path=merlict_config["executables"][
                "merlict_plenoscope_propagation_path"
            ],
            merlict_plenoscope_propagator_config_path=merlict_plenoscope_propagator_config_path,
            random_seed=star_config["seed"],
            photon_origins=True,
            stdout_path=run_path + ".o",
            stderr_path=run_path + ".e",
        )

        run = plenopy.Run(path=run_path)
        event = run[0]
        return event.raw_sensor_response


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
    I = corsika_primary.I

    assert size >= 0
    tmp_path = path + ".tmp"
    with corsika_primary.event_tape.EventTapeWriter(path=tmp_path) as run:
        runh = np.zeros(273, dtype=np.float32)
        runh[I.RUNH.MARKER] = I.RUNH.MARKER_FLOAT32
        runh[I.RUNH.RUN_NUMBER] = 1
        runh[I.RUNH.NUM_EVENTS] = 1

        evth = np.zeros(273, dtype=np.float32)
        evth[I.EVTH.MARKER] = I.EVTH.MARKER_FLOAT32
        evth[I.EVTH.EVENT_NUMBER] = 1
        evth[I.EVTH.PARTICLE_ID] = 1
        evth[I.EVTH.TOTAL_ENERGY_GEV] = 1.0
        evth[I.EVTH.RUN_NUMBER] = runh[I.RUNH.RUN_NUMBER]
        evth[I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1

        run.write_runh(runh)
        run.write_evth(evth)

        size_written = 0
        while size_written < size:
            block_size = BUFFER_SIZE
            if block_size + size_written > size:
                block_size = size - size_written
            size_written += block_size

            bunches = corsika_primary.calibration_light_source.draw_parallel_and_isochor_bunches(
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


def make_source_config_from_job(job):
    prng = np.random.Generator(np.random.PCG64(job["number"]))

    star_cfg = json_numpy.read(
        os.path.join(job["work_dir"], "config", "observations", "star.json")
    )

    (cx_deg, cy_deg,) = corsika_primary.random.distributions.draw_x_y_in_disc(
        prng=prng, radius=star_cfg["max_angle_off_optical_axis_deg"]
    )

    source_config = {
        "type": "star",
        "cx_deg": cx_deg,
        "cy_deg": cy_deg,
        "areal_photon_density_per_m2": star_cfg["areal_photon_density_per_m2"],
        "seed": job["number"],
    }
    return source_config
