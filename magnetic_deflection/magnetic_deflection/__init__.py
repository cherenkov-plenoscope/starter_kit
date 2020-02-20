import json
import os
import glob
import shutil
import numpy as np
import multiprocessing
import corsika_primary_wrapper as cpw
import subprocess
import tempfile
import sys
import time


EXAMPLE_SITE_CHILE = {
    "earth_magnetic_field_x_muT": 20.815,
    "earth_magnetic_field_z_muT": -11.366,
    "observation_level_asl_m": 5e3,
    "atmosphere_id": 26,
}

EXAMPLE_SITE_NAMIBIA = {
    'earth_magnetic_field_x_muT': 12.5,
    'earth_magnetic_field_z_muT': -25.9,
    'observation_level_asl_m': 2300,
    'atmosphere_id': 10
}

NUM_FLOATS_IN_EVENTSUMMARY = 25

PARTICLE_ZENITH_RAD = 0
PARTICLE_AZIMUTH_RAD = 1
NUM_PHOTONS = 2
XS_MEDIAN = 3
YS_MEDIAN = 4
CXS_MEDIAN = 5
CYS_MEDIAN = 6

EXAMPLE_CORSIKA_PRIMARY_MOD_PATH = os.path.abspath(
    os.path.join(
        'build',
        'corsika',
        'modified',
        'corsika-75600',
        'run',
        'corsika75600Linux_QGSII_urqmd'))

CORSIKA_ZENITH_LIMIT_DEG = 70.0


def _azimuth_range(azimuth_deg):
    # Enforce azimuth between -180deg and +180deg
    azimuth_deg = azimuth_deg % 360
    # force it to be the positive remainder, so that 0 <= angle < 360
    azimuth_deg = (azimuth_deg + 360) % 360
    # force into the minimum absolute value residue class,
    # so that -180 < angle <= 180
    if azimuth_deg > 180:
        azimuth_deg -= 360
    return azimuth_deg


def _az_zd_to_cx_cy(azimuth_deg, zenith_deg):
    azimuth_deg = _azimuth_range(azimuth_deg)
    # Adopted from CORSIKA
    az = np.deg2rad(azimuth_deg)
    zd = np.deg2rad(zenith_deg)
    cx = np.cos(az)*np.sin(zd)
    cy = np.sin(az)*np.sin(zd)
    _cz = np.cos(zd)
    return cx, cy


def _cx_cy_to_az_zd_deg(cx, cy):
    cz = np.sqrt(1.0 - cx**2 - cy**2)
    # 1 = sqrt(cx**2 + cy**2 + cz**2)
    az = np.arctan2(cy, cx)
    zd = np.arccos(cz)
    return np.rad2deg(az), np.rad2deg(zd)


def _great_circle_distance_alt_zd_deg(az1_deg, zd1_deg, az2_deg, zd2_deg):
    az1 = np.deg2rad(az1_deg)
    zd1 = np.deg2rad(zd1_deg)
    az2 = np.deg2rad(az2_deg)
    zd2 = np.deg2rad(zd2_deg)
    return np.rad2deg(_great_circle_distance_long_lat(
        lam_long1=az1,
        phi_alt1=np.pi/2 - zd1,
        lam_long2=az2,
        phi_alt2=np.pi/2 - zd2))


def _great_circle_distance_long_lat(lam_long1, phi_alt1, lam_long2, phi_alt2):
    delta_lam = np.abs(lam_long2 - lam_long1)
    delta_sigma = np.arccos(
        np.sin(phi_alt1) * np.sin(phi_alt2) +
        np.cos(phi_alt1) * np.cos(phi_alt2) * np.cos(delta_lam))
    return delta_sigma


def estimate_cherenkov_pool(
    corsika_primary_steering,
    corsika_primary_path,
    min_num_cherenkov_photons,
):
    with tempfile.TemporaryDirectory(prefix="mag_defl_") as tmp:
        corsika_output_path = os.path.join(tmp, "run.tario")
        cpw.corsika_primary(
            corsika_path=corsika_primary_path,
            steering_dict=corsika_primary_steering,
            output_path=corsika_output_path)
        cherenkov_pool_summaries = []
        run = cpw.Tario(corsika_output_path)
        for idx, airshower in enumerate(run):
            corsika_event_header, photon_bunches = airshower
            num_bunches = photon_bunches.shape[0]
            if num_bunches >= min_num_cherenkov_photons:
                cps = np.zeros(NUM_FLOATS_IN_EVENTSUMMARY, dtype=np.float32)
                cps[XS_MEDIAN] = np.median(photon_bunches[:, cpw.IX])
                cps[YS_MEDIAN] = np.median(photon_bunches[:, cpw.IY])
                cps[CXS_MEDIAN] = np.median(photon_bunches[:, cpw.ICX])
                cps[CYS_MEDIAN] = np.median(photon_bunches[:, cpw.ICY])
                ceh = corsika_event_header
                cps[PARTICLE_ZENITH_RAD] = cpw._evth_zenith_rad(ceh)
                cps[PARTICLE_AZIMUTH_RAD] = cpw._evth_azimuth_rad(ceh)
                cps[NUM_PHOTONS] = np.sum(photon_bunches[:, cpw.IBSIZE])
                cherenkov_pool_summaries.append(cps)

        actual_num_valid_pools = len(cherenkov_pool_summaries)
        expected_num_valid_pools = len(corsika_primary_steering['primaries'])
        if actual_num_valid_pools < 0.1*expected_num_valid_pools:
            raise RuntimeError("Too few Cherenkov-pools")

        return np.vstack(cherenkov_pool_summaries)


def _make_steering(
    run_id,
    site,
    num_events,
    primary_particle_id,
    primary_energy,
    primary_cx,
    primary_cy,
):
    steering = {}
    steering["run"] = {
        "run_id": int(run_id),
        "event_id_of_first_event": 1,
        "observation_level_asl_m": site["observation_level_asl_m"],
        "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
        "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
        "atmosphere_id": site["atmosphere_id"],
    }
    steering["primaries"] = []
    for event_id in range(num_events):
        az_deg, zd_deg = _cx_cy_to_az_zd_deg(cx=primary_cx, cy=primary_cy)
        prm = {
            "particle_id": int(primary_particle_id),
            "energy_GeV": float(primary_energy),
            "zenith_rad": np.deg2rad(zd_deg),
            "azimuth_rad": np.deg2rad(az_deg),
            "depth_g_per_cm2": 0.0,
            "random_seed": cpw._simple_seed(event_id + run_id*num_events),
        }
        steering["primaries"].append(prm)
    return steering


def _info_json(
    run_id,
    atmosphere_id,
    particle_id,
    energy,
    off_axis_deg,
    num_events,
    primary_cx,
    primary_cy,
    num_cherenkov_pools,
):
    prm_az_deg, prm_zd_deg = _cx_cy_to_az_zd_deg(cx=primary_cx, cy=primary_cy)
    s = '"time": {:d}, '.format(int(time.time()))
    s += '"atmosphere_id": {:d}, '.format(atmosphere_id)
    s += '"particle_id": {:d}, '.format(particle_id)
    s += '"energy_GeV": {:.2f}, '.format(energy)
    s += '"it": {:d}, '.format(run_id)
    s += '"airshower": {:d}, '.format(num_events)
    s += '"azimuth_deg": {:.2f}, '.format(prm_az_deg)
    s += '"zenith_deg": {:.2f}, '.format(prm_zd_deg)
    s += '"off_deg": {:.2f}, '.format(off_axis_deg)
    s += '"pools": {:d}'.format(num_cherenkov_pools)
    return '{' + s + '}'


def estimate_deflection(
    site,
    primary_energy,
    primary_particle_id,
    instrument_azimuth_deg,
    instrument_zenith_deg,
    max_off_axis_deg,
    initial_num_events_per_iteration=2**5,
    max_num_events_per_iteration=2**11,
    max_rel_uncertainty=0.1,
    max_iterations=100,
    corsika_primary_path=EXAMPLE_CORSIKA_PRIMARY_MOD_PATH,
    iteration_speed=0.5,
    min_num_cherenkov_photons_in_airshower=100,
    verbose=True,
):
    assert iteration_speed > 0
    assert iteration_speed <= 1.0
    instrument_cx, instrument_cy = _az_zd_to_cx_cy(
        azimuth_deg=instrument_azimuth_deg,
        zenith_deg=instrument_zenith_deg)
    primary_cx = float(instrument_cx)
    primary_cy = float(instrument_cy)
    num_events = int(initial_num_events_per_iteration)
    previous_off_axis_deg = 180.0

    run_id = 0
    while True:
        run_id += 1
        steering = _make_steering(
            run_id=run_id,
            site=site,
            num_events=num_events,
            primary_particle_id=primary_particle_id,
            primary_energy=primary_energy,
            primary_cx=primary_cx,
            primary_cy=primary_cy)
        cherenkov_pools = estimate_cherenkov_pool(
            corsika_primary_steering=steering,
            corsika_primary_path=corsika_primary_path,
            min_num_cherenkov_photons=min_num_cherenkov_photons_in_airshower)

        cer_cx = np.median(cherenkov_pools[:, CXS_MEDIAN])
        cer_cy = np.median(cherenkov_pools[:, CYS_MEDIAN])

        cer_cx_off = cer_cx - instrument_cx
        cer_cy_off = cer_cy - instrument_cy

        off_axis_deg = np.rad2deg(np.hypot(cer_cx_off, cer_cy_off))

        num_pools = len(cherenkov_pools)
        rel_uncertainty = np.sqrt(num_pools)/num_pools

        if verbose:
            print(_info_json(
                run_id,
                site['atmosphere_id'],
                primary_particle_id,
                primary_energy,
                off_axis_deg,
                num_events,
                primary_cx,
                primary_cy,
                num_pools))

        if previous_off_axis_deg < off_axis_deg:
            num_events *= 2

        if off_axis_deg >= max_off_axis_deg:
            primary_cx = primary_cx - iteration_speed*cer_cx_off
            primary_cy = primary_cy - iteration_speed*cer_cy_off
            previous_off_axis_deg = off_axis_deg
        else:
            if rel_uncertainty > max_rel_uncertainty:
                num_events *= 2
            else:
                break

        if run_id > max_iterations:
            raise RuntimeError(
                "Can not converge. "
                "Reached limit of {:d} iterations".format(max_iterations))
        if num_events > max_num_events_per_iteration:
            raise RuntimeError(
                "Can not converge. "
                "Reached limit of {:d} num. events per iteration".format(
                    max_num_events_per_iteration))

    primary_azimuth_deg, primary_zenith_deg = _cx_cy_to_az_zd_deg(
        cx=primary_cx,
        cy=primary_cy)

    cer_x = np.median(cherenkov_pools[:, XS_MEDIAN])*cpw.CM2M
    cer_y = np.median(cherenkov_pools[:, YS_MEDIAN])*cpw.CM2M

    return {
        "primary_azimuth_deg": float(primary_azimuth_deg),
        "primary_zenith_deg": float(primary_zenith_deg),
        "primary_cx": float(primary_cx),
        "primary_cy": float(primary_cy),
        "cherenkov_pool_x_m": float(cer_x),
        "cherenkov_pool_y_m": float(cer_y),
        "rel_uncertainty": float(rel_uncertainty)
    }


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def run_job(job):
    try:
        deflection = estimate_deflection(
            site=job['site'],
            primary_energy=job['primary_energy'],
            primary_particle_id=job['primary_particle_id'],
            instrument_azimuth_deg=job['instrument_azimuth_deg'],
            instrument_zenith_deg=job['instrument_zenith_deg'],
            max_off_axis_deg=job['max_off_axis_deg'],
            corsika_primary_path=job['corsika_primary_path'],
            iteration_speed=job['iteration_speed'],
            max_iterations=job['max_iterations'])
        deflection["valid"] = True
    except RuntimeError as whatever:
        eprint("RuntimeError:")
        eprint(job['site_key'], job['particle_key'], job['primary_energy'])
        eprint(whatever)
        deflection = {}
        deflection["valid"] = False
    deflection['site_key'] = job['site_key']
    deflection['particle_key'] = job['particle_key']
    deflection['particle_id'] = job['primary_particle_id']
    deflection['energy_GeV'] = job['primary_energy']
    return deflection
