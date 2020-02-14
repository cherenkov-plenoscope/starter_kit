import json
import os
import glob
import shutil
import numpy as np
import multiprocessing
import corsika_primary_wrapper as cpw
import subprocess
import tempfile


EXAMPLE_INITIAL_STATE = {
    "input": {
        "site": {
            "earth_magnetic_field_x_muT": 20.815,
            "earth_magnetic_field_z_muT": -11.366,
            "observation_level_asl_m": 5e3,
            "atmosphere_id": 26,
        },
        "primary": {
            "particle_id": 3,
            "energy_GeV": 10.,
            "azimuth_deg": 0.,
            "zenith_deg": 0.,
            "max_scatter_angle_deg": 5.,
        },
        "plenoscope": {
            "azimuth_deg": 0.0,
            "zenith_deg": 0.0,
            "field_of_view_radius_deg": 2.5,
        },
        "energy_iteration_factor": 0.95,
        "energy_thrown_per_iteration_GeV": 4e2,
    },
    "primary_energy_GeV": [],
    "primary_azimuth_deg": [],
    "primary_zenith_deg": [],
    "cherenkov_pool_x_m": [],
    "cherenkov_pool_y_m": [],
    "cherenkov_pool_major_x": [],
    "cherenkov_pool_major_y": [],
    "cherenkov_pool_minor_x": [],
    "cherenkov_pool_minor_y": [],
    "cherenkov_pool_major_std_m": [],
    "cherenkov_pool_minor_std_m": [],
}


def _az_zd_to_cx_cy(azimuth_deg, zenith_deg):
    # Adopted from CORSIKA
    az = np.deg2rad(azimuth_deg)
    zd = np.deg2rad(zenith_deg)
    cx = np.cos(az)*np.sin(zd)
    cy = np.sin(az)*np.sin(zd)
    _cz = np.cos(zd)
    return cx, cy


def _estimate_ellipse(xs, ys):
    x_mean = np.mean(xs)
    y_mean = np.mean(ys)
    cov_matrix = np.cov(np.c_[xs, ys].T)
    eigen_vals, eigen_vecs = np.linalg.eig(cov_matrix)
    major_idx = np.argmax(eigen_vals)
    if major_idx == 0:
        minor_idx = 1
    else:
        minor_idx = 0
    major_axis = eigen_vecs[:, major_idx]
    major_std = np.sqrt(eigen_vals[major_idx])
    minor_axis = eigen_vecs[:, minor_idx]
    minor_std = np.sqrt(eigen_vals[minor_idx])
    return {
        "x_mean": float(x_mean),
        "y_mean": float(y_mean),
        "major_x": float(major_axis[0]),
        "major_y": float(major_axis[1]),
        "minor_x": float(minor_axis[0]),
        "minor_y": float(minor_axis[1]),
        "major_std": float(major_std),
        "minor_std": float(minor_std)}


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


def init(work_dir, initial_state):
    abs_work_dir = os.path.abspath(work_dir)
    os.makedirs(work_dir, exist_ok=True)
    state_path = os.path.join(abs_work_dir, "{:06d}_state.json".format(0))
    with open(state_path, "wt") as f:
        f.write(json.dumps(initial_state, indent=4))


def _latest_state_number(work_dir):
    abs_work_dir = os.path.abspath(work_dir)
    state_paths = glob.glob(os.path.join(abs_work_dir, "*_state.json"))
    state_numbers = [
        int(os.path.basename(sp).split("_")[0]) for sp in state_paths]
    return np.max(state_numbers)


def _read_state(work_dir, state_number):
    abs_work_dir = os.path.abspath(work_dir)
    state_path = os.path.join(
        abs_work_dir, "{:06d}_state.json".format(state_number))
    with open(state_path, "rt") as f:
        s = json.loads(f.read())
    return s


def _write_state(work_dir, state, iteration):
    abs_work_dir = os.path.abspath(work_dir)
    state_path = os.path.join(
        abs_work_dir, "{:06d}_state.json".format(iteration))
    with open(state_path, "wt") as f:
        f.write(json.dumps(state, indent=4))


def _make_jobs(
    particle_id,
    energy,
    site,
    azimuth_deg,
    zenith_deg,
    max_scatter_angle_deg,
    num_events,
    num_jobs,
    corsika_primary_path,
):
    jobs = []
    for i in range(num_jobs):
        run_number = i + 1

        cpw_run_steering = {}
        cpw_run_steering["run"] = {
            "run_id": int(run_number),
            "event_id_of_first_event": 1,
            "observation_level_asl_m": site["observation_level_asl_m"],
            "earth_magnetic_field_x_muT": site["earth_magnetic_field_x_muT"],
            "earth_magnetic_field_z_muT": site["earth_magnetic_field_z_muT"],
            "atmosphere_id": site["atmosphere_id"],
        }
        cpw_run_steering["primaries"] = []
        for event_id in range(num_events):
            az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
                azimuth_rad=np.deg2rad(azimuth_deg),
                zenith_rad=np.deg2rad(zenith_deg),
                min_scatter_opening_angle_rad=0.,
                max_scatter_opening_angle_rad=np.deg2rad(
                    max_scatter_angle_deg))
            prm = {
                "particle_id": particle_id,
                "energy_GeV": energy,
                "zenith_rad": zd,
                "azimuth_rad": az,
                "depth_g_per_cm2": 0.0,
                "random_seed": cpw._simple_seed(event_id),
            }
            cpw_run_steering["primaries"].append(prm)
        job = {}
        job["corsika_primary_steering"] = cpw_run_steering
        job["corsika_primary_path"] = corsika_primary_path
        jobs.append(job)
    return jobs


NUM_FLOATS_IN_EVENTSUMMARY = 25

PARTICLE_ZENITH_RAD = 0
PARTICLE_AZIMUTH_RAD = 1
NUM_PHOTONS = 2
XS_MEDIAN = 3
YS_MEDIAN = 4
CXS_MEDIAN = 5
CYS_MEDIAN = 6


def _run_job(job):
    with tempfile.TemporaryDirectory(prefix="mag_defl_") as tmp:

        corsika_output_path = os.path.join(tmp, "run.tario")
        cpw.corsika_primary(
            corsika_path=job['corsika_primary_path'],
            steering_dict=job['corsika_primary_steering'],
            output_path=corsika_output_path)

        num_events = len(job['corsika_primary_steering']['primaries'])
        event_summaries = np.nan*np.ones(
            shape=(num_events, NUM_FLOATS_IN_EVENTSUMMARY),
            dtype=np.float32)
        es = event_summaries
        run = cpw.Tario(corsika_output_path)
        for idx, airshower in enumerate(run):
            corsika_event_header, photon_bunches = airshower
            ceh = corsika_event_header
            es[idx, PARTICLE_ZENITH_RAD] = cpw._evth_zenith_rad(ceh)
            es[idx, PARTICLE_AZIMUTH_RAD] = cpw._evth_azimuth_rad(ceh)
            es[idx, NUM_PHOTONS] = np.sum(photon_bunches[:, cpw.IBSIZE])
            num_bunches = photon_bunches.shape[0]
            if num_bunches > 0:
                es[idx, XS_MEDIAN] = np.median(photon_bunches[:, cpw.IX])
                es[idx, YS_MEDIAN] = np.median(photon_bunches[:, cpw.IY])
                es[idx, CXS_MEDIAN] = np.median(photon_bunches[:, cpw.ICX])
                es[idx, CYS_MEDIAN] = np.median(photon_bunches[:, cpw.ICY])
        assert idx+1 == num_events
        return es


EXAMPLE_CORSIKA_PRIMARY_MOD_PATH = os.path.abspath(
    os.path.join(
        'build',
        'corsika',
        'modified',
        'corsika-75600',
        'run',
        'corsika75600Linux_QGSII_urqmd'))


def one_more_iteration(
    work_dir,
    corsika_primary_path=EXAMPLE_CORSIKA_PRIMARY_MOD_PATH,
    num_jobs=4,
    pool=multiprocessing.Pool(4),
    max_subiterations=12,
):
    on_target_direction_threshold_deg = 0.5

    s = _read_state(
        work_dir=work_dir,
        state_number=_latest_state_number(work_dir))
    energy_iteration = len(s["primary_energy_GeV"])

    if energy_iteration == 0:
        energy = s["input"]["primary"]["energy_GeV"]
        azimuth_deg = s["input"]["primary"]["azimuth_deg"]
        zenith_deg = s["input"]["primary"]["zenith_deg"]
    else:
        last_energy = s["primary_energy_GeV"][-1]
        energy = last_energy*s["input"]["energy_iteration_factor"]
        azimuth_deg = s["primary_azimuth_deg"][-1]
        zenith_deg = s["primary_zenith_deg"][-1]

    expected_ratio_detected_over_thrown = \
        s["input"]["plenoscope"]["field_of_view_radius_deg"]**2 / \
        s["input"]["primary"]["max_scatter_angle_deg"]**2

    on_target = False
    sub_iteration = 0
    while not on_target:
        direction_converged = False
        position_converged = False
        if sub_iteration > max_subiterations:
            raise RuntimeError("Can not converge. Quit.")

        print("E: {:0.3f}GeV, It: ({:d},{:d})".format(
            energy,
            energy_iteration,
            sub_iteration))

        num_events_per_job = int(
            s["input"]["energy_thrown_per_iteration_GeV"]/energy)

        jobs = _make_jobs(
            particle_id=s["input"]["primary"]["particle_id"],
            energy=energy,
            site=s["input"]["site"],
            azimuth_deg=azimuth_deg,
            zenith_deg=zenith_deg,
            max_scatter_angle_deg=s["input"]["primary"][
                "max_scatter_angle_deg"],
            num_events=num_events_per_job,
            num_jobs=num_jobs,
            corsika_primary_path=corsika_primary_path)

        result_blocks = pool.map(_run_job, jobs)
        events = np.concatenate(result_blocks)

        num_thrown = events.shape[0]
        above100ph = events[:, NUM_PHOTONS] >= 100.
        num_above100ph = int(np.sum(above100ph))

        events_above100 = events[above100ph]

        cxs = events_above100[:, CXS_MEDIAN]
        cys = events_above100[:, CYS_MEDIAN]
        target_cx, target_cy = _az_zd_to_cx_cy(
            azimuth_deg=s["input"]["plenoscope"]["azimuth_deg"],
            zenith_deg=s["input"]["plenoscope"]["zenith_deg"])
        target_offset = np.hypot(cxs-target_cx, cys-target_cy)

        near_target = target_offset < \
            np.deg2rad(s["input"]["plenoscope"]["field_of_view_radius_deg"])
        print("target_offset: {:0.1f}deg".format(
            np.rad2deg(
                np.median(target_offset))))
        events_valid = events_above100[near_target]
        num_above100ph_and_on_target = events_valid.shape[0]

        print("thrown: {:d}, >100ph: {:d}, on target: {:d}.".format(
            num_thrown,
            num_above100ph,
            num_above100ph_and_on_target))

        num_events_expected = expected_ratio_detected_over_thrown*num_thrown
        min_num_events_expected = int(0.25*num_events_expected)

        if num_above100ph_and_on_target < min_num_events_expected:
            raise RuntimeError("Can not converge. Quit.")

        azimuth_deg_valid = float(
            np.rad2deg(
                np.median(
                    events_valid[:, PARTICLE_AZIMUTH_RAD])))
        zenith_deg_valid = float(
            np.rad2deg(
                np.median(
                    events_valid[:, PARTICLE_ZENITH_RAD])))

        cherenkov_core_xs = 1e-2*events_valid[:, XS_MEDIAN]
        cherenkov_core_ys = 1e-2*events_valid[:, YS_MEDIAN]

        x_valid = float(np.median(cherenkov_core_xs))
        y_valid = float(np.median(cherenkov_core_ys))
        x_std_valid = float(np.std(cherenkov_core_xs))
        y_std_valid = float(np.std(cherenkov_core_ys))
        xy_std_valid = np.hypot(x_std_valid, y_std_valid)

        cherenkov_core_ellipse = _estimate_ellipse(
            xs=cherenkov_core_xs,
            ys=cherenkov_core_ys)

        delta_directiong_deg = _great_circle_distance_alt_zd_deg(
            az1_deg=azimuth_deg_valid,
            zd1_deg=zenith_deg_valid,
            az2_deg=azimuth_deg,
            zd2_deg=zenith_deg)
        print("delta_directiong_deg: {:0.1f}".format(delta_directiong_deg))
        print("az: {:0.1f}, zd: {:0.1f}".format(
            azimuth_deg_valid,
            zenith_deg_valid))
        if delta_directiong_deg <= on_target_direction_threshold_deg:
            direction_converged = True

        if direction_converged:
            on_target = True
        else:
            azimuth_deg = (azimuth_deg + azimuth_deg_valid)/2
            zenith_deg = (zenith_deg + zenith_deg_valid)/2

        sub_iteration += 1

    s["primary_energy_GeV"].append(float(energy))
    s["primary_azimuth_deg"].append(float(azimuth_deg_valid))
    s["primary_zenith_deg"].append(float(zenith_deg_valid))

    cce = cherenkov_core_ellipse
    s["cherenkov_pool_x_m"].append(float(x_valid))
    s["cherenkov_pool_y_m"].append(float(y_valid))
    s["cherenkov_pool_major_x"].append(float(cce["major_x"]))
    s["cherenkov_pool_major_y"].append(float(cce["major_y"]))
    s["cherenkov_pool_minor_x"].append(float(cce["minor_x"]))
    s["cherenkov_pool_minor_y"].append(float(cce["minor_y"]))
    s["cherenkov_pool_major_std_m"].append(float(cce["major_std"]))
    s["cherenkov_pool_minor_std_m"].append(float(cce["minor_std"]))

    _write_state(work_dir=work_dir, state=s, iteration=energy_iteration)
