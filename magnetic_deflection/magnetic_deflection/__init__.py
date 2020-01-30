from . import EventSummary
import json
import os
import glob
import shutil
import numpy as np
import multiprocessing
import corsika_primary_wrapper
import subprocess
import tempfile


example_state = {
    "input": {
        "corsika_particle_id": 3,
        "site": {
            "earth_magnetic_field_x_muT": 20.815,
            "earth_magnetic_field_z_muT": -11.366,
            "observation_level_asl_m": 5e3,
            "corsika_atmosphere_id": 26,
        },
        "initial": {
            "energy": 14.,
            "energy_iteration_factor": 0.95,
            "instrument_radius": 1e3,
            "instrument_x": 0.,
            "instrument_y": 0.,
            "azimuth_phi_deg": 0.,
            "zenith_theta_deg": 0.,
            "scatter_angle_deg": 5.,
        },
        "energy_thrown_per_iteration": 4e2,
        "target_cx": 0.,
        "target_cy": 0.,
        "target_containment_angle_deg": 2.5,
    },
    "energy": [],
    "energy_iteration_factor": [],
    "instrument_radius": [],
    "instrument_x": [],
    "instrument_y": [],
    "azimuth_phi_deg": [],
    "zenith_theta_deg": [],
    "scatter_angle_deg": [],
    "cherenkov_core_spread_xy": [],
}


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
        "major_axis_x": float(major_axis[0]),
        "major_axis_y": float(major_axis[1]),
        "minor_axis_x": float(minor_axis[0]),
        "minor_axis_y": float(minor_axis[1]),
        "major_std": float(major_std),
        "minor_std": float(minor_std)}


def great_circle_distance_alt_zd_deg(az1_deg, zd1_deg, az2_deg, zd2_deg):
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


def _make_corsika_steering(
    random_seed,
    run_number,
    num_events,
    particle_id,
    energy,
    cone_azimuth_deg,
    cone_zenith_distance_deg,
    cone_max_scatter_angle_deg,
    earth_magnetic_field_x_muT,
    earth_magnetic_field_z_muT,
    atmosphere_id,
    observation_level_asl_m,
):
    steering = {}
    steering['run'] = {
        'run_id': run_number,
        'event_id_of_first_event': 1,
        "observation_level_asl_m": observation_level_asl_m,
        "earth_magnetic_field_x_muT": earth_magnetic_field_x_muT,
        "earth_magnetic_field_z_muT": earth_magnetic_field_z_muT,
        "atmosphere_id": atmosphere_id,
    }
    steering['primaries'] = []
    for idx in range(num_events):
        prm = {}
        np.random.seed(run_number)
        az, zd = cpw.random.draw_azimuth_zenith_in_viewcone(
            azimuth_rad=np.deg2rad(cone_azimuth_deg),
            zenith_rad=np.deg2rad(cone_zenith_distance_deg),
            min_scatter_opening_angle_rad=0.,
            max_scatter_opening_angle_rad=np.deg2rad(
                cone_max_scatter_angle_deg))
        prm["particle_id"] = int(particle_id)
        prm["energy_GeV"] = float(energy)
        prm["zenith_rad"] = zd
        prm["azimuth_rad"] = az
        prm["depth_g_per_cm2"] = 0.0
        prm["random_seed"] = cpw._simple_seed(run_number*10000 + event_id)
    steering['primaries'].append(prm)
    return steering


def _init_work_dir(work_dir, initial_state):
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


def _run_job(job):
    with tempfile.TemporaryDirectory(prefix="mag_defl_job_") as tmp:
        corsika_out_path = os.path.join(tmp, "run.tar")
        cor_rc = corsika_primary_wrapper.corsika(
            steering_dict=job['steering_dict'],
            output_path=corsika_out_path,
            corsika_path=job["corsika_primary_path"])

        run = corsika_primary_wrapper.Tario(corsika_out_path)
        for event in run:


        summary_block = EventSummary.read_event_summary_block(summary_out_path)
        return summary_block


def _one_iteration(
    work_dir,
    corsika_primary_path=os.path.abspath(
        './build/corsika/modified/corsika-75600/run/corsika75600Linux_QGSII_urqmd'),
    num_jobs=4,
    pool=multiprocessing.Pool(4),
    max_subiterations=12,
):
    on_target_direction_threshold_deg = 0.5

    s = _read_state(
        work_dir=work_dir,
        state_number=_latest_state_number(work_dir))
    energy_iteration = len(s["energy"])

    if energy_iteration == 0:
        energy_iteration_factor = s["input"]["initial"][
            "energy_iteration_factor"]
        energy = s["input"]["initial"]["energy"]
        azimuth_phi_deg = s["input"]["initial"]["azimuth_phi_deg"]
        zenith_theta_deg = s["input"]["initial"]["zenith_theta_deg"]
        scatter_angle_deg = s["input"]["initial"]["scatter_angle_deg"]
        instrument_x = s["input"]["initial"]["instrument_x"]
        instrument_y = s["input"]["initial"]["instrument_y"]
        instrument_radius = s["input"]["initial"]["instrument_radius"]
    else:
        last_energy = s["energy"][-1]
        energy_iteration_factor = s["energy_iteration_factor"][-1]
        energy = last_energy*energy_iteration_factor
        azimuth_phi_deg = s["azimuth_phi_deg"][-1]
        zenith_theta_deg = s["zenith_theta_deg"][-1]
        scatter_angle_deg = s["scatter_angle_deg"][-1]
        instrument_x = s["instrument_x"][-1]
        instrument_y = s["instrument_y"][-1]
        instrument_radius = s["instrument_radius"][-1]

    expected_ratio_detected_over_thrown = \
        s["input"]["target_containment_angle_deg"]**2 / \
        scatter_angle_deg**2

    on_target = False
    sub_iteration = 0
    while not on_target:
        direction_converged = False
        position_converged = False
        if sub_iteration > max_subiterations or energy_iteration_factor > 0.98:
            raise RuntimeError("Can not converge. Quit.")

        print("E: {:0.3f}, It: ({:d},{:d})".format(
            energy,
            energy_iteration,
            sub_iteration))

        num_events_per_job = int(
            s["input"]["energy_thrown_per_iteration"]/energy)

        jobs = _make_jobs(
            particle_id=s["input"]["corsika_particle_id"],
            energy=energy,
            site=s["input"]["site"],
            azimuth_phi_deg=azimuth_phi_deg,
            zenith_theta_deg=zenith_theta_deg,
            scatter_angle_deg=scatter_angle_deg,
            num_events=num_events_per_job,
            energy_iteration=energy_iteration,
            sub_iteration=sub_iteration,
            num_jobs=num_jobs,
            corsika_primary_path=corsika_primary_path)

        result_blocks = pool.map(_run_job, jobs)
        events = np.concatenate(result_blocks)

        num_thrown = events.shape[0]
        above100ph = events[:, EventSummary.num_photons] >= 100.
        num_above100ph = int(np.sum(above100ph))

        events_above100 = events[above100ph]

        cxs = events_above100[:, EventSummary.cxs_median]
        cys = events_above100[:, EventSummary.cys_median]
        target_offset = np.hypot(
            cxs - s["input"]["target_cx"],
            cys - s["input"]["target_cy"])

        near_target = target_offset < \
            np.deg2rad(s["input"]["target_containment_angle_deg"])
        events_valid = events_above100[near_target]
        num_above100ph_and_on_target = events_valid.shape[0]

        print(
            "thrown: {:d}, >100ph: {:d}, on target: {:d}.".format(
                num_thrown,
                num_above100ph,
                num_above100ph_and_on_target))

        num_events_expected = expected_ratio_detected_over_thrown*num_thrown
        min_num_events_expected = int(0.25*num_events_expected)

        if num_above100ph_and_on_target < min_num_events_expected:
            energy_iteration_factor = (energy_iteration_factor + 1.)/2.
            energy = last_energy*energy_iteration_factor
            print(
                "Expected valid events > {:d}, but found only {:d}.".format(
                    min_num_events_expected,
                    num_above100ph_and_on_target))
            print("Reducing energy_iteration_factor to: {:.4f}".format(
                energy_iteration_factor))
            sub_iteration += 1
            continue

        azimuth_phi_deg_valid = float(
            np.rad2deg(
                np.median(
                    events_valid[:, EventSummary.particle_azimuth_phi])))
        zenith_theta_deg_valid = float(
            np.rad2deg(
                np.median(
                    events_valid[:, EventSummary.particle_zenith_theta])))

        cherenkov_core_xs = 1e-2*events_valid[:, EventSummary.xs_median]
        cherenkov_core_ys = 1e-2*events_valid[:, EventSummary.ys_median]

        x_valid = float(np.median(cherenkov_core_xs))
        y_valid = float(np.median(cherenkov_core_ys))
        x_std_valid = float(np.std(cherenkov_core_xs))
        y_std_valid = float(np.std(cherenkov_core_ys))
        xy_std_valid = np.hypot(x_std_valid, y_std_valid)

        cherenkov_core_ellipse = _estimate_ellipse(
            xs=cherenkov_core_xs,
            ys=cherenkov_core_ys)

        delta_directiong_deg = great_circle_distance_alt_zd_deg(
            az1_deg=azimuth_phi_deg_valid,
            zd1_deg=zenith_theta_deg_valid,
            az2_deg=azimuth_phi_deg,
            zd2_deg=zenith_theta_deg)
        print("delta_directiong_deg: {:0.1f}".format(delta_directiong_deg))
        if delta_directiong_deg <= on_target_direction_threshold_deg:
            direction_converged = True

        delta_position = np.hypot(x_valid, y_valid)
        print(
            "delta_position: {:0.1f}+-{:0.1f}".format(
                delta_position,
                xy_std_valid))
        on_target_position_threshold = (1./2.)*xy_std_valid
        if delta_position <= on_target_position_threshold:
            position_converged = True

        if position_converged and direction_converged:
            on_target = True
        else:
            instrument_x += x_valid/2
            instrument_y += y_valid/2
            instrument_radius = np.min([
                np.max([instrument_radius, 2*xy_std_valid]),
                2.5e3])
            azimuth_phi_deg = (azimuth_phi_deg + azimuth_phi_deg_valid)/2
            zenith_theta_deg = (zenith_theta_deg + zenith_theta_deg_valid)/2

        sub_iteration += 1

    s["energy"].append(float(energy))
    s["energy_iteration_factor"].append(float(energy_iteration_factor))
    s["instrument_radius"].append(float(instrument_radius))
    s["instrument_x"].append(float(x_valid + instrument_x))
    s["instrument_y"].append(float(y_valid + instrument_y))
    s["azimuth_phi_deg"].append(float(azimuth_phi_deg_valid))
    s["zenith_theta_deg"].append(float(zenith_theta_deg_valid))
    s["scatter_angle_deg"].append(float(scatter_angle_deg))
    s["cherenkov_core_spread_xy"].append(cherenkov_core_ellipse)
    _write_state(work_dir=work_dir, state=s, iteration=energy_iteration)


def _make_jobs(
    particle_id,
    energy,
    site,
    azimuth_phi_deg,
    zenith_theta_deg,
    scatter_angle_deg,
    num_events,
    energy_iteration,
    sub_iteration,
    num_jobs,
    corsika_primary_path,
):
    jobs = []
    for i in range(num_jobs):
        run_number = i + 1
        job = {}
        job["steering_card"] = _make_corsika_steering(
            random_seed=energy_iteration,
            run_number=run_number,
            num_events=num_events,
            particle_id=particle_id,
            energy=energy,
            cone_azimuth_deg=azimuth_phi_deg,
            cone_zenith_distance_deg=zenith_theta_deg,
            cone_max_scatter_angle_deg=scatter_angle_deg,
            earth_magnetic_field_x_muT=site["earth_magnetic_field_x_muT"],
            earth_magnetic_field_z_muT=site["earth_magnetic_field_z_muT"],
            atmosphere_id=site["corsika_atmosphere_id"],
            observation_level_asl_m=site["observation_level_asl_m"])
        job["corsika_primary_path"] = corsika_primary_path
        jobs.append(job)
    return jobs
