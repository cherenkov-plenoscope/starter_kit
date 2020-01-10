import numpy as np
import os
import tempfile
import json
import tarfile
import io
import time
import subprocess
import corsika_primary_wrapper as cpw


CORSIKA_PRIMARY_PATH = os.path.abspath(
    os.path.join(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd"))

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = os.path.abspath(
    os.path.join(
        "build",
        "merlict",
        "merlict-plenoscope-propagation"))

LIGHT_FIELD_GEOMETRY_PATH = os.path.abspath(
    os.path.join(
        "run20190724_10",
        "light_field_calibration"))

EXAMPLE_PLENOSCOPE_SCENERY_PATH = os.path.abspath(
    os.path.join(
        "resources",
        "acp",
        "71m",
        "scenery"))

MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH = os.path.abspath(
    os.path.join(
        "resources",
        "acp",
        "merlict_propagation_config.json"))

VERSION_MAJOR = 0.
VERSION_MINOR = 0.

EXAMPLE_SITE = {
    "observation_level_altitude_asl": 2300,
    "earth_magnetic_field_x_muT": 12.5,
    "earth_magnetic_field_z_muT": -25.9,
    "atmosphere_id": 10,
}

EXAMPLE_PARTICLE = {
    "particle_id": 1,
    "energy_bin_edges": [0.25, 1, 10, 100, 1000],
    "max_zenith_deg_vs_energy": [10, 10, 10, 10, 10,],
    "max_depth_g_per_cm2_vs_energy": [0, 0, 0, 0, 0],
    "energy_power_law_slope": -1.5,
}

EXAMPLE_APERTURE_GRID = {
    "num_bins_radius": 512,
}

EXAMPLE_SUM_TRIGGER = {
    "patch_threshold": 103,
    "integration_time_in_slices": 10
}

EXAMPLE_JOB = {
    "run_id": 1,
    "num_air_showers": 100,
    "num_max_air_shower_reuse": 1,
    "particle": EXAMPLE_PARTICLE,
    "site": EXAMPLE_SITE,
    "aperture_grid_threshold_pe": 1,
    "sum_trigger": EXAMPLE_SUM_TRIGGER,
    "aperture_grid": EXAMPLE_APERTURE_GRID,
    "corsika_primary_path": CORSIKA_PRIMARY_PATH,
    "plenoscope_scenery_path": EXAMPLE_PLENOSCOPE_SCENERY_PATH,
    "merlict_plenoscope_propagator_path": MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
    "light_field_geometry_path": LIGHT_FIELD_GEOMETRY_PATH,
    "merlict_plenoscope_propagator_config_path":
        MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH,
    "logging_directory": ".",
}


def _read_plenoscope_geometry(scenery_path):
    with open(scenery_path, "rt") as f:
        _scenery = json.loads(f.read())
    children = _scenery['children']
    for child in children:
        if child["type"] == "Frame" and child["name"] == "Portal":
            protal = child.copy()
    for child in protal['children']:
        if child["type"] == "LightFieldSensor":
            light_field_sensor = child.copy()
    return light_field_sensor

"""
LEVEL 1
=======

Draw basic particle properties.
Create steering for CORSIKA-primary-mod.

"""
NUM_FIELDS_IN_LEVEL_BLOCK = 273

I_VERSION_1 = 0
I_VERSION_2 = 1
I_VERSION_3 = 2

I_LEVEL = 3

I_RUN_ID = 10
I_AIR_SHOWER_ID = 11

I_TRUE_PRIMARY_ID = 20
I_TRUE_PRIMARY_ENERGY_GEV = 21
I_TRUE_PRIMARY_AZIMUTH_RAD = 22
I_TRUE_PRIMARY_ZENITH_RAD = 23
I_TRUE_PRIMARY_DEPTH_G_PER_CM2 = 24

I_SITE_OBSERVATION_LEVEL_ALTITUDE_ASL_M = 200
I_SITE_EARTH_MAGNETIC_FIELD_X_MUT = 201
I_SITE_EARTH_MAGNETIC_FIELD_Z_MUT = 202
I_SITE_ATMOSPHERE_ID = 203


def _draw_power_law(lower_limit, upper_limit, power_slope, num_samples):
    # Adopted from CORSIKA
    rd = np.random.uniform(size=num_samples)
    if (power_slope != -1.):
        ll   = lower_limit ** (power_slope + 1.)
        ul   = upper_limit ** (power_slope + 1.)
        slex = 1./(power_slope + 1.)
        return (rd*ul + (1. - rd)*ll)**slex
    else:
        ll = upper_limit/lower_limit
        return lower_limit * ll**rd


def _draw_zenith_distance(
    min_zenith_distance,
    max_zenith_distance,
    num_samples=1
):
    v_min = (np.cos(min_zenith_distance) + 1) / 2
    v_max = (np.cos(max_zenith_distance) + 1) / 2
    v = np.random.uniform(low=v_min, high=v_max, size=num_samples)
    return np.arccos(2 * v - 1)

MAX_NUM_EVENTS_IN_RUN = 1000

def _random_seed_based_on(run_id, event_id):
    return run_id*MAX_NUM_EVENTS_IN_RUN + event_id


def draw_LEVEL_1_corsika_primary_steering(
    run_id=1,
    site=EXAMPLE_SITE,
    particle=EXAMPLE_PARTICLE,
    num_events=100
):
    particle_id = particle["particle_id"]
    energy_bin_edges = particle["energy_bin_edges"]
    max_zenith_deg_vs_energy = particle["max_zenith_deg_vs_energy"]
    max_depth_g_per_cm2_vs_energy = particle["max_depth_g_per_cm2_vs_energy"]
    energy_power_law_slope = particle["energy_power_law_slope"]

    assert(run_id > 0)
    assert(np.all(np.diff(energy_bin_edges) >= 0))
    assert(len(energy_bin_edges) == len(max_zenith_deg_vs_energy))
    assert(len(energy_bin_edges) == len(max_depth_g_per_cm2_vs_energy))
    max_zenith_vs_energy = np.deg2rad(max_zenith_deg_vs_energy)
    assert(num_events < MAX_NUM_EVENTS_IN_RUN + 1)

    np.random.seed(run_id)
    energies = _draw_power_law(
        lower_limit=np.min(energy_bin_edges),
        upper_limit=np.max(energy_bin_edges),
        power_slope=energy_power_law_slope,
        num_samples=num_events)
    steering = {}
    steering["run"] = {
        "run_id": int(run_id),
        "event_id_of_first_event": 1}
    for key in site:
        steering["run"][key] = site[key]

    steering["primaries"] = []
    for e in range(energies.shape[0]):
        event_id = e + 1
        primary = {}
        primary["particle_id"] = int(particle_id)
        primary["energy_GeV"] = float(energies[e])
        max_zenith_distance = np.interp(
            x=energies[e],
            xp=energy_bin_edges,
            fp=max_zenith_vs_energy)
        primary["zenith_rad"] = float(
            _draw_zenith_distance(
                min_zenith_distance=0.,
                max_zenith_distance=max_zenith_distance))
        primary["azimuth_rad"] = float(
            np.random.uniform(
                low=0,
                high=2*np.pi))
        max_depth_g_per_cm2 = np.interp(
            x=energies[e],
            xp=energy_bin_edges,
            fp=max_depth_g_per_cm2_vs_energy)
        primary["depth_g_per_cm2"] = float(
            np.random.uniform(
                low=0.0,
                high=max_depth_g_per_cm2))
        primary["random_seed"] = cpw._simple_seed(
            _random_seed_based_on(run_id=run_id, event_id=event_id))

        steering["primaries"].append(primary)
    return steering


def corsika_primary_steering_to_LEVEL_1_dict(corsika_primary_steering):
    level1 = []
    run_cfg = corsika_primary_steering["run"]
    for asidx, primary in enumerate(corsika_primary_steering["primaries"]):
        a = {}
        a["run_id"] = int(run_cfg["run_id"])
        a["air_shower_id"] = int(asidx + 1)

        a["true_particle_energy_GeV"] = float(primary["energy_GeV"])
        a["true_particle_azimuth_rad"] = float(primary["azimuth_rad"])
        a["true_particle_zenith_rad"] = float(primary["zenith_rad"])
        a["true_particle_depth_g_per_cm2"] = float(primary["depth_g_per_cm2"])

        a["site_observation_level_altitude_asl"] = float(
            run_cfg["observation_level_altitude_asl"])
        a["site_earth_magnetic_field_x_muT"] = float(
            run_cfg["earth_magnetic_field_x_muT"])
        a["site_earth_magnetic_field_z_muT"] = float(
            run_cfg["earth_magnetic_field_z_muT"])
        a["site_atmosphere_id"] = int(run_cfg["atmosphere_id"])
        level1.append(a)
    return level1


def corsika_primary_steering_to_LEVEL_1_array(corsika_primary_steering):
    num_primaries = len(corsika_primary_steering["primaries"])
    l = np.zeros(
        shape=(num_primaries, NUM_FIELDS_IN_LEVEL_BLOCK),
        dtype=np.float32)
    run_cfg = corsika_primary_steering["run"]
    for a, primary in enumerate(corsika_primary_steering["primaries"]):
        l[a, I_RUN_ID] = run_cfg["run_id"]
        l[a, I_AIR_SHOWER_ID] = a + 1

        l[a, I_TRUE_PRIMARY_ID] = primary["particle_id"]
        l[a, I_TRUE_PRIMARY_ENERGY_GEV] = primary["energy_GeV"]
        l[a, I_TRUE_PRIMARY_AZIMUTH_RAD] = primary["azimuth_rad"]
        l[a, I_TRUE_PRIMARY_ZENITH_RAD] = primary["zenith_rad"]
        l[a, I_TRUE_PRIMARY_DEPTH_G_PER_CM2] = primary["depth_g_per_cm2"]

        l[a, I_SITE_OBSERVATION_LEVEL_ALTITUDE_ASL_M] = run_cfg[
            "observation_level_altitude_asl"]
        l[a, I_SITE_EARTH_MAGNETIC_FIELD_X_MUT] = run_cfg[
            "earth_magnetic_field_x_muT"]
        l[a, I_SITE_EARTH_MAGNETIC_FIELD_Z_MUT] = run_cfg[
            "earth_magnetic_field_z_muT"]
        l[a, I_SITE_ATMOSPHERE_ID] = run_cfg["atmosphere_id"]
    return l


"""
LEVEL 2
=======

Run CORSIKA-primary-mod.
reuse showers.
"""

def _init_plenoscope_grid(
    plenoscope_diameter=36,
    num_bins_radius=512,
):
    """
     num_bins_radius = 2
     _______^______
    /              \
    +-------+-------+-------+-------+\
    |       |       |       |       | |
    |       |       |       |       | |
    |       |       |       |       | |
    +-------+-------+-------+-------+ |
    |       |       |       |       |  > num_bins_radius = 2
    |       |       |       |       | |
    |       |       |(0,0)  |       | |
    +-------+-------+-------+-------+/
    |       |       |       |       |
    |       |       |       |       |
    |       |       |       |       |
    +-------+-------+-------+-------+ <-- bin edge
    |       |       |       |       |
    |       |       |       |   X <-- bin center
    |       |       |       |       |
    +-------+-------+-------+-------+

    """
    assert num_bins_radius > 0
    assert plenoscope_diameter > 0.0
    g = {
        "plenoscope_diameter": plenoscope_diameter,
        "num_bins_radius": num_bins_radius}
    g["xy_bin_edges"] = np.linspace(
        -g["plenoscope_diameter"]*g["num_bins_radius"],
        g["plenoscope_diameter"]*g["num_bins_radius"],
        2*g["num_bins_radius"] + 1)
    g["num_bins_diameter"] = len(g["xy_bin_edges"] - 1)
    g["xy_bin_centers"] = .5*(g["xy_bin_edges"][:-1] + g["xy_bin_edges"][1:])
    return g


EXAMPLE_PLENOSCOPE_GRID = _init_plenoscope_grid(
    plenoscope_diameter=36,
    num_bins_radius=512)

def _power2_bin_edges(power):
    be = np.geomspace(1, 2**power, power+1)
    beiz = np.zeros(shape=(be.shape[0] + 1))
    beiz[1:] = be
    return beiz

PH_BIN_EDGES = _power2_bin_edges(16)

def _make_bunch_direction(cx, cy):
    d = np.zeros(shape=(cx.shape[0], 3))
    d[:, 0] = cx
    d[:, 1] = cy
    d[:, 2] = -1.0*np.sqrt(1.0 - cx**2 - cy**2)
    return d

def _make_angle_between(directions, direction):
    # expect normalized
    return np.arccos(np.dot(directions, direction))

EXAMPLE_STEERING = draw_LEVEL_1_corsika_primary_steering(
    num_events=100)

CM2M = 1e-2


def _reuse_id_based_on(bin_idx_x, bin_idx_y, num_bins_per_edge):
    return bin_idx_x*num_bins_per_edge + bin_idx_y


def _core_position(bin_idx, bin_centers, offset):
    return bin_centers[bin_idx] + offset


def _assign_plenoscope_grid(
    cherenkov_bunches,
    plenoscope_field_of_view_radius_deg=3.5,
    plenoscope_pointing_direction=[0, 0, 1],
    plenoscope_grid_geometry=EXAMPLE_PLENOSCOPE_GRID,
    threshold_1=10,
):
    pgg = plenoscope_grid_geometry

    # Directions
    # ----------
    bunch_directions = _make_bunch_direction(
        cx=cherenkov_bunches[:, cpw.ICX],
        cy=cherenkov_bunches[:, cpw.ICY])
    bunch_incidents = -1.0*bunch_directions

    angle_bunch_pointing = _make_angle_between(
        directions=bunch_incidents,
        direction=plenoscope_pointing_direction)

    mask_inside_field_of_view = angle_bunch_pointing < np.deg2rad(
        plenoscope_field_of_view_radius_deg)

    bunches_in_fov = cherenkov_bunches[mask_inside_field_of_view, :]

    # Supports
    # --------
    plenoscope_radius = .5*pgg["plenoscope_diameter"]
    x_offset, y_offset = np.random.uniform(
        low=-plenoscope_radius,
        high=plenoscope_radius,
        size=2)

    bunch_x_bin_idxs = np.digitize(
        CM2M*bunches_in_fov[:, cpw.IX] + x_offset,
        bins=pgg["xy_bin_edges"])
    bunch_y_bin_idxs = np.digitize(
        CM2M*bunches_in_fov[:, cpw.IY] + y_offset,
        bins=pgg["xy_bin_edges"])

    integrated_bins = np.histogram2d(
        CM2M*bunches_in_fov[:, cpw.IX] + x_offset,
        CM2M*bunches_in_fov[:, cpw.IY] + y_offset,
        bins=(
            pgg["xy_bin_edges"],
            pgg["xy_bin_edges"])
        )[0]

    bin_intensity_histogram = np.histogram(
        integrated_bins.flatten(),
        bins=PH_BIN_EDGES)[0]

    bin_idxs_above_threshold = np.where(integrated_bins > threshold_1)
    num_bins_above_threshold = bin_idxs_above_threshold[0].shape[0]

    if num_bins_above_threshold == 0:
        return None, bin_intensity_histogram
    else:
        reuse_bin = np.random.choice(np.arange(num_bins_above_threshold))
        bin_idx_x = bin_idxs_above_threshold[0][reuse_bin]
        bin_idx_y = bin_idxs_above_threshold[1][reuse_bin]
        evt = {}
        evt["reuse_id"] = int(_reuse_id_based_on(
            bin_idx_x=bin_idx_x,
            bin_idx_y=bin_idx_y,
            num_bins_per_edge=pgg["num_bins_diameter"]))
        evt["grid_bin_idx_x"] = int(bin_idx_x)
        evt["grid_bin_idx_y"] = int(bin_idx_y)
        evt["grid_x_offset"] = float(x_offset)
        evt["grid_y_offset"] = float(y_offset)
        evt["true_primary_core_x"] = pgg["xy_bin_centers"][bin_idx_x] - x_offset
        evt["true_primary_core_y"] = pgg["xy_bin_centers"][bin_idx_y] - y_offset
        match_bin_idx_x = bunch_x_bin_idxs == bin_idx_x
        match_bin_idx_y = bunch_y_bin_idxs == bin_idx_y
        match_bin = np.logical_and(match_bin_idx_x, match_bin_idx_y)
        evt["cherenkov_bunches"] = bunches_in_fov[match_bin, :]

        return evt, bin_intensity_histogram


def _addbinaryfile(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


class Timer:
    def __init__(self, task):
        self.history = []
        self.history.append({"time": time.time(), "task": task})

    def append(self, task):
        self.history.append({"time": time.time(), "task": task})

    def export_jsonl(self, path):
        with open(path, "wt") as f:
            for e, _ in enumerate(self.history):
                if e > 0:
                    d = {"s": (
                            self.history[e]["time"] -
                            self.history[e-1]["time"]),
                        "task": self.history[e]["task"]}
                else:
                    d = {"s": 0, "task": self.history[e]["task"]}
                f.write(json.dumps(d)+"\n")


def _merlict_plenoscope_propagator(
    corsika_run_path,
    output_path,
    light_field_geometry_path,
    merlict_plenoscope_propagator_path,
    merlict_plenoscope_propagator_config_path,
    random_seed,
    photon_origins=True,
    stdout_postfix=".stdout",
    stderr_postfix=".stderr",
):
    """
    Calls the merlict Cherenkov-plenoscope propagation
    and saves the stdout and stderr
    """
    with open(output_path+stdout_postfix, 'w') as out, \
        open(output_path+stderr_postfix, 'w') as err:
        call = [
            merlict_plenoscope_propagator_path,
            '-l', light_field_geometry_path,
            '-c', merlict_plenoscope_propagator_config_path,
            '-i', corsika_run_path,
            '-o', output_path,
            '-r', '{:d}'.format(random_seed)]
        if photon_origins:
            call.append('--all_truth')
        mct_rc = subprocess.call(call, stdout=out, stderr=err)
    return mct_rc


def run(job=EXAMPLE_JOB):
    _t = Timer("start")
    assert os.path.exists(job["corsika_primary_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_config_path"])
    assert os.path.exists(job["plenoscope_scenery_path"])
    assert os.path.exists(job["light_field_geometry_path"])

    with tempfile.TemporaryDirectory(prefix="plenoscope_irf_") as tmp_dir:
        run_id_str = "{:06d}".format(job["run_id"])
        tmp_dir = "/home/relleums/Desktop/work"
        os.makedirs(tmp_dir, exist_ok=True)

        # LEVEL 1
        # -------
        corsika_primary_steering = draw_LEVEL_1_corsika_primary_steering(
            run_id=job["run_id"],
            site=job["site"],
            particle=job["particle"],
            num_events=job["num_air_showers"])
        _t.append("draw primaries")

        lvl1_jsonl_path = os.path.join(tmp_dir, run_id_str+"_LEVEL1.jsonl")
        with open(lvl1_jsonl_path, "wt") as f:
            air_showers = corsika_primary_steering_to_LEVEL_1_dict(
                corsika_primary_steering)
            for air_shower in air_showers:
                f.write(json.dumps(air_shower) + "\n")

        lvl1_bin_path = os.path.join(tmp_dir, run_id_str+"_LEVEL1.273xfloat32")
        level1_bin = corsika_primary_steering_to_LEVEL_1_array(
            corsika_primary_steering)
        with open(lvl1_bin_path, "wb") as f:
            f.write(level1_bin.tobytes())

        _t.append("exporrt level 1")

        # LEVEL 2
        # -------
        corsika_run_path = os.path.join(tmp_dir, run_id_str+"_corsika.tar")
        if not os.path.exists(corsika_run_path):
            cpw_rc = cpw.corsika_primary(
                corsika_path=job["corsika_primary_path"],
                steering_dict=corsika_primary_steering,
                output_path=corsika_run_path,
                stdout_postfix=".stdout",
                stderr_postfix=".stderr")
        with open(corsika_run_path+".stdout", "rt") as f:
            assert cpw.stdout_ends_with_end_of_run_marker(f.read())
        _t.append("run CORSIKA")

        # set up plenoscope grid
        _scenery_path = os.path.join(
            job["plenoscope_scenery_path"],
            "scenery.json")
        _light_field_sensor_geometry = _read_plenoscope_geometry(_scenery_path)

        plenoscope_diameter = 2.0*_light_field_sensor_geometry[
            "expected_imaging_system_aperture_radius"]
        plenoscope_pointing_direction = np.array([0, 0, 1])
        plenoscope_field_of_view_radius_deg = 0.5*_light_field_sensor_geometry[
            "max_FoV_diameter_deg"]

        plenoscope_grid_geometry = _init_plenoscope_grid(
            plenoscope_diameter=plenoscope_diameter,
            num_bins_radius=job["aperture_grid"]["num_bins_radius"])

        # loop over air-showers
        lvl2 = []
        run = cpw.Tario(corsika_run_path)
        reuse_run_path = os.path.join(tmp_dir, run_id_str+"_reuse.tar")
        lvl2_jsonl_path =  os.path.join(tmp_dir, run_id_str+"_LEVEL2.jsonl")
        with tarfile.open(reuse_run_path, "w") as tarout, \
            open(lvl2_jsonl_path, "wt") as flvl2:

            _addbinaryfile(tarout, "runh.float32", run.runh.tobytes())

            for event_idx, event in enumerate(run):
                event_header, cherenkov_bunches = event

                # assert match
                run_id = int(cpw._evth_run_number(event_header))
                assert (run_id == corsika_primary_steering["run"]["run_id"])
                event_id = event_idx + 1
                assert (event_id == cpw._evth_event_number(event_header))
                event_steering = corsika_primary_steering["primaries"][event_idx]
                event_seed = event_steering["random_seed"][0]["SEED"]
                assert (event_seed == _random_seed_based_on(
                    run_id=run_id,
                    event_id=event_id))

                # set event's random-seed
                np.random.seed(event_seed)

                reuse_event, plenoscope_grid_histogram = _assign_plenoscope_grid(
                    cherenkov_bunches=cherenkov_bunches,
                    plenoscope_field_of_view_radius_deg=
                        plenoscope_field_of_view_radius_deg,
                    plenoscope_pointing_direction=plenoscope_pointing_direction,
                    plenoscope_grid_geometry=plenoscope_grid_geometry,
                    threshold_1=job["aperture_grid_threshold_pe"])

                #print(reuse_event)
                l2 = {}
                l2["run_id"] = int(run_id)
                l2["air_shower_id"] = int(event_id)

                l2["true_primary_momentum_x_GeV_per_c"] = float(
                    cpw._evth_px_momentum_in_x_direction_GeV_per_c(
                        event_header))
                l2["true_primary_momentum_y_GeV_per_c"] = float(
                    cpw._evth_py_momentum_in_y_direction_GeV_per_c(
                        event_header))
                l2["true_primary_momentum_z_GeV_per_c"] = float(
                    cpw._evth_pz_momentum_in_z_direction_GeV_per_c(
                        event_header))

                l2["true_total_number_cherenkov_photons"] = int(
                    cherenkov_bunches.shape[0])

                if l2["true_total_number_cherenkov_photons"] > 0:
                    l2["true_air_shower_cherenkov_median_asl_m"] = float(
                        CM2M*np.median(cherenkov_bunches[:, cpw.IZEM]))
                else:
                    l2["true_air_shower_cherenkov_median_asl_m"] = 0.

                for i in range(len(plenoscope_grid_histogram)):
                    l2["grid_hist_{:02d}".format(i)] = int(
                        plenoscope_grid_histogram[i])
                flvl2.write(json.dumps(l2)+"\n")

                if reuse_event is not None:
                    l3 = {}
                    l3["reuse_id"] = reuse_event["reuse_id"]

                    _addbinaryfile(
                        tarout=tarout,
                        file_name="{:09d}.evth.float32".format(event_id),
                        file_bytes=event_header.tobytes())
                    _addbinaryfile(
                        tarout=tarout,
                        file_name="{:09d}.cherenkov_bunches.Nx8_float32".format(event_id),
                        file_bytes=reuse_event["cherenkov_bunches"].tobytes())

        _t.append("reuse, grid")

        # LEVEL 3
        # -------
        merlict_run_path = os.path.join(tmp_dir, run_id_str+"_merlict.cp")
        if not os.path.exists(merlict_run_path):
            merlict_rc = _merlict_plenoscope_propagator(
                corsika_run_path=reuse_run_path,
                output_path=merlict_run_path,
                light_field_geometry_path=job[
                    "light_field_geometry_path"],
                merlict_plenoscope_propagator_path=job[
                    "merlict_plenoscope_propagator_path"],
                merlict_plenoscope_propagator_config_path=job[
                    "merlict_plenoscope_propagator_config_path"],
                random_seed=run_id)
            assert(merlict_rc == 0)

        _t.append("run merlict")





        _t.append("end")
        _t.export_jsonl(os.path.join(tmp_dir, "hist.jsonl"))
