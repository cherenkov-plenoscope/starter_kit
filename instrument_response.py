import numpy as np
import os
import shutil
import tempfile
import json
import tarfile
import io
import datetime
import subprocess
import corsika_primary_wrapper as cpw
import plenopy as pl


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
    "energy_bin_edges": [1, 10, 100, 1000],
    "max_zenith_deg_vs_energy": [3, 3, 3, 3],
    "max_depth_g_per_cm2_vs_energy": [0, 0, 0, 0],
    "energy_power_law_slope": -1.5,
}

EXAMPLE_APERTURE_GRID = {
    "num_bins_radius": 512,
}

EXAMPLE_SUM_TRIGGER = {
    "patch_threshold": 103,
    "integration_time_in_slices": 10,
    "min_num_neighbors": 3,
    "object_distances": [10e3, 15e3, 20e3],
}

EXAMPLE_LOG_DIRECTORY = os.path.join(".", "log")

EXAMPLE_JOB = {
    "run_id": 1,
    "num_air_showers": 100,
    "num_max_air_shower_reuse": 1,
    "particle": EXAMPLE_PARTICLE,
    "site": EXAMPLE_SITE,
    "aperture_grid_threshold_pe": 50,
    "sum_trigger": EXAMPLE_SUM_TRIGGER,
    "aperture_grid": EXAMPLE_APERTURE_GRID,
    "corsika_primary_path": CORSIKA_PRIMARY_PATH,
    "plenoscope_scenery_path": EXAMPLE_PLENOSCOPE_SCENERY_PATH,
    "merlict_plenoscope_propagator_path": MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
    "light_field_geometry_path": LIGHT_FIELD_GEOMETRY_PATH,
    "merlict_plenoscope_propagator_config_path":
        MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH,
    "log_dir": EXAMPLE_LOG_DIRECTORY,
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

I_RUN_ID = 10
I_AIR_SHOWER_ID = 11
I_REUSE_ID = 12

I_TRUE_PRIMARY_ID = 20
I_TRUE_PRIMARY_ENERGY_GEV = 21
I_TRUE_PRIMARY_AZIMUTH_RAD = 22
I_TRUE_PRIMARY_ZENITH_RAD = 23
I_TRUE_PRIMARY_DEPTH_G_PER_CM2 = 24

I_TRUE_PRIMARY_MOMENTUM_X_GEV_PER_C = 25
I_TRUE_PRIMARY_MOMENTUM_Y_GEV_PER_C = 26
I_TRUE_PRIMARY_MOMENTUM_Z_GEV_PER_C = 27

I_TRUE_NUM_CHERENKOV_BUNCHES = 28
I_TRUE_NUM_CHERENKOV_PHOTONS = 29

I_TRUE_CHERENKOV_MAXIMUM_ASL_M = 30

I_GRID_NUM_BINS_RADIUS = 100
I_GRID_BIN_RADIUS = 101
I_GRID_POINTING_X = 102
I_GRID_POINTING_Y = 103
I_GRID_POINTING_Z = 104
I_GRID_FIELD_OF_VIEW_RADIUS_DEG = 105
I_GRID_THRESHOLD = 106
I_GRID_HISTOGRAM_0_TO_1 = 110
I_GRID_HISTOGRAM_1_TO_2 = 111
I_GRID_HISTOGRAM_2_TO_4 = 112
I_GRID_HISTOGRAM_4_TO_8 = 113

I_SITE_OBSERVATION_LEVEL_ALTITUDE_ASL_M = 200
I_SITE_EARTH_MAGNETIC_FIELD_X_MUT = 201
I_SITE_EARTH_MAGNETIC_FIELD_Z_MUT = 202
I_SITE_ATMOSPHERE_ID = 203


def _draw_power_law(lower_limit, upper_limit, power_slope, num_samples):
    # Adopted from CORSIKA
    rd = np.random.uniform(size=num_samples)
    if (power_slope != -1.):
        ll = lower_limit**(power_slope + 1.)
        ul = upper_limit**(power_slope + 1.)
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


def _random_seed_based_on(run_id, event_id):
    return run_id*MAX_NUM_EVENTS_IN_RUN + event_id


MAX_NUM_EVENTS_IN_RUN = 1000


def draw_corsika_primary_steering(
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


"""
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
"""


def set_corsika_primary_steering_to_block(block, corsika_primary_steering):
    run_cfg = corsika_primary_steering["run"]
    for a, primary in enumerate(corsika_primary_steering["primaries"]):
        block[a, I_RUN_ID] = run_cfg["run_id"]
        block[a, I_AIR_SHOWER_ID] = a + 1

        block[a, I_TRUE_PRIMARY_ID] = primary["particle_id"]
        block[a, I_TRUE_PRIMARY_ENERGY_GEV] = primary["energy_GeV"]
        block[a, I_TRUE_PRIMARY_AZIMUTH_RAD] = primary["azimuth_rad"]
        block[a, I_TRUE_PRIMARY_ZENITH_RAD] = primary["zenith_rad"]
        block[a, I_TRUE_PRIMARY_DEPTH_G_PER_CM2] = primary["depth_g_per_cm2"]

        block[a, I_SITE_OBSERVATION_LEVEL_ALTITUDE_ASL_M] = run_cfg[
            "observation_level_altitude_asl"]
        block[a, I_SITE_EARTH_MAGNETIC_FIELD_X_MUT] = run_cfg[
            "earth_magnetic_field_x_muT"]
        block[a, I_SITE_EARTH_MAGNETIC_FIELD_Z_MUT] = run_cfg[
            "earth_magnetic_field_z_muT"]
        block[a, I_SITE_ATMOSPHERE_ID] = run_cfg["atmosphere_id"]
    return block


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
    /              -
    +-------+-------+-------+-------+-
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
    g["num_bins_diameter"] = len(g["xy_bin_edges"]) - 1
    g["xy_bin_centers"] = .5*(g["xy_bin_edges"][:-1] + g["xy_bin_edges"][1:])
    return g


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


CM2M = 1e-2
M2CM = 1./CM2M


def _reuse_id_based_on(bin_idx_x, bin_idx_y, num_bins_per_edge):
    return bin_idx_x*num_bins_per_edge + bin_idx_y


def _core_position(bin_idx, bin_centers, offset):
    return bin_centers[bin_idx] + offset


def _assign_plenoscope_grid(
    cherenkov_bunches,
    plenoscope_field_of_view_radius_deg,
    plenoscope_pointing_direction,
    plenoscope_grid_geometry,
    threshold_1,
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
    assert integrated_bins.shape[0] == pgg["num_bins_diameter"]
    assert integrated_bins.shape[1] == pgg["num_bins_diameter"]

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

        num_bunches_in_integrated_bin = integrated_bins[bin_idx_x, bin_idx_y]

        evt = {}
        evt["reuse_id"] = int(_reuse_id_based_on(
            bin_idx_x=bin_idx_x,
            bin_idx_y=bin_idx_y,
            num_bins_per_edge=pgg["num_bins_diameter"]))
        evt["grid_bin_idx_x"] = int(bin_idx_x)
        evt["grid_bin_idx_y"] = int(bin_idx_y)
        evt["grid_x_offset"] = float(x_offset)
        evt["grid_y_offset"] = float(y_offset)
        evt["grid_x_bin_center"] = float(pgg["xy_bin_centers"][bin_idx_x])
        evt["grid_y_bin_center"] = float(pgg["xy_bin_centers"][bin_idx_y])

        evt["true_primary_core_x"] = (
            pgg["xy_bin_centers"][bin_idx_x] - x_offset)
        evt["true_primary_core_y"] = (
            pgg["xy_bin_centers"][bin_idx_y] - y_offset)

        match_bin_idx_x = bunch_x_bin_idxs - 1 == bin_idx_x
        match_bin_idx_y = bunch_y_bin_idxs - 1 == bin_idx_y
        match_bin = np.logical_and(match_bin_idx_x, match_bin_idx_y)
        assert np.sum(match_bin) == num_bunches_in_integrated_bin

        evt["cherenkov_bunches"] = bunches_in_fov[match_bin, :].copy()
        evt["cherenkov_bunches"][:, cpw.IX] -= M2CM*evt["true_primary_core_x"]
        evt["cherenkov_bunches"][:, cpw.IY] -= M2CM*evt["true_primary_core_y"]
        print("true_primary_core_x/m", evt["true_primary_core_x"])
        print("true_primary_core_y/m", evt["true_primary_core_y"])
        print("IX/m", CM2M*evt["cherenkov_bunches"][:, cpw.IX])
        print("IY/m", CM2M*evt["cherenkov_bunches"][:, cpw.IY])

        return evt, bin_intensity_histogram


def _addbinaryfile(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


class JsonlLog:
    def __init__(self, path):
        self.last_log_time = datetime.datetime.now()
        self.path = path
        self.log("start")

    def log(self, msg):
        now = datetime.datetime.now()
        with open(self.path, "at") as f:
            d = {
                "t": now.strftime("%Y-%m-%d_%H:%M:%S"),
                "delta_t": (now - self.last_log_time).total_seconds(),
                "msg": msg}
            f.write(json.dumps(d)+"\n")
        self.last_log_time = now


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


def _summarize_trigger_response(
    unique_id,
    trigger_responses,
    detector_truth,
):
    tr = unique_id.copy()
    tr["true_pe_cherenkov"] = int(detector_truth.number_air_shower_pulses())
    tr["trigger_response"] = int(np.max(
        [layer['patch_threshold'] for layer in trigger_responses]))
    for o in range(len(trigger_responses)):
        tr["trigger_{:d}_object_distance".format(o)] = float(
            trigger_responses[o]['object_distance'])
        tr["trigger_{:d}_respnse".format(o)] = int(
            trigger_responses[o]['patch_threshold'])
    return tr


"""
IDs:
1.) run_id
2.) airshower_id
3.) reuse_id
"""


def run(job=EXAMPLE_JOB):
    os.makedirs(job["log_dir"], exist_ok=True)
    run_id_str = "{:06d}".format(job["run_id"])
    _t = JsonlLog(os.path.join(job["log_dir"], run_id_str+".josnl"))

    assert os.path.exists(job["corsika_primary_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_config_path"])
    assert os.path.exists(job["plenoscope_scenery_path"])
    assert os.path.exists(job["light_field_geometry_path"])
    _t.log("assert resource-paths exist.")

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

    table_primaries = []
    table_airshowers = []
    table_grid_histograms = []

    with tempfile.TemporaryDirectory(prefix="plenoscope_irf_") as tmp_dir:
        tmp_dir = "/home/relleums/Desktop/work"
        os.makedirs(tmp_dir, exist_ok=True)
        _t.log("make temp_dir:'{:s}'".format(tmp_dir))

        corsika_primary_steering = draw_corsika_primary_steering(
            run_id=job["run_id"],
            site=job["site"],
            particle=job["particle"],
            num_events=job["num_air_showers"])
        _t.log("draw primaries")

        corsika_run_path = os.path.join(tmp_dir, run_id_str+"_corsika.tar")
        if not os.path.exists(corsika_run_path):
            cpw_rc = cpw.corsika_primary(
                corsika_path=job["corsika_primary_path"],
                steering_dict=corsika_primary_steering,
                output_path=corsika_run_path,
                stdout_postfix=".stdout",
                stderr_postfix=".stderr")
        shutil.copy(
            corsika_run_path+".stdout",
            os.path.join(job["log_dir"], run_id_str+"_corsika.stdout"))
        shutil.copy(
            corsika_run_path+".stderr",
            os.path.join(job["log_dir"], run_id_str+"_corsika.stderr"))
        _t.log("run CORSIKA")
        with open(corsika_run_path+".stdout", "rt") as f:
            assert cpw.stdout_ends_with_end_of_run_marker(f.read())
        _t.log("assert CORSIKA quit ok")

        # loop over air-showers
        run = cpw.Tario(corsika_run_path)
        reuse_run_path = os.path.join(tmp_dir, run_id_str+"_reuse.tar")
        with tarfile.open(reuse_run_path, "w") as tarout:
            _addbinaryfile(tarout, "runh.float32", run.runh.tobytes())
            for event_idx, event in enumerate(run):
                event_header, cherenkov_bunches = event

                # assert match
                run_id = int(cpw._evth_run_number(event_header))
                assert (run_id == corsika_primary_steering["run"]["run_id"])
                event_id = event_idx + 1
                assert (event_id == cpw._evth_event_number(event_header))
                primary = corsika_primary_steering["primaries"][event_idx]
                event_seed = primary["random_seed"][0]["SEED"]
                ide = {"run_id":  int(run_id), "airshower_id": int(event_id)}
                assert (event_seed == _random_seed_based_on(
                    run_id=run_id,
                    event_id=event_id))
                # set event's random-seed
                np.random.seed(event_seed)

                (
                    reuse_event,
                    plenoscope_grid_histogram
                ) = _assign_plenoscope_grid(
                    cherenkov_bunches=cherenkov_bunches,
                    plenoscope_field_of_view_radius_deg=(
                        plenoscope_field_of_view_radius_deg),
                    plenoscope_pointing_direction=(
                        plenoscope_pointing_direction),
                    plenoscope_grid_geometry=plenoscope_grid_geometry,
                    threshold_1=job["aperture_grid_threshold_pe"])

                # primary
                # -------
                pre = ide.copy()
                pre["particle_id"] = float(primary["particle_id"])
                pre["energy_GeV"] = float(primary["energy_GeV"])
                pre["azimuth_rad"] = float(primary["azimuth_rad"])
                pre["zenith_rad"] = float(primary["zenith_rad"])
                pre["depth_g_per_cm2"] = float(primary["depth_g_per_cm2"])
                pre["momentum_x_GeV_per_c"] = float(
                    cpw._evth_px_momentum_in_x_direction_GeV_per_c(
                        event_header))
                pre["momentum_y_GeV_per_c"] = float(
                    cpw._evth_py_momentum_in_y_direction_GeV_per_c(
                        event_header))
                pre["momentum_z_GeV_per_c"] = float(
                    cpw._evth_pz_momentum_in_z_direction_GeV_per_c(
                        event_header))
                table_primaries.append(pre)

                # shower statistics
                # -----------------
                ase = ide.copy()
                ase["num_bunches"] = int(cherenkov_bunches.shape[0])
                ase["num_photons"] = float(
                    np.sum(cherenkov_bunches[:, cpw.IBSIZE]))
                if cherenkov_bunches.shape[0] > 0:
                    ase["maximum_asl_m"] = float(
                        CM2M*np.median(cherenkov_bunches[:, cpw.IZEM]))
                    ase["wavelength_median_nm"] = float(
                        np.median(cherenkov_bunches[:, cpw.IWVL]))
                    ase["cx_median_rad"] = float(
                        np.median(cherenkov_bunches[:, cpw.ICX]))
                    ase["cy_median_rad"] = float(
                        np.median(cherenkov_bunches[:, cpw.ICY]))
                    ase["x_median_m"] = float(
                        CM2M*np.median(cherenkov_bunches[:, cpw.IX]))
                    ase["y_median_m"] = float(
                        CM2M*np.median(cherenkov_bunches[:, cpw.IY]))
                    ase["bunch_size_median"] = float(
                        np.median(cherenkov_bunches[:, cpw.IBSIZE]))
                else:
                    ase["maximum_asl_m"] = float("nan")
                    ase["wavelength_median_nm"] = float("nan")
                    ase["cx_median_rad"] = float("nan")
                    ase["cy_median_rad"] = float("nan")
                    ase["x_median_m"] = float("nan")
                    ase["y_median_m"] = float("nan")
                    ase["bunch_size_median"] = float("nan")
                table_airshowers.append(ase)

                # grid statistics
                # ---------------
                gre = ide.copy()
                gre["num_bins_radius"] = int(
                    plenoscope_grid_geometry["num_bins_radius"])
                gre["plenoscope_diameter"] = float(
                    plenoscope_grid_geometry["plenoscope_diameter"])
                for i in range(len(plenoscope_grid_histogram)):
                    gre["hist_{:02d}".format(i)] = int(
                        plenoscope_grid_histogram[i])
                table_grid_histograms.append(gre)

                if reuse_event is not None:
                    _addbinaryfile(
                        tarout=tarout,
                        file_name="{:09d}.evth.float32".format(event_id),
                        file_bytes=event_header.tobytes())
                    _addbinaryfile(
                        tarout=tarout,
                        file_name="{:09d}.cherenkov_bunches.Nx8_float32".format(event_id),
                        file_bytes=reuse_event["cherenkov_bunches"].tobytes())

        _t.log("reuse, grid")

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
                random_seed=run_id,
                stdout_postfix=".stdout",
                stderr_postfix=".stderr")
        shutil.copy(
            merlict_run_path+".stdout",
            os.path.join(job["log_dir"], run_id_str+"_merlict.stdout"))
        shutil.copy(
            merlict_run_path+".stderr",
            os.path.join(job["log_dir"], run_id_str+"_merlict.stderr"))
        _t.log("run merlict")
        assert(merlict_rc == 0)

        run = pl.Run(merlict_run_path)
        trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
            light_field_geometry=run.light_field_geometry,
            object_distances=job["sum_trigger"]["object_distances"])

        _t.log("prepare_refocus_sum_trigger")

        trigger_truth_table = []
        past_trigger_table = []

        past_trigger_path = os.path.join(tmp_dir, "past_trigger")
        os.makedirs(past_trigger_path, exist_ok=True)

        for event in run:
            trigger_responses = pl.trigger.apply_refocus_sum_trigger(
                event=event,
                trigger_preparation=trigger_preparation,
                min_number_neighbors=job["sum_trigger"]["min_num_neighbors"],
                integration_time_in_slices=job["sum_trigger"]["integration_time_in_slices"])
            with open(os.path.join(event._path, "refocus_sum_trigger.json"), "wt") as f:
                f.write(json.dumps(trigger_responses, indent=4))

            crunh = event.simulation_truth.event.corsika_run_header.raw
            cevth = event.simulation_truth.event.corsika_event_header.raw
            run_id = int(cpw._evth_run_number(cevth))
            event_id = int(cpw._evth_event_number(cevth))

            trigger_truth = _summarize_trigger_response(
                unique_id={},
                trigger_responses=trigger_responses,
                detector_truth=event.simulation_truth.detector)
            trigger_truth_table.append(trigger_truth)

            if trigger_truth["trigger_response"] >= job["sum_trigger"]["patch_threshold"]:

                event_filename = '{run_id:06d}{event_id:06d}'.format(
                    run_id=run_id,
                    event_id=event_id)
                event_path = os.path.join(
                    past_trigger_path,
                    event_filename)
                shutil.copytree(event._path, event_path)
                pl.tools.acp_format.compress_event_in_place(event_path)
        _t.log("run sum-trigger")

        with open(os.path.join(tmp_dir, "trigger_truth.jsonl"), 'wt') as f:
            for e in trigger_truth_table:
                f.write(json.dumps(e)+"\n")

        _t.log("end")
