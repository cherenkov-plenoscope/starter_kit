import numpy as np
import os
from os import path as op
import shutil
import errno
import tempfile
import json
import tarfile
import io
import datetime
import subprocess
import PIL
import corsika_primary_wrapper as cpw
import plenopy as pl

"""
I think I have an efficient and very simple algorithm

0) Pick a threshold photon number T1 where trigger curve starts rising
(for a given type of primary)

1) Generate shower such that particle direction hits ground at 0,0;
shower direction spread over large solid angle Omega (energy-dep.)
(for charged particles)
{could also pick (0,0) at some height, but I believe for z=0 the photon
scatter is smallest}

2) Divide ground in grid of spacing = mirror diameter; could e.g. without
too much trouble use up to M x M = 1000 x 1000 grid cells = 70 x 70 km^2;
grid area is A, grid centered on (0,0)

3) Reset photon counter for each cell

3) For each shower, shift grid randomly in x,y by 1/2 mirror diameter

4) Loop over shower photons
   4.1) reject photon if angle outside FOV
   4.2) for each photon, calculate grid cell index ix, iy
        {easy since square grid}
   4.3) calculate distance of photon from cell center;
        keep photon if distance < R_Mirror
   4.4) increment photon counter for cell
   4.5) optionally save photon in a buffer

5) Loop over grid cells
   5.1) count cells with photons > T1: N_1
   5.2) using trigger curve for given particle type;
        calculate trigger prob. for (real) trigger
        and randomly reject events: keep N_2
        {or simply use a 2nd threshold where trigger prob=0.5}
   5.3) Increment event counters by N_1, N_2
        Increment error counters by N_1^2, N_2^2

6) For detailed simulation, optionally output photons for
   few randomly selected T1-triggered cells
   (up to 10 should be fine, given that
   probably only one of 10 triggers the detailed simulation)

7) Toy effective area (x solid angle): (N_1 event counter/M^2/Nevent)*A*Omega
   error = sqrt(error counter) ...
   Somewhat better effective area: N_2 event counter ...
   Final eff. area: N1_eff area x fraction of events kept in detailed sim.

Cheers
Werner



Coordinate system
=================
                                  | z
                                  |                               starting pos.
                                  |                                  ___---O
                                  |                            ___---    / |
                                  |                      ___---     n  /   |
                                  |                ___---         io /     |
                                  |          ___---             ct /       |
                                  |    ___---                 re /         |
              starting altitude __|_---                     di /           |
                                  |                       y- /             |
                                  | _-------__          ar /               |
                                  |-    th    |_      im /                 |
                                  |   ni        |_  pr /                   |
                                  | ze            |  /                     |
                                  |               |/                       |
                      ____________|______________/________________________ |
                     /            |            /            /            / |
                    /            /|          //            /            /  |
                  3/            / |        / /            /            /   |
                  /            /  |      /  /            /            /    |
                 /____________/___|____/___/____________/____________/     |
                /            /    |  /    /            /            /      |
obs. level     /            /     |/     /    grid    /            /       |
altitude -  -2/-  -  -  -  /  -  -X-----/  <-shift y /            /        |
             /            /      /|    /            /            /         |
            /____________/______/_____/____________/____________/          |
           /            /     -|  |  /            /            /           |
          /            /      /   | /            /            /            |
        1/            /  grid     |/            /            /             |
        /            /  shift x   /            /            /              |
       /____________/____________/____________/____________/               |
      /            /            / |          /            /                |
     /            /            /  |         /            /                 |
   0/            /            /   |        /            /                  |
   /            /            /    |       /            /                   |
  /____________/____________/____________/____________/                    |
        0            1           2|             3                          |
                                  |                                  ___---O
                                  |                            ___---
                                  |                      ___--- |
                                  |                ___---        |
                                  |          ___---               |
                                  |    ___---                     |
                                  |_---         azimuth           |
                                  |                               /
                 sea leavel z=0   |______________________________/___ x
                                  / (0,0,0)
                                 /
                                /
                               /
                              /
                             /
                            /
                           /
                          /
                         /
                        /
                       / y
Drawn by Sebastian
"""


def date_dict_now():
    dt = datetime.datetime.now()
    out = {}
    for key in ["year", "month", "day", "hour", "minute", "second"]:
        out[key] = int(dt.__getattribute__(key))
    return out


CORSIKA_PRIMARY_PATH = os.path.abspath(
    op.join(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd"))

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = os.path.abspath(
    op.join(
        "build",
        "merlict",
        "merlict-plenoscope-propagation"))

LIGHT_FIELD_GEOMETRY_PATH = os.path.abspath(
    op.join(
        "run20190724_10",
        "light_field_calibration"))

EXAMPLE_PLENOSCOPE_SCENERY_PATH = os.path.abspath(
    op.join(
        "resources",
        "acp",
        "71m",
        "scenery"))

MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH = os.path.abspath(
    op.join(
        "resources",
        "acp",
        "merlict_propagation_config.json"))

EXAMPLE_SITE = {
    "observation_level_asl_m": 2300,
    "earth_magnetic_field_x_muT": 12.5,
    "earth_magnetic_field_z_muT": -25.9,
    "atmosphere_id": 10,
}

EXAMPLE_PARTICLE = {
    "particle_id": 14,
    "energy_bin_edges": [5, 10, 100, 1000],
    "max_zenith_deg_vs_energy": [30, 30, 30, 30],
    "max_depth_g_per_cm2_vs_energy": [0, 0, 0, 0],
    "energy_power_law_slope": -2.0,
}

EXAMPLE_GRID = {
    "num_bins_radius": 512,
    "threshold_num_photons": 50
}

EXAMPLE_SUM_TRIGGER = {
    "patch_threshold": 103,
    "integration_time_in_slices": 10,
    "min_num_neighbors": 3,
    "object_distances": [10e3, 15e3, 20e3],
}

EXAMPLE_LOG_DIRECTORY = op.join(".", "_log3")
EXAMPLE_PAST_TRIGGER_DIRECTORY = op.join(".", "_past_trigger3")
EXAMPLE_FEATURE_DIRECTORY = op.join(".", "_features3")
EXAMPLE_WORK_DIR = op.join(".", "_work3")

EXAMPLE_JOB = {
    "run_id": 1,
    "num_air_showers": 100,
    "particle": EXAMPLE_PARTICLE,
    "site": EXAMPLE_SITE,
    "grid": EXAMPLE_GRID,
    "sum_trigger": EXAMPLE_SUM_TRIGGER,
    "corsika_primary_path": CORSIKA_PRIMARY_PATH,
    "plenoscope_scenery_path": EXAMPLE_PLENOSCOPE_SCENERY_PATH,
    "merlict_plenoscope_propagator_path": MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
    "light_field_geometry_path": LIGHT_FIELD_GEOMETRY_PATH,
    "merlict_plenoscope_propagator_config_path":
        MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH,
    "log_dir": EXAMPLE_LOG_DIRECTORY,
    "past_trigger_dir": EXAMPLE_PAST_TRIGGER_DIRECTORY,
    "feature_dir": EXAMPLE_FEATURE_DIRECTORY,
    "non_temp_work_dir": EXAMPLE_WORK_DIR,
    "date": date_dict_now(),
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
IX = {}
IX["marker"] = 0
IX["version"] = 1

IX["run_id"] = 5
IX["airshower_id"] = 6
IX["primary_id"] = 20
IX["primary_energy_gev"] = 21
IX["primary_azimuth_rad"] = 22
IX["primary_zenith_rad"] = 23
IX["primary_depth_g_per_cm2"] = 24
IX["primary_height_asl_m"] = 25
IX["primary_momentum_x_gev_per_c"] = 26
IX["primary_momentum_y_gev_per_c"] = 27
IX["primary_momentum_z_gev_per_c"] = 28
IX["primary_first_interaction_height_asl_m"] = 29

IX["full_shower_num_bunches"] = 40
IX["full_shower_num_photons"] = 41
IX["full_shower_maximum_asl_m"] = 42
IX["full_shower_wavelength_median_nm"] = 43
IX["full_shower_cx_median_rad"] = 44
IX["full_shower_cy_median_rad"] = 45
IX["full_shower_x_median_m"] = 46
IX["full_shower_y_median_m"] = 47
IX["full_shower_bunch_size_median"] = 48

IX["grid_histogram_0"] = 60
IX["grid_histogram_1"] = 61
IX["grid_histogram_2"] = 62
IX["grid_histogram_3"] = 63
IX["grid_histogram_16"] = 76
# and so on
IX["grid_random_shift_x_m"] = 78
IX["grid_random_shift_y_m"] = 79
IX["grid_plenoscope_pointing_direction_x"] = 80
IX["grid_plenoscope_pointing_direction_y"] = 81
IX["grid_plenoscope_pointing_direction_z"] = 82
IX["grid_plenoscope_field_of_view_radius_deg"] = 83

IX[""] = 0
"""


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


def ray_plane_x_y_intersection(support, direction, plane_z):
    direction = np.array(direction)
    support = np.array(support)
    direction_norm = direction/np.linalg.norm(direction)
    ray_parameter = -(support[2] - plane_z)/direction_norm[2]
    intersection = support + ray_parameter*direction_norm
    assert np.abs(intersection[2] - plane_z) < 1e-3
    return intersection


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
    assert(num_events <= MAX_NUM_EVENTS_IN_RUN)

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


def _assign_plenoscope_grid(
    cherenkov_bunches,
    plenoscope_field_of_view_radius_deg,
    plenoscope_pointing_direction,
    plenoscope_grid_geometry,
    grid_random_shift_x,
    grid_random_shift_y,
    threshold_num_photons,
    FIELD_OF_VIEW_OVERHEAD=1.1,
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
        plenoscope_field_of_view_radius_deg*FIELD_OF_VIEW_OVERHEAD)

    bunches_in_fov = cherenkov_bunches[mask_inside_field_of_view, :]

    # Supports
    # --------
    bunch_x_bin_idxs = np.digitize(
        CM2M*bunches_in_fov[:, cpw.IX] + grid_random_shift_x,
        bins=pgg["xy_bin_edges"])
    bunch_y_bin_idxs = np.digitize(
        CM2M*bunches_in_fov[:, cpw.IY] + grid_random_shift_y,
        bins=pgg["xy_bin_edges"])

    # Add under-, and overflow bin-edges
    _xy_bin_edges = [-np.inf] + pgg["xy_bin_edges"].tolist() + [np.inf]

    # histogram num photons, i.e. use bunchsize weights.
    grid_histogram_flow = np.histogram2d(
        x=CM2M*bunches_in_fov[:, cpw.IX] + grid_random_shift_x,
        y=CM2M*bunches_in_fov[:, cpw.IY] + grid_random_shift_y,
        bins=(_xy_bin_edges, _xy_bin_edges),
        weights=bunches_in_fov[:, cpw.IBSIZE])[0]

    # cut out the inner grid, use outer rim to estimate under-, and overflow
    grid_histogram = grid_histogram_flow[1:-1, 1:-1]
    assert grid_histogram.shape[0] == pgg["num_bins_diameter"]
    assert grid_histogram.shape[1] == pgg["num_bins_diameter"]

    bin_idxs_above_threshold = np.where(grid_histogram > threshold_num_photons)
    num_bins_above_threshold = bin_idxs_above_threshold[0].shape[0]

    if num_bins_above_threshold == 0:
        choice = None
    else:
        _choice_bin = np.random.choice(np.arange(num_bins_above_threshold))
        bin_idx_x = bin_idxs_above_threshold[0][_choice_bin]
        bin_idx_y = bin_idxs_above_threshold[1][_choice_bin]
        num_photons_in_bin = grid_histogram[bin_idx_x, bin_idx_y]
        choice = {}
        choice["bin_idx_x"] = int(bin_idx_x)
        choice["bin_idx_y"] = int(bin_idx_y)
        choice["core_x_m"] = float(
            pgg["xy_bin_centers"][bin_idx_x] - grid_random_shift_x)
        choice["core_y_m"] = float(
            pgg["xy_bin_centers"][bin_idx_y] - grid_random_shift_y)
        match_bin_idx_x = bunch_x_bin_idxs - 1 == bin_idx_x
        match_bin_idx_y = bunch_y_bin_idxs - 1 == bin_idx_y
        match_bin = np.logical_and(match_bin_idx_x, match_bin_idx_y)
        assert 1e-2 > np.abs(
            np.sum(bunches_in_fov[match_bin, cpw.IBSIZE]) -
            num_photons_in_bin)
        choice["cherenkov_bunches"] = bunches_in_fov[match_bin, :].copy()
        choice["cherenkov_bunches"][:, cpw.IX] -= M2CM*choice["core_x_m"]
        choice["cherenkov_bunches"][:, cpw.IY] -= M2CM*choice["core_y_m"]

    out = {}
    out["random_choice"] = choice
    out["histogram"] = grid_histogram
    out["overflow_x"] = np.sum(grid_histogram_flow[-1, :])
    out["underflow_x"] = np.sum(grid_histogram_flow[0, :])
    out["overflow_y"] = np.sum(grid_histogram_flow[:, -1])
    out["underflow_y"] = np.sum(grid_histogram_flow[:, 0])
    out["intensity_histogram"] = np.histogram(
        grid_histogram.flatten(),
        bins=PH_BIN_EDGES)[0]
    out["num_bins_above_threshold"] = num_bins_above_threshold
    return out


def tar_append(tarout, file_name, file_bytes):
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


def _append_trigger_truth(
    trigger_dict,
    trigger_responses,
    detector_truth,
):
    tr = trigger_dict
    tr["num_cherenkov_pe"] = int(detector_truth.number_air_shower_pulses())
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
"""


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    ase["num_bunches"] = int(cb.shape[0])
    ase["num_photons"] = float(np.sum(cb[:, cpw.IBSIZE]))
    if cb.shape[0] > 0:
        ase["maximum_asl_m"] = float(CM2M*np.median(cb[:, cpw.IZEM]))
        ase["wavelength_median_nm"] = float(np.abs(np.median(cb[:, cpw.IWVL])))
        ase["cx_median_rad"] = float(np.median(cb[:, cpw.ICX]))
        ase["cy_median_rad"] = float(np.median(cb[:, cpw.ICY]))
        ase["x_median_m"] = float(CM2M*np.median(cb[:, cpw.IX]))
        ase["y_median_m"] = float(CM2M*np.median(cb[:, cpw.IY]))
        ase["bunch_size_median"] = float(np.median(cb[:, cpw.IBSIZE]))
    else:
        ase["maximum_asl_m"] = float("nan")
        ase["wavelength_median_nm"] = float("nan")
        ase["cx_median_rad"] = float("nan")
        ase["cy_median_rad"] = float("nan")
        ase["x_median_m"] = float("nan")
        ase["y_median_m"] = float("nan")
        ase["bunch_size_median"] = float("nan")
    return ase


def write_jsonl(path, list_of_dicts):
    with open(path+".tmp", "wt") as f:
        for d in list_of_dicts:
            f.write(json.dumps(d)+"\n")
    shutil.move(path+".tmp", path)


def image_to_8bit_png_logscale_bytes(img):
    hist = img.copy()
    above_zero = hist > 0
    hist[above_zero] = np.log2(hist[above_zero])
    _max = int(np.max(hist))
    if _max > 0:
        hist = 255.*hist/_max
    hist = hist.astype(np.uint8)
    with io.BytesIO() as f:
        image = PIL.Image.fromarray(hist)
        image.save(f, format="PNG")
        png_bytes = f.getvalue()
    return png_bytes


def save_copy(src, dst):
    try:
        shutil.copytree(src, dst+".tmp")
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy2(src, dst+".tmp")
        else:
            raise
    shutil.move(dst+".tmp", dst)


def run(job=EXAMPLE_JOB):
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)
    run_id_str = "{:06d}".format(job["run_id"])
    logger = JsonlLog(op.join(job["log_dir"], run_id_str+"_log.josnl"))
    job_path = op.join(job["feature_dir"], run_id_str+"_job.json")
    with open(job_path, "wt") as f:
        f.write(json.dumps(job, indent=4))

    # assert resources exist
    # ----------------------
    assert os.path.exists(job["corsika_primary_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_path"])
    assert os.path.exists(job["merlict_plenoscope_propagator_config_path"])
    assert os.path.exists(job["plenoscope_scenery_path"])
    assert os.path.exists(job["light_field_geometry_path"])
    logger.log("assert resource-paths exist.")

    # set up plenoscope grid
    # ----------------------
    _scenery_path = op.join(job["plenoscope_scenery_path"], "scenery.json")
    _light_field_sensor_geometry = _read_plenoscope_geometry(_scenery_path)
    plenoscope_diameter = 2.0*_light_field_sensor_geometry[
        "expected_imaging_system_aperture_radius"]
    plenoscope_radius = .5*plenoscope_diameter
    plenoscope_pointing_direction = np.array([0, 0, 1])
    plenoscope_field_of_view_radius_deg = 0.5*_light_field_sensor_geometry[
        "max_FoV_diameter_deg"]
    plenoscope_grid_geometry = _init_plenoscope_grid(
        plenoscope_diameter=plenoscope_diameter,
        num_bins_radius=job["grid"]["num_bins_radius"])
    logger.log("set plenoscope-grid")

    # draw primaries
    # --------------
    corsika_primary_steering = draw_corsika_primary_steering(
        run_id=job["run_id"],
        site=job["site"],
        particle=job["particle"],
        num_events=job["num_air_showers"])
    logger.log("draw primaries")

    with tempfile.TemporaryDirectory(prefix="plenoscope_irf_") as tmp_dir:
        if job["non_temp_work_dir"] is not None:
            tmp_dir = job["non_temp_work_dir"]
            os.makedirs(tmp_dir, exist_ok=True)
        logger.log("make temp_dir:'{:s}'".format(tmp_dir))

        # run CORSIKA
        # -----------
        corsika_run_path = op.join(tmp_dir, run_id_str+"_corsika.tar")
        if not os.path.exists(corsika_run_path):
            cpw_rc = cpw.corsika_primary(
                corsika_path=job["corsika_primary_path"],
                steering_dict=corsika_primary_steering,
                output_path=corsika_run_path,
                stdout_postfix=".stdout",
                stderr_postfix=".stderr")
            save_copy(
                corsika_run_path+".stdout",
                op.join(job["log_dir"], run_id_str+"_corsika.stdout"))
            save_copy(
                corsika_run_path+".stderr",
                op.join(job["log_dir"], run_id_str+"_corsika.stderr"))
            logger.log("run CORSIKA")

        with open(corsika_run_path+".stdout", "rt") as f:
            assert cpw.stdout_ends_with_end_of_run_marker(f.read())
        logger.log("assert CORSIKA quit ok")

        # loop over air-showers
        # ---------------------
        table_prim = []
        table_fase = []
        table_grhi = []
        table_rase = []
        table_rcor = []

        run = cpw.Tario(corsika_run_path)
        reuse_run_path = op.join(tmp_dir, run_id_str+"_reuse.tar")
        tmp_imgtar_path = op.join(tmp_dir, "grid_histogram_images.tar")
        with tarfile.open(reuse_run_path, "w") as tarout,\
                tarfile.open(tmp_imgtar_path, "w") as imgtar:
            tar_append(tarout, cpw.TARIO_RUNH_FILENAME, run.runh.tobytes())
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

                np.random.seed(event_seed)
                grid_random_shift_x, grid_random_shift_y = np.random.uniform(
                    low=-plenoscope_radius,
                    high=plenoscope_radius,
                    size=2)

                # export primary table
                # --------------------
                prim = ide.copy()
                prim["particle_id"] = int(primary["particle_id"])
                prim["energy_GeV"] = float(primary["energy_GeV"])
                prim["azimuth_rad"] = float(primary["azimuth_rad"])
                prim["zenith_rad"] = float(primary["zenith_rad"])
                prim["depth_g_per_cm2"] = float(primary["depth_g_per_cm2"])
                prim["momentum_x_GeV_per_c"] = float(
                    cpw._evth_px_momentum_in_x_direction_GeV_per_c(
                        event_header))
                prim["momentum_y_GeV_per_c"] = float(
                    cpw._evth_py_momentum_in_y_direction_GeV_per_c(
                        event_header))
                prim["momentum_z_GeV_per_c"] = float(
                    -1.0*cpw._evth_pz_momentum_in_z_direction_GeV_per_c(
                        event_header))
                prim["first_interaction_height_asl_m"] = float(
                    CM2M*-1.0*cpw._evth_z_coordinate_of_first_interaction_cm(
                        event_header))
                prim["starting_height_asl_m"] = float(
                    CM2M*cpw._evth_starting_height_cm(event_header))
                obs_lvl_intersection = ray_plane_x_y_intersection(
                    support=[0, 0, prim["starting_height_asl_m"]],
                    direction=[
                        prim["momentum_x_GeV_per_c"],
                        prim["momentum_y_GeV_per_c"],
                        prim["momentum_z_GeV_per_c"]],
                    plane_z=job["site"]["observation_level_asl_m"])
                prim["starting_x_m"] = -float(obs_lvl_intersection[0])
                prim["starting_y_m"] = -float(obs_lvl_intersection[1])
                table_prim.append(prim)

                # export full shower statistics
                # -----------------------------
                fase = ide.copy()
                fase = _append_bunch_statistics(
                    airshower_dict=fase,
                    cherenkov_bunches=cherenkov_bunches)
                table_fase.append(fase)

                # assign grid
                # -----------
                grid_result = _assign_plenoscope_grid(
                    cherenkov_bunches=cherenkov_bunches,
                    plenoscope_field_of_view_radius_deg=(
                        plenoscope_field_of_view_radius_deg),
                    plenoscope_pointing_direction=(
                        plenoscope_pointing_direction),
                    plenoscope_grid_geometry=plenoscope_grid_geometry,
                    grid_random_shift_x=grid_random_shift_x,
                    grid_random_shift_y=grid_random_shift_y,
                    threshold_num_photons=job["grid"]["threshold_num_photons"])

                tar_append(
                    tarout=imgtar,
                    file_name="run{:06d}_shower{:06d}_max{:06d}pe".format(
                        run_id,
                        event_id,
                        int(np.max(grid_result["histogram"]))),
                    file_bytes=image_to_8bit_png_logscale_bytes(
                        grid_result["histogram"]))

                # grid statistics
                # ---------------
                grhi = ide.copy()
                grhi["num_bins_radius"] = int(
                    plenoscope_grid_geometry["num_bins_radius"])
                grhi["plenoscope_diameter_m"] = float(
                    plenoscope_grid_geometry["plenoscope_diameter"])
                grhi["plenoscope_field_of_view_radius_deg"] = float(
                    plenoscope_field_of_view_radius_deg)
                grhi["plenoscope_pointing_direction_x"] = float(
                    plenoscope_pointing_direction[0])
                grhi["plenoscope_pointing_direction_y"] = float(
                    plenoscope_pointing_direction[1])
                grhi["plenoscope_pointing_direction_z"] = float(
                    plenoscope_pointing_direction[2])
                grhi["random_shift_x_m"] = grid_random_shift_x
                grhi["random_shift_y_m"] = grid_random_shift_y
                for i in range(len(grid_result["intensity_histogram"])):
                    grhi["hist_{:02d}".format(i)] = int(
                        grid_result["intensity_histogram"][i])
                grhi["num_bins_above_threshold"] = int(
                    grid_result["num_bins_above_threshold"])
                grhi["overflow_x"] = int(grid_result["overflow_x"])
                grhi["underflow_x"] = int(grid_result["underflow_x"])
                grhi["overflow_y"] = int(grid_result["overflow_y"])
                grhi["underflow_y"] = int(grid_result["underflow_y"])
                table_grhi.append(grhi)

                reuse_event = grid_result["random_choice"]
                if reuse_event is not None:
                    IEVTH_NUM_REUSES = 98-1
                    IEVTH_CORE_X = IEVTH_NUM_REUSES + 1
                    IEVTH_CORE_Y = IEVTH_NUM_REUSES + 11
                    reuse_evth = event_header.copy()
                    reuse_evth[IEVTH_NUM_REUSES] = 1.0
                    reuse_evth[IEVTH_CORE_X] = M2CM*reuse_event["core_x_m"]
                    reuse_evth[IEVTH_CORE_Y] = M2CM*reuse_event["core_y_m"]

                    tar_append(
                        tarout=tarout,
                        file_name=cpw.TARIO_EVTH_FILENAME.format(event_id),
                        file_bytes=reuse_evth.tobytes())
                    tar_append(
                        tarout=tarout,
                        file_name=cpw.TARIO_BUNCHES_FILENAME.format(event_id),
                        file_bytes=reuse_event["cherenkov_bunches"].tobytes())

                    rase = ide.copy()
                    rase = _append_bunch_statistics(
                        airshower_dict=rase,
                        cherenkov_bunches=reuse_event["cherenkov_bunches"])
                    table_rase.append(rase)

                    rcor = ide.copy()
                    rcor["bin_idx_x"] = int(reuse_event["bin_idx_x"])
                    rcor["bin_idx_y"] = int(reuse_event["bin_idx_y"])
                    rcor["core_x_m"] = float(reuse_event["core_x_m"])
                    rcor["core_y_m"] = float(reuse_event["core_y_m"])
                    table_rcor.append(rcor)
        logger.log("reuse, grid")

        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l1_primary.jsonl"),
            table_prim)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l1_airshower.jsonl"),
            table_fase)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l1_grid.jsonl"),
            table_grhi)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l2_airshower.jsonl"),
            table_rase)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l2_core.jsonl"),
            table_rcor)
        save_copy(
            tmp_imgtar_path,
            op.join(job["feature_dir"], run_id_str+"_grid_images.tar"))

        logger.log("export, level 1, and level 2")

        # run merlict
        # -----------
        merlict_run_path = op.join(tmp_dir, run_id_str+"_merlict.cp")
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
            shutil.move(
                merlict_run_path+".stdout",
                op.join(job["log_dir"], run_id_str+"_merlict.stdout"))
            shutil.move(
                merlict_run_path+".stderr",
                op.join(job["log_dir"], run_id_str+"_merlict.stderr"))
            logger.log("run merlict")
            assert(merlict_rc == 0)

        # prepare trigger
        # ---------------
        merlict_run = pl.Run(merlict_run_path)
        trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
            light_field_geometry=merlict_run.light_field_geometry,
            object_distances=job["sum_trigger"]["object_distances"])
        logger.log("prepare refocus-sum-trigger")

        table_trigger_truth = []
        table_past_trigger = []
        table_past_trigger_paths = []

        # loop over sensor responses
        # --------------------------
        for event in merlict_run:
            trigger_responses = pl.trigger.apply_refocus_sum_trigger(
                event=event,
                trigger_preparation=trigger_preparation,
                min_number_neighbors=job["sum_trigger"]["min_num_neighbors"],
                integration_time_in_slices=(
                    job["sum_trigger"]["integration_time_in_slices"]))
            sum_trigger_info_path = op.join(
                event._path,
                "refocus_sum_trigger.json")
            with open(sum_trigger_info_path, "wt") as f:
                f.write(json.dumps(trigger_responses, indent=4))

            cevth = event.simulation_truth.event.corsika_event_header.raw
            run_id = int(cpw._evth_run_number(cevth))
            airshower_id = int(cpw._evth_event_number(cevth))
            ide = {"run_id": run_id, "airshower_id": airshower_id}

            trigger_truth = ide.copy()
            trigger_truth = _append_trigger_truth(
                trigger_dict=trigger_truth,
                trigger_responses=trigger_responses,
                detector_truth=event.simulation_truth.detector)
            table_trigger_truth.append(trigger_truth)

            if (trigger_truth["trigger_response"] >=
                    job["sum_trigger"]["patch_threshold"]):
                table_past_trigger_paths.append(event._path)
                pl.tools.acp_format.compress_event_in_place(event._path)
                final_event_filename = '{run_id:06d}{airshower_id:06d}'.format(
                    run_id=run_id,
                    airshower_id=airshower_id)
                final_event_path = op.join(
                    job["past_trigger_dir"],
                    final_event_filename)
                save_copy(event._path, final_event_path)
                past_trigger = ide.copy()
                table_past_trigger.append(past_trigger)
        logger.log("run sum-trigger")

        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l2_trigger_truth.jsonl"),
            table_trigger_truth)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l3_past_trigger.jsonl"),
            table_past_trigger)

        # Cherenkov classification
        # ------------------------
        table_cherenkov_classification_scores = []
        for past_trigger_event_path in table_past_trigger_paths:
            event = pl.Event(
                path=past_trigger_event_path,
                light_field_geometry=merlict_run.light_field_geometry)
            roi = pl.classify.center_for_region_of_interest(event)
            photons = pl.classify.RawPhotons.from_event(event)
            (
                cherenkov_photons,
                roi_settings
            ) = pl.classify.cherenkov_photons_in_roi_in_image(
                roi=roi,
                photons=photons)
            pl.classify.write_dense_photon_ids_to_event(
                event_path=os.path.abspath(event._path),
                photon_ids=cherenkov_photons.photon_ids,
                settings=roi_settings)
            score = pl.classify.benchmark(
                pulse_origins=event.simulation_truth.detector.pulse_origins,
                photon_ids_cherenkov=cherenkov_photons.photon_ids)
            score["run_id"] = int(
                event.simulation_truth.event.corsika_run_header.number)
            score["airshower_id"] = int(
                event.simulation_truth.event.corsika_event_header.number)
            table_cherenkov_classification_scores.append(score)
        write_jsonl(
            op.join(
                job["feature_dir"],
                run_id_str+"_l3_cherenkov_classification.jsonl"),
            table_cherenkov_classification_scores)
        logger.log("Cherenkov classification")

        # extracting features
        # -------------------
        lfg = merlict_run.light_field_geometry
        lfg_addon = {}
        lfg_addon["paxel_radius"] = (
            lfg.sensor_plane2imaging_system.
            expected_imaging_system_max_aperture_radius /
            lfg.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal)
        lfg_addon["nearest_neighbor_paxel_enclosure_radius"] = \
            3*lfg_addon["paxel_radius"]
        lfg_addon["paxel_neighborhood"] = (
            pl.features.estimate_nearest_neighbors(
                x=lfg.paxel_pos_x,
                y=lfg.paxel_pos_y,
                epsilon=lfg_addon["nearest_neighbor_paxel_enclosure_radius"]))
        lfg_addon["fov_radius"] = \
            .5*lfg.sensor_plane2imaging_system.max_FoV_diameter
        lfg_addon["fov_radius_leakage"] = 0.9*lfg_addon["fov_radius"]
        lfg_addon["num_pixel_on_diagonal"] = \
            np.floor(2*np.sqrt(lfg.number_pixel/np.pi))
        logger.log("create light_field_geometry addons")

        table_features = []
        for event_path in table_past_trigger_paths:
            event = pl.Event(path=event_path, light_field_geometry=lfg)
            run_id = int(
                event.simulation_truth.event.corsika_run_header.number)
            airshower_id = int(
                event.simulation_truth.event.corsika_event_header.number)
            try:
                cp = event.cherenkov_photons
                if cp is None:
                    raise RuntimeError("No Cherenkov-photons classified yet.")
                f = pl.features.extract_features(
                    cherenkov_photons=cp,
                    light_field_geometry=lfg,
                    light_field_geometry_addon=lfg_addon)
                f["run_id"] = int(run_id)
                f["airshower_id"] = int(airshower_id)
                table_features.append(f)
            except Exception as e:
                print(
                    "run_id {:d}, airshower_id: {:d} :".format(
                        run_id,
                        airshower_id),
                    e)
        write_jsonl(
            op.join(job["feature_dir"], run_id_str+"_l4_features.jsonl"),
            table_features)
        logger.log("extract features from light-field")

        logger.log("end")
