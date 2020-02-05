from . import random
from . import table
from . import grid
from . import merlict
from . import logging

import numpy as np
import os
from os import path as op
import shutil
import errno
import uuid
import tempfile
import json
import tarfile
import io
import datetime
import subprocess
import PIL
import pandas as pd
import corsika_primary_wrapper as cpw
import plenopy as pl
import gzip

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
                                  |    ___---       azimuth       |
                sea leavel z=0    |_---__________________________/______ x
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


def absjoin(*args):
    return op.abspath(op.join(*args))


def date_dict_now():
    dt = datetime.datetime.now()
    out = {}
    for key in ["year", "month", "day", "hour", "minute", "second"]:
        out[key] = int(dt.__getattribute__(key))
    return out


CORSIKA_PRIMARY_PATH = absjoin(
        "build",
        "corsika",
        "modified",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd")

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = absjoin(
        "build",
        "merlict",
        "merlict-plenoscope-propagation")

LIGHT_FIELD_GEOMETRY_PATH = absjoin(
        "run",
        "light_field_geometry")

EXAMPLE_PLENOSCOPE_SCENERY_PATH = absjoin(
        "resources",
        "acp",
        "71m",
        "scenery")

MERLICT_PLENOSCOPE_PROPAGATOR_CONFIG_PATH = absjoin(
        "resources",
        "acp",
        "merlict_propagation_config.json")

EXAMPLE_SITE = {
    "observation_level_asl_m": 5000,
    "earth_magnetic_field_x_muT": 20.815,
    "earth_magnetic_field_z_muT": -11.366,
    "atmosphere_id": 26,
}

EXAMPLE_PLENOSCOPE_POINTING = {
    "azimuth_deg": 0.,
    "zenith_deg": 0.
}

EXAMPLE_PARTICLE = {
    "particle_id": 14,
    "energy_bin_edges_GeV": [5, 100],
    "max_scatter_angle_deg": 30,
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
    "plenoscope_pointing": EXAMPLE_PLENOSCOPE_POINTING,
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
    "keep_tmp": True,
    "tmp_dir": EXAMPLE_WORK_DIR,
    "date": date_dict_now(),
}


def contains_same_bytes(path_a, path_b):
    with open(path_a, 'rb') as fa, open(path_b, 'rb') as fb:
        a_bytes = fa.read()
        b_bytes = fb.read()
        return a_bytes == b_bytes


def _cone_solid_angle(cone_radial_opening_angle_rad):
    cap_hight = (1.0 - np.cos(cone_radial_opening_angle_rad))
    return 2.0*np.pi*cap_hight


def ray_plane_x_y_intersection(support, direction, plane_z):
    direction = np.array(direction)
    support = np.array(support)
    direction_norm = direction/np.linalg.norm(direction)
    ray_parameter = -(support[2] - plane_z)/direction_norm[2]
    intersection = support + ray_parameter*direction_norm
    assert np.abs(intersection[2] - plane_z) < 1e-3
    return intersection


def draw_corsika_primary_steering(
    run_id=1,
    site=EXAMPLE_SITE,
    particle=EXAMPLE_PARTICLE,
    plenoscope_pointing=EXAMPLE_PLENOSCOPE_POINTING,
    num_events=100
):
    particle_id = particle["particle_id"]
    energy_bin_edges_GeV = particle["energy_bin_edges_GeV"]
    max_scatter_angle_deg = particle["max_scatter_angle_deg"]
    energy_power_law_slope = particle["energy_power_law_slope"]

    assert(run_id > 0)
    assert(np.all(np.diff(energy_bin_edges_GeV) >= 0))
    assert(len(energy_bin_edges_GeV) == 2)
    max_scatter_rad = np.deg2rad(max_scatter_angle_deg)
    assert(num_events <= table.MAX_NUM_EVENTS_IN_RUN)

    np.random.seed(run_id)
    energies = random.draw_power_law(
        lower_limit=np.min(energy_bin_edges_GeV),
        upper_limit=np.max(energy_bin_edges_GeV),
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
        az, zd = random.draw_azimuth_zenith_in_viewcone(
            azimuth_rad=np.deg2rad(plenoscope_pointing["azimuth_deg"]),
            zenith_rad=np.deg2rad(plenoscope_pointing["zenith_deg"]),
            min_scatter_opening_angle_rad=0.,
            max_scatter_opening_angle_rad=max_scatter_rad)
        primary["max_scatter_rad"] = max_scatter_rad
        primary["zenith_rad"] = zd
        primary["azimuth_rad"] = az
        primary["depth_g_per_cm2"] = 0.0
        primary["random_seed"] = cpw._simple_seed(
            table.random_seed_based_on(run_id=run_id, airshower_id=event_id))

        steering["primaries"].append(primary)
    return steering


def tar_append(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


def _append_trigger_truth(
    trigger_dict,
    trigger_responses,
    detector_truth,
):
    tr = trigger_dict
    tr["num_cherenkov_pe"] = int(detector_truth.number_air_shower_pulses())
    tr["response_pe"] = int(np.max(
        [layer['patch_threshold'] for layer in trigger_responses]))
    for o in range(len(trigger_responses)):
        tr["refocus_{:d}_object_distance_m".format(o)] = float(
            trigger_responses[o]['object_distance'])
        tr["refocus_{:d}_respnse_pe".format(o)] = int(
            trigger_responses[o]['patch_threshold'])
    return tr


def _append_bunch_ssize(cherenkovsise_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = cherenkovsise_dict
    ase["num_bunches"] = int(cb.shape[0])
    ase["num_photons"] = float(np.sum(cb[:, cpw.IBSIZE]))
    return ase


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    assert cb.shape[0] > 0
    ase["maximum_asl_m"] = float(cpw.CM2M*np.median(cb[:, cpw.IZEM]))
    ase["wavelength_median_nm"] = float(np.abs(np.median(cb[:, cpw.IWVL])))
    ase["cx_median_rad"] = float(np.median(cb[:, cpw.ICX]))
    ase["cy_median_rad"] = float(np.median(cb[:, cpw.ICY]))
    ase["x_median_m"] = float(cpw.CM2M*np.median(cb[:, cpw.IX]))
    ase["y_median_m"] = float(cpw.CM2M*np.median(cb[:, cpw.IY]))
    ase["bunch_size_median"] = float(np.median(cb[:, cpw.IBSIZE]))
    return ase


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir+".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


def safe_copy(src, dst):
    copy_id = uuid.uuid4().__str__()
    tmp_dst = "{:s}.{:s}.tmp".format(dst, copy_id)
    try:
        shutil.copytree(src, tmp_dst)
    except OSError as exc:
        if exc.errno == errno.ENOTDIR:
            shutil.copy2(src, tmp_dst)
        else:
            raise
    os.rename(tmp_dst, dst)


def safe_move(src, dst):
    try:
        os.rename(src, dst)
    except OSError as err:
        if err.errno == errno.EXDEV:
            safe_copy(src, dst)
            os.unlink(src)
        else:
            raise


def run_job(job=EXAMPLE_JOB):
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)
    run_id_str = "{:06d}".format(job["run_id"])
    time_log_path = op.join(job["log_dir"], run_id_str+"_runtime.jsonl")
    logger = logging.JsonlLog(time_log_path+".tmp")
    job_path = op.join(job["log_dir"], run_id_str+"_job.json")
    with open(job_path+".tmp", "wt") as f:
        f.write(json.dumps(job, indent=4))
    safe_move(job_path+".tmp", job_path)
    print('{{"run_id": {:d}"}}\n'.format(job["run_id"]))

    # assert resources exist
    # ----------------------
    assert op.exists(job["corsika_primary_path"])
    assert op.exists(job["merlict_plenoscope_propagator_path"])
    assert op.exists(job["merlict_plenoscope_propagator_config_path"])
    assert op.exists(job["plenoscope_scenery_path"])
    assert op.exists(job["light_field_geometry_path"])
    logger.log("assert_resource_paths_exist.")

    # set up plenoscope grid
    # ----------------------
    assert job["plenoscope_pointing"]["zenith_deg"] == 0.
    assert job["plenoscope_pointing"]["azimuth_deg"] == 0.
    plenoscope_pointing_direction = np.array([0, 0, 1])  # For now this is fix.

    _scenery_path = op.join(job["plenoscope_scenery_path"], "scenery.json")
    _light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        merlict_scenery_path=_scenery_path)
    plenoscope_diameter = 2.0*_light_field_sensor_geometry[
        "expected_imaging_system_aperture_radius"]
    plenoscope_radius = .5*plenoscope_diameter
    plenoscope_field_of_view_radius_deg = 0.5*_light_field_sensor_geometry[
        "max_FoV_diameter_deg"]
    plenoscope_grid_geometry = grid.init(
        plenoscope_diameter=plenoscope_diameter,
        num_bins_radius=job["grid"]["num_bins_radius"])
    logger.log("init_plenoscope_grid")

    # draw primaries
    # --------------
    corsika_primary_steering = draw_corsika_primary_steering(
        run_id=job["run_id"],
        site=job["site"],
        particle=job["particle"],
        plenoscope_pointing=job["plenoscope_pointing"],
        num_events=job["num_air_showers"])
    logger.log("draw_primaries")

    # tmp dir
    # -------
    if job['tmp_dir'] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job['tmp_dir'], run_id_str)
        os.makedirs(tmp_dir, exist_ok=True)
    logger.log("make_temp_dir:'{:s}'".format(tmp_dir))

    # run corsika
    # -----------
    corsika_run_path = op.join(tmp_dir, run_id_str+"_corsika.tar")
    if not op.exists(corsika_run_path):
        cpw_rc = cpw.corsika_primary(
            corsika_path=job["corsika_primary_path"],
            steering_dict=corsika_primary_steering,
            output_path=corsika_run_path,
            stdout_postfix=".stdout",
            stderr_postfix=".stderr")
        safe_copy(
            corsika_run_path+".stdout",
            op.join(job["log_dir"], run_id_str+"_corsika.stdout"))
        safe_copy(
            corsika_run_path+".stderr",
            op.join(job["log_dir"], run_id_str+"_corsika.stderr"))
        logger.log("corsika")
    with open(corsika_run_path+".stdout", "rt") as f:
        assert cpw.stdout_ends_with_end_of_run_marker(f.read())
    logger.log("assert_corsika_ok")
    corsika_run_size = os.stat(corsika_run_path).st_size
    logger.log("corsika_run_size:{:d}".format(corsika_run_size))

    # loop over air-showers
    # ---------------------
    evttab = {}
    for level_key in table.CONFIG_LEVELS_KEYS:
        evttab[level_key] = []
    run = cpw.Tario(corsika_run_path)
    reuse_run_path = op.join(tmp_dir, run_id_str+"_reuse.tar")
    grid_histogram_filename = run_id_str+"_grid.tar"
    tmp_grid_histogram_path = op.join(tmp_dir, grid_histogram_filename)
    with tarfile.open(reuse_run_path, "w") as tarout,\
            tarfile.open(tmp_grid_histogram_path, "w") as imgtar:
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
            assert (event_seed == table.random_seed_based_on(
                run_id=run_id,
                airshower_id=event_id))

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
            prim["max_scatter_rad"] = float(primary["max_scatter_rad"])
            prim["solid_angle_thrown_sr"] = float(_cone_solid_angle(
                prim["max_scatter_rad"]))
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
                -1.0*cpw.CM2M *
                cpw._evth_z_coordinate_of_first_interaction_cm(
                    event_header))
            prim["starting_height_asl_m"] = float(
                cpw.CM2M*cpw._evth_starting_height_cm(event_header))
            obs_lvl_intersection = ray_plane_x_y_intersection(
                support=[0, 0, prim["starting_height_asl_m"]],
                direction=[
                    prim["momentum_x_GeV_per_c"],
                    prim["momentum_y_GeV_per_c"],
                    prim["momentum_z_GeV_per_c"]],
                plane_z=job["site"]["observation_level_asl_m"])
            prim["starting_x_m"] = -float(obs_lvl_intersection[0])
            prim["starting_y_m"] = -float(obs_lvl_intersection[1])
            evttab["primary"].append(prim)

            # cherenkov size
            # --------------
            crsz = ide.copy()
            crsz = _append_bunch_ssize(crsz, cherenkov_bunches)
            evttab["cherenkovsize"].append(crsz)

            # assign grid
            # -----------
            grid_result = grid.assign(
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
                file_name="{:06d}{:06d}.f4.gz".format(run_id, event_id),
                file_bytes=grid.histogram_to_bytes(
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
            grhi["area_thrown_m2"] = float(plenoscope_grid_geometry[
                "total_area"])
            evttab["grid"].append(grhi)

            # cherenkov statistics
            # --------------------
            if cherenkov_bunches.shape[0] > 0:
                fase = ide.copy()
                fase = _append_bunch_statistics(
                    airshower_dict=fase,
                    cherenkov_bunches=cherenkov_bunches)
                evttab["cherenkovpool"].append(fase)

            reuse_event = grid_result["random_choice"]
            if reuse_event is not None:
                IEVTH_NUM_REUSES = 98-1
                IEVTH_CORE_X = IEVTH_NUM_REUSES + 1
                IEVTH_CORE_Y = IEVTH_NUM_REUSES + 11
                reuse_evth = event_header.copy()
                reuse_evth[IEVTH_NUM_REUSES] = 1.0
                reuse_evth[IEVTH_CORE_X] = cpw.M2CM*reuse_event["core_x_m"]
                reuse_evth[IEVTH_CORE_Y] = cpw.M2CM*reuse_event["core_y_m"]
                tar_append(
                    tarout=tarout,
                    file_name=cpw.TARIO_EVTH_FILENAME.format(event_id),
                    file_bytes=reuse_evth.tobytes())
                tar_append(
                    tarout=tarout,
                    file_name=cpw.TARIO_BUNCHES_FILENAME.format(event_id),
                    file_bytes=reuse_event["cherenkov_bunches"].tobytes())
                crszp = ide.copy()
                crszp = _append_bunch_ssize(crszp, cherenkov_bunches)
                evttab["cherenkovsizepart"].append(crszp)
                rase = ide.copy()
                rase = _append_bunch_statistics(
                    airshower_dict=rase,
                    cherenkov_bunches=reuse_event["cherenkov_bunches"])
                evttab["cherenkovpoolpart"].append(rase)
                rcor = ide.copy()
                rcor["bin_idx_x"] = int(reuse_event["bin_idx_x"])
                rcor["bin_idx_y"] = int(reuse_event["bin_idx_y"])
                rcor["core_x_m"] = float(reuse_event["core_x_m"])
                rcor["core_y_m"] = float(reuse_event["core_y_m"])
                evttab["core"].append(rcor)
    logger.log("grid")

    if not job["keep_tmp"]:
        os.remove(corsika_run_path)

    # run merlict
    # -----------
    merlict_run_path = op.join(tmp_dir, run_id_str+"_merlict.cp")
    if not op.exists(merlict_run_path):
        merlict_rc = merlict.plenoscope_propagator(
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
        safe_copy(
            merlict_run_path+".stdout",
            op.join(job["log_dir"], run_id_str+"_merlict.stdout"))
        safe_copy(
            merlict_run_path+".stderr",
            op.join(job["log_dir"], run_id_str+"_merlict.stderr"))
        assert(merlict_rc == 0)
    logger.log("merlict")

    if not job["keep_tmp"]:
        os.remove(reuse_run_path)

    # prepare trigger
    # ---------------
    merlict_run = pl.Run(merlict_run_path)
    trigger_preparation = pl.trigger.prepare_refocus_sum_trigger(
        light_field_geometry=merlict_run.light_field_geometry,
        object_distances=job["sum_trigger"]["object_distances"])
    logger.log("prepare_trigger")

    # loop over sensor responses
    # --------------------------
    table_past_trigger = []
    tmp_past_trigger_dir = op.join(tmp_dir, "past_trigger")
    os.makedirs(tmp_past_trigger_dir, exist_ok=True)

    for event in merlict_run:
        # id
        # --
        cevth = event.simulation_truth.event.corsika_event_header.raw
        run_id = int(cpw._evth_run_number(cevth))
        airshower_id = int(cpw._evth_event_number(cevth))
        ide = {"run_id": run_id, "airshower_id": airshower_id}

        # apply trigger
        # -------------
        trigger_responses = pl.trigger.apply_refocus_sum_trigger(
            event=event,
            trigger_preparation=trigger_preparation,
            min_number_neighbors=job["sum_trigger"]["min_num_neighbors"],
            integration_time_in_slices=(
                job["sum_trigger"]["integration_time_in_slices"]))
        trg_resp_path = op.join(event._path, "refocus_sum_trigger.json")
        with open(trg_resp_path, "wt") as f:
            f.write(json.dumps(trigger_responses, indent=4))

        # export trigger-truth
        # --------------------
        trgtru = ide.copy()
        trgtru = _append_trigger_truth(
            trigger_dict=trgtru,
            trigger_responses=trigger_responses,
            detector_truth=event.simulation_truth.detector)
        evttab["trigger"].append(trgtru)

        # passing trigger
        # ---------------
        if (trgtru["response_pe"] >= job["sum_trigger"]["patch_threshold"]):
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp["unique_id_str"] = '{run_id:06d}{airshower_id:06d}'.format(
                run_id=ptp["run_id"],
                airshower_id=ptp["airshower_id"])
            table_past_trigger.append(ptp)

            # export past trigger
            # -------------------
            ptrg = ide.copy()
            evttab["pasttrigger"].append(ptrg)
    logger.log("trigger")

    # Cherenkov classification
    # ------------------------
    for pt in table_past_trigger:
        event = pl.Event(
            path=pt["tmp_path"],
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
            event_path=op.abspath(event._path),
            photon_ids=cherenkov_photons.photon_ids,
            settings=roi_settings)
        crcl = pl.classify.benchmark(
            pulse_origins=event.simulation_truth.detector.pulse_origins,
            photon_ids_cherenkov=cherenkov_photons.photon_ids)
        crcl["run_id"] = pt["run_id"]
        crcl["airshower_id"] = pt["airshower_id"]
        evttab["cherenkovclassification"].append(crcl)
    logger.log("cherenkov_classification")

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
    logger.log("light_field_geometry_addons")

    for pt in table_past_trigger:
        event = pl.Event(path=pt["tmp_path"], light_field_geometry=lfg)
        try:
            lfft = pl.features.extract_features(
                cherenkov_photons=event.cherenkov_photons,
                light_field_geometry=lfg,
                light_field_geometry_addon=lfg_addon)
            lfft["run_id"] = pt["run_id"]
            lfft["airshower_id"] = pt["airshower_id"]
            evttab["features"].append(lfft)
        except Exception as excep:
            print(
                "run_id {:d}, airshower_id: {:d}:".format(
                    pt["run_id"],
                    pt["airshower_id"]),
                excep)
    logger.log("feature_extraction")

    # compress and tar
    # ----------------
    for pt in table_past_trigger:
        pl.tools.acp_format.compress_event_in_place(pt["tmp_path"])
        final_tarname = pt["unique_id_str"]+'.tar'
        plenoscope_event_dir_to_tar(
            event_dir=pt["tmp_path"],
            output_tar_path=op.join(tmp_past_trigger_dir, final_tarname))
    logger.log("past_trigger_gz_tar")

    # export event-table
    # ------------------
    table_filename = run_id_str+"_event_table.tar"
    with tarfile.open(op.join(tmp_dir, table_filename+".tmp"), "w") as tarfout:
        for level_key in table.CONFIG_LEVELS_KEYS:
            level_csv = table.level_records_to_csv(
                level_records=evttab[level_key],
                config=table.CONFIG,
                level=level_key)
            with io.BytesIO() as buff:
                level_csv_bytes = str.encode(level_csv)
                buff.write(level_csv_bytes)
                buff.seek(0)
                tarinfo = tarfile.TarInfo()
                tarinfo.name = level_key+"."+table.FORMAT_SUFFIX
                tarinfo.size = len(buff.getvalue())
                tarfout.addfile(tarinfo=tarinfo, fileobj=buff)
    safe_move(
        src=op.join(tmp_dir, table_filename+".tmp"),
        dst=op.join(tmp_dir, table_filename))
    safe_copy(
        src=op.join(tmp_dir, table_filename),
        dst=op.join(job["feature_dir"], table_filename))
    logger.log("export_event_table")

    # export grid histograms
    # ----------------------
    safe_copy(
        src=tmp_grid_histogram_path,
        dst=op.join(job["feature_dir"], grid_histogram_filename))
    logger.log("export_grid_histograms")

    # export past trigger
    # -------------------
    for pt in table_past_trigger:
        final_tarname = pt["unique_id_str"]+'.tar'
        safe_copy(
            src=op.join(tmp_past_trigger_dir, final_tarname),
            dst=op.join(job["past_trigger_dir"], final_tarname))
    logger.log("export_past_trigger")

    # end
    # ---
    logger.log("end")
    safe_move(time_log_path+".tmp", time_log_path)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)
