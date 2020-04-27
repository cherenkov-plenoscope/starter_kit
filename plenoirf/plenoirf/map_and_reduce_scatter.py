from . import table
from . import grid
from . import merlict
from . import logging
from . import network_file_system as nfs

import numpy as np
import os
from os import path as op
import shutil

import tempfile
import json
import tarfile
import io
import datetime
import subprocess
import PIL
import pandas as pd
import corsika_primary_wrapper as cpw
import magnetic_deflection as mdfl
import plenopy as pl
import gzip
import sparse_table as spt


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
        "2020-04-08_run",
        "light_field_geometry")

EXAMPLE_SITE_PARTICLE_DEFLECTION_PATH = absjoin(
        "2020-04-08_run",
        "magnetic_deflection",
        "result",
        "chile_electron.csv")

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
    "particle_id": 3,
    "energy_bin_edges_GeV": [ 0.5,    1,   10,  100, 1000],
    "max_scatter_radius_m": [3000, 3000, 2000, 1200, 1200],
    "max_scatter_angle_deg": 6.5,
    "energy_power_law_slope": -2.0,
}

EXAMPLE_SITE_PARTICLE_DEFLECTION = {
    "energy_GeV": [5, 1000],
    "primary_azimuth_deg": [0.0, 0.0],
    "primary_zenith_deg": [0.0, 0.0],
    "cherenkov_pool_x_m": [0.0, 0.0],
    "cherenkov_pool_y_m": [0.0, 0.0],
}

EXAMPLE_SITE_PARTICLE_DEFLECTION = pd.read_csv(
    EXAMPLE_SITE_PARTICLE_DEFLECTION_PATH
).to_dict(orient='list')

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

EXAMPLE_LOG_DIRECTORY = op.join(".", "_log")
EXAMPLE_PAST_TRIGGER_DIRECTORY = op.join(".", "_past_trigger")
EXAMPLE_FEATURE_DIRECTORY = op.join(".", "_features")
EXAMPLE_WORK_DIR = op.join(".", "_work")

EXAMPLE_JOB = {
    "run_id": 2,
    "num_air_showers": 999,
    "particle": EXAMPLE_PARTICLE,
    "site_particle_deflection": EXAMPLE_SITE_PARTICLE_DEFLECTION,
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


def _assert_deflection(site_particle_deflection):
    for example_key in EXAMPLE_SITE_PARTICLE_DEFLECTION:
        assert example_key in site_particle_deflection
    assert len(site_particle_deflection["energy_GeV"]) >= 2
    for _key in site_particle_deflection:
        assert (
            len(site_particle_deflection["energy_GeV"]) ==
            len(site_particle_deflection[_key]))
    for energy in site_particle_deflection["energy_GeV"]:
        assert energy > 0.
    for zenith_deg in site_particle_deflection["primary_zenith_deg"]:
        assert zenith_deg >= 0.
    assert(np.all(np.diff(site_particle_deflection["energy_GeV"]) >= 0))


def _assert_site(site):
    for _key in EXAMPLE_SITE:
        assert _key in site


def _assert_particle(particle):
    for _key in EXAMPLE_PARTICLE:
        assert _key in particle
    assert(np.all(np.diff(particle["energy_bin_edges_GeV"]) >= 0))
    assert(
        len(particle["energy_bin_edges_GeV"]) ==
        len(particle["max_scatter_radius_m"])
    )


def draw_corsika_primary_steering(
    run_id=1,
    site=EXAMPLE_SITE,
    particle=EXAMPLE_PARTICLE,
    site_particle_deflection=EXAMPLE_SITE_PARTICLE_DEFLECTION,
    num_events=100
):
    assert table.is_valid_run_id(run_id)
    _assert_site(site)
    _assert_particle(particle)
    _assert_deflection(site_particle_deflection)
    assert num_events <= table.NUM_AIRSHOWER_IDS_IN_RUN

    max_scatter_rad = np.deg2rad(particle["max_scatter_angle_deg"])

    min_common_energy = np.max([
        np.min(particle["energy_bin_edges_GeV"]),
        np.min(site_particle_deflection["energy_GeV"])
    ])

    np.random.seed(run_id)
    energies = cpw.random_distributions.draw_power_law(
        lower_limit=min_common_energy,
        upper_limit=np.max(particle["energy_bin_edges_GeV"]),
        power_slope=particle["energy_power_law_slope"],
        num_samples=num_events)

    steering = {}
    steering["run"] = {}
    steering["run"]["run_id"] = int(run_id)
    steering["run"]["event_id_of_first_event"] = 1

    for key in site:
        steering["run"][key] = site[key]

    steering["primaries"] = []
    for e in range(energies.shape[0]):
        primary = {}
        primary["particle_id"] = particle["particle_id"]
        primary["energy_GeV"] = energies[e]

        # magnetic deflection
        primary["magnet_azimuth_rad"] = np.deg2rad(np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["primary_azimuth_deg"]))
        primary["magnet_zenith_rad"] = np.deg2rad(np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["primary_zenith_deg"]))
        primary["magnet_cherenkov_pool_x_m"] = np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["cherenkov_pool_x_m"])
        primary["magnet_cherenkov_pool_y_m"] = np.interp(
            x=primary["energy_GeV"],
            xp=site_particle_deflection["energy_GeV"],
            fp=site_particle_deflection["cherenkov_pool_y_m"])

        # direction
        az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
            azimuth_rad=primary["magnet_azimuth_rad"],
            zenith_rad=primary["magnet_zenith_rad"],
            min_scatter_opening_angle_rad=0.,
            max_scatter_opening_angle_rad=max_scatter_rad)
        primary["max_scatter_rad"] = max_scatter_rad
        primary["zenith_rad"] = zd
        primary["azimuth_rad"] = az

        # position
        primary["max_scatter_radius_wrt_cherenkov_pool_m"] = np.interp(
            x=primary["energy_GeV"],
            xp=particle["energy_bin_edges_GeV"],
            fp=particle["max_scatter_radius_m"])
        core_x, core_y = cpw.random_distributions.draw_x_y_in_disc(
            radius=primary["max_scatter_radius_wrt_cherenkov_pool_m"])
        primary["scatter_wrt_cherenkov_pool_x_m"] = core_x
        primary["scatter_wrt_cherenkov_pool_y_m"] = core_y

        # depth
        primary["depth_g_per_cm2"] = 0.0

        # seed
        primary["random_seed"] = cpw.simple_seed(
            table.random_seed_based_on(
                run_id=run_id,
                airshower_id=e + 1))

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


def _append_bunch_size(cherenkovsise_dict, cherenkov_bunches):
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


STRUCTURE = {}

STRUCTURE['primary'] = {
    'particle_id': {'dtype': '<i8', 'comment': 'CORSIKA particle-id'},
    'energy_GeV': {'dtype': '<f8', 'comment': ''},
    'azimuth_rad': {'dtype': '<f8', 'comment':
        'Direction of the primary particle w.r.t. magnetic north.'},
    'zenith_rad': {'dtype': '<f8', 'comment':
        'Direction of the primary particle.'},

    'magnet_azimuth_rad': {'dtype': '<f8', 'comment':
        'The azimuth direction that the primary particle needs to have '
        'in order to induce an air-shower that emits its Cherenkov-light '
        'head on the pointing of the plenoscope.'},
    'magnet_zenith_rad': {'dtype': '<f8', 'comment':
        'The zenith direction that the primary particle needs to have '
        'in order to induce an air-shower that emits its Cherenkov-light '
        'head on the pointing of the plenoscope.'},
    'magnet_cherenkov_pool_x_m': {'dtype': '<f8', 'comment':
        'This offset must be added to the core-position, where '
        'the trajectory of the primary particle intersects the '
        'observation-level, in order for the plenoscope to stand in '
        'the typical center of the Cherenkov-pool.'},
    'magnet_cherenkov_pool_y_m': {'dtype': '<f8', 'comment': ''},

    'depth_g_per_cm2': {'dtype': '<f8', 'comment': ''},
    'momentum_x_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'momentum_y_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'momentum_z_GeV_per_c': {'dtype': '<f8', 'comment': ''},
    'first_interaction_height_asl_m': {'dtype': '<f8', 'comment': ''},
    'starting_height_asl_m': {'dtype': '<f8', 'comment': ''},
    'starting_x_m': {'dtype': '<f8', 'comment': ''},
    'starting_y_m': {'dtype': '<f8', 'comment': ''},

    'max_scatter_rad': {'dtype': '<f8', 'comment': ''},
    'solid_angle_thrown_sr': {'dtype': '<f8', 'comment': ''},

    'max_scatter_radius_wrt_cherenkov_pool_m': {'dtype': '<f8', 'comment': ''},
    'scatter_wrt_cherenkov_pool_x_m': {'dtype': '<f8', 'comment': ''},
    'scatter_wrt_cherenkov_pool_y_m': {'dtype': '<f8', 'comment': ''},
    'area_thrown_m2': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovsize'] = {
    'num_bunches': {'dtype': '<i8', 'comment': ''},
    'num_photons': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovpool'] = {
    'maximum_asl_m': {'dtype': '<f8', 'comment': ''},
    'wavelength_median_nm': {'dtype': '<f8', 'comment': ''},
    'cx_median_rad': {'dtype': '<f8', 'comment': ''},
    'cy_median_rad': {'dtype': '<f8', 'comment': ''},
    'x_median_m': {'dtype': '<f8', 'comment': ''},
    'y_median_m': {'dtype': '<f8', 'comment': ''},
    'bunch_size_median': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['cherenkovsizepart'] = STRUCTURE['cherenkovsize'].copy()
STRUCTURE['cherenkovpoolpart'] = STRUCTURE['cherenkovpool'].copy()

STRUCTURE['cherenkovpoolcore'] = {
    'x_m': {'dtype': '<i8', 'comment': ''},
    'y_m': {'dtype': '<f8', 'comment': ''},
}

STRUCTURE['trigger'] = {
    'num_cherenkov_pe': {'dtype': '<i8', 'comment': ''},
    'response_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_0_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_0_respnse_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_1_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_1_respnse_pe': {'dtype': '<i8', 'comment': ''},
    'refocus_2_object_distance_m': {'dtype': '<f8', 'comment': ''},
    'refocus_2_respnse_pe': {'dtype': '<i8', 'comment': ''},
}

STRUCTURE['pasttrigger'] = {}

STRUCTURE['cherenkovclassification'] = {
    'num_true_positives': {'dtype': '<i8', 'comment': ''},
    'num_false_negatives': {'dtype': '<i8', 'comment': ''},
    'num_false_positives': {'dtype': '<i8', 'comment': ''},
    'num_true_negatives': {'dtype': '<i8', 'comment': ''},
}

STRUCTURE['features'] = {
    'num_photons': {'dtype': '<i8', 'comment': ''},
    'paxel_intensity_peakness_std_over_mean': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_peakness_max_over_mean': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_median_x': {'dtype': '<f8', 'comment': ''},
    'paxel_intensity_median_y': {'dtype': '<f8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_2':
        {'dtype': '<i8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_4':
        {'dtype': '<i8', 'comment': ''},
    'aperture_num_islands_watershed_rel_thr_8':
        {'dtype': '<i8', 'comment': ''},
    'light_front_cx': {'dtype': '<f8', 'comment': ''},
    'light_front_cy': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cx_mean': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cy_mean': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cx_std': {'dtype': '<f8', 'comment': ''},
    'image_infinity_cy_std': {'dtype': '<f8', 'comment': ''},
    'image_infinity_num_photons_on_edge_field_of_view':
        {'dtype': '<i8', 'comment': ''},
    'image_smallest_ellipse_object_distance': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_solid_angle': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_half_depth': {'dtype': '<f8', 'comment': ''},
    'image_half_depth_shift_cx': {'dtype': '<f8', 'comment': ''},
    'image_half_depth_shift_cy': {'dtype': '<f8', 'comment': ''},
    'image_smallest_ellipse_num_photons_on_edge_field_of_view':
        {'dtype': '<i8', 'comment': ''},
    'image_num_islands': {'dtype': '<i8', 'comment': ''},
}


def run_job(job=EXAMPLE_JOB):
    run_id_str = "{:06d}".format(job["run_id"])
    print('{{"run_id": {:d}"}}\n'.format(job["run_id"]))

    # create output dirs
    # ------------------
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)

    # export job instruction
    # ----------------------
    job_path = op.join(job["log_dir"], run_id_str+"_job.json")
    with open(job_path+".tmp", "wt") as f:
        f.write(json.dumps(job, indent=4))
    nfs.move(job_path+".tmp", job_path)

    # logger
    # ------
    time_log_path = op.join(job["log_dir"], run_id_str+"_runtime.jsonl")
    logger = logging.JsonlLog(time_log_path+".tmp")

    # assert resources exist
    # ----------------------
    assert op.exists(job["corsika_primary_path"])
    assert op.exists(job["merlict_plenoscope_propagator_path"])
    assert op.exists(job["merlict_plenoscope_propagator_config_path"])
    assert op.exists(job["plenoscope_scenery_path"])
    assert op.exists(job["light_field_geometry_path"])
    logger.log("assert_resource_paths_exist.")

    # set up plenoscope
    # -----------------
    _scenery_path = op.join(job["plenoscope_scenery_path"], "scenery.json")
    _light_field_sensor_geometry = merlict.read_plenoscope_geometry(
        merlict_scenery_path=_scenery_path)
    assert job["plenoscope_pointing"]["zenith_deg"] == 0.
    assert job["plenoscope_pointing"]["azimuth_deg"] == 0.

    plenoscope = {}
    plenoscope['zenith_deg'] = job["plenoscope_pointing"]["zenith_deg"]
    plenoscope['azimuth_deg'] = job["plenoscope_pointing"]["azimuth_deg"]
    plenoscope['pointing_direction_wrt_root'] = np.array([0, 0, 1])
    plenoscope['radius'] = _light_field_sensor_geometry[
        "expected_imaging_system_aperture_radius"]
    plenoscope['diameter'] = 2.0*plenoscope['radius']
    plenoscope['field_of_view_radius_deg'] = 0.5*_light_field_sensor_geometry[
        "max_FoV_diameter_deg"]
    logger.log("init_plenoscope")

    # set up grid
    # -----------
    grid_geometry = grid.init(
        plenoscope_diameter=plenoscope['diameter'],
        num_bins_radius=job["grid"]["num_bins_radius"])

    # draw primaries
    # --------------
    corsika_primary_steering = draw_corsika_primary_steering(
        run_id=job["run_id"],
        site=job["site"],
        particle=job["particle"],
        site_particle_deflection=job["site_particle_deflection"],
        num_events=job["num_air_showers"])
    logger.log("draw_primaries")

    # make tmp dir
    # ------------
    if job['tmp_dir'] is None:
        tmp_dir = tempfile.mkdtemp(prefix="plenoscope_irf_")
    else:
        tmp_dir = op.join(job['tmp_dir'], run_id_str)
        os.makedirs(tmp_dir, exist_ok=True)
    logger.log("make_temp_dir:'{:s}'".format(tmp_dir))

    # prepare output table
    # --------------------
    tabrec = {}
    for level_key in STRUCTURE:
        tabrec[level_key] = []

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
        nfs.copy(
            corsika_run_path+".stdout",
            op.join(job["log_dir"], run_id_str+"_corsika.stdout"))
        nfs.copy(
            corsika_run_path+".stderr",
            op.join(job["log_dir"], run_id_str+"_corsika.stderr"))
        logger.log("corsika")
    with open(corsika_run_path+".stdout", "rt") as f:
        assert cpw.stdout_ends_with_end_of_run_marker(f.read())
    logger.log("assert_corsika_ok")
    corsika_run_size = os.stat(corsika_run_path).st_size
    logger.log("corsika_run_size:{:d}".format(corsika_run_size))

    # loop over raw air-showers
    # -------------------------
    corsika_run_part_filename = run_id_str+"_corsika_part.tar"
    tmp_corsika_run_part_path = op.join(tmp_dir, corsika_run_part_filename)
    grid_histogram_filename = run_id_str+"_grid.tar"
    tmp_grid_histogram_path = op.join(tmp_dir, grid_histogram_filename)

    corsika_run = cpw.Tario(corsika_run_path)
    run_part_tarfile = tarfile.open(tmp_corsika_run_part_path, "w")
    grid_tarfile = tarfile.open(tmp_grid_histogram_path, "w")

    tar_append(
        tarout=run_part_tarfile,
        file_name=cpw.TARIO_RUNH_FILENAME,
        file_bytes=corsika_run.runh.tobytes())

    for airshower_idx, corsika_airshower in enumerate(corsika_run):
        event_header, raw_cherenkov_bunches = corsika_airshower

        # assert corsika_airshower matches primary
        # ----------------------------------------
        run_id = int(event_header[cpw.I_EVTH_RUN_NUMBER])
        airshower_id = int(event_header[cpw.I_EVTH_EVENT_NUMBER])
        airshower_seed = table.random_seed_based_on(
            run_id=run_id,
            airshower_id=airshower_id)

        primary = corsika_primary_steering["primaries"][airshower_idx]

        assert run_id == corsika_primary_steering["run"]["run_id"]
        assert airshower_id == airshower_idx + 1
        assert airshower_seed == primary["random_seed"][0]["SEED"]

        print("--airshower--", airshower_seed)

        # export primary table
        # --------------------
        prim = {}
        prim[spt.IDX] = airshower_seed
        prim["particle_id"] = int(primary["particle_id"])
        prim["energy_GeV"] = float(primary["energy_GeV"])
        prim["azimuth_rad"] = float(primary["azimuth_rad"])
        prim["zenith_rad"] = float(primary["zenith_rad"])
        prim["max_scatter_rad"] = float(primary["max_scatter_rad"])
        prim["solid_angle_thrown_sr"] = float(_cone_solid_angle(
            prim["max_scatter_rad"]))
        prim["depth_g_per_cm2"] = float(primary["depth_g_per_cm2"])
        prim["momentum_x_GeV_per_c"] = float(
            event_header[cpw.I_EVTH_PX_MOMENTUM_GEV_PER_C])
        prim["momentum_y_GeV_per_c"] = float(
            event_header[cpw.I_EVTH_PY_MOMENTUM_GEV_PER_C])
        prim["momentum_z_GeV_per_c"] = float(
            -1.0*event_header[cpw.I_EVTH_PZ_MOMENTUM_GEV_PER_C])
        prim["first_interaction_height_asl_m"] = float(
            -1.0*cpw.CM2M*event_header[
                cpw.I_EVTH_Z_FIRST_INTERACTION_CM])
        prim["starting_height_asl_m"] = float(
            cpw.CM2M*event_header[cpw.I_EVTH_STARTING_HEIGHT_CM])
        obs_lvl_intersection = ray_plane_x_y_intersection(
            support=[0, 0, prim["starting_height_asl_m"]],
            direction=[
                prim["momentum_x_GeV_per_c"],
                prim["momentum_y_GeV_per_c"],
                prim["momentum_z_GeV_per_c"]],
            plane_z=job["site"]["observation_level_asl_m"])
        prim["starting_x_m"] = -float(obs_lvl_intersection[0])
        prim["starting_y_m"] = -float(obs_lvl_intersection[1])
        prim["magnet_azimuth_rad"] = float(primary["magnet_azimuth_rad"])
        prim["magnet_zenith_rad"] = float(primary["magnet_zenith_rad"])
        prim["magnet_cherenkov_pool_x_m"] = float(
            primary["magnet_cherenkov_pool_x_m"])
        prim["magnet_cherenkov_pool_y_m"] = float(
            primary["magnet_cherenkov_pool_y_m"])

        prim["max_scatter_radius_wrt_cherenkov_pool_m"] = float(
            primary["max_scatter_radius_wrt_cherenkov_pool_m"])
        prim["scatter_wrt_cherenkov_pool_x_m"] = float(
            primary["scatter_wrt_cherenkov_pool_x_m"])
        prim["scatter_wrt_cherenkov_pool_y_m"] = float(
            primary["scatter_wrt_cherenkov_pool_y_m"])
        prim["area_thrown_m2"] = float(
            np.pi*prim["max_scatter_radius_wrt_cherenkov_pool_m"]**2
        )

        tabrec["primary"].append(prim)

        # cherenkov size of full airshower
        # ----------------------------------
        crsz = {}
        crsz[spt.IDX] = airshower_seed
        crsz = _append_bunch_size(crsz, raw_cherenkov_bunches)
        tabrec["cherenkovsize"].append(crsz)
        print("full", crsz['num_bunches'])

        if crsz['num_bunches'] < 50:
            continue

        crst = {}
        crst[spt.IDX] = airshower_seed
        crst = _append_bunch_statistics(crst, raw_cherenkov_bunches)
        tabrec["cherenkovpool"].append(crst)


        # reject cherenkov-bunches outside of fov
        # ---------------------------------------
        OVERHEAD = 1.1
        bunch_directions = grid._make_bunch_direction(
            cx=raw_cherenkov_bunches[:, cpw.ICX],
            cy=raw_cherenkov_bunches[:, cpw.ICY]
        )
        bunch_incidents = -1.0*bunch_directions
        angle_bunch_pointing = grid._make_angle_between(
            directions=bunch_incidents,
            direction=plenoscope['pointing_direction_wrt_root']
        )
        mask_inside_field_of_view = (
            angle_bunch_pointing <
            np.deg2rad(plenoscope['field_of_view_radius_deg'])*OVERHEAD
        )
        bunches_fov = raw_cherenkov_bunches[mask_inside_field_of_view, :]

        if bunches_fov.shape[0] < 50:
            continue

        # translate cherenkov-pool to x=0, y=0 wrt. obs.-level
        # ----------------------------------------------------
        bunches_fov_x_median_cm = np.median(bunches_fov[:, cpw.IX])
        bunches_fov_y_median_cm = np.median(bunches_fov[:, cpw.IY])
        bunches_fov[:, cpw.IX] -= bunches_fov_x_median_cm
        bunches_fov[:, cpw.IY] -= bunches_fov_y_median_cm

        cercore = {}
        cercore[spt.IDX] = airshower_seed
        cercore["x_m"] = 1e-2*bunches_fov_x_median_cm
        cercore["y_m"] = 1e-2*bunches_fov_y_median_cm
        tabrec["cherenkovpoolcore"].append(cercore)

        # intensity histogram on observation-level
        # ----------------------------------------
        intensity_histogram = np.histogram2d(
            x=1e-2*bunches_fov[:, cpw.IX],
            y=1e-2*bunches_fov[:, cpw.IY],
            bins=(
                grid_geometry['xy_bin_edges'],
                grid_geometry['xy_bin_edges']
            )
        )[0]
        tar_append(
            tarout=grid_tarfile,
            file_name=table.SEED_TEMPLATE_STR.format(
                seed=airshower_seed)+".f4.gz",
            file_bytes=grid.histogram_to_bytes(intensity_histogram)
        )

        # add scatter
        # -----------
        bunches_fov[:, cpw.IX] += 1e2*primary["scatter_wrt_cherenkov_pool_x_m"]
        bunches_fov[:, cpw.IY] += 1e2*primary["scatter_wrt_cherenkov_pool_y_m"]

        # inside aperture
        # ---------------
        bunch_x_m = 1e-2*bunches_fov[:, cpw.IX]
        bunch_y_m = 1e-2*bunches_fov[:, cpw.IY]
        bunch_radius_m = np.hypot(bunch_x_m, bunch_y_m)
        mask_inside_aperture = bunch_radius_m < plenoscope['radius']*OVERHEAD

        # cut relevant part for plenoscope
        # --------------------------------
        bunches_part = bunches_fov[mask_inside_aperture]

        # cherenkov size of airshower part
        # --------------------------------
        crszp = {}
        crszp[spt.IDX] = airshower_seed
        crszp = _append_bunch_size(crszp, bunches_part)
        tabrec["cherenkovsizepart"].append(crszp)
        print("part", crszp['num_bunches'])

        # pre trigger
        # -----------
        if crszp['num_photons'] > job["grid"]["threshold_num_photons"]:

            # cherenkov statistics of airshower part
            # --------------------------------------
            crstp = {}
            crstp[spt.IDX] = airshower_seed
            crstp = _append_bunch_statistics(crstp, bunches_part)
            tabrec["cherenkovpoolpart"].append(crstp)

            evth_out = event_header.copy()
            evth_out[cpw.I_EVTH_NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
            evth_out[cpw.I_EVTH_X_CORE_CM(reuse=1)] = 1e2*primary[
                "scatter_wrt_cherenkov_pool_x_m"]
            evth_out[cpw.I_EVTH_Y_CORE_CM(reuse=1)] = 1e2*primary[
                "scatter_wrt_cherenkov_pool_y_m"]
            tar_append(
                tarout=run_part_tarfile,
                file_name=cpw.TARIO_EVTH_FILENAME.format(airshower_id),
                file_bytes=evth_out.tobytes())
            tar_append(
                tarout=run_part_tarfile,
                file_name=cpw.TARIO_BUNCHES_FILENAME.format(airshower_id),
                file_bytes=bunches_part.tobytes())

    run_part_tarfile.close()
    grid_tarfile.close()

    logger.log("grid")

    if not job["keep_tmp"]:
        os.remove(corsika_run_path)

    # run merlict
    # -----------
    merlict_run_path = op.join(tmp_dir, run_id_str+"_merlict.cp")
    if not op.exists(merlict_run_path):
        merlict_rc = merlict.plenoscope_propagator(
            corsika_run_path=tmp_corsika_run_part_path,
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
        nfs.copy(
            merlict_run_path+".stdout",
            op.join(job["log_dir"], run_id_str+"_merlict.stdout"))
        nfs.copy(
            merlict_run_path+".stderr",
            op.join(job["log_dir"], run_id_str+"_merlict.stderr"))
        assert(merlict_rc == 0)
    logger.log("merlict")

    if not job["keep_tmp"]:
        os.remove(tmp_corsika_run_part_path)

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
        run_id = int(cevth[cpw.I_EVTH_RUN_NUMBER])
        airshower_id = int(cevth[cpw.I_EVTH_EVENT_NUMBER])
        ide = {
            spt.IDX: table.random_seed_based_on(
                run_id=run_id,
                airshower_id=airshower_id)}

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
        tabrec["trigger"].append(trgtru)

        # passing trigger
        # ---------------
        if (trgtru["response_pe"] >= job["sum_trigger"]["patch_threshold"]):
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp["unique_id_str"] = table.SEED_TEMPLATE_STR.format(
                seed=ptp[spt.IDX])
            table_past_trigger.append(ptp)

            # export past trigger
            # -------------------
            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)
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
        crcl[spt.IDX] = pt[spt.IDX]
        tabrec["cherenkovclassification"].append(crcl)
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
            lfft[spt.IDX] = pt[spt.IDX]
            tabrec["features"].append(lfft)
        except Exception as excep:
            print("idx:", pt[spt.IDX], excep)
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
    event_table = spt.table_of_records_to_sparse_table(
        table_records=tabrec,
        structure=STRUCTURE)
    spt.write(
        path=op.join(tmp_dir, table_filename),
        table=event_table,
        structure=STRUCTURE)
    nfs.copy(
        src=op.join(tmp_dir, table_filename),
        dst=op.join(job["feature_dir"], table_filename))
    logger.log("export_event_table")

    # export grid histograms
    # ----------------------
    nfs.copy(
        src=tmp_grid_histogram_path,
        dst=op.join(job["feature_dir"], grid_histogram_filename))
    logger.log("export_grid_histograms")

    # export past trigger
    # -------------------
    for pt in table_past_trigger:
        final_tarname = pt["unique_id_str"]+'.tar'
        nfs.copy(
            src=op.join(tmp_past_trigger_dir, final_tarname),
            dst=op.join(job["past_trigger_dir"], final_tarname))
    logger.log("export_past_trigger")

    # end
    # ---
    logger.log("end")
    nfs.move(time_log_path+".tmp", time_log_path)

    if not job["keep_tmp"]:
        shutil.rmtree(tmp_dir)
