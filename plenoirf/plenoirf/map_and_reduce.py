from . import table
from . import random_seed
from . import grid
from . import merlict
from . import logging
from . import network_file_system as nfs

import sys
import numpy as np
import os
from os import path as op
import shutil

import tempfile
import pandas
import json
import tarfile
import io
import datetime
import corsika_primary_wrapper as cpw
import magnetic_deflection
import plenopy as pl
import sparse_numeric_table as spt


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
    "corsika75600Linux_QGSII_urqmd"
)

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = absjoin(
    "build",
    "merlict",
    "merlict-plenoscope-propagation"
)

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
    "energy_bin_edges_GeV": [20, 200],
    "max_scatter_angle_deg": 13,
    "energy_power_law_slope": -1.0,
}

EXAMPLE_SITE_PARTICLE_DEFLECTION = {
    "energy_GeV": [5, 1000],
    "primary_azimuth_deg": [0.0, 0.0],
    "primary_zenith_deg": [0.0, 0.0],
    "cherenkov_pool_x_m": [0.0, 0.0],
    "cherenkov_pool_y_m": [0.0, 0.0],
}

EXAMPLE_GRID = {
    "num_bins_radius": 512,
    "threshold_num_photons": 50,
    "field_of_view_overhead": 1.1,
    "bin_width_overhead": 1.1,
}

EXAMPLE_SUM_TRIGGER = {
    "object_distances_m": [
        5000.0,
        6164.0,
        7600.0,
        9369.0,
        11551.0,
        14240.0,
        17556.0,
        21644.0,
        26683.0,
        32897.0,
        40557.0,
        50000.0
    ],
    "threshold_pe": 107,
    "integration_time_slices": 10,
    "image": {
        "image_outer_radius_deg": 3.216665,
        "pixel_spacing_deg": 0.06667,
        "pixel_radius_deg": 0.146674,
        "max_number_nearest_lixel_in_pixel": 7
    }
}

EXAMPLE_CHERENKOV_CLASSIFICATION = {
    "region_of_interest": {
        "time_offset_start_s": -10e-9,
        "time_offset_stop_s": 10e-9,
        "direction_radius_deg": 2.0,
        "object_distance_offsets_m": [
            4000.,
            2000.,
            0.,
            -2000.,
        ],
    },
    "min_num_photons": 17,
    "neighborhood_radius_deg": 0.075,
    "direction_to_time_mixing_deg_per_s": 0.375e9
}

ARTIFICIAL_CORE_LIMITATION = {
    "gamma": {
        "energy_GeV":           [0.23, 0.8, 3.0, 35,   81,   432,  1000],
        "max_scatter_radius_m": [150,  150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy_GeV":           [0.23, 1.0,  10,  100,  1000],
        "max_scatter_radius_m": [150,  150,  500, 1100, 2600],
    },
    "proton": {
        "energy_GeV":           [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    }
}
ARTIFICIAL_CORE_LIMITATION['helium'] = ARTIFICIAL_CORE_LIMITATION[
    'proton'].copy()

# ARTIFICIAL_CORE_LIMITATION = None


def make_example_job(run_dir, num_air_showers=25, example_dirname="_testing"):
    particle_key = "proton"
    site_key = "namibia"

    deflection_table = magnetic_deflection.read(
        work_dir=op.join(run_dir, "magnetic_deflection"),
        style="dict",
    )
    test_dir = op.join(run_dir, example_dirname)

    job = {
        "run_id": 1,
        "num_air_showers": num_air_showers,
        "particle": EXAMPLE_PARTICLE,
        "plenoscope_pointing": EXAMPLE_PLENOSCOPE_POINTING,
        "site": EXAMPLE_SITE,
        "grid": EXAMPLE_GRID,
        "sum_trigger": EXAMPLE_SUM_TRIGGER,
        "corsika_primary_path": CORSIKA_PRIMARY_PATH,
        "plenoscope_scenery_path": op.join(
            run_dir,
            'light_field_geometry',
            'input',
            'scenery'),
        "merlict_plenoscope_propagator_path": MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
        "light_field_geometry_path": op.join(
            run_dir,
            'light_field_geometry'),
        "trigger_geometry_path": op.join(
            run_dir,
            'trigger_geometry'),
        "merlict_plenoscope_propagator_config_path": op.join(
            run_dir,
            'input',
            'merlict_propagation_config.json'),
        "site_particle_deflection": deflection_table[site_key][particle_key],
        "cherenkov_classification": EXAMPLE_CHERENKOV_CLASSIFICATION,
        "log_dir": op.join(test_dir, "log"),
        "past_trigger_dir": op.join(test_dir, "past_trigger"),
        "feature_dir": op.join(test_dir, "features"),
        "keep_tmp": True,
        "tmp_dir": op.join(test_dir, "tmp"),
        "date": date_dict_now(),
        "artificial_core_limitation": ARTIFICIAL_CORE_LIMITATION[particle_key]
    }
    return job


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
    assert(len(particle["energy_bin_edges_GeV"]) == 2)


def draw_corsika_primary_steering(
    run_id=1,
    site=EXAMPLE_SITE,
    particle=EXAMPLE_PARTICLE,
    site_particle_deflection=EXAMPLE_SITE_PARTICLE_DEFLECTION,
    num_events=100
):
    assert(run_id > 0)
    _assert_site(site)
    _assert_particle(particle)
    _assert_deflection(site_particle_deflection)
    assert(num_events <= random_seed.NUM_AIRSHOWER_IDS_IN_RUN)

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
    steering["run"] = {
        "run_id": int(run_id),
        "event_id_of_first_event": 1}
    for key in site:
        steering["run"][key] = site[key]

    steering["primaries"] = []
    for e in range(energies.shape[0]):
        event_id = e + 1
        primary = {}
        primary["particle_id"] = particle["particle_id"]
        primary["energy_GeV"] = energies[e]

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

        az, zd = cpw.random_distributions.draw_azimuth_zenith_in_viewcone(
            azimuth_rad=primary["magnet_azimuth_rad"],
            zenith_rad=primary["magnet_zenith_rad"],
            min_scatter_opening_angle_rad=0.,
            max_scatter_opening_angle_rad=max_scatter_rad)

        primary["max_scatter_rad"] = max_scatter_rad
        primary["zenith_rad"] = zd
        primary["azimuth_rad"] = az
        primary["depth_g_per_cm2"] = 0.0
        primary["random_seed"] = cpw.simple_seed(
            random_seed.random_seed_based_on(
                run_id=run_id,
                airshower_id=event_id)
        )

        steering["primaries"].append(primary)
    return steering


def tar_append(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


def _append_bunch_ssize(cherenkovsise_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = cherenkovsise_dict
    ase["num_bunches"] = cb.shape[0]
    ase["num_photons"] = np.sum(cb[:, cpw.IBSIZE])
    return ase


def _append_bunch_statistics(airshower_dict, cherenkov_bunches):
    cb = cherenkov_bunches
    ase = airshower_dict
    assert cb.shape[0] > 0
    ase["maximum_asl_m"] = cpw.CM2M*np.median(cb[:, cpw.IZEM])
    ase["wavelength_median_nm"] = np.abs(np.median(cb[:, cpw.IWVL]))
    ase["cx_median_rad"] = np.median(cb[:, cpw.ICX])
    ase["cy_median_rad"] = np.median(cb[:, cpw.ICY])
    ase["x_median_m"] = cpw.CM2M*np.median(cb[:, cpw.IX])
    ase["y_median_m"] = cpw.CM2M*np.median(cb[:, cpw.IY])
    ase["bunch_size_median"] = np.median(cb[:, cpw.IBSIZE])
    return ase


def plenoscope_event_dir_to_tar(event_dir, output_tar_path=None):
    if output_tar_path is None:
        output_tar_path = event_dir+".tar"
    with tarfile.open(output_tar_path, "w") as tarfout:
        tarfout.add(event_dir, arcname=".")


def run_job(job):
    os.makedirs(job["log_dir"], exist_ok=True)
    os.makedirs(job["past_trigger_dir"], exist_ok=True)
    os.makedirs(job["feature_dir"], exist_ok=True)
    run_id_str = "{:06d}".format(job["run_id"])
    time_log_path = op.join(job["log_dir"], run_id_str+"_runtime.jsonl")
    logger = logging.JsonlLog(time_log_path+".tmp")
    job_path = op.join(job["log_dir"], run_id_str+"_job.json")
    with open(job_path+".tmp", "wt") as f:
        f.write(json.dumps(job, indent=4))
    nfs.move(job_path+".tmp", job_path)
    print('{{"run_id": {:d}"}}\n'.format(job["run_id"]))

    # assert resources exist
    # ----------------------
    assert op.exists(job["corsika_primary_path"])
    assert op.exists(job["merlict_plenoscope_propagator_path"])
    assert op.exists(job["merlict_plenoscope_propagator_config_path"])
    assert op.exists(job["plenoscope_scenery_path"])
    assert op.exists(job["light_field_geometry_path"])
    assert op.exists(job["trigger_geometry_path"])
    logger.log("assert_resource_paths_exist.")

    # draw primaries
    # --------------
    corsika_primary_steering = draw_corsika_primary_steering(
        run_id=job["run_id"],
        site=job["site"],
        particle=job["particle"],
        site_particle_deflection=job["site_particle_deflection"],
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

    # set up grid geometry
    # --------------------
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

    grid_geometry = grid.init_geometry(
        instrument_aperture_outer_diameter=plenoscope_diameter,
        bin_width_overhead=job["grid"]["bin_width_overhead"],
        instrument_field_of_view_outer_radius_deg=(
            plenoscope_field_of_view_radius_deg),
        instrument_pointing_direction=plenoscope_pointing_direction,
        field_of_view_overhead=job["grid"]["field_of_view_overhead"],
        num_bins_radius=job["grid"]["num_bins_radius"],
    )
    logger.log("init_grid_geometry")

    # loop over air-showers
    # ---------------------
    tabrec = {}
    for level_key in table.STRUCTURE:
        tabrec[level_key] = []
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
            run_id = int(event_header[cpw.I_EVTH_RUN_NUMBER])
            assert (run_id == corsika_primary_steering["run"]["run_id"])
            event_id = event_idx + 1
            assert (event_id == event_header[cpw.I_EVTH_EVENT_NUMBER])
            primary = corsika_primary_steering["primaries"][event_idx]
            event_seed = primary["random_seed"][0]["SEED"]
            ide = {spt.IDX: event_seed}
            assert (event_seed == random_seed.random_seed_based_on(
                run_id=run_id,
                airshower_id=event_id)
            )

            # random seed
            # -----------
            np.random.seed(event_seed)

            # export primary table
            # --------------------
            prim = ide.copy()
            prim["particle_id"] = primary["particle_id"]
            prim["energy_GeV"] = primary["energy_GeV"]
            prim["azimuth_rad"] = primary["azimuth_rad"]
            prim["zenith_rad"] = primary["zenith_rad"]
            prim["max_scatter_rad"] = primary["max_scatter_rad"]
            prim["solid_angle_thrown_sr"] = _cone_solid_angle(
                prim["max_scatter_rad"])
            prim["depth_g_per_cm2"] = primary["depth_g_per_cm2"]
            prim["momentum_x_GeV_per_c"] = event_header[
                cpw.I_EVTH_PX_MOMENTUM_GEV_PER_C]
            prim["momentum_y_GeV_per_c"] = event_header[
                cpw.I_EVTH_PY_MOMENTUM_GEV_PER_C]
            prim["momentum_z_GeV_per_c"] = -1.*event_header[
                cpw.I_EVTH_PZ_MOMENTUM_GEV_PER_C]
            prim["first_interaction_height_asl_m"] = -1.*cpw.CM2M*event_header[
                cpw.I_EVTH_Z_FIRST_INTERACTION_CM]
            prim["starting_height_asl_m"] = cpw.CM2M*event_header[
                cpw.I_EVTH_STARTING_HEIGHT_CM]
            obs_lvl_intersection = ray_plane_x_y_intersection(
                support=[0, 0, prim["starting_height_asl_m"]],
                direction=[
                    prim["momentum_x_GeV_per_c"],
                    prim["momentum_y_GeV_per_c"],
                    prim["momentum_z_GeV_per_c"]],
                plane_z=job["site"]["observation_level_asl_m"]
            )
            prim["starting_x_m"] = -1.*obs_lvl_intersection[0]
            prim["starting_y_m"] = -1.*obs_lvl_intersection[1]
            prim["magnet_azimuth_rad"] = primary["magnet_azimuth_rad"]
            prim["magnet_zenith_rad"] = primary["magnet_zenith_rad"]
            prim["magnet_cherenkov_pool_x_m"] = primary[
                "magnet_cherenkov_pool_x_m"]
            prim["magnet_cherenkov_pool_y_m"] = primary[
                "magnet_cherenkov_pool_y_m"]
            tabrec["primary"].append(prim)

            # cherenkov size
            # --------------
            crsz = ide.copy()
            crsz = _append_bunch_ssize(crsz, cherenkov_bunches)
            tabrec["cherenkovsize"].append(crsz)

            # assign grid
            # -----------
            grid_random_shift_x, grid_random_shift_y = np.random.uniform(
                low=-0.5*grid_geometry["bin_width"],
                high=0.5*grid_geometry["bin_width"],
                size=2
            )

            grhi = ide.copy()
            if job['artificial_core_limitation']:
                _max_core_scatter_radius = np.interp(
                    x=primary['energy_GeV'],
                    xp=job['artificial_core_limitation']['energy_GeV'],
                    fp=job['artificial_core_limitation']['max_scatter_radius_m']
                )
                grid_bin_idxs_limitation = grid.where_grid_idxs_within_radius(
                    grid_geometry=grid_geometry,
                    radius=_max_core_scatter_radius,
                    center_x=-1.0*grid_random_shift_x,
                    center_y=-1.0*grid_random_shift_y
                )
                grhi["artificial_core_limitation"] = 1
                grhi["artificial_core_limitation_radius_m"] = (
                    _max_core_scatter_radius)
                grhi["num_bins_thrown"] = len(grid_bin_idxs_limitation[0])
                grhi["area_thrown_m2"] = grhi["num_bins_thrown"]*grid_geometry[
                    "bin_area"]
                logger.log("artificial core limitation is ON")
            else:
                grid_bin_idxs_limitation = None
                grhi["artificial_core_limitation"] = 0
                grhi["artificial_core_limitation_radius_m"] = -1.0
                grhi["num_bins_thrown"] = grid_geometry["total_num_bins"]
                grhi["area_thrown_m2"] = grid_geometry["total_area"]

            grhi["bin_width_m"] = grid_geometry["bin_width"]
            grhi["field_of_view_radius_deg"] = grid_geometry[
                "field_of_view_radius_deg"]
            grhi["pointing_direction_x"] = grid_geometry[
                "pointing_direction"][0]
            grhi["pointing_direction_y"] = grid_geometry[
                "pointing_direction"][1]
            grhi["pointing_direction_z"] = grid_geometry[
                "pointing_direction"][2]
            grhi["random_shift_x_m"] = grid_random_shift_x
            grhi["random_shift_y_m"] = grid_random_shift_y
            grhi["magnet_shift_x_m"] = -1.*primary["magnet_cherenkov_pool_x_m"]
            grhi["magnet_shift_y_m"] = -1.*primary["magnet_cherenkov_pool_x_m"]
            grhi["total_shift_x_m"] = (
                grhi["random_shift_x_m"] + grhi["magnet_shift_x_m"]
            )
            grhi["total_shift_y_m"] = (
                grhi["random_shift_y_m"] + grhi["magnet_shift_y_m"]
            )

            grid_result = grid.assign(
                cherenkov_bunches=cherenkov_bunches,
                grid_geometry=grid_geometry,
                shift_x=grhi["total_shift_x_m"],
                shift_y=grhi["total_shift_y_m"],
                threshold_num_photons=job["grid"]["threshold_num_photons"],
                bin_idxs_limitation=grid_bin_idxs_limitation,
            )
            tar_append(
                tarout=imgtar,
                file_name=random_seed.SEED_TEMPLATE_STR.format(
                    seed=event_seed)+".f4.gz",
                file_bytes=grid.histogram_to_bytes(grid_result["histogram"])
            )

            # grid statistics
            # ---------------
            grhi["num_bins_above_threshold"] = grid_result[
                "num_bins_above_threshold"]
            grhi["overflow_x"] = grid_result["overflow_x"]
            grhi["underflow_x"] = grid_result["underflow_x"]
            grhi["overflow_y"] = grid_result["overflow_y"]
            grhi["underflow_y"] = grid_result["underflow_y"]
            tabrec["grid"].append(grhi)

            # cherenkov statistics
            # --------------------
            if cherenkov_bunches.shape[0] > 0:
                fase = ide.copy()
                fase = _append_bunch_statistics(
                    airshower_dict=fase,
                    cherenkov_bunches=cherenkov_bunches)
                tabrec["cherenkovpool"].append(fase)

            reuse_event = grid_result["random_choice"]
            if reuse_event is not None:
                reuse_evth = event_header.copy()
                reuse_evth[cpw.I_EVTH_NUM_REUSES_OF_CHERENKOV_EVENT] = 1.0
                reuse_evth[
                    cpw.I_EVTH_X_CORE_CM(reuse=1)] = cpw.M2CM*reuse_event[
                        "core_x_m"]
                reuse_evth[
                    cpw.I_EVTH_Y_CORE_CM(reuse=1)] = cpw.M2CM*reuse_event[
                        "core_y_m"]
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
                tabrec["cherenkovsizepart"].append(crszp)
                rase = ide.copy()
                rase = _append_bunch_statistics(
                    airshower_dict=rase,
                    cherenkov_bunches=reuse_event["cherenkov_bunches"])
                tabrec["cherenkovpoolpart"].append(rase)
                rcor = ide.copy()
                rcor["bin_idx_x"] = reuse_event["bin_idx_x"]
                rcor["bin_idx_y"] = reuse_event["bin_idx_y"]
                rcor["core_x_m"] = reuse_event["core_x_m"]
                rcor["core_y_m"] = reuse_event["core_y_m"]
                tabrec["core"].append(rcor)
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
        nfs.copy(
            merlict_run_path+".stdout",
            op.join(job["log_dir"], run_id_str+"_merlict.stdout"))
        nfs.copy(
            merlict_run_path+".stderr",
            op.join(job["log_dir"], run_id_str+"_merlict.stderr"))
        assert(merlict_rc == 0)
    logger.log("merlict")

    if not job["keep_tmp"]:
        os.remove(reuse_run_path)

    # prepare trigger
    # ---------------
    light_field_geometry = pl.LightFieldGeometry(
        path=job["light_field_geometry_path"]
    )
    trigger_geometry = pl.simple_trigger.io.read_trigger_geometry_from_path(
        path=job["trigger_geometry_path"]
    )
    logger.log("prepare_trigger")

    # loop over sensor responses
    # --------------------------
    merlict_run = pl.Run(merlict_run_path)
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
            spt.IDX: random_seed.random_seed_based_on(
                run_id=run_id,
                airshower_id=airshower_id)}

        # apply trigger
        # -------------
        (
            trigger_responses,
            max_response_in_focus_vs_timeslices
        ) = pl.simple_trigger.estimate.first_stage(
            raw_sensor_response=event.raw_sensor_response,
            light_field_geometry=light_field_geometry,
            trigger_geometry=trigger_geometry,
            integration_time_slices=(
                job["sum_trigger"]["integration_time_slices"]
            ),
        )

        trg_resp_path = op.join(event._path, "refocus_sum_trigger.json")
        with open(trg_resp_path, "wt") as f:
            f.write(json.dumps(trigger_responses, indent=4))

        trg_maxr_path = op.join(
            event._path,
            "refocus_sum_trigger.focii_x_time_slices.uint32"
        )
        with open(trg_maxr_path, "wb") as f:
            f.write(max_response_in_focus_vs_timeslices.tobytes())

        # export trigger-truth
        # --------------------
        trgtru = ide.copy()
        trgtru["num_cherenkov_pe"] = int(
            event.simulation_truth.detector.number_air_shower_pulses()
        )
        trgtru["response_pe"] = int(
            np.max(
                [focus['response_pe'] for focus in trigger_responses]
            )
        )
        for o in range(len(trigger_responses)):
            trgtru["focus_{:02d}_response_pe".format(o)] = int(
                trigger_responses[o]['response_pe']
            )
        tabrec["trigger"].append(trgtru)

        # passing trigger
        # ---------------
        if (trgtru["response_pe"] >= job["sum_trigger"]["threshold_pe"]):
            ptp = ide.copy()
            ptp["tmp_path"] = event._path
            ptp["unique_id_str"] = random_seed.SEED_TEMPLATE_STR.format(
                seed=ptp[spt.IDX])
            table_past_trigger.append(ptp)

            # export past trigger
            # -------------------
            ptrg = ide.copy()
            tabrec["pasttrigger"].append(ptrg)
    logger.log("trigger")

    # Cherenkov classification
    # ------------------------
    roi_cfg = job['cherenkov_classification']['region_of_interest']
    dbscan_cfg = job['cherenkov_classification']

    for pt in table_past_trigger:
        event = pl.Event(
            path=pt["tmp_path"],
            light_field_geometry=light_field_geometry
        )
        trigger_responses = pl.simple_trigger.io. \
            read_trigger_response_from_path(
                path=os.path.join(event._path, 'refocus_sum_trigger.json')
            )
        roi = pl.simple_trigger.region_of_interest.from_trigger_response(
            trigger_response=trigger_responses,
            trigger_geometry=trigger_geometry,
            time_slice_duration=event.raw_sensor_response.time_slice_duration,
        )
        photons = pl.classify.RawPhotons.from_event(event)
        (
            cherenkov_photons,
            roi_settings
        ) = pl.classify.cherenkov_photons_in_roi_in_image(
            roi=roi,
            photons=photons,
            roi_time_offset_start=roi_cfg['time_offset_start_s'],
            roi_time_offset_stop=roi_cfg['time_offset_stop_s'],
            roi_cx_cy_radius=np.deg2rad(
                roi_cfg['direction_radius_deg']),
            roi_object_distance_offsets=roi_cfg['object_distance_offsets_m'],
            dbscan_epsilon_cx_cy_radius=np.deg2rad(
                dbscan_cfg['neighborhood_radius_deg']),
            dbscan_min_number_photons=dbscan_cfg['min_num_photons'],
            dbscan_deg_over_s=dbscan_cfg['direction_to_time_mixing_deg_per_s']
        )
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
    event_table = spt.table_of_records_to_sparse_numeric_table(
        table_records=tabrec,
        structure=table.STRUCTURE)
    spt.write(
        path=op.join(tmp_dir, table_filename),
        table=event_table,
        structure=table.STRUCTURE)
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


def run_bundle(bundle):
    results = []
    for j, job in enumerate(bundle):
        msg = '\n#bundle {:d} of {:d}\n'.format((j+1), len(bundle))
        print(msg, file=sys.stdout)
        print(msg, file=sys.stderr)
        try:
            result = run_job(job=job)
        except Exception as exception_msg:
            print(exception_msg, file=sys.stderr)
            result = 0
        results.append(result)
    return results
