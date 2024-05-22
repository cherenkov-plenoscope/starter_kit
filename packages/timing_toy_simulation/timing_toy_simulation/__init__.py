import os
import glob
import numpy as np
import corsika_primary as cpw
import json_utils
import plenoirf
import rename_after_writing as rnw
import spherical_coordinates
import sparse_numeric_table as snt
import atmospheric_cherenkov_response as acr
from . import table


PARTICLES_CORSIKA_ID = {
    "gamma": 1,
    "proton": 14,
    "helium": 408,
}

CONFIG = {
    "particle": {
        "type": "gamma",
        "energy_range": {"start_GeV": 0.5, "stop_GeV": 1.5, "power_slope": 0},
    },
    "flux": {
        "azimuth_deg": 0.0,
        "zenith_deg": 0.0,
        "radial_angle_deg": 60.0,
    },
    "site": {"namibia": acr.sites.init("namibia")},
    "scatter": {
        "direction": {
            "radial_angle_deg": 3.25,
        },
        "position": {
            "radius_m": 640.0,
        },
    },
    "statistics": {
        "num_showers_per_run": 1280,
        "num_runs": 1280,
    },
    "instrument": {
        "radius_m": 35.0,
        "field_of_view_deg": 6.5,
        "photon_detection_efficiency": 0.8 * 0.3,
    },
    "corsika": {
        "path": os.path.join(
            "build",
            "corsika",
            "modified",
            "corsika-75600",
            "run",
            "corsika75600Linux_QGSII_urqmd",
        ),
    },
}

SPEED_OF_LIGHT = 299792458.0

r2d = np.rad2deg
d2r = np.deg2rad


def init(work_dir, config=None):
    os.makedirs(work_dir, exist_ok=True)
    if config == None:
        config = CONFIG
    json_utils.write(os.path.join(work_dir, "config.json"), CONFIG)


def make(work_dir, parallel_pool):
    jobs = make_jobs(work_dir=work_dir)
    parallel_pool.map(run_job, jobs)
    reduce_jobs(work_dir=work_dir)


def make_jobs(work_dir):
    work_dir = os.path.abspath(work_dir)
    config = json_utils.read(os.path.join(work_dir, "config.json"))
    jobs = []
    for run_id in np.arange(1, 1 + config["statistics"]["num_runs"]):
        job = {}
        job["job_id"] = run_id
        job["work_dir"] = work_dir
        job["corsika_path"] = os.path.abspath(config["corsika"]["path"])
        jobs.append(job)
    return jobs


def _make_corsika_steering(job, config, prng):
    i8 = np.int64
    f8 = np.float64

    STARTING_DEPTH_WHEN_ENTERING_ATMOSPHERE_FROM_SPACE_G_PER_CM2 = 0.0

    cor = {
        "run": {
            "run_id": i8(job["job_id"]),
            "event_id_of_first_event": i8(1),
            "observation_level_asl_m": f8(
                config["site"]["observation_level_asl_m"]
            ),
            "earth_magnetic_field_x_muT": f8(
                config["site"]["earth_magnetic_field_x_muT"]
            ),
            "earth_magnetic_field_z_muT": f8(
                config["site"]["earth_magnetic_field_z_muT"]
            ),
            "atmosphere_id": i8(config["site"]["atmosphere_id"]),
            "energy_range": {
                "start_GeV": f8(
                    config["particle"]["energy_range"]["start_GeV"]
                ),
                "stop_GeV": f8(config["particle"]["energy_range"]["stop_GeV"]),
            },
            "random_seed": cpw.random.seed.make_simple_seed(job["job_id"]),
        },
        "primaries": [],
    }

    for i in range(config["statistics"]["num_showers_per_run"]):
        energy_GeV = cpw.random.distributions.draw_power_law(
            prng=prng,
            lower_limit=config["particle"]["energy_range"]["start_GeV"],
            upper_limit=config["particle"]["energy_range"]["stop_GeV"],
            power_slope=config["particle"]["energy_range"]["power_slope"],
            num_samples=1,
        )
        (
            azimuth_rad,
            zenith_rad,
        ) = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
            prng=prng,
            azimuth_rad=d2r(config["flux"]["azimuth_deg"]),
            zenith_rad=d2r(config["flux"]["zenith_deg"]),
            min_scatter_opening_angle_rad=0.0,
            max_scatter_opening_angle_rad=d2r(
                config["flux"]["radial_angle_deg"]
            ),
        )

        pp = {
            "particle_id": f8(
                PARTICLES_CORSIKA_ID[config["particle"]["type"]]
            ),
            "energy_GeV": f8(energy_GeV),
            "zenith_rad": f8(zenith_rad),
            "azimuth_rad": f8(azimuth_rad),
            "depth_g_per_cm2": f8(
                STARTING_DEPTH_WHEN_ENTERING_ATMOSPHERE_FROM_SPACE_G_PER_CM2
            ),
        }
        cor["primaries"].append(pp)

    return cor


def make_cz(cx, cy):
    return np.sqrt(1.0 - cx**2 - cy**2)


def make_instrument_pointing_direction_vector(ins_cx, ins_cy):
    return np.array([ins_cx, ins_cy, -make_cz(cx=ins_cx, cy=ins_cy)])


def calculate_distance_to_origin(cx, x_m, cy, y_m):
    cz = make_cz(cx=cx, cy=cy)

    hh_m = cx * x_m + cy * y_m

    # closest_point
    # -------------
    _cpx = x_m + hh_m * cx
    _cpy = y_m + hh_m * cy
    _cpz = 0.0 + hh_m * cz
    cp_m = np.array([_cpx, _cpy, _cpz])
    return np.linalg.norm(cp_m, axis=0)


def _bunches_calculate_arrival_time_wrt_origin(bunches_cgs):
    num_bunches = bunches_cgs.shape[0]

    arrival_times = np.zeros(num_bunches)

    MOMENTUM_TO_INCIDENT = -1.0
    for i in range(num_bunches):
        cx = MOMENTUM_TO_INCIDENT * bunches_cgs[i, cpw.I.BUNCH.UX_1]
        cy = MOMENTUM_TO_INCIDENT * bunches_cgs[i, cpw.I.BUNCH.VY_1]
        x = cpw.CM2M * bunches_cgs[i, cpw.I.BUNCH.X_CM]
        y = cpw.CM2M * bunches_cgs[i, cpw.I.BUNCH.Y_CM]
        hh = cx * x + cy * y
        ht = hh / SPEED_OF_LIGHT
        arrival_times[i] = 1e-9 * bunches_cgs[i, cpw.I.BUNCH.TIME_NS] - ht

    return arrival_times


def _make_uid(evth):
    run_id = int(evth[cpw.I.EVTH.RUN_NUMBER])
    event_id = int(evth[cpw.I.EVTH.EVENT_NUMBER])
    return plenoirf.unique.make_uid(run_id=run_id, event_id=event_id)


def _calculate_primary_ray_wrt_starting_position(evth):
    primary_direction_uxvywz = cpw.I.EVTH.get_direction_uxvywz(evth=evth)

    primary_starting_z_m = cpw.CM2M * evth[cpw.I.EVTH.STARTING_HEIGHT_CM]

    assert 1 == evth[cpw.I.EVTH.NUM_OBSERVATION_LEVELS]
    primary_core_z_m = cpw.CM2M * evth[cpw.I.EVTH.HEIGHT_OBSERVATION_LEVEL(1)]

    obs_lvl_intersection_m = acr.utils.ray_plane_x_y_intersection(
        support=[0, 0, primary_starting_z_m],
        direction=primary_direction_uxvywz,
        plane_z=primary_core_z_m,
    )

    primary_starting_position = [
        -1.0 * obs_lvl_intersection_m[0],
        -1.0 * obs_lvl_intersection_m[1],
        primary_starting_z_m,
    ]

    return primary_starting_position, primary_direction_uxvywz


def _bunches_translate_into_instrument_frame(
    bunches_cgs, instrument_x_m, instrument_y_m
):
    bunches_cgs[:, cpw.I.BUNCH.X_CM] = (
        bunches_cgs[:, cpw.I.BUNCH.X_CM] - cpw.M2CM * instrument_x_m
    )
    bunches_cgs[:, cpw.I.BUNCH.Y_CM] = (
        bunches_cgs[:, cpw.I.BUNCH.Y_CM] - cpw.M2CM * instrument_y_m
    )
    return bunches_cgs


def _bunches_calculate_distance_to_origin_m(bunches_cgs):
    sphcor = spherical_coordinates

    return calculate_distance_to_origin(
        cx=sphcor.corsika.ux_to_cx(ux=bunches_cgs[:, cpw.I.BUNCH.UX_1]),
        x_m=cpw.CM2M * bunches_cgs[:, cpw.I.BUNCH.X_CM],
        cy=sphcor.corsika.vy_to_cy(vy=bunches_cgs[:, cpw.I.BUNCH.VY_1]),
        y_m=cpw.CM2M * bunches_cgs[:, cpw.I.BUNCH.Y_CM],
    )


def _bunches_calculate_angle_between_rad(bunches_cgs, azimuth_rad, zenith_rad):
    sphcor = spherical_coordinates

    ins_cx, ins_cy = sphcor.az_zd_to_cx_cy(
        azimuth_rad=azimuth_rad,
        zenith_rad=zenith_rad,
    )

    return spherical_coordinates.angle_between_cx_cy(
        cx1=sphcor.corsika.ux_to_cx(ux=bunches_cgs[:, cpw.I.BUNCH.UX_1]),
        cy1=sphcor.corsika.vy_to_cy(vy=bunches_cgs[:, cpw.I.BUNCH.VY_1]),
        cx2=ins_cx,
        cy2=ins_cy,
    )


def _bunches_mask_inside_instruments_etendue(
    bunches_cgs,
    instrument_azimith_rad,
    instrument_zenith_rad,
    instrument_radius_m,
    instrument_field_of_view_deg,
):
    bunches_radius_m = _bunches_calculate_distance_to_origin_m(
        bunches_cgs=bunches_cgs
    )
    bunches_angle_rad = _bunches_calculate_angle_between_rad(
        bunches_cgs=bunches_cgs,
        azimuth_rad=instrument_azimith_rad,
        zenith_rad=instrument_zenith_rad,
    )
    mask_radius = bunches_radius_m <= instrument_radius_m
    mask_angle = bunches_angle_rad <= (
        (1.0 / 2.0) * np.deg2rad(instrument_field_of_view_deg)
    )
    return np.logical_and(mask_radius, mask_angle)


def _export_event_table(path, tabrec):
    event_table = snt.table_of_records_to_sparse_numeric_table(
        table_records=tabrec, structure=table.STRUCTURE
    )
    tmp_path = path + ".incomplete"
    snt.write(
        path=tmp_path,
        table=event_table,
        structure=table.STRUCTURE,
    )
    rnw.move(src=tmp_path, dst=path)


def run_job(job):
    prng = np.random.Generator(np.random.PCG64(job["job_id"]))
    config = json_utils.read(os.path.join(job["work_dir"], "config.json"))

    map_dir = os.path.join(job["work_dir"], "map")
    run_id_str = plenoirf.unique.RUN_ID_FORMAT_STR.format(job["job_id"])
    job_dir = os.path.join(map_dir, run_id_str)
    os.makedirs(job_dir, exist_ok=True)

    steering = _make_corsika_steering(job=job, config=config, prng=prng)

    steering_path = os.path.join(job_dir, "steering.json")
    with open(steering_path + ".incomplete", "wt") as f:
        f.write(json_utils.dumps(steering))
    rnw.move(steering_path + ".incomplete", steering_path)

    tabrec = table.init_records()

    corsika_o_path = os.path.join(job_dir, "corska.o")
    corsika_e_path = os.path.join(job_dir, "corska.e")

    t_deltas = []

    with cpw.CorsikaPrimary(
        corsika_path=job["corsika_path"],
        steering_dict=steering,
        stdout_path=corsika_o_path + ".incomplete",
        stderr_path=corsika_e_path + ".incomplete",
        particle_output_path=os.path.join(job_dir, "particles.dat"),
    ) as run:
        for shower in run:
            evth, cherenkov_reader = shower

            bunches_cgs = np.vstack([b for b in cherenkov_reader])

            # unique id
            # ---------
            uid = {snt.IDX: _make_uid(evth=evth)}

            # everything that is known before Cherenkov emission
            # --------------------------------------------------
            obs_level_m = (
                cpw.CM2M * evth[cpw.I.EVTH.HEIGHT_OBSERVATION_LEVEL(1)]
            )
            base = uid.copy()
            base["primary_particle_id"] = evth[cpw.I.EVTH.PARTICLE_ID]
            base["primary_energy_GeV"] = evth[cpw.I.EVTH.TOTAL_ENERGY_GEV]

            prm_az, prm_zd = cpw.I.EVTH.get_pointing_az_zd(evth=evth)

            base["primary_azimuth_rad"] = prm_az
            base["primary_zenith_rad"] = prm_zd

            p_s, p_d = _calculate_primary_ray_wrt_starting_position(evth=evth)

            base["primary_start_x_m"] = p_s[0]
            base["primary_start_y_m"] = p_s[1]
            base["primary_start_z_m"] = p_s[2]

            base["primary_direction_x"] = p_d[0]
            base["primary_direction_y"] = p_d[1]
            base["primary_direction_z"] = p_d[2]

            # cross-check
            core_position_xy_m = acr.utils.ray_plane_x_y_intersection(
                support=[
                    base["primary_start_x_m"],
                    base["primary_start_y_m"],
                    base["primary_start_z_m"],
                ],
                direction=[
                    base["primary_direction_x"],
                    base["primary_direction_y"],
                    base["primary_direction_z"],
                ],
                plane_z=obs_level_m,
            )
            assert np.abs(core_position_xy_m[0]) < 1e-1
            assert np.abs(core_position_xy_m[1]) < 1e-1

            # instrument
            # ----------
            (
                base["instrument_x_m"],
                base["instrument_y_m"],
            ) = cpw.random.distributions.draw_x_y_in_disc(
                prng=prng,
                radius=config["scatter"]["position"]["radius_m"],
            )
            base["instrument_z_m"] = obs_level_m
            (
                base["instrument_azimuth_rad"],
                base["instrument_zenith_rad"],
            ) = cpw.random.distributions.draw_azimuth_zenith_in_viewcone(
                prng=prng,
                azimuth_rad=evth[cpw.I.EVTH.AZIMUTH_RAD],
                zenith_rad=evth[cpw.I.EVTH.ZENITH_RAD],
                min_scatter_opening_angle_rad=0.0,
                max_scatter_opening_angle_rad=d2r(
                    config["scatter"]["direction"]["radial_angle_deg"]
                ),
            )

            # true arrival time
            # -----------------
            base[
                "primary_distance_to_closest_point_to_instrument_m"
            ] = plenoirf.utils.ray_parameter_for_closest_distance_to_point(
                ray_direction=[
                    base["primary_direction_x"],
                    base["primary_direction_y"],
                    base["primary_direction_z"],
                ],
                ray_support=[
                    base["primary_start_x_m"],
                    base["primary_start_y_m"],
                    base["primary_start_z_m"],
                ],
                point=[
                    base["instrument_x_m"],
                    base["instrument_y_m"],
                    base["instrument_z_m"],
                ],
            )

            base["primary_time_to_closest_point_to_instrument_s"] = (
                base["primary_distance_to_closest_point_to_instrument_m"]
                / SPEED_OF_LIGHT
            )

            tabrec["base"].append(base)

            # Cherenkov-pool-size
            # -------------------
            cers = uid.copy()
            cers["num_bunches"] = bunches_cgs.shape[0]
            cers["num_photons"] = np.sum(
                bunches_cgs[:, cpw.I.BUNCH.BUNCH_SIZE_1]
            )
            tabrec["cherenkov_size"].append(cers)

            if cers["num_bunches"] == 0:
                continue

            # Cherenkov-pool-properties
            # -------------------------
            cerp = uid.copy()
            plenoirf.instrument_response._append_bunch_statistics(
                airshower_dict=cerp,
                cherenkov_bunches=bunches_cgs,
            )
            tabrec["cherenkov_pool"].append(cerp)

            # translate bunches into instrument's frame
            # ------------------------------------------
            bunches_wrt_intrument_cgs = (
                _bunches_translate_into_instrument_frame(
                    bunches_cgs=bunches_cgs,
                    instrument_x_m=base["instrument_x_m"],
                    instrument_y_m=base["instrument_y_m"],
                )
            )

            mask_visible = _bunches_mask_inside_instruments_etendue(
                bunches_cgs=bunches_wrt_intrument_cgs,
                instrument_azimith_rad=base["instrument_azimuth_rad"],
                instrument_zenith_rad=base["instrument_zenith_rad"],
                instrument_radius_m=config["instrument"]["radius_m"],
                instrument_field_of_view_deg=config["instrument"][
                    "field_of_view_deg"
                ],
            )

            # visible Cherenkov-pool-size
            # ---------------------------
            bunches_visible_cgs = bunches_wrt_intrument_cgs[mask_visible]

            cerps = uid.copy()
            cerps["num_bunches"] = bunches_visible_cgs.shape[0]
            cerps["num_photons"] = np.sum(
                bunches_visible_cgs[:, cpw.I.BUNCH.BUNCH_SIZE_1]
            )
            tabrec["cherenkov_visible_size"].append(cerps)

            if cerps["num_bunches"] == 0:
                continue

            # visible Cherenkov-pool-properties
            # ---------------------------------
            cerpp = uid.copy()
            plenoirf.instrument_response._append_bunch_statistics(
                airshower_dict=cerpp,
                cherenkov_bunches=bunches_visible_cgs,
            )
            tabrec["cherenkov_visible_pool"].append(cerpp)

            # detected Cherenkov-pool
            # -----------------------
            assert np.all(
                bunches_visible_cgs[:, cpw.I.BUNCH.BUNCH_SIZE_1] <= 1.0
            ), "Expected bunch-size <= 1.0"

            mask_bunch_is_photon = (
                prng.uniform(size=bunches_visible_cgs.shape[0])
                < bunches_visible_cgs[:, cpw.I.BUNCH.BUNCH_SIZE_1]
            )

            photons_cgs = bunches_visible_cgs[mask_bunch_is_photon]

            mask_detected = (
                prng.uniform(size=photons_cgs.shape[0])
                < config["instrument"]["photon_detection_efficiency"]
            )

            detected_photons_cgs = photons_cgs[mask_detected]

            cerds = uid.copy()
            cerds["num_bunches"] = detected_photons_cgs.shape[0]
            cerds["num_photons"] = np.sum(
                detected_photons_cgs[:, cpw.I.BUNCH.BUNCH_SIZE_1]
            )
            tabrec["cherenkov_detected_size"].append(cerds)

            if cerds["num_bunches"] == 0:
                continue

            cerdp = uid.copy()
            plenoirf.instrument_response._append_bunch_statistics(
                airshower_dict=cerdp,
                cherenkov_bunches=detected_photons_cgs,
            )
            tabrec["cherenkov_detected_pool"].append(cerdp)

            # reconstructed arrival time
            # --------------------------
            bunches_arrival_times = _bunches_calculate_arrival_time_wrt_origin(
                bunches_cgs=detected_photons_cgs
            )

            reco = uid.copy()
            reco["arrival_time_median_s"] = np.median(bunches_arrival_times)
            reco["arrival_time_stddev_s"] = np.std(bunches_arrival_times)
            tabrec["reconstruction"].append(reco)

            """
            t_delta = (
                reco["arrival_time_median_s"] -
                base["primary_time_to_closest_point_to_instrument_s"]
            )
            if cerds["num_bunches"] > 25:
                t_deltas.append(t_delta)
                print(
                    "{: 6.1f}ns +- {: 6.1f}ns".format(t_delta*1e9, 1e9*reco["arrival_time_stddev_s"]),
                    ", avg: ",
                    "{: 6.1f}ns, ".format(1e9 * np.std(t_deltas)),
                    "zd {: 6.1f}deg".format(np.rad2deg(base["primary_zenith_rad"])),
                    "d {: 6.3f}km".format(base["primary_distance_to_closest_point_to_instrument_m"] * 1e-3)
                )
            """

    _export_event_table(
        path=os.path.join(job_dir, "result.tar"),
        tabrec=tabrec,
    )

    rnw.move(corsika_o_path + ".incomplete", corsika_o_path)
    rnw.move(corsika_e_path + ".incomplete", corsika_e_path)


def reduce_jobs(work_dir):
    _run_paths = glob.glob(os.path.join(work_dir, "map", "*"))
    result_paths = []
    for _run_path in _run_paths:
        result_path = os.path.join(_run_path, "result.tar")
        if os.path.exists(result_path):
            result_paths.append(result_path)

    result_table = snt.concatenate_files(
        list_of_table_paths=result_paths, structure=table.STRUCTURE
    )

    result_path = os.path.join(work_dir, "result.tar")

    snt.write(
        path=result_path + ".incomplete",
        table=result_table,
        structure=table.STRUCTURE,
    )
    rnw.move(src=result_path + ".incomplete", dst=result_path)
