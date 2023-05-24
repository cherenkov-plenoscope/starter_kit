import json_line_logger
import os
import json_numpy
import plenopy
from .. import sources


def run(work_dir, pool, logger=json_line_logger.LoggerStdout()):
    logger.info("Obs: Make observations")
    ojobs = _observations_make_jobs(work_dir=work_dir)

    logger.info("Obs: {:d} jobs to do".format(len(ojobs)))

    pool.map(_observations_run_job, ojobs)
    logger.info("Obs: Observations done")


def make_response_to_source(
    source_config, light_field_geometry_path, merlict_config,
):
    if source_config["type"] == "star":
        return sources.star.make_response_to_star(
            star_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "mesh":
        return sources.mesh.make_response_to_mesh(
            mesh_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    elif source_config["type"] == "point":
        return sources.point.make_response_to_point(
            point_config=source_config,
            light_field_geometry_path=light_field_geometry_path,
            merlict_config=merlict_config,
        )
    else:
        raise AssertionError("Type of source is not known")


def _observations_make_jobs(work_dir):
    return _tasks_make_jobs(work_dir=work_dir, task_key="responses", suffix="")


def _tasks_make_jobs(work_dir, task_key, suffix):
    cfg_dir = os.path.join(work_dir, "config")
    config = json_numpy.read_tree(cfg_dir)

    jobs = []

    for instrument_key in config["observations"]["instruments"]:

        if instrument_key not in config["instruments"]:
            continue

        for observation_key in config["observations"]["instruments"][
            instrument_key
        ]:
            if observation_key == "star":
                stars = config["observations"]["star"]
                for n in range(stars["num_stars"]):
                    nkey = "{:06d}".format(n)
                    outpath = os.path.join(
                        work_dir,
                        task_key,
                        instrument_key,
                        observation_key,
                        nkey + suffix,
                    )
                    if not os.path.exists(outpath):
                        job = {
                            "work_dir": work_dir,
                            "instrument_key": instrument_key,
                            "observation_key": observation_key,
                            "number": n,
                        }
                        jobs.append(job)

            elif observation_key == "point":
                points = config["observations"]["point"]
                for n in range(points["num_points"]):
                    nkey = "{:06d}".format(n)
                    outpath = os.path.join(
                        work_dir,
                        task_key,
                        instrument_key,
                        observation_key,
                        nkey + suffix,
                    )
                    if not os.path.exists(outpath):
                        job = {
                            "work_dir": work_dir,
                            "instrument_key": instrument_key,
                            "observation_key": observation_key,
                            "number": n,
                        }
                        jobs.append(job)
            elif observation_key == "phantom":
                phantom = config["observations"]["phantom"]
                n = 0
                nkey = "{:06d}".format(n)

                outpath = os.path.join(
                    work_dir,
                    task_key,
                    instrument_key,
                    observation_key,
                    nkey + suffix,
                )
                if not os.path.exists(outpath):
                    job = {
                        "work_dir": work_dir,
                        "instrument_key": instrument_key,
                        "observation_key": observation_key,
                        "number": n,
                    }
                    jobs.append(job)
    return jobs


def _observations_run_job(job):
    outdir = os.path.join(
        job["work_dir"],
        "responses",
        job["instrument_key"],
        job["observation_key"],
    )
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "{:06d}".format(job["number"]))
    light_field_geometry_path = os.path.join(
        job["work_dir"],
        "instruments",
        job["instrument_key"],
        "light_field_geometry",
    )

    merlict_config = json_numpy.read_tree(
        os.path.join(job["work_dir"], "config", "merlict")
    )

    if job["observation_key"] == "star":
        source_config = sources.star.make_source_config_from_job(job=job)
    elif job["observation_key"] == "point":
        source_config = sources.point.make_source_config_from_job(job=job)
    elif job["observation_key"] == "phantom":
        source_config = sources.mesh.make_source_config_from_job(job=job)
    else:
        raise AssertionError("Bad observation_key")

    raw_sensor_response = make_response_to_source(
        source_config=source_config,
        light_field_geometry_path=light_field_geometry_path,
        merlict_config=merlict_config,
    )

    # export truth
    # ------------
    outtruthpath = outpath + ".json"
    json_numpy.write(
        outtruthpath + ".incomplete", source_config,
    )
    os.rename(outtruthpath + ".incomplete", outtruthpath)

    # export raw sensor resposnse
    # ---------------------------
    with open(outpath + ".incomplete", "wb") as f:
        plenopy.raw_light_field_sensor_response.write(
            f=f, raw_sensor_response=raw_sensor_response
        )
    os.rename(outpath + ".incomplete", outpath)
