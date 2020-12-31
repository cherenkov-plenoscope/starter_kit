#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import pandas
import plenopy as pl
import glob
import multiprocessing
import scipy
import time

PARALLEL = True

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

loph_chunk_base_dir = os.path.join(
    pa["summary_dir"], "0068_prepare_loph_passed_trigger_and_quality"
)

fuzzy_model_config = {
    "min_num_photons": 3,
    "min_time_slope_ns_per_deg": 5.0,
    "max_time_slope_ns_per_deg": 7.0,
}

plenoscope_fov_opening_angle_deg = 0.5 * np.rad2deg(
    lfg.sensor_plane2imaging_system.max_FoV_diameter
)

fuzzy_binning = {
    "radius_deg": 1.0 + plenoscope_fov_opening_angle_deg,
    "num_bins": 128,
}

fuzz_img_gaussian_kernel = pl.fuzzy.discrete_kernel.gauss2d(num_steps=5)


def make_jobs(
    loph_chunk_dir,
    quality,
    site_key,
    particle_key
):
    chunk_paths = glob.glob(os.path.join(loph_chunk_dir, "*.tar"))
    jobs = []
    for chunk_path in chunk_paths:
        job = {}
        job["loph_path"] = str(chunk_path)
        job["quality"] = dict(quality)
        job["site_key"] = str(site_key)
        job["particle_key"] = str(particle_key)
        jobs.append(job)
    return jobs


def run_job(job):
    run = pl.photon_stream.loph.LopfTarReader(job["loph_path"])

    result = []
    ii = 0
    for event in run:
        ii += 1
        start_time = time.time()
        airshower_id, loph_record = event


        fit, debug = irf.reconstruction.trajectory.estimate(
            loph_record=loph_record,
            light_field_geometry=lfg,
            shower_maximum_object_distance=shower_maximum_object_distance[
                airshower_id
            ],
            fuzzy_config=fuzzy_config,
            model_fit_config=long_fit_cfg,
        )

        stop_time = time.time()

        reco = {
            spt.IDX: airshower_id,
            "cx": fit["primary_particle_cx"],
            "cy": fit["primary_particle_cy"],
            "x": fit["primary_particle_x"],
            "y": fit["primary_particle_y"],
            "time": stop_time - start_time
        }
        result.append(reco)

        print("{:03d} {:09d} {:0.2f}".format(
                ii, airshower_id, stop_time - start_time
            )
        )

    return result


fuzzy_config = irf.reconstruction.fuzzy_method.compile_user_config(
    user_config=sum_config["reconstruction"]["trajectory"]["fuzzy_method"]
)

long_fit_cfg = irf.reconstruction.model_fit.compile_user_config(
    user_config=sum_config["reconstruction"]["trajectory"]["core_axis_fit"]
)

for sk in irf_config["config"]["sites"]:
    for pk in ["gamma"]:  # irf_config["config"]["particles"]:

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        _event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar",
            ),
            structure=irf.table.STRUCTURE,
        )

        print("read prev. estimated object_distance")
        shower_maximum_object_distance = spt.get_column_as_dict_by_index(
            table=_event_table,
            level_key="features",
            column_key="image_smallest_ellipse_object_distance"
        )

        print("make jobs")
        jobs = make_jobs(
            loph_chunk_dir=os.path.join(loph_chunk_base_dir, sk, pk, "chunks"),
            quality=sum_config["quality"],
            site_key=sk,
            particle_key=pk,
        )

        print("run jobs")
        if PARALLEL:
            print("PARALLEL")
            pool = multiprocessing.Pool(8)
            _results = pool.map(run_job, jobs)
        else:
            print("SEQUENTIEL")
            _results = []
            for job in jobs:
                print("job", job)
                _results.append(run_job(job))
        results = []
        for chunk in _results:
            for r in chunk:
                results.append(r)

        reco_df = pandas.DataFrame(results)
        reco_di = reco_df.to_dict(orient="list")

        irf.json_numpy.write(
            path=os.path.join(site_particle_dir, "reco" + ".json"),
            out_dict=reco_di,
        )
