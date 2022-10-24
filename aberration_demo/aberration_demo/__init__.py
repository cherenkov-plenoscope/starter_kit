"""
simulate different light-field-sensors
"""
import numpy as np
import plenoirf
import multiprocessing
import json_numpy
import os
import shutil
import plenopy
import sebastians_matplotlib_addons as sebplt
from . import merlict
from . import analysis
from . import calibration_source
import json_line_logger

HEXAGON_INNER_OVER_OUTER_RADIUS = np.sqrt(3) * 0.5

CONFIG = {}
CONFIG["seed"] = 42

CONFIG["executables"] = {
    "merlict_plenoscope_propagator_path": os.path.abspath(
        os.path.join("build", "merlict", "merlict-plenoscope-propagation")
    ),
    "merlict_plenoscope_calibration_map_path": os.path.abspath(
        os.path.join("build", "merlict", "merlict-plenoscope-calibration-map")
    ),
    "merlict_plenoscope_calibration_reduce_path": os.path.abspath(
        os.path.join(
            "build", "merlict", "merlict-plenoscope-calibration-reduce"
        )
    ),
}

CONFIG["sources"] = {}
CONFIG["sources"]["off_axis_angles_deg"] = np.linspace(0.0, 8.0, 9)
CONFIG["sources"]["num_photons"] = 1000 * 1000

CONFIG["mirror"] = {}
CONFIG["mirror"]["keys"] = [
    "sphere_monolith",
    "davies_cotton",
    "parabola_segmented",
]
CONFIG["mirror"]["focal_length"] = 106.5
CONFIG["mirror"]["outer_radius"] = 41.0
CONFIG["mirror"]["inner_radius"] = (
    HEXAGON_INNER_OVER_OUTER_RADIUS * CONFIG["mirror"]["outer_radius"]
)
CONFIG["sensor"] = {}
CONFIG["sensor"]["fov_radius_deg"] = 3.25
CONFIG["sensor"]["housing_overhead"] = 1.1
CONFIG["sensor"]["hex_pixel_fov_flat2flat_deg"] = 0.06667
CONFIG["sensor"]["num_paxel_on_diagonal"] = [1, 3, 9]
CONFIG["light_field_geometry"] = {}
CONFIG["light_field_geometry"]["num_blocks"] = 5
CONFIG["light_field_geometry"]["num_photons_per_block"] = 1000 * 1000
CONFIG["binning"] = analysis.BINNING


ANGLE_FMT = "angle{:06d}"
PAXEL_FMT = "paxel{:06d}"


def init(work_dir, config=CONFIG):
    """
    Initialize the work_dir, i.e. the base of all operations to explore how
    plenoptics can compensate distortions and aberrations.

    When initialized, you might want to adjust the work_dir/config.json.
    Then you call run(work_dir) to run the exploration.

    Parameters
    ----------
    work_dir : str
        The path to the work_dir.
    config : dict
        The configuration of our explorations.
        Configure the different geometries of the mirrors,
        the different off-axis-angles,
        and the number of photo-sensors in the light-field-sensor.
    """
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(json_numpy.dumps(config, indent=4))

    with open(
        os.path.join(work_dir, "merlict_propagation_config.json"), "wt"
    ) as f:
        f.write(json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4))


def run(
    work_dir,
    map_and_reduce_pool,
    logger=json_line_logger.LoggerStdout(),
    desired_num_bunbles=400,
):
    """
    Runs the entire exploration.
        - Makes the sceneries with the optics
        - Estimates the light-field-geometries of the optical instruments.
        - Makes the calibration-source.
        - Makes the response of each instrument to the calibration-source.
        - Analyses each response.
        - Makes overview plots of the responses.

    Parameters
    ----------
    work_dir : str
        The path to the work_dir.
    map_and_reduce_pool : pool
        Must have a map()-function. Used for parallel computing.
    logger : logging.Logger
        A logger of your choice.
    """
    logger.info("Start")

    logger.info("Make sceneries")
    make_sceneries_for_light_field_geometires(work_dir=work_dir)

    logger.info("Estimate light_field_geometry")
    make_light_field_geometires(
        work_dir=work_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
        desired_num_bunbles=desired_num_bunbles,
    )

    logger.info("Make calibration source")
    make_source(work_dir=work_dir)

    logger.info("Make responses to calibration source")
    make_responses(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.info("Make analysis")
    make_analysis(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.info("Plot analysis")
    make_plots(work_dir=work_dir)

    logger.info("Stop")


def read_config(work_dir):
    """
    Returns the config in work_dir/config.json.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())
    return config


def LightFieldGeometry(path, off_axis_angle_deg):
    """
    Returns a plenopy.LightFieldGeometry(path) but de-rotated by
    off_axis_angle_deg in cx.

    Parameters
    ----------
    path : str
        Path to the plenopy.LightFieldGeometry
    off_axis_angle_deg : float
        The off-axis-angle of the light.
    """
    lfg = plenopy.LightFieldGeometry(path=path)
    lfg.cx_mean += np.deg2rad(off_axis_angle_deg)
    return lfg


def guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
    num_paxel_on_diagonal,
):
    return num_paxel_on_diagonal * num_paxel_on_diagonal


def make_responses(
    work_dir, map_and_reduce_pool,
):
    """
    Makes the responses of the instruments to the calibration-sources.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    map_and_reduce_pool : pool
        Used for parallel computing.
    """
    jobs = _responses_make_jobs(work_dir=work_dir)
    _ = map_and_reduce_pool.map(_responses_run_job, jobs)


def _responses_make_jobs(work_dir):
    jobs = []

    config = read_config(work_dir=work_dir)

    runningseed = int(config["seed"])
    for mkey in config["mirror"]["keys"]:

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                job = {}
                job["work_dir"] = work_dir
                job["mkey"] = mkey
                job["pkey"] = pkey
                job["akey"] = akey
                job["merlict_plenoscope_propagator_path"] = config[
                    "executables"
                ]["merlict_plenoscope_propagator_path"]
                job["seed"] = runningseed
                jobs.append(job)

                runningseed += 1

    return jobs


def _responses_run_job(job):
    adir = os.path.join(
        job["work_dir"], "responses", job["mkey"], job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    response_event_path = os.path.join(adir, "1")

    if not os.path.exists(response_event_path):
        plenoirf.production.merlict.plenoscope_propagator(
            corsika_run_path=os.path.join(job["work_dir"], "source.tar"),
            output_path=adir,
            light_field_geometry_path=os.path.join(
                job["work_dir"],
                "geometries",
                job["mkey"],
                job["pkey"],
                job["akey"],
                "light_field_geometry",
            ),
            merlict_plenoscope_propagator_path=job[
                "merlict_plenoscope_propagator_path"
            ],
            merlict_plenoscope_propagator_config_path=os.path.join(
                job["work_dir"], "merlict_propagation_config.json"
            ),
            random_seed=job["seed"],
            photon_origins=True,
            stdout_path=adir + ".o",
            stderr_path=adir + ".e",
        )

    left_over_input_dir = os.path.join(adir, "input")
    if os.path.exists(left_over_input_dir):
        shutil.rmtree(left_over_input_dir)

    return 1


def _analysis_make_jobs(
    work_dir, containment_percentile=80, object_distance_m=1e6,
):
    assert 0.0 < containment_percentile <= 100.0
    assert object_distance_m > 0.0

    config = read_config(work_dir=work_dir)

    jobs = []
    runningseed = int(config["seed"])

    for mkey in config["mirror"]["keys"]:
        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)
                angle_deg = config["sources"]["off_axis_angles_deg"][ofa]

                job = {}
                job["work_dir"] = work_dir
                job["mkey"] = mkey
                job["pkey"] = pkey
                job["akey"] = akey
                job["off_axis_angle_deg"] = angle_deg
                job["seed"] = runningseed
                job["object_distance_m"] = object_distance_m
                job["containment_percentile"] = containment_percentile
                jobs.append(job)
                runningseed += 1

    return jobs


def _analysis_run_job(job):
    adir = os.path.join(
        job["work_dir"], "analysis", job["mkey"], job["pkey"], job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    summary_path = os.path.join(adir, "summary.json")

    if os.path.exists(summary_path):
        return 1

    config = read_config(work_dir=job["work_dir"])
    prng = np.random.Generator(np.random.PCG64(job["seed"]))

    light_field_geometry = LightFieldGeometry(
        path=os.path.join(
            job["work_dir"],
            "geometries",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "light_field_geometry",
        ),
        off_axis_angle_deg=job["off_axis_angle_deg"],
    )

    event = plenopy.Event(
        path=os.path.join(
            job["work_dir"],
            "responses",
            job["mkey"],
            job["pkey"],
            job["akey"],
            "1",
        ),
        light_field_geometry=light_field_geometry,
    )

    print(job["mkey"], job["pkey"], job["akey"])

    calibrated_response = analysis.calibrate_plenoscope_response(
        light_field_geometry=light_field_geometry,
        event=event,
        object_distance=job["object_distance_m"],
    )

    cres = calibrated_response

    print("image encirclement2d")
    psf_cx, psf_cy, psf_angle80 = analysis.encirclement2d(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        prng=prng,
        percentile=job["containment_percentile"],
        num_sub_samples=1,
    )

    thisbinning = dict(config["binning"])
    thisbinning["image"]["center"]["cx_deg"] = job["off_axis_angle_deg"]
    thisbinning["image"]["center"]["cy_deg"] = 0.0
    thisimg_bin_edges = analysis.binning_image_bin_edges(binning=thisbinning)

    print("image histogram2d_std")
    imgraw = analysis.histogram2d_std(
        x=cres["image_beams"]["cx"],
        y=cres["image_beams"]["cy"],
        x_std=cres["image_beams"]["cx_std"],
        y_std=cres["image_beams"]["cy_std"],
        weights=cres["image_beams"]["weights"],
        bins=thisimg_bin_edges,
        prng=prng,
        num_sub_samples=1000,
    )[0]

    print("time encirclement1d")
    time_80_start, time_80_stop = analysis.encirclement1d(
        x=cres["time"]["bin_centers"],
        f=cres["time"]["weights"],
        percentile=job["containment_percentile"],
    )
    print("time full_width_half_maximum")
    (time_fwhm_start, time_fwhm_stop,) = analysis.full_width_half_maximum(
        x=cres["time"]["bin_centers"], f=cres["time"]["weights"],
    )

    print("export")
    out = {}
    out["statistics"] = {}
    out["statistics"]["image_beams"] = {}
    out["statistics"]["image_beams"][
        "total"
    ] = light_field_geometry.number_lixel
    out["statistics"]["image_beams"]["valid"] = np.sum(
        cres["image_beams"]["valid"]
    )
    out["statistics"]["photons"] = {}
    out["statistics"]["photons"][
        "total"
    ] = event.raw_sensor_response.number_photons
    out["statistics"]["photons"]["valid"] = np.sum(
        cres["image_beams"]["weights"]
    )

    out["time"] = cres["time"]
    out["time"]["fwhm"] = {}
    out["time"]["fwhm"]["start"] = time_fwhm_start
    out["time"]["fwhm"]["stop"] = time_fwhm_stop
    out["time"]["containment80"] = {}
    out["time"]["containment80"]["start"] = time_80_start
    out["time"]["containment80"]["stop"] = time_80_stop

    out["image"] = {}
    out["image"]["angle80"] = psf_angle80
    out["image"]["binning"] = thisbinning
    out["image"]["raw"] = imgraw

    with open(summary_path, "wt") as f:
        f.write(json_numpy.dumps(out))

    return 1


def make_analysis(
    work_dir,
    map_and_reduce_pool,
    object_distance_m=1e6,
    containment_percentile=80,
):
    jobs = _analysis_make_jobs(
        work_dir=work_dir,
        object_distance_m=object_distance_m,
        containment_percentile=containment_percentile,
    )
    _ = map_and_reduce_pool.map(_analysis_run_job, jobs)


def make_source(work_dir):
    """
    Makes the calibration-source.
    This is a bundle of parallel photons coming from zenith.
    It is written to work_dir/source.tar and is in the CORSIKA-like format
    EvnetTape.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    config = read_config(work_dir=work_dir)

    source_path = os.path.join(work_dir, "source.tar")

    if not os.path.exists(source_path):
        prng = np.random.Generator(np.random.PCG64(config["seed"]))
        calibration_source.write_photon_bunches(
            cx=0.0,
            cy=0.0,
            size=config["sources"]["num_photons"],
            path=source_path,
            prng=prng,
            aperture_radius=1.2 * config["mirror"]["outer_radius"],
        )


def make_sceneries_for_light_field_geometires(work_dir):
    """
    Makes the sceneries of the instruments for each combination of:
    - mirror-geometry
    - photo-sensor-density
    - off-axis-angle

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    config = read_config(work_dir=work_dir)

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for mkey in config["mirror"]["keys"]:
        mdir = os.path.join(geometries_dir, mkey)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            pdir = os.path.join(mdir, pkey)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)
                adir = os.path.join(pdir, akey)

                scenery_dir = os.path.join(adir, "input", "scenery")
                os.makedirs(scenery_dir, exist_ok=True)
                with open(
                    os.path.join(scenery_dir, "scenery.json"), "wt"
                ) as f:
                    s = merlict.make_plenoscope_scenery_for_merlict(
                        mirror_key=mkey,
                        num_paxel_on_diagonal=npax,
                        config=config,
                        off_axis_angles_deg=config["sources"][
                            "off_axis_angles_deg"
                        ][ofa],
                    )
                    f.write(json_numpy.dumps(s, indent=4))


def make_light_field_geometires(
    work_dir, map_and_reduce_pool, logger, desired_num_bunbles=400,
):
    logger.info("lfg: Make jobs to estimate light-field-geometries.")

    jobs, rjobs = _light_field_geometries_make_jobs_and_rjobs(
        work_dir=work_dir,
    )

    logger.info(
        "lfg: num jobs: mapping {:d}, reducing {:d}".format(
            len(jobs), len(rjobs)
        )
    )

    bundle_jobs = plenoirf.bundle.make_jobs_in_bundles(
        jobs=jobs, desired_num_bunbles=desired_num_bunbles,
    )

    logger.info("lfg: Map")

    _ = map_and_reduce_pool.map(
        _light_field_geometries_run_jobs_in_bundles, bundle_jobs
    )

    logger.info("lfg: Reduce")
    _ = map_and_reduce_pool.map(_light_field_geometries_run_rjob, rjobs)
    logger.info("lfg: Done")


def _light_field_geometries_run_jobs_in_bundles(bundle):
    results = []
    for job in bundle:
        result = plenoirf.production.light_field_geometry.run_job(job)
        results.append(result)
    return results


def _light_field_geometries_make_jobs_and_rjobs(work_dir):
    config = read_config(work_dir=work_dir)

    jobs = []
    rjobs = []

    for mkey in config["mirror"]["keys"]:
        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                adir = os.path.join(work_dir, "geometries", mkey, pkey, akey,)

                out_dir = os.path.join(adir, "light_field_geometry")

                if not os.path.exists(out_dir):

                    # mapping
                    # -------
                    map_dir = os.path.join(adir, "light_field_geometry.map")
                    os.makedirs(map_dir, exist_ok=True)

                    _num_blocks = config["light_field_geometry"]["num_blocks"]
                    _num_blocks *= guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
                        num_paxel_on_diagonal=npax
                    )

                    _num_photons_per_block = config["light_field_geometry"][
                        "num_photons_per_block"
                    ]

                    _jobs = plenoirf.production.light_field_geometry.make_jobs(
                        merlict_map_path=config["executables"][
                            "merlict_plenoscope_calibration_map_path"
                        ],
                        scenery_path=os.path.join(adir, "input", "scenery"),
                        map_dir=map_dir,
                        num_photons_per_block=_num_photons_per_block,
                        num_blocks=_num_blocks,
                        random_seed=0,
                    )

                    jobs += _jobs

                    # reducing
                    # --------
                    rjob = {}
                    rjob["work_dir"] = work_dir
                    rjob["mkey"] = mkey
                    rjob["pkey"] = pkey
                    rjob["akey"] = akey
                    rjobs.append(rjob)

    return jobs, rjobs


def _light_field_geometries_run_rjob(rjob):
    config = read_config(work_dir=rjob["work_dir"])

    adir = os.path.join(
        rjob["work_dir"],
        "geometries",
        rjob["mkey"],
        rjob["pkey"],
        rjob["akey"],
    )

    map_dir = os.path.join(adir, "light_field_geometry.map")
    out_dir = os.path.join(adir, "light_field_geometry")

    rc = plenoirf.production.light_field_geometry.reduce(
        merlict_reduce_path=config["executables"][
            "merlict_plenoscope_calibration_reduce_path"
        ],
        map_dir=map_dir,
        out_dir=out_dir,
    )

    if rc == 0:
        shutil.rmtree(map_dir)

    return rc


def read_analysis(work_dir):
    config = read_config(work_dir=work_dir)

    coll = {}
    for mkey in config["mirror"]["keys"]:
        coll[mkey] = {}

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            coll[mkey][pkey] = {}

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)

                summary_path = os.path.join(
                    work_dir, "analysis", mkey, pkey, akey, "summary.json",
                )
                if not os.path.exists(summary_path):
                    print("Expected summary:", summary_path)
                    continue

                with open(summary_path, "rt") as f:
                    out = json_numpy.loads(f.read())
                coll[mkey][pkey][akey] = out
    return coll


def make_plots(work_dir):
    config = read_config(work_dir=work_dir)

    plot_dir = os.path.join(work_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    coll = read_analysis(work_dir=work_dir)
    for mkey in coll:
        for pkey in coll[mkey]:
            for akey in coll[mkey][pkey]:

                tcoll = coll[mkey][pkey][akey]
                scenario_key = mkey + "_" + pkey + "_" + akey

                bin_edges_cx, bin_edges_cy = analysis.binning_image_bin_edges(
                    binning=tcoll["image"]["binning"]
                )

                img_path = os.path.join(
                    plot_dir, "image_" + scenario_key + ".jpg"
                )
                if not os.path.exists(img_path):
                    fig = sebplt.figure(sebplt.FIGURE_4_3)
                    ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
                    ax_cb = sebplt.add_axes(
                        fig=fig, span=[0.85, 0.1, 0.02, 0.8]
                    )
                    img_raw_norm = tcoll["image"]["raw"] / np.max(
                        tcoll["image"]["raw"]
                    )
                    cmap_psf = ax.pcolormesh(
                        np.rad2deg(bin_edges_cx),
                        np.rad2deg(bin_edges_cy),
                        np.transpose(img_raw_norm),
                        cmap="Greys",
                        norm=sebplt.plt_colors.PowerNorm(gamma=0.33),
                    )
                    sebplt.plt.colorbar(cmap_psf, cax=ax_cb, extend="max")

                    ax.grid(
                        color="k", linestyle="-", linewidth=0.66, alpha=0.33
                    )
                    sebplt.ax_add_circle(
                        ax=ax,
                        x=tcoll["image"]["binning"]["image"]["center"][
                            "cx_deg"
                        ],
                        y=tcoll["image"]["binning"]["image"]["center"][
                            "cy_deg"
                        ],
                        r=np.rad2deg(tcoll["image"]["angle80"]),
                        linewidth=0.5,
                        linestyle="-",
                        color="r",
                        alpha=1,
                        num_steps=360,
                    )
                    ax.set_aspect("equal")
                    ax.set_xlabel(r"$c_x$ / $1^\circ{}$")
                    ax.set_ylabel(r"$c_y$ / $1^\circ{}$")
                    fig.savefig(img_path)
                    sebplt.close(fig)

                time_path = os.path.join(
                    plot_dir, "time_" + scenario_key + ".jpg"
                )
                if not os.path.exists(time_path):
                    fig = sebplt.figure(sebplt.FIGURE_1_1)
                    ax = sebplt.add_axes(fig, [0.1, 0.1, 0.8, 0.8])
                    sebplt.ax_add_histogram(
                        ax=ax,
                        bin_edges=tcoll["time"]["bin_edges"],
                        bincounts=tcoll["time"]["weights"]
                        / np.max(tcoll["time"]["weights"]),
                        face_color="k",
                        face_alpha=0.1,
                        draw_bin_walls=True,
                    )
                    ax.plot(
                        [
                            tcoll["time"]["fwhm"]["start"],
                            tcoll["time"]["fwhm"]["stop"],
                        ],
                        [0.5, 0.5],
                        "r-",
                    )
                    ax.semilogy()
                    ax.set_xlabel("time / s")
                    fig.savefig(time_path)
                    sebplt.close(fig)
