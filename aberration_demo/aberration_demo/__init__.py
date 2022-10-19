"""
simulate different light-field-sensors
"""
import numpy as np
import plenoirf
import queue_map_reduce
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

CFG = {}
CFG["seed"] = 42

CFG["executables"] = {
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

CFG["sources"] = {}
CFG["sources"]["off_axis_angles_deg"] = [0.0, 4.0, 8.0]
CFG["sources"]["num_photons"] = 1000 * 1000

CFG["mirror"] = {}
CFG["mirror"]["keys"] = [
    "sphere_monolith",
    "davies_cotton",
    "parabola_segmented",
]
CFG["mirror"]["focal_length"] = 106.5
CFG["mirror"]["inner_radius"] = 35.5
CFG["mirror"]["outer_radius"] = (2 / np.sqrt(3)) * CFG["mirror"][
    "inner_radius"
]
CFG["sensor"] = {}
CFG["sensor"]["fov_radius_deg"] = 6.5
CFG["sensor"]["housing_overhead"] = 1.1
CFG["sensor"]["hex_pixel_fov_flat2flat_deg"] = 0.06667
CFG["sensor"]["num_paxel_on_diagonal"] = [1, 3, 9]
CFG["light_field_geometry"] = {}
CFG["light_field_geometry"]["num_blocks"] = 24
CFG["light_field_geometry"]["num_photons_per_block"] = 1000 * 1000
CFG["binning"] = analysis.BINNING


ANGLE_FMT = "angle{:06d}"
PAXEL_FMT = "paxel{:06d}"

def init(work_dir, config=CFG):
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(json_numpy.dumps(config, indent=4))

    with open(
        os.path.join(work_dir, "merlict_propagation_config.json"), "wt"
    ) as f:
        f.write(json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4))


def LightFieldGeometry(path, off_axis_angle_deg):
    lfg = plenopy.LightFieldGeometry(path=path)
    lfg.cx_mean += np.deg2rad(off_axis_angle_deg)
    return lfg


def run(
    work_dir,
    map_and_reduce_pool=multiprocessing.Pool(4),
    logger=json_line_logger.LoggerStdout(),
):
    logger.debug("Start")

    logger.debug("Make sceneries")
    make_sceneries_for_light_field_geometires(work_dir=work_dir)

    logger.debug("Estimate light_field_geometry")
    make_light_field_geometires(
        work_dir=work_dir,
        map_and_reduce_pool=map_and_reduce_pool,
        logger=logger,
    )

    logger.debug("Make calibration source")
    make_source(work_dir=work_dir)

    logger.debug("Make responses to calibration source")
    make_responses(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.debug("Make analysis")
    make_analysis(work_dir=work_dir, map_and_reduce_pool=map_and_reduce_pool)

    logger.debug("Plot analysis")
    make_plots(work_dir=work_dir)

    logger.debug("Stop")


def make_responses(
    work_dir,
    map_and_reduce_pool=multiprocessing.Pool(4),
):
    jobs =  _responses_make_jobs(work_dir=work_dir)
    _ = map_and_reduce_pool.map(_responses_run_job, jobs)


def _responses_make_jobs(work_dir):
    jobs = []

    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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
        job["work_dir"],
        "responses",
        job["mkey"],
        job["pkey"],
        job["akey"]
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
    work_dir,
    containment_percentile=80,
    object_distance_m=1e6,
):
    assert 0.0 < containment_percentile <= 100.0
    assert object_distance_m > 0.0

    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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
        job["work_dir"],
        "analysis",
        job["mkey"],
        job["pkey"],
        job["akey"]
    )
    os.makedirs(adir, exist_ok=True)
    summary_path = os.path.join(adir, "summary.json")

    if os.path.exists(summary_path):
        return 1

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
    thisimg_bin_edges = analysis.binning_image_bin_edges(
        binning=thisbinning
    )

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
        percentile=CONTAINMENT_PERCENTILE,
    )
    print("time full_width_half_maximum")
    (
        time_fwhm_start,
        time_fwhm_stop,
    ) = analysis.full_width_half_maximum(
        x=cres["time"]["bin_centers"],
        f=cres["time"]["weights"],
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
    object_distance_m=1e6,
    containment_percentile=80,
    map_and_reduce_pool=multiprocessing.Pool(4),
):
    jobs = _analysis_make_jobs(
        work_dir=work_dir,
        object_distance_m=object_distance_m,
        containment_percentile=containment_percentile,
    )
    _ map_and_reduce_pool.map(_responses_run_job, jobs)


def make_source(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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
                        mkey=mkey,
                        num_paxel_on_diagonal=npax,
                        config=config,
                        off_axis_angles_deg=config["sources"]["off_axis_angles_deg"][ofa]
                    )
                    f.write(json_numpy.dumps(s, indent=4))



def make_light_field_geometires(
    work_dir,
    map_and_reduce_pool=multiprocessing.Pool(4),
    logger=json_line_logger.LoggerStdout(),
):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for mkey in config["mirror"]["keys"]:
        mdir = os.path.join(geometries_dir, mkey)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            pkey = PAXEL_FMT.format(npax)
            pdir = os.path.join(mdir, pkey)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                akey = ANGLE_FMT.format(ofa)
                angle_deg = config["sources"]["off_axis_angles_deg"][ofa]
                adir = os.path.join(pdir, akey)

                logger.debug(
                    "Estimate geometry, "
                    "mirror: {:s}, ".format(mkey) +
                    "num. paxel: {:d}, ".format(npax) +
                    "angle: {:.2f}deg".format(angle_deg)
                )

                plenoirf._estimate_light_field_geometry_of_plenoscope(
                    config={
                        "light_field_geometry": {
                            "num_blocks": config["light_field_geometry"][
                                "num_blocks"
                            ],
                            "num_photons_per_block": config[
                                "light_field_geometry"
                            ]["num_photons_per_block"],
                        }
                    },
                    run_dir=adir,
                    map_and_reduce_pool=map_and_reduce_pool,
                    executables=config["executables"],
                    logger=logger,
                    make_plots=(npax < 3),
                )


def read_analysis(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

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


# def make_summary_plots(work_dir):
