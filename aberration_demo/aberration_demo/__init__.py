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
import corsika_primary as cpw
import plenopy
import sebastians_matplotlib_addons as sebplt
from . import merlict
from . import analysis


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
CFG["sources"]["off_axis_angles_deg"] = [[0.0, 0.0], [4.0, 0.0], [8.0, 0.0]]
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
CFG["sensor"]["fov_radius_deg"] = 9.0
CFG["sensor"]["housing_overhead"] = 1.1
CFG["sensor"]["hex_pixel_fov_flat2flat_deg"] = 0.1
CFG["sensor"]["num_paxel_on_diagonal"] = [1, 3, 9]
CFG["light_field_geometry"] = {}
CFG["light_field_geometry"]["num_blocks"] = 24
CFG["light_field_geometry"]["num_photons_per_block"] = 1000 * 1000
CFG["binning"] = analysis.BINNING


def init(work_dir, config=CFG):
    os.makedirs(work_dir, exist_ok=True)

    with open(os.path.join(work_dir, "config.json"), "wt") as f:
        f.write(json_numpy.dumps(config, indent=4))

    with open(
        os.path.join(work_dir, "merlict_propagation_config.json"), "wt"
    ) as f:
        f.write(json_numpy.dumps(merlict.PROPAGATION_CONFIG, indent=4))


def make_responses(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    with open(
        os.path.join(work_dir, "merlict_propagation_config.json"), "wt"
    ) as f:
        f.write(json_numpy.dumps(merlict.PROPAGATION_CONFIG))

    sources_dir = os.path.join(work_dir, "sources")
    geometries_dir = os.path.join(work_dir, "geometries")
    responses_dir = os.path.join(work_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)

    runningseed = int(config["seed"])
    for mkey in config["mirror"]["keys"]:
        mirror_dir = os.path.join(responses_dir, mkey)
        os.makedirs(mirror_dir, exist_ok=True)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            paxel_dir = os.path.join(mirror_dir, "paxel{:d}".format(npax))
            os.makedirs(paxel_dir, exist_ok=True)

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                ofa_dir = os.path.join(paxel_dir, "{:03d}".format(ofa))

                if not os.path.exists(ofa_dir):
                    plenoirf.production.merlict.plenoscope_propagator(
                        corsika_run_path=os.path.join(
                            sources_dir, "{:03d}.tar".format(ofa)
                        ),
                        output_path=ofa_dir,
                        light_field_geometry_path=os.path.join(
                            geometries_dir,
                            mkey,
                            "paxel{:d}".format(npax),
                            "light_field_geometry",
                        ),
                        merlict_plenoscope_propagator_path=config[
                            "executables"
                        ]["merlict_plenoscope_propagator_path"],
                        merlict_plenoscope_propagator_config_path=os.path.join(
                            work_dir, "merlict_propagation_config.json"
                        ),
                        random_seed=runningseed,
                        photon_origins=True,
                        stdout_path=ofa_dir + ".o",
                        stderr_path=ofa_dir + ".e",
                    )
                    shutil.rmtree(os.path.join(ofa_dir, "input"))
                runningseed += 1


def make_calibration_source_run(cx, cy, size, path, prng, aperture_radius):
    with cpw.event_tape.EventTapeWriter(path=path) as run:
        runh = np.zeros(273, dtype=np.float32)
        runh[cpw.I.RUNH.MARKER] = cpw.I.RUNH.MARKER_FLOAT32
        runh[cpw.I.RUNH.RUN_NUMBER] = 1
        runh[cpw.I.RUNH.NUM_EVENTS] = 1

        evth = np.zeros(273, dtype=np.float32)
        evth[cpw.I.EVTH.MARKER] = cpw.I.EVTH.MARKER_FLOAT32
        evth[cpw.I.EVTH.EVENT_NUMBER] = 1
        evth[cpw.I.EVTH.PARTICLE_ID] = 1
        evth[cpw.I.EVTH.TOTAL_ENERGY_GEV] = 1.0
        evth[cpw.I.EVTH.RUN_NUMBER] = runh[cpw.I.RUNH.RUN_NUMBER]
        evth[cpw.I.EVTH.NUM_REUSES_OF_CHERENKOV_EVENT] = 1

        run.write_runh(runh)
        run.write_evth(evth)
        run.write_bunches(
            cpw.calibration_light_source.draw_parallel_and_isochor_bunches(
                cx=-1.0 * cx,
                cy=-1.0 * cy,
                aperture_radius=aperture_radius,
                wavelength=433e-9,
                size=size,
                prng=prng,
                speed_of_light=299792458,
            )
        )


def make_analyse(work_dir):
    CONTAINMENT_PERCENTILE = 80
    OBJECT_DISTANCE = 1e6

    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    prng = np.random.Generator(np.random.PCG64(config["seed"]))

    analysis_dir = os.path.join(work_dir, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    for mkey in config["mirror"]["keys"]:
        mirror_dir = os.path.join(analysis_dir, mkey)
        os.makedirs(mirror_dir, exist_ok=True)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            paxkey = "paxel{:d}".format(npax)
            paxel_dir = os.path.join(mirror_dir, paxkey)
            os.makedirs(paxel_dir, exist_ok=True)

            light_field_geometry = plenopy.LightFieldGeometry(
                path=os.path.join(
                    work_dir,
                    "geometries",
                    mkey,
                    paxkey,
                    "light_field_geometry",
                )
            )

            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                ofakey = "{:03d}".format(ofa)
                ofa_dir = os.path.join(paxel_dir, ofakey)
                os.makedirs(ofa_dir, exist_ok=True)
                summary_path = os.path.join(ofa_dir, "summary.json")

                if not os.path.exists(summary_path):

                    event = plenopy.Event(
                        path=os.path.join(
                            work_dir, "responses", mkey, paxkey, ofakey, "1",
                        ),
                        light_field_geometry=light_field_geometry,
                    )

                    print(mkey, paxkey, ofakey)

                    calibrated_response = analysis.calibrate_plenoscope_response(
                        light_field_geometry=light_field_geometry,
                        event=event,
                        object_distance=OBJECT_DISTANCE,
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
                        percentile=CONTAINMENT_PERCENTILE,
                        num_sub_samples=1,
                    )

                    thisbinning = dict(config["binning"])
                    thisbinning["image"]["center"]["cx_deg"] = config[
                        "sources"
                    ]["off_axis_angles_deg"][ofa][0]
                    thisbinning["image"]["center"]["cy_deg"] = config[
                        "sources"
                    ]["off_axis_angles_deg"][ofa][1]
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


def make_sources(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    sources_dir = os.path.join(work_dir, "sources")
    os.makedirs(sources_dir, exist_ok=True)

    for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
        source_path = os.path.join(sources_dir, "{:03d}.tar".format(ofa))

        if not os.path.exists(source_path):
            prng = np.random.Generator(np.random.PCG64(config["seed"] + ofa))
            cx_deg, cy_deg = config["sources"]["off_axis_angles_deg"][ofa]
            make_calibration_source_run(
                cx=np.deg2rad(cx_deg),
                cy=np.deg2rad(cy_deg),
                size=config["sources"]["num_photons"],
                path=source_path,
                prng=prng,
                aperture_radius=1.2 * config["mirror"]["outer_radius"],
            )


def make_light_field_geometires(work_dir, map_and_reduce_pool):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    geometries_dir = os.path.join(work_dir, "geometries")
    os.makedirs(geometries_dir, exist_ok=True)

    for mkey in config["mirror"]["keys"]:
        mirror_dir = os.path.join(geometries_dir, mkey)
        os.makedirs(mirror_dir, exist_ok=True)

        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            paxel_dir = os.path.join(mirror_dir, "paxel{:d}".format(npax))

            if not os.path.exists(paxel_dir):
                os.makedirs(paxel_dir, exist_ok=True)

                scenery_dir = os.path.join(paxel_dir, "input", "scenery")
                os.makedirs(scenery_dir, exist_ok=True)
                with open(
                    os.path.join(scenery_dir, "scenery.json"), "wt"
                ) as f:
                    s = merlict.make_plenoscope_scenery_for_merlict(
                        mirror_key=mkey,
                        num_paxel_on_diagonal=npax,
                        cfg=config,
                    )
                    f.write(json_numpy.dumps(s))

                lfg_path = os.path.join(paxel_dir, "light_field_geometry")

                plenoirf._estimate_light_field_geometry_of_plenoscope(
                    cfg={
                        "light_field_geometry": {
                            "num_blocks": config["light_field_geometry"][
                                "num_blocks"
                            ]
                            * npax ** 2,
                            "num_photons_per_block": config[
                                "light_field_geometry"
                            ]["num_photons_per_block"],
                        }
                    },
                    out_absdir=paxel_dir,
                    map_and_reduce_pool=map_and_reduce_pool,
                    executables=config["executables"],
                )


def read_analysis(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    coll = {}
    for mkey in config["mirror"]["keys"]:
        coll[mkey] = {}
        for npax in config["sensor"]["num_paxel_on_diagonal"]:
            paxkey = "paxel{:d}".format(npax)
            coll[mkey][paxkey] = {}
            for ofa in range(len(config["sources"]["off_axis_angles_deg"])):
                ofakey = "{:03d}".format(ofa)

                summary_path = os.path.join(
                    work_dir, "analysis", mkey, paxkey, ofakey, "summary.json",
                )
                if not os.path.exists(summary_path):
                    print("Expected summary:", summary_path)
                    continue

                with open(summary_path, "rt") as f:
                    out = json_numpy.loads(f.read())
                coll[mkey][paxkey][ofakey] = out
    return coll


def make_plots(work_dir):
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())

    plot_dir = os.path.join(work_dir, "plot")
    os.makedirs(plot_dir, exist_ok=True)

    coll = read_analysis(work_dir=work_dir)
    for mkey in coll:
        for paxkey in coll[mkey]:
            for ofakey in coll[mkey][paxkey]:

                tcoll = coll[mkey][paxkey][ofakey]
                scenario_key = mkey + "_" + paxkey + "_" + ofakey

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
