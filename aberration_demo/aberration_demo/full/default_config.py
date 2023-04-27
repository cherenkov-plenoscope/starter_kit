import os
import json_numpy
import copy
import phantom_source
from .. import deformations
from .. import portal
from .. import merlict


def write_default_config(cfg_dir, minimal):
    os.makedirs(cfg_dir, exist_ok=True)

    write_merlict_config(cfg_dir=cfg_dir)
    write_statistics_config(cfg_dir=cfg_dir, minimal=minimal)

    write_mirrors_config(cfg_dir=cfg_dir)
    write_mirror_deformations(cfg_dir=cfg_dir)

    write_sensors_config(cfg_dir=cfg_dir)
    write_sensors_transformations(cfg_dir=cfg_dir)

    write_instruments_config(cfg_dir=cfg_dir, minimal=minimal)

    write_observations_config(cfg_dir=cfg_dir, minimal=minimal)


def write_instruments_config(cfg_dir, minimal):
    cfg_inst_dir = os.path.join(cfg_dir, "instruments")
    os.makedirs(cfg_inst_dir, exist_ok=True)

    # for off-axis demo and for deformation demo
    # ------------------------------------------
    paxel_configurations = (
        ["diag3", "diag1"] if minimal else ["diag9", "diag3", "diag1"]
    )

    tkey = "default"
    for dkey in ["default", "perlin55mm"]:
        for skey in paxel_configurations:
            json_numpy.write(
                os.path.join(
                    cfg_inst_dir,
                    "{:s}_{:s}_{:s}.json".format(skey, dkey, tkey),
                ),
                {
                    "mirror": "71m",
                    "mirror_deformation": dkey,
                    "sensor": skey,
                    "sensor_transformation": tkey,
                },
            )

    # for misalignment demo
    # ---------------------
    dkey = "default"
    tkey = "gentle"
    for skey in paxel_configurations:
        json_numpy.write(
            os.path.join(
                cfg_inst_dir, "{:s}_{:s}_{:s}.json".format(skey, dkey, tkey)
            ),
            {
                "mirror": "71m",
                "mirror_deformation": dkey,
                "sensor": skey,
                "sensor_transformation": tkey,
            },
        )

    # for phantom demo
    # ----------------
    dkey = "perlin55mm"
    tkey = "gentle"
    for skey in paxel_configurations:
        json_numpy.write(
            os.path.join(
                cfg_inst_dir, "{:s}_{:s}_{:s}.json".format(skey, dkey, tkey)
            ),
            {
                "mirror": "71m",
                "mirror_deformation": dkey,
                "sensor": skey,
                "sensor_transformation": tkey,
            },
        )


def write_merlict_config(cfg_dir):
    cfg_merl_dir = os.path.join(cfg_dir, "merlict")
    os.makedirs(cfg_merl_dir, exist_ok=True)
    json_numpy.write(
        os.path.join(cfg_merl_dir, "executables.json"), merlict.EXECUTABLES
    )
    json_numpy.write(
        os.path.join(cfg_merl_dir, "merlict_propagation_config.json"),
        merlict.PROPAGATION_CONFIG,
    )


def write_mirror_deformations(cfg_dir):
    cfg_mdef_dir = os.path.join(cfg_dir, "mirror_deformations")
    os.makedirs(cfg_mdef_dir, exist_ok=True)
    json_numpy.write(
        os.path.join(cfg_mdef_dir, "default.json"),
        deformations.deformation_map.ZERO_MIRROR_DEFORMATION,
    )
    json_numpy.write(
        os.path.join(cfg_mdef_dir, "perlin55mm.json"),
        deformations.deformation_map.EXAMPLE_MIRROR_DEFORMATION,
    )


def write_statistics_config(cfg_dir, minimal):
    cfg_stat_dir = os.path.join(cfg_dir, "statistics")
    os.makedirs(cfg_stat_dir, exist_ok=True)
    json_numpy.write(
        os.path.join(cfg_stat_dir, "light_field_geometry.json"),
        {
            "num_blocks": 1 if minimal else 16,
            "num_photons_per_block": 1000 * 100 if minimal else 1000,
        },
    )


def write_mirrors_config(cfg_dir):
    cfg_mirg_dir = os.path.join(cfg_dir, "mirrors")
    os.makedirs(cfg_mirg_dir, exist_ok=True)
    json_numpy.write(os.path.join(cfg_mirg_dir, "71m.json"), portal.MIRROR)


def write_sensors_config(cfg_dir):
    cfg_lfsg_dir = os.path.join(cfg_dir, "sensors")
    os.makedirs(cfg_lfsg_dir, exist_ok=True)
    _p61 = copy.deepcopy(portal.SENSOR)
    _p61["num_paxel_on_pixel_diagonal"] = 9
    json_numpy.write(os.path.join(cfg_lfsg_dir, "diag9.json"), _p61)
    _p7 = copy.deepcopy(portal.SENSOR)
    _p7["num_paxel_on_pixel_diagonal"] = 3
    json_numpy.write(os.path.join(cfg_lfsg_dir, "diag3.json"), _p7)
    _t1 = copy.deepcopy(portal.SENSOR)
    _t1["num_paxel_on_pixel_diagonal"] = 1
    json_numpy.write(os.path.join(cfg_lfsg_dir, "diag1.json"), _t1)


def write_sensors_transformations(cfg_dir):
    cfg_stra_dir = os.path.join(cfg_dir, "sensor_transformations")
    os.makedirs(cfg_stra_dir, exist_ok=True)
    json_numpy.write(
        os.path.join(cfg_stra_dir, "default.json"),
        portal.SENSOR_TRANSFORMATION_DEFAULT,
    )
    json_numpy.write(
        os.path.join(cfg_stra_dir, "gentle.json"),
        portal.SENSOR_TRANSFORMATION_GENTLE,
    )


def write_observations_config(cfg_dir, minimal):
    cfg_obsv_dir = os.path.join(cfg_dir, "observations")
    os.makedirs(cfg_obsv_dir, exist_ok=True)

    json_numpy.write(
        os.path.join(cfg_obsv_dir, "start.json"),
        {
            "num_stars": 50 if minimal else 500,
            "max_angle_off_optical_axis_deg": 4.0,
        },
    )

    json_numpy.write(
        os.path.join(cfg_obsv_dir, "point.json"),
        {
            "num_points": 40 if minimal else 4096,
            "max_angle_off_optical_axis_deg": 3.25,
        },
    )

    cfg_phan_dir = os.path.join(cfg_obsv_dir, "phantom")
    os.makedirs(cfg_phan_dir, exist_ok=True)

    (
        mesch_scn,
        mesh_img,
        mesh_depth,
    ) = phantom_source.demonstration._make_mesh_for_phantom_source(
        intensity=36 if minimal else 360
    )
    json_numpy.write(
        os.path.join(cfg_phan_dir, "phantom_source_meshes.json"), mesch_scn,
    )
    json_numpy.write(
        os.path.join(cfg_phan_dir, "phantom_source_meshes_img.json"), mesh_img,
    )
    json_numpy.write(
        os.path.join(cfg_phan_dir, "phantom_source_meshes_depth.json"),
        mesh_depth,
    )
