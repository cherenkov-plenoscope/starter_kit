import os
import json_numpy
import plenopy
import pkg_resources
import subprocess


ANGLE_FMT = "angle{:06d}"
PAXEL_FMT = "paxel{:06d}"


def read_config(work_dir):
    """
    Returns the config in work_dir/config.json.

    Parameters
    ----------
    work_dir : str
        Path to the work_dir
    """
    return read_json(path=os.path.join(work_dir, "config.json"))


def read_json(path):
    with open(path, "rt") as f:
        cont = json_numpy.loads(f.read())
    return cont


def guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
    num_paxel_on_pixel_diagonal,
):
    return num_paxel_on_pixel_diagonal * num_paxel_on_pixel_diagonal


def get_instrument_geometry_from_light_field_geometry(
    light_field_geometry=None, light_field_geometry_path=None,
):
    if light_field_geometry_path:
        assert light_field_geometry is None
        geom_path = os.path.join(
            light_field_geometry_path, "light_field_sensor_geometry.header.bin"
        )
        geom_header = plenopy.corsika.utils.hr.read_float32_header(geom_path)
        geom = plenopy.light_field_geometry.PlenoscopeGeometry(raw=geom_header)
    else:
        geom = light_field_geometry.sensor_plane2imaging_system
    return class_members_to_dict(c=geom)


def class_members_to_dict(c):
    member_keys = []
    for key in dir(c):
        if not callable(getattr(c, key)):
            if not str.startswith(key, "__"):
                member_keys.append(key)
    out = {}
    for key in member_keys:
        out[key] = getattr(c, key)
    return out


def run_script(script_path, argv):
    if not script_path.endswith(".py"):
        script_path += ".py"
    explicit_script_path = pkg_resources.resource_filename(
        "aberration_demo", script_path,
    )
    args = []
    args.append("python")
    args.append(explicit_script_path)
    args += argv
    return subprocess.call(args)
