import os
import json_numpy


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
    num_paxel_on_diagonal,
):
    return num_paxel_on_diagonal * num_paxel_on_diagonal
