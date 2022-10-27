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
    with open(os.path.join(work_dir, "config.json"), "rt") as f:
        config = json_numpy.loads(f.read())
    return config


def guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
    num_paxel_on_diagonal,
):
    return num_paxel_on_diagonal * num_paxel_on_diagonal
