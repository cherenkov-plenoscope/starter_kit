import subprocess
import json


def plenoscope_propagator(
    corsika_run_path,
    output_path,
    light_field_geometry_path,
    merlict_plenoscope_propagator_path,
    merlict_plenoscope_propagator_config_path,
    random_seed,
    photon_origins,
    stdout_path,
    stderr_path,
):
    """
    Calls the merlict Cherenkov-plenoscope propagation
    and saves the stdout and stderr
    """
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        call = [
            merlict_plenoscope_propagator_path,
            "-l",
            light_field_geometry_path,
            "-c",
            merlict_plenoscope_propagator_config_path,
            "-i",
            corsika_run_path,
            "-o",
            output_path,
            "-r",
            "{:d}".format(random_seed),
        ]
        if photon_origins:
            call.append("--all_truth")
        mct_rc = subprocess.call(call, stdout=out, stderr=err)
    return mct_rc


def read_plenoscope_geometry(merlict_scenery_path):
    with open(merlict_scenery_path, "rt") as f:
        _scenery = json.loads(f.read())
    children = _scenery["children"]
    for child in children:
        if child["type"] == "Frame" and child["name"] == "Portal":
            protal = child.copy()
    for child in protal["children"]:
        if child["type"] == "LightFieldSensor":
            light_field_sensor = child.copy()
    return light_field_sensor
