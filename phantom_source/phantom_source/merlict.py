import numpy as np
import os
import subprocess
import shutil
import tempfile
import plenopy


def propagate_photons(
    input_path,
    output_path,
    light_field_geometry_path,
    merlict_propagate_photons_path,
    merlict_propagate_config_path,
    random_seed=0,
):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mct_propagate_call = [
        merlict_propagate_photons_path,
        "-l",
        light_field_geometry_path,
        "-c",
        merlict_propagate_config_path,
        "-i",
        input_path,
        "-o",
        output_path,
        "--all_truth",
        "-r",
        str(random_seed),
    ]

    o_path = output_path + ".stdout.txt"
    e_path = output_path + ".stderr.txt"
    with open(o_path, "wt") as fo, open(e_path, "wt") as fe:
        rc = subprocess.call(mct_propagate_call, stdout=fo, stderr=fe)
    return rc


"""
[0] id,
[1] [2] [3] support
[4] [5] [6] direction
[7] wavelength
"""

def append_photons_to_space_seperated_values(
    path, ids, supports, directions, wavelengths
):
    with open(path, "at") as f:
        for i in range(len(ids)):
            f.write(
                "{:d} {:.3e} {:.3e} {:.3e} {:.9e} {:.9e} {:.9e} {:.3e}".format(
                    ids[i],
                    supports[i, 0],
                    supports[i, 1],
                    supports[i, 2],
                    directions[i, 0],
                    directions[i, 1],
                    directions[i, 2],
                    wavelengths[i],
                )
            )
            f.write("\n")


def write_light_fields_to_space_seperated_values(light_fields, path):
    curid = 0
    for lf in light_fields:
        sups = lf[0]
        dirs = lf[1]
        ids = np.arange(curid, curid + len(sups))
        curid += len(sups)

        append_photons_to_space_seperated_values(
            path=path,
            ids=ids,
            supports=sups,
            directions=dirs,
            wavelengths=np.ones(len(sups)) * 433e-9,
        )


def make_plenopy_event_and_read_light_field_geometry(
    light_fields,
    light_field_geometry_path,
    merlict_propagate_photons_path,
    merlict_propagate_config_path,
    random_seed=0,
):
    with tempfile.TemporaryDirectory(prefix="phantom_source_") as tmpdir:
        photons_path = os.path.join(tmpdir, "photons.ssv")
        run_dir = os.path.join(tmpdir, "run")

        write_light_fields_to_space_seperated_values(
            light_fields=light_fields,
            path=photons_path,
        )

        rc = propagate_photons(
            input_path=photons_path,
            output_path=run_dir,
            light_field_geometry_path=light_field_geometry_path,
            merlict_propagate_photons_path=merlict_propagate_photons_path,
            merlict_propagate_config_path=merlict_propagate_config_path,
            random_seed=0,
        )

        light_field_geometry = plenopy.LightFieldGeometry(light_field_geometry_path)
        event = plenopy.Event(
            os.path.join(run_dir, "1"),
            light_field_geometry=light_field_geometry,
        )

        return event, light_field_geometry
