"""
I must be called in a fermipy environment!
"""

import os
import subprocess
import astropy.io.fits
import numpy as np
import fermipy
import json
import tempfile
import datetime

fermipy_version = fermipy.get_git_version()
creation_time = datetime.datetime.now().isoformat()

day = 3600 * 24
year = 365 * day
observation_times = np.geomspace(60, 10 * year, 27)

cfg = {
    "--glon": {
        "value": 120.0,
        "help": "Galactic longitude in deg at which the sensitivity "
        "will be evaluated. Also sets the center of the sensitivity map "
        "for the `wcs` map type.",
    },
    "--glat": {
        "value": 60.0,
        "help": "Galactic latitude in deg at which the sensitivity "
        "will be evaluated. Also sets the center of the sensitivity "
        "map for the `wcs` map type.",
    },
    "--ltcube": {
        "value": None,
        "help": "Set the path to the livetime cube. If no livetime cube "
        "is provided the calculation will use an idealized observation "
        "profile for a uniform all-sky survey with no Earth obscuration "
        "or deadtime.",
    },
    "--galdiff": {
        "value": "data/gll_iem_v06.fits",
        "help": "Set the path to the galactic diffuse model used for fitting. "
        "This can be used to assess the impact of IEM systematics  "
        "from fitting with the wrong model.  If none "
        "then the same model will be used for data and fit.",
    },
    "--event_class": {
        "value": "P8R2_SOURCE_V6",
        "help": "Set the IRF name (e.g. P8R2_SOURCE_V6).",
    },
    "--min_counts": {
        "value": 10.0,
        "help": "Set the minimum number of counts.",
    },
    "--ts_thresh": {
        "value": 25.0,
        "help": "Set the test statistic (TS) detection threshold.",
    },
    "--obs_time_yr": {
        "value": 1,
        "help": "Rescale the livetime cube to this observation time in years. "
        "If none then the calculation will use the intrinsic observation "
        "time of the livetime cube.",
    },
}

COMAND_NAME = "fermipy-flux-sensitivity"

for i in range(len(observation_times)):
    output_path = "lat_sensitivity_{:06d}s.fits".format(i)

    if not os.path.exists(output_path):

        output_part_path = output_path + ".part"

        cmd = [COMAND_NAME]
        cmd.append("--glon={:f}".format(cfg["--glon"]["value"]))
        cmd.append("--glat={:f}".format(cfg["--glat"]["value"]))
        cmd.append("--output={:s}".format(output_part_path))
        cmd.append("--galdiff={:s}".format(cfg["--galdiff"]["value"]))
        cmd.append("--event_class={:s}".format(cfg["--event_class"]["value"]))
        cmd.append("--ts_thresh={:f}".format(cfg["--ts_thresh"]["value"]))
        cmd.append("--min_counts={:f}".format(cfg["--min_counts"]["value"]))
        cmd.append("--obs_time_yr={:f}".format(observation_times[i] / year))
        if cfg["--ltcube"]["value"] is not None:
            cmd.append("--ltcube={:s}".format(cfg["--ltcube"]["value"]))

        print(output_path)
        subprocess.call(cmd)
        os.rename(output_part_path, output_path)

FIELDS_OF_INTEREST = ["e_min", "e_max", "dnde"]
HDU_OF_INTEREST = "DIFF_FLUX"

units = {}
bundle = []
for i in range(len(observation_times)):
    path = "lat_sensitivity_{:06d}s.fits".format(i)
    bundle.append({})
    with astropy.io.fits.open(path) as ff:
        hdu = ff[HDU_OF_INTEREST]
        for column in hdu.columns:
            if column.name in FIELDS_OF_INTEREST:
                if column.name in units:
                    assert units[column.name] == column.unit
                else:
                    units[column.name] = column.unit
        bundle[i] = {}
        for key in FIELDS_OF_INTEREST:
            bundle[i][key] = hdu.data[key]


# compactify energy-bin-edges
# ===========================

# assert all energy-bins are equal
e_min = bundle[0]["e_min"]
e_max = bundle[0]["e_max"]
assert units["e_min"] == units["e_max"]
for i in range(len(observation_times)):
    np.testing.assert_array_equal(e_min, bundle[i]["e_min"])
    np.testing.assert_array_equal(e_max, bundle[i]["e_max"])

# assert bin edges are touching each other
np.testing.assert_array_equal(e_min[1:], e_max[:-1])

energy_bin_edges = [float(e) for e in e_min]
energy_bin_edges.append(e_max[-1])

# create matrix
# =============
dnde_vs_observation_time_vs_energy = []
for i in range(len(observation_times)):
    dnde_vs_observation_time_vs_energy.append(bundle[i]["dnde"].tolist())

# export
# ======
out = {
    "energy_bin_edges": {
        "value": energy_bin_edges,
        "unit": units["e_min"],
    },
    "observation_times": {
        "value": observation_times.tolist(),
        "unit": "s",
    },
    "dnde": {
        "value": dnde_vs_observation_time_vs_energy,
        "unit": units["dnde"],
        "axes": [
            "observation_time",
            "energy",
        ],
    },
    "provenance": {
        "creation_time": creation_time,
        "author": "Sebastian A. Meuller",
        "comment": "Calling fermipy {:s} once for each observation_time.".format(
            COMAND_NAME
        ),
        "fermipy": fermipy_version,
        "command": COMAND_NAME,
        "options": cfg,
    },
}

with open("dnde_vs_observation_time_vs_energy.json", "wt") as f:
    f.write(json.dumps(out, indent=4))
