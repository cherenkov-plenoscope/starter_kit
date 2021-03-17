import os
from os import path as op
import json

import magnetic_deflection
from .. import utils

def absjoin(*args):
    return op.abspath(op.join(*args))


CORSIKA_PRIMARY_PATH = absjoin(
    "build",
    "corsika",
    "modified",
    "corsika-75600",
    "run",
    "corsika75600Linux_QGSII_urqmd",
)

MERLICT_PLENOSCOPE_PROPAGATOR_PATH = absjoin(
    "build", "merlict", "merlict-plenoscope-propagation"
)

EXAMPLE_SITE = {
    "observation_level_asl_m": 5000,
    "earth_magnetic_field_x_muT": 20.815,
    "earth_magnetic_field_z_muT": -11.366,
    "atmosphere_id": 26,
}

EXAMPLE_PLENOSCOPE_POINTING = {"azimuth_deg": 0.0, "zenith_deg": 0.0}

EXAMPLE_PARTICLE = {
    "particle_id": 14,
    "energy_bin_edges_GeV": [20, 200],
    "max_scatter_angle_deg": 13,
    "energy_power_law_slope": -1.0,
}

EXAMPLE_SITE_PARTICLE_DEFLECTION = {
    "energy_GeV": [5, 1000],
    "primary_azimuth_deg": [0.0, 0.0],
    "primary_zenith_deg": [0.0, 0.0],
    "cherenkov_pool_x_m": [0.0, 0.0],
    "cherenkov_pool_y_m": [0.0, 0.0],
}

EXAMPLE_GRID = {
    "num_bins_radius": 512,
    "threshold_num_photons": 50,
    "field_of_view_overhead": 1.1,
    "bin_width_overhead": 1.1,
}

EXAMPLE_SUM_TRIGGER = {
    "object_distances_m": [
        5000.0,
        6164.0,
        7600.0,
        9369.0,
        11551.0,
        14240.0,
        17556.0,
        21644.0,
        26683.0,
        32897.0,
        40557.0,
        50000.0,
    ],
    "threshold_pe": 107,
    "integration_time_slices": 10,
    "image": {
        "image_outer_radius_deg": 3.216665,
        "pixel_spacing_deg": 0.06667,
        "pixel_radius_deg": 0.146674,
        "max_number_nearest_lixel_in_pixel": 7,
    },
}

EXAMPLE_CHERENKOV_CLASSIFICATION = {
    "region_of_interest": {
        "time_offset_start_s": -10e-9,
        "time_offset_stop_s": 10e-9,
        "direction_radius_deg": 2.0,
        "object_distance_offsets_m": [4000.0, 2000.0, 0.0, -2000.0,],
    },
    "min_num_photons": 17,
    "neighborhood_radius_deg": 0.075,
    "direction_to_time_mixing_deg_per_s": 0.375e9,
}

ARTIFICIAL_CORE_LIMITATION = {
    "gamma": {
        "energy_GeV": [0.23, 0.8, 3.0, 35, 81, 432, 1000],
        "max_scatter_radius_m": [150, 150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy_GeV": [0.23, 1.0, 10, 100, 1000],
        "max_scatter_radius_m": [150, 150, 500, 1100, 2600],
    },
    "proton": {
        "energy_GeV": [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    },
}
ARTIFICIAL_CORE_LIMITATION["helium"] = ARTIFICIAL_CORE_LIMITATION[
    "proton"
].copy()

# ARTIFICIAL_CORE_LIMITATION = None


def make_example_job(
    run_dir,
    num_air_showers=25,
    example_dirname="_testing",
    particle_key="proton",
    site_key="namibia",
):
    deflection_table = magnetic_deflection.read(
        work_dir=op.join(run_dir, "magnetic_deflection"), style="dict",
    )
    test_dir = op.join(run_dir, example_dirname)
    with open(op.join(run_dir, "input", "config.json"), "rt") as fin:
        config = json.loads(fin.read())

    job = {
        "run_id": 1,
        "num_air_showers": num_air_showers,
        "particle": config["particles"][particle_key],
        "plenoscope_pointing": config["plenoscope_pointing"],
        "site": config["sites"][site_key],
        "grid": config["grid"],
        "sum_trigger": config["sum_trigger"],
        "corsika_primary_path": CORSIKA_PRIMARY_PATH,
        "plenoscope_scenery_path": op.join(
            run_dir, "light_field_geometry", "input", "scenery"
        ),
        "merlict_plenoscope_propagator_path": MERLICT_PLENOSCOPE_PROPAGATOR_PATH,
        "light_field_geometry_path": op.join(run_dir, "light_field_geometry"),
        "trigger_geometry_path": op.join(run_dir, "trigger_geometry"),
        "merlict_plenoscope_propagator_config_path": op.join(
            run_dir, "input", "merlict_propagation_config.json"
        ),
        "site_particle_deflection": deflection_table[site_key][particle_key],
        "cherenkov_classification": config["cherenkov_classification"],
        "log_dir": op.join(test_dir, "log"),
        "past_trigger_dir": op.join(test_dir, "past_trigger"),
        "past_trigger_reconstructed_cherenkov_dir": op.join(
            test_dir, "past_trigger_reconstructed_cherenkov"
        ),
        "feature_dir": op.join(test_dir, "features"),
        "keep_tmp": True,
        "tmp_dir": op.join(test_dir, "tmp"),
        "date": utils.date_dict_now(),
        "artificial_core_limitation": ARTIFICIAL_CORE_LIMITATION[particle_key],
    }
    return job
