import plenoirf
import rename_after_writing as rnw
import json_utils
import numpy as np
import os

storage_dir = os.path.join("/", "lfs", "l8", "hin", "relleums", "portal")

production_date_str = "2020-08-09"

PHD_CORE_LIMITATION = {
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
    "helium": {
        "energy_GeV": [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    },
}

NO_CORE_LIMITATION = {
    "gamma": None,
    "electron": None,
    "proton": None,
    "helium": None,
}

CONFIG = {
    "light_field_geometry": {
        "num_photons_per_block": 4000000,
        "num_blocks": 360,
    },
    "plenoscope_pointing": {"azimuth_deg": 0.0, "zenith_deg": 0.0},
    "sites": {
        "chile": {
            "observation_level_asl_m": 5000,
            "earth_magnetic_field_x_muT": 1e-9,
            "earth_magnetic_field_z_muT": 1e-9,
            "atmosphere_id": 26,
            "geomagnetic_cutoff_rigidity_GV": 10.0,
        }
    },
    "particles": {
        "gamma": {
            "particle_id": 1,
            "energy_bin_edges_GeV": [0.25, 1000],
            "max_scatter_angle_deg": 3.25,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": 0.0,
            "magnetic_deflection_max_off_axis_deg": 0.25,
        },
        "electron": {
            "particle_id": 3,
            "energy_bin_edges_GeV": [0.25, 1000],
            "max_scatter_angle_deg": 6.5,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": -1.0,
            "magnetic_deflection_max_off_axis_deg": 0.5,
        },
        "proton": {
            "particle_id": 14,
            "energy_bin_edges_GeV": [5, 1000],
            "max_scatter_angle_deg": 13,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": 1.0,
            "magnetic_deflection_max_off_axis_deg": 1.5,
        },
        "helium": {
            "particle_id": 402,
            "energy_bin_edges_GeV": [10, 1000],
            "max_scatter_angle_deg": 13,
            "energy_power_law_slope": -1.5,
            "electric_charge_qe": 2.0,
            "magnetic_deflection_max_off_axis_deg": 1.5,
        },
    },
    "grid": {
        "field_of_view_overhead": 1.1,
        "bin_width_overhead": 1.1,
        "num_bins_radius": 512,
        "threshold_num_photons": 50,
    },
    "sum_trigger": {
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
    },
    "cherenkov_classification": {
        "region_of_interest": {
            "time_offset_start_s": -1e-08,
            "time_offset_stop_s": 1e-08,
            "direction_radius_deg": 2.0,
            "object_distance_offsets_m": [4000.0, 2000.0, 0.0, -2000.0],
        },
        "min_num_photons": 17,
        "neighborhood_radius_deg": 0.075,
        "direction_to_time_mixing_deg_per_s": 375000000.0,
    },
    "runs": {
        "gamma": {"num": 64, "first_run_id": 1},
        "electron": {"num": 64, "first_run_id": 1},
        "proton": {"num": 64, "first_run_id": 1},
        "helium": {"num": 64, "first_run_id": 1},
    },
    "magnetic_deflection": {"num_energy_supports": 512, "max_energy_GeV": 64},
    "num_airshowers_per_run": 100,
    "artificial_core_limitation": {
        "gamma": None,
        "electron": None,
        "proton": None,
        "helium": None,
    },
}

SCENERY = {
    "functions": [
        {
            "name": "mirror_reflectivity_vs_wavelength",
            "argument_versus_value": [
                [2.40e-07, 2.523e-01],
                [2.49e-07, 4.664e-01],
                [2.59e-07, 6.657e-01],
                [2.68e-07, 8.117e-01],
                [2.78e-07, 8.775e-01],
                [2.87e-07, 9.185e-01],
                [2.96e-07, 9.514e-01],
                [3.06e-07, 9.606e-01],
                [3.15e-07, 9.633e-01],
                [3.24e-07, 9.680e-01],
                [3.34e-07, 9.755e-01],
                [3.43e-07, 9.653e-01],
                [3.53e-07, 9.542e-01],
                [3.62e-07, 9.732e-01],
                [3.71e-07, 9.872e-01],
                [3.81e-07, 9.865e-01],
                [3.90e-07, 9.782e-01],
                [4.00e-07, 9.808e-01],
                [4.09e-07, 9.795e-01],
                [4.18e-07, 9.623e-01],
                [4.28e-07, 9.560e-01],
                [4.37e-07, 9.801e-01],
                [4.47e-07, 9.775e-01],
                [4.56e-07, 9.508e-01],
                [4.65e-07, 9.437e-01],
                [4.75e-07, 9.713e-01],
                [4.84e-07, 9.820e-01],
                [4.93e-07, 9.769e-01],
                [5.03e-07, 9.763e-01],
                [5.12e-07, 9.822e-01],
                [5.22e-07, 9.745e-01],
                [5.31e-07, 9.722e-01],
                [5.40e-07, 9.821e-01],
                [5.50e-07, 9.821e-01],
                [5.59e-07, 9.771e-01],
                [5.69e-07, 9.767e-01],
                [5.78e-07, 9.828e-01],
                [5.87e-07, 9.828e-01],
                [5.97e-07, 9.746e-01],
                [6.06e-07, 9.421e-01],
                [6.16e-07, 7.582e-01],
                [6.25e-07, 5.498e-01],
                [6.34e-07, 5.072e-01],
                [6.44e-07, 3.086e-01],
                [6.53e-07, 1.295e-01],
                [6.62e-07, 1.914e-01],
                [6.72e-07, 3.118e-01],
                [6.81e-07, 2.781e-01],
                [6.91e-07, 1.611e-01],
                [7.01e-07, 1.214e-01],
            ],
            "comment": "https://arxiv.org/abs/1310.1713, MST dielectric mirror",
        },
        {
            "name": "lens_refraction_vs_wavelength",
            "argument_versus_value": [
                [240e-9, 1.5133],
                [280e-9, 1.4942],
                [320e-9, 1.4827],
                [360e-9, 1.4753],
                [400e-9, 1.4701],
                [486e-9, 1.4631],
                [546e-9, 1.4601],
                [633e-9, 1.4570],
                [694e-9, 1.4554],
                [753e-9, 1.4542],
            ],
            "comment": "Hereaus Quarzglas GmbH and Co. KG, Quarzstr. 8, 63450 Hanau, Suprasil Family 311/312/313",
        },
    ],
    "colors": [],
    "children": [
        {
            "type": "Frame",
            "name": "Portal",
            "pos": [0, 0, 0],
            "rot": [0, 0, 0],
            "children": [
                {
                    "type": "SegmentedReflector",
                    "name": "reflector",
                    "pos": [0, 0, 0],
                    "rot": [0, 0, 0],
                    "focal_length": 106.5,
                    "max_outer_aperture_radius": 35.5,
                    "min_inner_aperture_radius": 3.05,
                    "DaviesCotton_over_parabolic_mixing_factor": 0.0,
                    "facet_inner_hex_radius": 0.75,
                    "gap_between_facets": 0.025,
                    "surface": {
                        "outer_reflection": "mirror_reflectivity_vs_wavelength"
                    },
                    "children": [],
                },
                {
                    "type": "LightFieldSensor",
                    "name": "light_field_sensor",
                    "pos": [0, 0, 106.5],
                    "rot": [0, 0, 0],
                    "expected_imaging_system_focal_length": 106.5,
                    "expected_imaging_system_aperture_radius": 35.5,
                    "max_FoV_diameter_deg": 6.5,
                    "hex_pixel_FoV_flat2flat_deg": 0.06667,
                    "num_paxel_on_pixel_diagonal": 9,
                    "housing_overhead": 1.1,
                    "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
                    "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
                    "children": [],
                },
            ],
        }
    ],
}

MERLICT_PROPAGATION_CONFIG = {
    "night_sky_background_ligth": {
        "flux_vs_wavelength": [
            [2.40e-07, 1.651e18],
            [2.70e-07, 2.018e18],
            [3.00e-07, 2.428e18],
            [3.30e-07, 2.357e18],
            [3.40e-07, 2.914e18],
            [3.94e-07, 3.772e18],
            [3.95e-07, 1.819e19],
            [3.96e-07, 3.753e18],
            [4.33e-07, 4.577e18],
            [4.35e-07, 9.225e18],
            [4.37e-07, 4.535e18],
            [4.62e-07, 5.362e18],
            [4.85e-07, 6.129e18],
            [5.25e-07, 7.550e18],
            [5.55e-07, 8.659e18],
            [5.58e-07, 8.425e19],
            [5.61e-07, 8.213e18],
            [5.85e-07, 8.893e18],
            [5.91e-07, 3.554e19],
            [5.95e-07, 1.216e19],
            [6.00e-07, 1.073e19],
            [6.05e-07, 8.680e18],
            [6.27e-07, 1.272e19],
            [6.30e-07, 6.692e19],
            [6.32e-07, 1.137e19],
            [6.38e-07, 2.881e19],
            [6.40e-07, 8.128e18],
            [6.45e-07, 8.065e18],
            [6.47e-07, 1.072e19],
            [6.60e-07, 1.051e19],
            [6.70e-07, 7.764e18],
            [6.78e-07, 7.673e18],
            [6.85e-07, 1.570e19],
            [7.00e-07, 9.895e18],
        ],
        "exposure_time": 50e-9,
        "comment": "Night Sky Background Analysis for the Cherenkov Telescope Array using the Atmoscope instrument, Markus Gaug, arXiv preprint arXiv:1307.3053, based on Benn Ch.R., Ellison S.R., La Palma tech. note, 2007; wavelength[m] flux[1/(s m^2 sr m)]",
    },
    "photo_electric_converter": {
        "quantum_efficiency_vs_wavelength": [
            [240e-9, 0.0673],
            [260e-9, 0.1812],
            [280e-9, 0.2758],
            [300e-9, 0.3686],
            [320e-9, 0.3866],
            [340e-9, 0.3975],
            [360e-9, 0.4017],
            [380e-9, 0.4083],
            [400e-9, 0.4010],
            [420e-9, 0.3764],
            [440e-9, 0.3576],
            [460e-9, 0.3143],
            [480e-9, 0.2782],
            [500e-9, 0.2480],
            [540e-9, 0.1257],
            [580e-9, 0.0837],
            [620e-9, 0.0396],
            [660e-9, 0.0107],
            [701e-9, 0.0001],
        ],
        "dark_rate": 1e-3,
        "probability_for_second_puls": 0.0,
        "comment": "CTA LST PMT Hamamatsu R11920-100-05 Collection efficiency assumed: 0.95 Novel Photo Multiplier Tubes for the Cherenkov Telescope Array Project ICRC 2013, Takeshi Toyama wavelength[m] pde[1]",
    },
    "photon_stream": {
        "time_slice_duration": 0.5e-9,
        "single_photon_arrival_time_resolution": 0.416e-9,
    },
}

NUM_RUNS = 1000

scenarios = {
    "portal": {
        "artificial_core_limitation": NO_CORE_LIMITATION,
        "runs": {
            "gamma": {"num": 2 * NUM_RUNS, "first_run_id": 1},
            "electron": {"num": NUM_RUNS, "first_run_id": 1},
            "proton": {"num": NUM_RUNS, "first_run_id": 1},
            "helium": {"num": NUM_RUNS, "first_run_id": 1},
        },
    },
    "portal_limit_core_seed_1": {
        "artificial_core_limitation": PHD_CORE_LIMITATION,
        "runs": {
            "gamma": {"num": 2 * NUM_RUNS, "first_run_id": 1},
            "electron": {"num": NUM_RUNS, "first_run_id": 1},
            "proton": {"num": NUM_RUNS, "first_run_id": 1},
            "helium": {"num": NUM_RUNS, "first_run_id": 1},
        },
    },
    "portal_limit_core_seed_2": {
        "artificial_core_limitation": PHD_CORE_LIMITATION,
        "runs": {
            "gamma": {"num": 2 * NUM_RUNS, "first_run_id": 1 + 2 * NUM_RUNS},
            "electron": {"num": NUM_RUNS, "first_run_id": 1 + NUM_RUNS},
            "proton": {"num": NUM_RUNS, "first_run_id": 1 + NUM_RUNS},
            "helium": {"num": NUM_RUNS, "first_run_id": 1 + NUM_RUNS},
        },
    },
}

tmp_cfg_dir = os.path.join(storage_dir, production_date_str + "_tmp")
os.makedirs(tmp_cfg_dir, exist_ok=True)
merlict_cfg_files = {
    "merlict_plenoscope_propagator_config_path": os.path.join(
        tmp_cfg_dir, "merlict_config.json"
    ),
    "plenoscope_scenery_path": os.path.join(tmp_cfg_dir, "scenery"),
}
json_utils.write(
    merlict_cfg_files["merlict_plenoscope_propagator_config_path"],
    MERLICT_PROPAGATION_CONFIG,
)
if not os.path.exists(merlict_cfg_files["plenoscope_scenery_path"]):
    rnw.copy(
        src=os.path.join(
            tmp_cfg_dir, "light_field_geometry", "input", "scenery"
        ),
        dst=merlict_cfg_files["plenoscope_scenery_path"],
    )

for scenario_key in scenarios:

    config = CONFIG.copy()
    config["artificial_core_limitation"] = scenarios[scenario_key][
        "artificial_core_limitation"
    ].copy()
    config["runs"] = scenarios[scenario_key]["runs"].copy()

    scenario_path = os.path.join(
        storage_dir, production_date_str + "_" + scenario_key
    )

    plenoirf.init(
        out_dir=scenario_path, config=config, cfg_files=merlict_cfg_files
    )

    rnw.copy(
        src=os.path.join(tmp_cfg_dir, "light_field_geometry"),
        dst=os.path.join(scenario_path, "light_field_geometry"),
    )

    rnw.copy(
        src=os.path.join(tmp_cfg_dir, "magnetic_deflection"),
        dst=os.path.join(scenario_path, "magnetic_deflection"),
    )

    rnw.copy(
        src=os.path.join(tmp_cfg_dir, "trigger_geometry"),
        dst=os.path.join(scenario_path, "trigger_geometry"),
    )


for scenario_key in scenarios:
    scenario_path = os.path.join(
        storage_dir, production_date_str + "_" + scenario_key
    )
    plenoirf.run(path=scenario_path)
