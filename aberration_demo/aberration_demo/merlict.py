import os


EXECUTABLES = {
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


PROPAGATION_CONFIG = {
    "night_sky_background_ligth": {
        "flux_vs_wavelength": [[250.0e-9, 1.0], [700.0e-9, 1.0]],
        "exposure_time": 50e-9,
        "comment": "Night sky brightness is off. In Photons/(sr s m^2 m), last 'm' is for the wavelength wavelength[m] flux[1/(s m^2 sr m)",
    },
    "photo_electric_converter": {
        "quantum_efficiency_vs_wavelength": [[240e-9, 1.0], [701e-9, 1.0]],
        "dark_rate": 1e-3,
        "probability_for_second_puls": 0.0,
        "comment": "perfect detection",
    },
    "photon_stream": {
        "time_slice_duration": 0.5e-9,
        "single_photon_arrival_time_resolution": 0.416e-9,
    },
}