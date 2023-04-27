import os
from . import portal

EXECUTABLES = {
    "merlict_plenoscope_propagation_path": os.path.abspath(
        os.path.join("build", "merlict", "merlict-plenoscope-propagation")
    ),
    "merlict_plenoscope_raw_photon_propagation_path": os.path.abspath(
        os.path.join(
            "build", "merlict", "merlict-plenoscope-raw-photon-propagation"
        )
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


def find_first_child_by_type(children, child_type):
    """
    Search for certain child_type in the children of a merlict scenery.
    """
    for child in children:
        if child["type"] == child_type:
            return child
        else:
            res = find_first_child_by_type(
                children=child["children"], child_type=child_type
            )
            if res:
                return res


def make_mirror_and_sensor_dimensions_from_merlict_scenery(scenery):

    _mirror_dimensions = find_first_child_by_type(
        children=scenery["children"], child_type="SegmentedReflector",
    )

    _sensor_dimensions = find_first_child_by_type(
        children=scenery["children"], child_type="LightFieldSensor",
    )

    mirror_dimensions = {}
    for key in portal.MIRROR:
        mirror_dimensions[key] = _mirror_dimensions[key]

    sensor_dimensions = {}
    for key in portal.SENSOR:
        sensor_dimensions[key] = _sensor_dimensions[key]

    return mirror_dimensions, sensor_dimensions
