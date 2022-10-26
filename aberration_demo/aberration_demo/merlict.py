import numpy as np
import os
from . import portal


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


def make_mirror_davies_cotton(focal_length, outer_radius):
    return {
        "type": "SegmentedReflector",
        "pos": [0, 0, 0],
        "rot": [0, 0, 1.570796],
        "focal_length": focal_length,
        "max_outer_aperture_radius": outer_radius,
        "min_inner_aperture_radius": 0.0,
        "outer_aperture_shape_hex": True,
        "DaviesCotton_over_parabolic_mixing_factor": 1.0,
        "facet_inner_hex_radius": 0.75,
        "gap_between_facets": 0.025,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


def make_mirror_parabola_segmented(focal_length, outer_radius):
    return {
        "type": "SegmentedReflector",
        "pos": [0, 0, 0],
        "rot": [0, 0, 1.570796],
        "focal_length": focal_length,
        "max_outer_aperture_radius": outer_radius,
        "min_inner_aperture_radius": 0.0,
        "outer_aperture_shape_hex": True,
        "DaviesCotton_over_parabolic_mixing_factor": 0.0,
        "facet_inner_hex_radius": 0.75,
        "gap_between_facets": 0.025,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


def make_mirror_spherical_monolith(focal_length, outer_radius):
    return {
        "type": "SphereCapWithHexagonalBound",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "curvature_radius": 2.0 * focal_length,
        "outer_radius": outer_radius,
        "name": "reflector",
        "surface": {"outer_reflection": "mirror_reflectivity_vs_wavelength"},
        "children": [],
    }


MIRRORS = {
    "sphere_monolith": make_mirror_spherical_monolith,
    "davies_cotton": make_mirror_davies_cotton,
    "parabola_segmented": make_mirror_parabola_segmented,
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


def make_plenoscope_scenery_for_merlict(
    mirror_key, num_paxel_on_diagonal, config, off_axis_angles_deg,
):
    mirror_frame = MIRRORS[mirror_key](
        focal_length=config["mirror"]["focal_length"],
        outer_radius=config["mirror"]["outer_radius"],
    )

    # ------------------------------------------
    alpha = np.deg2rad(off_axis_angles_deg)
    f = config["mirror"]["focal_length"]

    posx = f * np.sin(alpha)
    posy = 0.0
    posz = f * np.cos(alpha)

    sensor_frame = {
        "type": "LightFieldSensor",
        "name": "light_field_sensor",
        "pos": [posx, posy, posz],
        "rot": [0, -alpha, 0],
        "expected_imaging_system_focal_length": config["mirror"][
            "focal_length"
        ],
        "expected_imaging_system_aperture_radius": config["mirror"][
            "inner_radius"
        ],
        "max_FoV_diameter_deg": 2.0 * config["sensor"]["fov_radius_deg"],
        "hex_pixel_FoV_flat2flat_deg": config["sensor"][
            "hex_pixel_fov_flat2flat_deg"
        ],
        "num_paxel_on_pixel_diagonal": num_paxel_on_diagonal,
        "housing_overhead": config["sensor"]["housing_overhead"],
        "lens_refraction_vs_wavelength": "lens_refraction_vs_wavelength",
        "bin_reflection_vs_wavelength": "mirror_reflectivity_vs_wavelength",
        "children": [],
    }

    scn = {
        "functions": [
            portal.MIRROR_REFLECTIVITY,
            portal.LENS_REFRACTION,
        ],
        "colors": [
        ],
        "children": [
            {
                "type": "Frame",
                "name": "Portal",
                "pos": [0, 0, 0],
                "rot": [0, -alpha, 0],
                "children": [mirror_frame, sensor_frame],
            }
        ],
    }

    return scn
