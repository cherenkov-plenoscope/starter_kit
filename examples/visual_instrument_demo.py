#! /usr/bin/env python
import cable_robo_mount as rs
import os
import subprocess as sp
import numpy as np
from io import BytesIO
from skimage import io
import json_numpy

WORKD_DIR = os.path.join("examples", "visual")
SCENERY_PATH = os.path.join(WORKD_DIR, "scenery.json")
VISUAL_CONFIG_PATH = os.path.join(WORKD_DIR, "visual_config.json")
MERLICT_CAMERA_SERVER = os.path.join("build", "merlict", "merlict-cameraserver")
os.makedirs(WORKD_DIR, exist_ok=True)

def read_ppm_image(fstream):
    magic = fstream.readline()
    assert magic[0] == 80
    assert magic[1] == 54
    comment = fstream.readline()
    image_size = fstream.readline()
    num_columns, num_rows = image_size.decode().split()
    num_columns = int(num_columns)
    num_rows = int(num_rows)
    max_color = int(fstream.readline().decode())
    assert max_color == 255
    count = num_columns*num_rows*3
    raw = np.frombuffer(fstream.read(count), dtype=np.uint8)
    img = raw.reshape((num_rows, num_columns, 3))
    return img


def camera_command(
    position,
    orientation,
    object_distance,
    sensor_size,
    field_of_view,
    f_stop,
    num_columns,
    num_rows,
    noise_level,
):
    """
    Returns a binary command-string to be fed into merlict's cmaera-server via
    std-in.

    Parameters
    ----------

        noise_level             From 0 (no noise) to 255 (strong noise).
                                Lower noise_levels give better looking images
                                but take longer to be computed. Strong
                                noise_levels are fast to compute.
    """
    fout = BytesIO()
    fout.write(np.uint64(645).tobytes())  # MAGIC

    fout.write(np.float64(position[0]).tobytes())  # position
    fout.write(np.float64(position[1]).tobytes())
    fout.write(np.float64(position[2]).tobytes())

    fout.write(np.float64(orientation[0]).tobytes())  # Tait–Bryan-angles
    fout.write(np.float64(orientation[1]).tobytes())
    fout.write(np.float64(orientation[2]).tobytes())

    fout.write(np.float64(object_distance).tobytes())
    fout.write(np.float64(sensor_size).tobytes())  # sensor-size
    fout.write(np.float64(field_of_view).tobytes())  # fov
    fout.write(np.float64(f_stop).tobytes())  # f-stop

    fout.write(np.uint64(num_columns).tobytes())
    fout.write(np.uint64(num_rows).tobytes())
    fout.write(np.uint64(noise_level).tobytes())

    fout.seek(0)
    return fout.read()


acp_config = {
    'pointing': {
        'azimuth': 20.0,
        'zenith_distance': 30.0,
        },
    'camera': {
        'expected_imaging_system_focal_length': 106.05,
        'expected_imaging_system_aperture_radius': 35.35,
        'max_FoV_diameter_deg': 6.5,
        'hex_pixel_FoV_flat2flat_deg': 0.083333,
        'housing_overhead': 1.1,
        'number_of_paxel_on_pixel_diagonal': 9,
        'sensor_distance_to_principal_aperture_plane': 106.05,
        'offset_position': [0, 0, 0],
        'offset_rotation_tait_bryan': [0, 0, 0],
        },
    'system': {
        'merlict': {
            'hostname': '192.168.56.101',
            'username': 'spiros',
            'key_path': 'C:\\Users\\Spiros Daglas\\Desktop\\ssh\\spiros',
            'run_path_linux': '/home/spiros/Desktop/run',
            'ray_tracer_propagation_path_linux':
                '/home/spiros/Desktop/build/mctPropagate'
            },
        'sap2000': {
            'path':
                'C:\Program Files\Computers and Structures\SAP2000 19\sap2000.exe',
            'working_directory':
                'C:\\Users\\Spiros Daglas\\Desktop\\SAP2000_working_directory\\example_1'
            }
        },
    'structure_spatial_position': {
        'translational_vector_xyz': [0.0, 0.0, 0.0],
        # not used anymore. created from the tait bryan angle Ry
        'rotational_vector_Rx_Ry_Rz': [0.0, 0.0, 0.0]
        },
    'reflector': {
        'main': {
            'max_outer_radius': 40.8187,
            'min_inner_radius': 2.5,
            'number_of_layers': 3,
            'x_over_z_ratio': 1.66,
            # for truss function always keep it between 1.36 and 2.26
            'security_distance_from_ground': 2.6
            },
        'optics': {
            'focal_length': 106.05,
            'davies_cotton_over_parabola_ratio': 0.0
            },
        'facet': {
            'gap_in_between': 0.025,
            'inner_hex_radius': 0.75,  # CTA LST facet size
            'surface_weight': 20.0,
            'actuator_weight': 0.0
            },
        'material': {
            'specific_weight': 78.5,
            'e_modul': 210e6,
            'yielding_point': 1460000.0,
            'ultimate_point': 1360000.0,
            'security_factor': 1.05
            },
        'bars': {
            'outer_diameter': 0.10,
            'thickness': 0.0025,
            'imperfection_factor': 0.49,
            'buckling_length_factor': 0.9
            }
        },
    'tension_ring': {
        'width': 1.1,
        'support_position': 10,
        'material': {
            'specific_weight': 78.5,
            'e_modul': 210e6,
            'yielding_point': 1460000.0,
            'ultimate_point': 1360000.0,
            'security_factor': 1.05
            },
        'bars': {
            'outer_diameter': 0.081,
            'thickness': 0.005,
            'imperfection_factor': 0.49,
            'buckling_length_factor': 0.9
            }
        },
    'cables': {
        'material': {
            'e_modul': 95e6,
            # according to Bridon Endurance Dyform 18 PI
            'specific_weight': 89.9,
            # according to Bridon Endurance Dyform 18 PI
            'yielding_point': 1671000.0,
            'ultimate_point': 1671000.0,
            'security_factor': 1.05
            },
        'cross_section_area': 0.000221
    },
    'load_scenario': {
        'security_factor': {
            'dead': 1.00,
            'live': 1.00,
            'wind': 1.00
            },
        'wind': {
            'direction': 0.0,
            # OK
            'speed': 55,
            # m/s.OK
            'terrain_factor': 1,
            # Terrain 1.OK
            'orography_factor': 1,
            # No increase of the wind due to mountains etc.OK
            'K1': 1,
            # Turbulence factor. No accurate information available.OK
            'CsCd': 1.2,
            # usually 1. But our structure very prone to dynamic efects,
            # so Cd very conservative 1.2.OK
            'wind_density': 1.25,
            # wind density.OK
            'cpei': 1.5
            # according to EC1-4 Z.7.3(freistehende Dächer) und Z. 7.2 Tab.7.4a
            # (big?, although a preciser definition is impossible), OK
            },
        'seismic': {
            'acceleration': 3.6
            }
        },
    'star_light_analysis': {
        'photons_per_square_meter': 1000,
        'sensor': {
            'bin_width_deg': 0.0005,
            'region_of_interest_deg': 0.5
            },
        'ground': {
            'bin_width_m': 0.1
            }
        }
    }

if not os.path.exists(SCENERY_PATH):
    geometry = rs.Geometry(acp_config)
    reflector = rs.factory.generate_reflector(geometry)
    out = rs.mctracer_bridge.merlict_json.visual_scenery(reflector)
    rs.mctracer_bridge.merlict_json.write_json(out, SCENERY_PATH)
    json_numpy.write(SCENERY_PATH, out)


merlict_visual_config = {
    "max_interaction_depth": 41,
    "preview": {
        "cols": 128,
        "rows": 72,
        "scale": 10
    },
    "snapshot": {
        "cols": 1920,
        "rows":  1080,
        "noise_level": 25,
        "focal_length_over_aperture_diameter": 0.95,
        "image_sensor_size_along_a_row": 0.07
    },
    "global_illumination": {
        "on": True,
        "incoming_direction": [-0.15, -0.2, 1.0]
    },
    "sky_dome": {
        "path": "",
        "color": [255, 255, 255]
    },
    "photon_trajectories": {
        "radius": 0.15
    }
}

json_numpy.write(VISUAL_CONFIG_PATH, merlict_visual_config)

image_general_config = {
    "sensor_size": 0.06,
    "f_stop": 0.95,
    "num_columns": 512,
    "num_rows": 288,
    "noise_level": 128
}

image_configs = {
    "top": {
        "position": [-1.256e+00, 0.000e+00, 1.200e+03],
        "orientation": np.deg2rad([0 ,-1.800e+02, -7.249e+01]),
        "object_distance": 1200,
        "field_of_view": np.deg2rad(14),
    },
    "top_total": {
        "position": [-1.256e+00, 0.000e+00, 1.200e+03],
        "orientation": np.deg2rad([0 ,-1.800e+02, 45.0]),
        "object_distance": 1200,
        "field_of_view": np.deg2rad(30),
    },
    "mirror_closeup": {
        "position": [3.609e+00,  9.980e+01,  1.825e+01],
        "orientation": np.deg2rad([0 ,-8.508e+01,  1.032e+02]),
        "object_distance": 90,
        "field_of_view": np.deg2rad(44.4),
    },
    "sensor_closeup_mirror_background": {
        "position": [5.431e+01,  3.093e+00,  1.530e+02],
        "orientation": np.deg2rad([0 ,-1.461e+02,  1.726e+02]),
        "object_distance": 150,
        "field_of_view": np.deg2rad(5.909e+01),
    },
}

call = [
    MERLICT_CAMERA_SERVER,
    '--scenery',
    SCENERY_PATH,
    '--config',
    VISUAL_CONFIG_PATH
]

merlict = sp.Popen(call, stdin=sp.PIPE, stdout=sp.PIPE)
for imgkey in image_configs:
    imgpath = os.path.join(WORKD_DIR, '{:s}.tiff'.format(imgkey))
    imgcfgpath = os.path.join(WORKD_DIR, '{:s}.json'.format(imgkey))

    if not os.path.exists(imgpath):
        full_config = dict(image_general_config)
        full_config.update(image_configs[imgkey])
        w = merlict.stdin.write(camera_command(**full_config))
        w = merlict.stdin.flush()
        img = read_ppm_image(merlict.stdout)
        io.imsave(imgpath, img)
        json_numpy.write(imgcfgpath, full_config)

merlict.kill()
