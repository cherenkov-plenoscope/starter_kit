import os
import plenopy


def guess_scaling_of_num_photons_used_to_estimate_light_field_geometry(
    num_paxel_on_pixel_diagonal,
):
    return num_paxel_on_pixel_diagonal * num_paxel_on_pixel_diagonal


def get_instrument_geometry_from_light_field_geometry(
    light_field_geometry=None, light_field_geometry_path=None,
):
    if light_field_geometry_path:
        assert light_field_geometry is None
        geom_path = os.path.join(
            light_field_geometry_path, "light_field_sensor_geometry.header.bin"
        )
        geom_header = plenopy.corsika.utils.hr.read_float32_header(geom_path)
        geom = plenopy.light_field_geometry.PlenoscopeGeometry(raw=geom_header)
    else:
        geom = light_field_geometry.sensor_plane2imaging_system
    return class_members_to_dict(c=geom)


def class_members_to_dict(c):
    member_keys = []
    for key in dir(c):
        if not callable(getattr(c, key)):
            if not str.startswith(key, "__"):
                member_keys.append(key)
    out = {}
    for key in member_keys:
        out[key] = getattr(c, key)
    return out
