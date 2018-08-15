import numpy as np
import os
from os.path import join
import subprocess as sp
import shutil
import tempfile


def point_source_in_plenoscope(
    cx,
    cy,
    object_distance,
    illumination_radius_on_ground,
    number_of_photons,
    light_field_geometry_path,
    output_path,
    mct_propagate_raw_photons_path,
    config_path,
    random_seed=0,
):
    theta_open = np.arctan(illumination_radius_on_ground/object_distance)

    pos_x = object_distance * np.tan(cx)
    pos_y = object_distance * np.tan(cy)
    pos_z = object_distance

    rot_y = np.pi - np.hypot(cx, cy)
    rot_z = - np.arctan2(cy, cx)

    with tempfile.TemporaryDirectory() as tmp:

        input_path = join(tmp, 'light.xml')
        ls =  '<lightsource>\n'
        ls += '  <point_source\n'
        ls += '    opening_angle_in_deg="{:0.6f}"\n'.format(
            np.rad2deg(theta_open))
        ls += '    number_of_photons="{:0.6f}"\n'.format(number_of_photons)
        ls += '    rot_in_deg="[0.0, {ry:f}, {rz:f}]"\n'.format(
            ry=np.rad2deg(rot_y), rz=np.rad2deg(rot_z))
        ls += '    pos="[{x:f}, {y:f}, {z:f}]"\n'.format(
            x=pos_x, y=pos_y, z=pos_z)
        ls += '  />\n'
        ls += '</lightsource>\n'
        with open(input_path, 'wt') as fout:
            fout.write(ls)

        mct_propagate_call = [
            mct_propagate_raw_photons_path,
            '-l', light_field_geometry_path,
            '-c', config_path,
            '-i', input_path,
            '-o', output_path,
            '-r', str(random_seed),]

        o_path = output_path + '.stdout.txt'
        e_path = output_path + '.stderr.txt'
        with open(o_path, 'wt') as fo, open(e_path, 'wt') as fe:
            rc = sp.call(mct_propagate_call, stdout=fo, stderr=fe)
        return rc