#! /usr/bin/env python
import os
from os.path import join
import subprocess as sp
import shutil
import plenopy as pl
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as xmlElementTree
import json
import glob
import wrapp_mct_photon_propagation as mctw
import tempfile
import functools


def triple2mct(x, y, z):
    return '[{x:f}, {y:f}, {z:f}]'.format(x=x, y=y, z=z)


def read_light_field_sensor_specifications(path):
    tree = xmlElementTree.parse(path)
    scenery = tree.getroot()
    frame = scenery.find('frame')
    lfs = frame.find('light_field_sensor')
    slfs = lfs.find('set_light_field_sensor')
    sf =  lfs.find('set_frame')
    return {
        'expected_imaging_system_focal_length': float(
            slfs.attrib['expected_imaging_system_focal_length']),
        'expected_imaging_system_aperture_radius': float(
            slfs.attrib['expected_imaging_system_aperture_radius']),
        'max_FoV_diameter_deg': float(
            slfs.attrib['max_FoV_diameter_deg']),
        'hex_pixel_FoV_flat2flat_deg': float(
            slfs.attrib['hex_pixel_FoV_flat2flat_deg']),
        'housing_overhead': float(
            slfs.attrib['housing_overhead']),
        'number_of_paxel_on_pixel_diagonal': float(
            slfs.attrib['number_of_paxel_on_pixel_diagonal']),
        'pos': json.loads(sf.attrib['pos']),
        'rot': json.loads(sf.attrib['rot']),
    }


def write_misaligned_plenoscope_scenery(
    pos_x,
    pos_y,
    pos_z,
    rot_x,
    rot_y,
    rot_z,
    path,
    template_scenery_path
):
    tree = xmlElementTree.parse(template_scenery_path)
    scenery = tree.getroot()
    frame = scenery.find('frame')
    light_field_sensor = frame.find('light_field_sensor')
    light_field_sensor.find('set_frame').attrib['pos'] = triple2mct(
        pos_x,
        pos_y,
        pos_z)
    light_field_sensor.find('set_frame').attrib['rot'] = triple2mct(
        rot_x,
        rot_y,
        rot_z)
    tree.write(path)


def esitmate_light_field_geometry(
    pos_x,
    pos_y,
    pos_z,
    rot_x,
    rot_y,
    rot_z,
    path,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        write_misaligned_plenoscope_scenery(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            path=join(tmp_dir, 'scenery.xml'),
            template_scenery_path=template_scenery_path)

        sp.call([
            join('.', 'build', 'mctracer', 'mctPlenoscopeCalibration'),
            '-s', tmp_dir,
            '-n', '{:d}'.format(number_mega_photons),
            '-o', path])

out_dir = join('examples', 'compensating_misalignments')
os.makedirs(out_dir, exist_ok=True)

template_scenery_path = join(
    'resources',
    'acp',
    '71m',
    'scenery',
    'scenery.xml')

# Estimating light-field-geometries for various misalignments
# -----------------------------------------------------------
number_mega_photons = 1

# translation parallel
light_field_sensor_specs = read_light_field_sensor_specifications(
    template_scenery_path)
focal_length = light_field_sensor_specs['expected_imaging_system_focal_length']
pixel_fov = np.deg2rad(light_field_sensor_specs['hex_pixel_FoV_flat2flat_deg'])
diameter = 2*light_field_sensor_specs[
    'expected_imaging_system_aperture_radius']

target_object_distance = 10e3
target_sensor_plane_distance = 1/(1/focal_length - 1/target_object_distance)

projected_pixel = focal_length*np.tan(pixel_fov)

g_minus = target_object_distance*(
    1 - projected_pixel * target_object_distance/(2*focal_length*diameter))
g_plus = target_object_distance*(
    1 + projected_pixel * target_object_distance/(2*focal_length*diameter))

d_minus = 1/(1/focal_length - 1/g_minus)
d_plus = 1/(1/focal_length - 1/g_plus)

delta_d_minus = d_minus - target_sensor_plane_distance
delta_d_plus = d_plus - target_sensor_plane_distance

para_tra_dir = join(out_dir, 'translation_parallel')
os.makedirs(para_tra_dir, exist_ok=True)
z_positions = np.linspace(
    target_sensor_plane_distance + 10 * delta_d_plus,
    target_sensor_plane_distance + 10 * delta_d_minus,
    4)

for z_pos in z_positions:
    out_path = join(para_tra_dir, '{:.0f}mm'.format(z_pos*1e3))
    if not os.path.exists(out_path):
        esitmate_light_field_geometry(
            pos_x=0,
            pos_y=0,
            pos_z=z_pos,
            rot_x=0,
            rot_y=0,
            rot_z=0,
            path=out_path)

# rotation perpendicular
# tollerance: 0.76deg -> 10 fold -> 7.6deg
perp_rot_dir = join(out_dir, 'rotation_perpendicular')
os.makedirs(perp_rot_dir, exist_ok=True)

fov = np.deg2rad(light_field_sensor_specs['max_FoV_diameter_deg'])
sensor_plane_radius = focal_length * np.tan(fov/2)

delta_trans = np.min(np.abs([delta_d_minus, delta_d_plus]))

delta_rot_perp = delta_trans/sensor_plane_radius
y_rotations = np.linspace(
    0,
    10 * delta_rot_perp,
    4)[1:]

for y_rot in y_rotations:
    out_path = join(perp_rot_dir, '{:.0f}mdeg'.format(np.rad2deg(y_rot)*1e3))
    if not os.path.exists(out_path):
        esitmate_light_field_geometry(
            pos_x=0,
            pos_y=0,
            pos_z=target_sensor_plane_distance,
            rot_x=0,
            rot_y=y_rot,
            rot_z=0,
            path=out_path)


# All bad misalignment
composition_all_bad_path = join(out_dir, 'composition_all_bad')
if not os.path.exists(composition_all_bad_path):
    esitmate_light_field_geometry(
        pos_x=0.3,
        pos_y=0.5,
        pos_z=target_sensor_plane_distance + 1.0,
        rot_x=0,
        rot_y=np.deg2rad(5),
        rot_z=np.deg2rad(15),
        path=composition_all_bad_path)

# READ IN all light_field_geometries
# ----------------------------------

light_field_geometries = {
    'rotation_perpendicular': [],
    'translation_parallel': [],
    'composition_all_bad': pl.LightFieldGeometry(composition_all_bad_path)
}

for lfg_path in glob.glob(join(perp_rot_dir, '*')):
    light_field_geometries['rotation_perpendicular'].append(
        pl.LightFieldGeometry(lfg_path))

def rot_y_of_sensor_plane(light_field_geometry):
    lfg = light_field_geometry
    comp_x = lfg.sensor_plane2imaging_system.sensor_plane2imaging_system[
        0:3, 0][0]
    comp_z = lfg.sensor_plane2imaging_system.sensor_plane2imaging_system[
        0:3, 0][2]
    return np.arctan2(comp_z, comp_x)

def compare_rot_y(lfg1, lfg2):
    return rot_y_of_sensor_plane(lfg1) - rot_y_of_sensor_plane(lfg2)

light_field_geometries['rotation_perpendicular'].sort(
    key=functools.cmp_to_key(compare_rot_y))

for lfg_path in glob.glob(join(para_tra_dir, '*')):
    light_field_geometries['translation_parallel'].append(
        pl.LightFieldGeometry(lfg_path))

def compare_pos_z(lfg1, lfg2):
    return (
        lfg1.sensor_plane2imaging_system.sensor_plane_distance -
        lfg2.sensor_plane2imaging_system.sensor_plane_distance)

light_field_geometries['translation_parallel'].sort(
    key=functools.cmp_to_key(compare_pos_z))


# Interpret
# ---------
