#! /usr/bin/env python
import os
from os.path import join
import subprocess as sp
import shutil
import hashlib
import plenopy as pl
# import corsika_wrapper as cw
import numpy as np
import matplotlib.pyplot as plt
import xml
import json
import glob


def triple2mct(x, y, z):
    return '[{x:f}, {y:f}, {z:f}]'.format(x=x, y=y, z=z)


def write_misaligned_plenoscope_scenery(
    x_in_units_of_f,
    y_in_units_of_f,
    z_in_units_of_f,
    rot_x,
    rot_y,
    rot_z,
    path,
    template_scenery_path=join(
        'resources', 'acp', '71m', 'scenery', 'scenery.xml')
):
    tree = xml.etree.ElementTree.parse(template_scenery_path)
    scenery = tree.getroot()
    frame = scenery.find('frame')
    light_field_sensor = frame.find('light_field_sensor')
    expected_imaging_system_focal_length = float(light_field_sensor.find(
        'set_light_field_sensor').attrib[
            'expected_imaging_system_focal_length'])
    light_field_sensor.find('set_frame').attrib['pos'] = triple2mct(
        expected_imaging_system_focal_length * x_in_units_of_f,
        expected_imaging_system_focal_length * y_in_units_of_f,
        expected_imaging_system_focal_length * z_in_units_of_f)
    light_field_sensor.find('set_frame').attrib['rot'] = triple2mct(
        rot_x,
        rot_y,
        rot_z)
    tree.write(path)

    m = hashlib.sha256()
    m.update('{:.6f}'.format(x_in_units_of_f).encode())
    m.update('{:.6f}'.format(y_in_units_of_f).encode())
    m.update('{:.6f}'.format(z_in_units_of_f).encode())
    m.update('{:.6f}'.format(rot_x).encode())
    m.update('{:.6f}'.format(rot_y).encode())
    m.update('{:.6f}'.format(rot_z).encode())
    return m.hexdigest()


def read_misalignment_from_scenery(path):
    tree = xml.etree.ElementTree.parse(path)
    scenery = tree.getroot()
    frame = scenery.find('frame')
    light_field_sensor = frame.find('light_field_sensor')
    pos_string = light_field_sensor.find('set_frame').attrib['pos']
    rot_string = light_field_sensor.find('set_frame').attrib['rot']
    pos = json.loads(pos_string)
    rot = json.loads(rot_string)
    return pos, rot


d2r = np.deg2rad

out_dir = join('examples', 'compensating_misalignments')
os.makedirs(out_dir, exist_ok=True)

# Estimating light-field-geometries for various misalignments
# -----------------------------------------------------------
number_mega_photons = 100

light_field_geometries_dir = join(out_dir, 'light_field_geometries')

y_rotations = d2r(np.linspace(0.0, 5.0, 7))
z_translations = np.linspace(0.97, 1.03, 7)

i = 0
for y_rot in y_rotations:
    for z_trans in z_translations:
        scenery_dir = join(
            light_field_geometries_dir,
            '{:03d}_portal_plenoscope'.format(i))
        os.makedirs(scenery_dir, exist_ok=True)

        misalignment_hash = write_misaligned_plenoscope_scenery(
            x_in_units_of_f=0,
            y_in_units_of_f=0,
            z_in_units_of_f=z_trans,
            rot_x=0,
            rot_y=y_rot,
            rot_z=0,
            path=join(scenery_dir, 'scenery.xml'))

        light_field_geometry_path = join(
            light_field_geometries_dir,
            '{i:03d}_light_field_geometry_{ha:s}'.format(
                i=i,
                ha=misalignment_hash))

        if not os.path.exists(light_field_geometry_path):
            sp.call([
                join('.', 'build', 'mctracer', 'mctPlenoscopeCalibration'),
                '-s', scenery_dir,
                '-n', '{:d}'.format(number_mega_photons),
                '-o', light_field_geometry_path ])

        shutil.rmtree(scenery_dir)
        i += 1


# Analyse and visualize different misalignments
# ---------------------------------------------
lfgs = []
poss = []
rots = []
for lfg_path in glob.glob(join(light_field_geometries_dir, '*')):
    lfg = pl.LightFieldGeometry(lfg_path)
    lfgs.append(lfg)
    pos, rot = read_misalignment_from_scenery(
        join(lfg_path, 'input', 'scenery', 'scenery.xml'))
    poss.append(pos)
    rots.append(rot)


"""
out_dir = join('examples', 'sensor_misalignment_demo')
os.makedirs(out_dir, exist_ok=True)

steering_card = cw.read_steering_card(
    join('resources', 'acp', '71m',
        'high_energy_example_helium_corsika_steering_card.txt'))

if not os.path.exists(join(out_dir, 'He.evtio')):
    cw.corsika(
        steering_card=steering_card,
        output_path=join(out_dir, 'He.evtio'),
        save_stdout=True)

if not os.path.exists(join(out_dir, 'target_He.acp')):
    call([
        join('build', 'mctracer', 'mctPlenoscopePropagation'),
        '--lixel', join('resources', 'acp', '71m', 'light_field_calibration'),
        '--input', join(out_dir, 'He.evtio'),
        '--config', join('resources', 'acp', 'mct_propagation_config.xml'),
        '--output', join(out_dir, 'target_He.acp'),
        '--random_seed', '0',
        '--all_truth'])

if not os.path.exists(join(out_dir, 'misaligned_He.acp')):
    call([
        join('build', 'mctracer', 'mctPlenoscopePropagation'),
        '--lixel', join('resources', 'acp', '71m',
                        'light_field_calibration_off_target_example'),
        '--input', join(out_dir, 'He.evtio'),
        '--config', join('resources', 'acp', 'mct_propagation_config.xml'),
        '--output', join(out_dir, 'misaligned_He.acp'),
        '--random_seed', '0',
        '--all_truth'])

# 1,4,6,18,21,22,23*,25
runpaths = [
    join(out_dir, 'target_He.acp'),
    join(out_dir, 'misaligned_He.acp')]

lixel_efficiencies = [
    pl.Run(run_path).light_field_geometry.efficiency for run_path in runpaths]
min_efficiency = np.minimum(lixel_efficiencies[0], lixel_efficiencies[1])
rel_min_efficiency = min_efficiency/min_efficiency.max()
valid = rel_min_efficiency > 0.5

target_light_field_geometry = pl.Run(runpaths[0]).light_field_geometry
target_cx = target_light_field_geometry.pixel_pos_cx
target_cy = target_light_field_geometry.pixel_pos_cy

refocussed_images = []

for run_path in runpaths:
    run = pl.Run(run_path)
    event = run[22]

    # event.show()

    image_rays = pl.image.ImageRays(run.light_field_geometry)
    image_rays.pixel_pos_tree = run.light_field_geometry.pixel_pos_tree
    object_distance = 2.5e3
    lixels2pixel = image_rays.pixel_ids_of_lixels_in_object_distance(
        object_distance)

    image_sequence = np.zeros(
        shape=(
            event.light_field.number_time_slices,
            event.light_field.number_pixel),
        dtype=np.uint16)

    for lix, lixel2pixel in enumerate(lixels2pixel):
        if valid[lix]:
            image_sequence[:, lixel2pixel] += (
                event.light_field.sequence[:, lix])

    t_m = pl.light_field.sequence.time_slice_with_max_intensity(image_sequence)
    ts = np.max([t_m-1, 0])
    te = np.min([t_m+1, image_sequence.shape[0]-1])

    intense = np.sum(image_sequence[ts:te, :], axis=0)

    refocussed_image = pl.Image(
        intense,
        run.light_field_geometry.pixel_pos_cx,
        run.light_field_geometry.pixel_pos_cy,)
    refocussed_images.append(refocussed_image)

    raw_image = pl.Image(
        (event.light_field.pixel_sequence())[ts:te].sum(axis=0),
        event.light_field.pixel_pos_cx,
        event.light_field.pixel_pos_cy)

    fig, ax = plt.subplots()
    plt.title('raw '+run_path)
    pl.image.plot.add_pixel_image_to_ax(raw_image, ax)

    fig, ax = plt.subplots()
    plt.title('post '+run_path)
    pl.image.plot.add_pixel_image_to_ax(refocussed_image, ax)
    plt.savefig(run_path+'.jpg')

diff = pl.Image(
    refocussed_images[1].intensity.astype('float') -
    refocussed_images[0].intensity.astype('float'),
    target_cx,
    target_cy)

fig, ax = plt.subplots()
plt.title('Difference')
pl.image.plot.add_pixel_image_to_ax(diff, ax)

plt.show()
"""