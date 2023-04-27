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
    with tempfile.TemporaryDirectory(prefix='acp_') as tmp_dir:
        write_misaligned_plenoscope_scenery(
            pos_x=pos_x,
            pos_y=pos_y,
            pos_z=pos_z,
            rot_x=rot_x,
            rot_y=rot_y,
            rot_z=rot_z,
            path=join(tmp_dir, 'scenery.json'),
            template_scenery_path=template_scenery_path)

        sp.call([
            join('.', 'build', 'merlict', 'merlict-plenoscope-calibration'),
            '-s', tmp_dir,
            '-n', '{:d}'.format(number_mega_photons),
            '-o', path])

out_dir = join('examples', 'compensating_misalignments')
os.makedirs(out_dir, exist_ok=True)

lfgs_dir = join(out_dir, 'light_field_geometries')
os.makedirs(lfgs_dir, exist_ok=True)

template_scenery_path = join(
    'resources',
    'acp',
    '71m',
    'scenery',
    'scenery.json')

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

para_tra_dir = join(lfgs_dir, 'translation_parallel')
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
perp_rot_dir = join(lfgs_dir, 'rotation_perpendicular')
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


# Export summary txt
# ------------------
su = {
    'expected_imaging_system_focal_length': focal_length,
    'expected_imaging_system_aperture_radius': diameter,
    'target_object_distance': target_object_distance,
    'target_sensor_plane_distance': target_sensor_plane_distance,
    'pixel_fov_deg': np.rad2deg(pixel_fov),
    'fov_deg': np.rad2deg(fov),
    'projected_pixel': projected_pixel,
    'g_minus': g_minus,
    'g_plus': g_plus,
    'd_minus': d_minus,
    'd_plus': d_plus,
    'delta_d_minus': delta_d_minus,
    'delta_d_plus': delta_d_plus,
    'z_positions': z_positions.tolist(),
    'z_positions_relative_to_target': (
        z_positions - target_sensor_plane_distance).tolist(),
    'object_distances': (1/(1/focal_length - (1/z_positions))).tolist(),
    'sensor_plane_radius': sensor_plane_radius,
    'delta_trans_para': delta_trans,
    'delta_trans_perp': projected_pixel/2,
    'delta_rot_para_deg': np.rad2deg((projected_pixel/2)/(sensor_plane_radius)),
    'delta_rot_perp_deg': np.rad2deg(delta_rot_perp),
    'y_rotations_deg': np.rad2deg(y_rotations).tolist(),
    'all_bad_pos_y': 0.8,
    'all_bad_pos_z': target_sensor_plane_distance + 1.2,
    'all_bad_rot_y_deg': 8,
    'all_bad_rot_z_deg': 17,
}

su['all_bad_pos_z_relative_to_target'] = (
    su['all_bad_pos_z'] - target_sensor_plane_distance)

with open(join(out_dir, 'summary.json'), 'wt') as fout:
    fout.write(json.dumps(su, indent=4))

# All bad misalignment
composition_all_bad_path = join(lfgs_dir, 'composition_all_bad')
if not os.path.exists(composition_all_bad_path):
    esitmate_light_field_geometry(
        pos_x=0,
        pos_y=su['all_bad_pos_y'],
        pos_z=su['all_bad_pos_z'],
        rot_x=0,
        rot_y=np.deg2rad(su['all_bad_rot_y_deg']),
        rot_z=np.deg2rad(su['all_bad_rot_z_deg']),
        path=composition_all_bad_path)


# The target-alignment
# -------------------------
original_target_alignment_path = join('run', 'light_field_calibration')
target_alignment_path = join(lfgs_dir, 'target_alignment')
if not os.path.exists(target_alignment_path):
    shutil.copytree(original_target_alignment_path, target_alignment_path)


# READ IN all light_field_geometries
# ----------------------------------
light_field_geometries = {
    'target_alignment':{
        'path': target_alignment_path,
        'lfg': pl.LightFieldGeometry(target_alignment_path)},
    'rotation_perpendicular': [],
    'translation_parallel': [],
    'composition_all_bad': {
        'path': composition_all_bad_path,
        'lfg': pl.LightFieldGeometry(composition_all_bad_path)}
}

for lfg_path in glob.glob(join(perp_rot_dir, '*')):
    light_field_geometries['rotation_perpendicular'].append({
        'path': lfg_path,
        'lfg': pl.LightFieldGeometry(lfg_path)})

def rot_y_of_sensor_plane(light_field_geometry):
    lfg = light_field_geometry
    comp_x = lfg.sensor_plane2imaging_system.sensor_plane2imaging_system[
        0:3, 0][0]
    comp_z = lfg.sensor_plane2imaging_system.sensor_plane2imaging_system[
        0:3, 0][2]
    return np.arctan2(comp_z, comp_x)

def compare_rot_y(lfg1, lfg2):
    return (
        rot_y_of_sensor_plane(lfg1['lfg']) -
        rot_y_of_sensor_plane(lfg2['lfg']))

light_field_geometries['rotation_perpendicular'].sort(
    key=functools.cmp_to_key(compare_rot_y))

for lfg_path in glob.glob(join(para_tra_dir, '*')):
    light_field_geometries['translation_parallel'].append({
        'path': lfg_path,
        'lfg': pl.LightFieldGeometry(lfg_path)})

def compare_pos_z(lfg1, lfg2):
    return (
        lfg1['lfg'].sensor_plane2imaging_system.sensor_plane_distance -
        lfg2['lfg'].sensor_plane2imaging_system.sensor_plane_distance)

light_field_geometries['translation_parallel'].sort(
    key=functools.cmp_to_key(compare_pos_z))


# Make responses
# --------------
phantom_path = join('examples', 'phantom', 'phantom_photons.csv')
mct_propagator_path = join(
    'build',
    'merlict',
    'merlict-plenoscope-raw-photon-propagation')
propagation_config_path = join(
    'resources',
    'acp',
    'merlict_propagation_config_no_night_sky_background.json')

response_dir = join(out_dir, 'responses')
os.makedirs(response_dir, exist_ok=True)

all_scenarios = []
for s in light_field_geometries['rotation_perpendicular']:
    all_scenarios.append(s)
for s in light_field_geometries['translation_parallel']:
    all_scenarios.append(s)
all_scenarios.append(light_field_geometries['target_alignment'])
all_scenarios.append(light_field_geometries['composition_all_bad'])

for scenario in all_scenarios:
    scenario['response_path'] = join(
        response_dir,
        os.path.relpath(scenario['path'], lfgs_dir))
    if not os.path.exists(scenario['response_path']):
        os.makedirs(os.path.dirname(scenario['response_path']), exist_ok=True)
        mctw.propagate_photons(
            input_path=phantom_path,
            output_path=scenario['response_path'],
            light_field_geometry_path=scenario['path'],
            mct_propagate_raw_photons_path=mct_propagator_path,
            config_path=propagation_config_path,
            random_seed=0,)


# Read responses and make images
# ------------------------------
for scenario in all_scenarios:
    run = pl.Run(scenario['response_path'])
    scenario['response'] = run[0]

refocus_object_distances = np.array([4.2e3, 7.1e3, 11.9e3])


def reconstruct_classic_image(event):
    light_field_intensity_sequence = pl.light_field_sequence.make_isochor_image(
        raw_sensor_response=event.raw_sensor_response,
        time_delay_image_mean=event.light_field_geometry.time_delay_image_mean,
    )
    light_field_intensity = np.sum(light_field_intensity_sequence, axis=0)
    light_field_intensity_pix_pax = np.reshape(
        light_field_intensity , (
            event.light_field_geometry.number_pixel,
            event.light_field_geometry.number_paxel))
    image_intensity = np.sum(light_field_intensity_pix_pax, axis=1)
    return pl.image.Image(
        intensity=image_intensity,
        positions_x=event.light_field_geometry.pixel_pos_cx,
        positions_y=event.light_field_geometry.pixel_pos_cy)


def write_image(img, path):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes(ax_span)
    pl.plot.image.add2ax(
        ax=ax,
        I=img.intensity,
        px=np.rad2deg(img.pixel_pos_x),
        py=np.rad2deg(img.pixel_pos_y),
        colormap=colormap,
        colorbar=False)
    ax.set_aspect('equal')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-fovr, fovr)
    ax.set_ylim(-fovr, fovr)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.savefig(path)


for scenario in all_scenarios:
    event = scenario['response']

    classic_image = reconstruct_classic_image(event)

    refocused_images = pl.plot.refocus.refocus_images(
        light_field_geometry=event.light_field_geometry,
        photon_lixel_ids=event.photon_arrival_times_and_lixel_ids()[1],
        object_distances=refocus_object_distances)

    scenario['phantom'] = {
        'classic_image': classic_image,
        'refocused_images': refocused_images,
        'refocus_object_distances': refocus_object_distances.copy()
    }

# plotting
# --------
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plot_dir = join(out_dir, 'plot')
os.makedirs(plot_dir, exist_ok=True)

fovr = 3.25
figsize = (5, 5)
dpi = 200
ax_span = (0.1, 0.1, 0.9, 0.9)
colormap = 'inferno'
xlabel = r'$c_x$/deg'
ylabel = r'$c_y$/deg'
plt.close('all')


# Target-alignment detailed
# -------------------------
plot_dir_target_alignment = join(plot_dir, 'target_alignment')
if not os.path.exists(plot_dir_target_alignment):
    os.makedirs(plot_dir_target_alignment, exist_ok=True)

    ta = light_field_geometries['target_alignment']
    event = ta['response']

    detailed_refocus_object_distances = np.logspace(
        np.log10(2e3),
        np.log10(25e3),
        5*3*2)

    detailed_refocused_images = pl.plot.refocus.refocus_images(
        light_field_geometry=event.light_field_geometry,
        photon_lixel_ids=event.photon_arrival_times_and_lixel_ids()[1],
        object_distances=detailed_refocus_object_distances)

    vmax = 0
    for img in detailed_refocused_images:
        imax = np.max(img.intensity)
        if imax > vmax:
            vmax = imax


    for i, obj in enumerate(detailed_refocus_object_distances):
        img = detailed_refocused_images[i]
        fig = plt.figure(figsize=(4,4), dpi=dpi)
        ax = fig.add_axes((0, 0, 1, 1))
        pl.plot.image.add2ax(
            ax=ax,
            I=img.intensity,
            px=np.rad2deg(img.pixel_pos_x),
            py=np.rad2deg(img.pixel_pos_y),
            colormap=colormap,
            colorbar=False,
            vmin=0,
            vmax=vmax)
        ax.set_aspect('equal')
        ax.set_xlim(-fovr, fovr)
        ax.set_ylim(-fovr, fovr)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.text(
            x=-fovr*0.99,
            y=-fovr*0.99,
            s='${:.1f}\,$km'.format(obj/1e3),
            fontdict={
                'family': 'serif',
                'color':  'black',
                'weight': 'normal',
                'size': 24,})
        fig.savefig(
            join(plot_dir_target_alignment, '{:01d}.jpg'.format(i)))
        plt.close('all')

    write_image(
        img=ta['phantom']['classic_image'],
        path=join(
            plot_dir_target_alignment,
            'classic_image.jpg'))

    for i, obj in enumerate(ta['phantom']['refocus_object_distances']):
        write_image(
            img=ta['phantom']['refocused_images'][i],
            path=join(
                plot_dir_target_alignment,
                'refocused_{obj:d}m.jpg'.format(
                    obj=int(obj))))
    plt.close('all')


# rotation perpendicular
# -------------------------
for r, rot in enumerate(light_field_geometries['rotation_perpendicular']):
    rot_angle = rot_y_of_sensor_plane(rot['lfg'])
    rot_angle_str = '{:d}mdeg'.format(int(np.rad2deg(rot_angle)*1e3))

    write_image(
        img=rot['phantom']['classic_image'],
        path=join(
            plot_dir,
            'rotation_perpendicular_{:s}_classic_image.jpg'.format(
                rot_angle_str
                )))

    for i, obj in enumerate(refocus_object_distances):
        write_image(
            img=rot['phantom']['refocused_images'][i],
            path=join(
                plot_dir,
                'rotation_perpendicular_{rot_angle_str:s}_refocused_{obj:d}m.jpg'.format(
                    rot_angle_str=rot_angle_str,
                    obj=int(obj))))
    plt.close('all')


# translation paralle
# -------------------------
for t, tra in enumerate(light_field_geometries['translation_parallel']):
    sensor_plane_z = tra['lfg'].sensor_plane2imaging_system.sensor_plane_distance
    sensor_plane_z_str = '{:d}mm'.format(int(sensor_plane_z*1e3))

    write_image(
        img=tra['phantom']['classic_image'],
        path=join(
            plot_dir,
            'translation_parallel_{:s}_classic_image.jpg'.format(
                sensor_plane_z_str
                )))

    for i, obj in enumerate(refocus_object_distances):
        write_image(
            img=tra['phantom']['refocused_images'][i],
            path=join(
                plot_dir,
                'translation_parallel_{sensor_plane_z_str:s}_refocused_{obj:d}m.jpg'.format(
                    sensor_plane_z_str=sensor_plane_z_str,
                    obj=int(obj))))
    plt.close('all')

# composition
# -----------
plot_dir_compo_alignment = join(plot_dir, 'composition_all_bad')

if not os.path.exists(plot_dir_compo_alignment):
    os.makedirs(plot_dir_compo_alignment, exist_ok=True)
    co = light_field_geometries['composition_all_bad']

    write_image(
        img=co['phantom']['classic_image'],
        path=join(
            plot_dir_compo_alignment,
            'classic_image.jpg'))

    for i, obj in enumerate(co['phantom']['refocus_object_distances']):
        write_image(
            img=co['phantom']['refocused_images'][i],
            path=join(
                plot_dir_compo_alignment,
                'refocused_{obj:d}m.jpg'.format(
                    obj=int(obj))))
    plt.close('all')
