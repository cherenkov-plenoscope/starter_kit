import numpy as np
import os
import wrapp_mct_photon_propagation as mctw
import subprocess as sp
import tempfile
import json
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import plenopy as pl


out_dir = os.path.join('examples', 'small_camera_lens_psf')
os.makedirs(out_dir, exist_ok=True)

# scenery file
# ------------

outer_radius = 0.0715447
inner_radius = 0.0619595
focal_length = 0.17438
curvature_radius = 0.18919


MIRROR_WALLS = True

scenery = {}
scenery["functions"] = [
    {
        "name": "mirror_reflection",
        "argument_versus_value": [
            [200e-9, 0.95],
            [1200e-9, 0.95]]
    },
    {
        "name": "glas_refraction",
        "argument_versus_value": [
            [200e-9, 1.46832],
            [1200e-9, 1.46832]]
    },
]
scenery["colors"] = [
    {"name": "red", "rgb": [255, 0, 0]},
    {"name": "brown", "rgb": [128, 150, 0]},
    {"name": "green", "rgb": [0, 200, 0]},
    {"name": "lens_white", "rgb": [255, 255, 255]}
]
scenery["children"] = []

scenery["children"].append(
    {
        "type": "BiConvexLensHex",
        "name": "lens",
        "pos": [0, 0, focal_length],
        "rot": [0, 0, np.pi/2],
        "curvature_radius": curvature_radius,
        "outer_radius": outer_radius,
        "surface": {
            "inner_color": "lens_white",
            "outer_color": "lens_white",
            "inner_refraction": "glas_refraction",
        },
        "children": [],
    })

stop_centers = np.zeros(shape=(6, 2))
for i, phi in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
    stop_centers[i, :] = 2*inner_radius*np.array([
        np.sin(phi + np.pi/2),
        np.cos(phi + np.pi/2)])

for idx, pos in enumerate(stop_centers):
    scenery["children"].append(
        {
            "type": "HexPlane",
            "name": "stop_{:d}".format(idx),
            "pos": [pos[0], pos[1], focal_length],
            "rot": [0, 0, np.pi/2],
            "outer_radius": outer_radius,
            "surface": {
                "inner_color": "brown",
                "outer_color": "brown"},
            "children": [],
        })

if MIRROR_WALLS:
    wall_centers = np.zeros(shape=(6, 2))
    for i, phi in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
        wall_centers[i, :] = inner_radius*np.array([
            np.sin(phi + np.pi/2),
            np.cos(phi + np.pi/2)])
        scenery["children"].append(
            {
                "type": "Plane",
                "name": "wall_{:d}".format(i),
                "pos": [wall_centers[i, 0], wall_centers[i, 1], 0.025],
                "rot": [1.5707, 0, phi + np.pi/2],
                "x_width": outer_radius,
                "y_width": 0.05,
                "surface": {
                    "inner_color": "green",
                    "outer_color": "green",
                    "outer_reflection": "mirror_reflection",
                    "inner_reflection": "mirror_reflection",
                },
                "children": [],
            }
        )

scenery["children"].append(
    {
        "type": "Disc",
        "name": "sensor",
        "pos": [0, 0, 0],
        "rot": [0, 0, 0],
        "radius": outer_radius*1.5,
        "sensor_id": 0,
        "children": [],
        "surface": {
            "inner_color": "red",
            "outer_color": "red"},
    })

with open(os.path.join(out_dir, 'optical-table_for_lens.json'), 'wt') as fout:
    fout.write(json.dumps(scenery, indent=4))

sensor_responses = []
focal_ratio_imaging_reflector = 1.5
max_incident_angle = np.arctan(0.5/focal_ratio_imaging_reflector)

incident_directions = np.linspace(0, max_incident_angle, 6)
for idx, incident_direction in enumerate(incident_directions):

    # photons
    # -------
    np.random.seed(0)
    num_photons = 1000*1000

    supports = np.zeros(shape=(num_photons, 3))
    supports[:, 2] = 1.3*focal_length
    supports[:, 0] = np.random.uniform(
        low=-outer_radius,
        high=outer_radius,
        size=num_photons)
    supports[:, 1] = np.random.uniform(
        low=-outer_radius,
        high=outer_radius,
        size=num_photons)
    area_exposed = (outer_radius*2)**2
    areal_photon_density = num_photons/area_exposed

    directions = np.zeros(shape=(num_photons, 3))
    directions[:, 0] = incident_direction
    directions[:, 2] = - np.sqrt(1 - incident_direction**2)

    direction_length = np.linalg.norm(directions[:, :], axis=1)
    np.testing.assert_allclose(direction_length, 1.0, atol=1e-3)

    wavelengths = 433e-9*np.ones(num_photons)

    with tempfile.TemporaryDirectory(suffix="acp_lens_psf") as tmp_dir:
        photons_path = os.path.join(
            tmp_dir, 'photons_{idx:d}.csv'.format(idx=idx))
        photons_result_path = os.path.join(
            tmp_dir, 'photons_result_{idx:d}.csv'.format(idx=idx))
        mctw.write_ascii_table_of_photons(
            photons_path,
            supports=supports,
            directions=directions,
            wavelengths=wavelengths)
        sp.call([
            os.path.join(
                ".",
                "build",
                "merlict",
                "merlict-propagate"),
            "-s", os.path.join(
                "examples",
                "small_camera_lens_psf",
                "optical-table_for_lens.json"),
            "-i", photons_path,
            "-o", photons_result_path,
            "-c", os.path.join(
                "merlict_development_kit",
                "merlict_tests",
                "apps",
                "examples",
                "settings.json")])
        photons_result_path += "1_0"
        result = np.genfromtxt(photons_result_path)
        r = {}
        r['incident_direction'] = incident_direction
        r['areal_photon_density'] = areal_photon_density
        r['x'] = result[:, 0]
        r['y'] = result[:, 1]
        r['cx'] = result[:, 2]
        r['cy'] = result[:, 3]
        r['wavelength'] = result[:, 4]
        r['arrival_time'] = result[:, 5]
        sensor_responses.append(r)


sensor_radius = outer_radius
num_bins = 300
xy_bin_edges = np.linspace(-sensor_radius, sensor_radius, num_bins + 1)
max_intensity = 0
for sensor_response in sensor_responses:
    psf = np.histogram2d(
        x=sensor_response['x'],
        y=sensor_response['y'],
        bins=[xy_bin_edges, xy_bin_edges])[0]
    sensor_response['point_spread_function'] = psf
    sensor_response['xy_bin_edges'] = xy_bin_edges
    if np.max(psf) > max_intensity:
        max_intensity = np.max(psf)

lfg_path = os.path.join('run', 'light_field_calibration')
if os.path.exists(lfg_path):
    lfg = pl.LightFieldGeometry()
    lixel_r = np.hypot(lfg.lixel_positions_x, lfg.lixel_positions_y)
    pixel_r = (
        lfg.sensor_plane2imaging_system.expected_imaging_system_focal_length *
        np.tan(lfg.sensor_plane2imaging_system.pixel_FoV_hex_flat2flat/2))
    mask = lixel_r < pixel_r
    lixel_x = lfg.lixel_positions_x[mask]
    lixel_y = lfg.lixel_positions_y[mask]
    lixel_outer_radius = lfg.lixel_outer_radius


def add_hexagon(
    ax,
    x=0,
    y=0,
    outer_radius=1,
    theta=0,
    color='k',
    linewidth=1,
    alpha=1
):
    hexagon = np.zeros(shape=(6, 2))
    for i, phi in enumerate(np.linspace(0, 2*np.pi, 6, endpoint=False)):
        hexagon[i, 0] = x + outer_radius*np.cos(phi + theta)
        hexagon[i, 1] = y + outer_radius*np.sin(phi + theta)

    for i in range(hexagon.shape[0]):
        s = hexagon[i, :]
        if i + 1 >= hexagon.shape[0]:
            e = hexagon[0, :]
        else:
            e = hexagon[i + 1, :]
        ax.plot(
            [s[0], e[0]],
            [s[1], e[1]],
            color=color,
            linewidth=linewidth,
            alpha=alpha)


for sensor_response in sensor_responses:
    fig = plt.figure(figsize=(4, 4), dpi=250)
    ax = fig.add_axes([0, 0, 1, 1])
    [s.set_visible(False) for s in ax.spines.values()]
    [t.set_visible(False) for t in ax.get_xticklines()]
    [t.set_visible(False) for t in ax.get_yticklines()]
    im = ax.pcolor(
        1e3*sensor_response['xy_bin_edges'],
        1e3*sensor_response['xy_bin_edges'],
        sensor_response['point_spread_function'],
        cmap='binary',
        norm=colors.PowerNorm(gamma=1./3.),
        vmax=max_intensity)
    ax.grid(color='k', linestyle='-', linewidth=1, alpha=0.1)
    ax.set_aspect('equal')
    add_hexagon(
        ax=ax,
        x=0,
        y=0,
        outer_radius=1e3*outer_radius,
        color='g',
        linewidth=1.5,
        alpha=0.5)
    if os.path.exists(lfg_path):
        for j in range(lfg.number_lixel//lfg.number_pixel):
            add_hexagon(
                ax=ax,
                x=1e3*lixel_x[j],
                y=1e3*lixel_y[j],
                outer_radius=1e3*lixel_outer_radius,
                theta=np.pi/6,
                color='r',
                linewidth=1,
                alpha=0.3)
    fig.savefig(
        os.path.join(
            out_dir,
            'psf_{:d}mdeg.png'.format(
                int(1000*np.rad2deg(sensor_response['incident_direction'])))))
    plt.close('all')

fig = plt.figure(figsize=(6, .5), dpi=250)
cax = fig.add_axes((0.1, 0.5, 0.8, 0.8))
cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
fig.savefig(os.path.join(out_dir, 'colorbar.png'))
plt.close('all')
