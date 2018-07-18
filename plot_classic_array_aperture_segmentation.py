import matplotlib.pyplot as plt
import plenopy as pl
import numpy as np
import os
from matplotlib import patches
import scipy
from descartes import PolygonPatch
from shapely.geometry import Point

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

def make_hex_grid(spacing, radius):
    x = []
    y = []
    ux = spacing*np.array([1.0, 0.0])
    uy = spacing*np.array([0.5, np.sqrt(3)/2])
    w = int(np.ceil((2*radius)/spacing))
    for ix in np.linspace(-w, w, 2*w + 1):
        for iy in np.linspace(-w, w, 2*w + 1):
            pos = ix*ux + iy*uy
            if np.hypot(pos[0], pos[1]) <= radius:
                x.append(pos[0])
                y.append(pos[1])
    return np.c_[np.array(x), np.array(y)]


light_field_geometry = pl.LightFieldGeometry(
    os.path.join('run', 'light_field_calibration'))

out_dir = os.path.join('examples', 'classic_view')
os.makedirs(out_dir, exist_ok=True)


fig = plt.figure(figsize=(6, 6), dpi=320)
ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
R = light_field_geometry.expected_aperture_radius_of_imaging_system
w = 1.19
r = R/light_field_geometry.sensor_plane2imaging_system.number_of_paxel_on_pixel_diagonal
paxels = make_hex_grid(
    spacing=w*2*r,
    radius=w*R*0.9)
for j in range(paxels.shape[0]):
    pax = patches.RegularPolygon(
        (paxels[j, 0], paxels[j, 1]),
        numVertices=6,
        radius=w*r*(2/np.sqrt(3)),
        orientation=np.deg2rad(0),
        color='k',
        fill=False)
    ax.add_patch(pax)
outer_ring = Point(0, 0).buffer(2*R)
inner_ring = Point(0, 0).buffer(R)
_1 = PolygonPatch(outer_ring, fc='white', ec='gray', alpha=0.2, zorder=1)
_2 = PolygonPatch(inner_ring, fc='white', ec='gray', alpha=0.2, zorder=1)
annulus = outer_ring.difference(inner_ring)
annulus_patch = PolygonPatch(annulus, fc='white', ec='k', alpha=0.9, zorder=2)
ax.add_patch(annulus_patch)
ax.set_xlim(-R*1.03, R*1.03)
ax.set_ylim(-R*1.03, R*1.03)
ax.set_xlabel('$x$/m')
ax.set_ylabel('$y$/m')
plt.savefig(os.path.join(out_dir, 'aperture_segmentation_equal_spacing.png'))
plt.close('all')


fig = plt.figure(figsize=(6, 6), dpi=320)
ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
circ = plt.Circle((0, 0), radius=R, color='k', fill=False)
ax.add_patch(circ)
vor = scipy.spatial.Voronoi(
    1.05*np.c_[
        light_field_geometry.paxel_pos_x,
        light_field_geometry.paxel_pos_y],
    furthest_site=False)
for vi in range(len(vor.ridge_vertices)):
    start_i = vor.ridge_vertices[vi][0]
    end_i = vor.ridge_vertices[vi][1]
    if start_i >= 0 and end_i >= 0:
        ax.plot(
            [vor.vertices[start_i, 0], vor.vertices[end_i, 0]],
            [vor.vertices[start_i, 1], vor.vertices[end_i, 1]],
             'k')
    elif end_i >= 0:
        end_x = vor.vertices[end_i, 0]
        end_y = vor.vertices[end_i, 1]
        end_2_center = np.hypot(end_x, end_y)
        phi = np.arctan2(end_y, end_x)
        start_x = end_x/end_2_center * R
        start_y = end_y/end_2_center * R
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            'k')
ax.set_xlim(-R*1.03, R*1.03)
ax.set_ylim(-R*1.03, R*1.03)
ax.set_xlabel('$x$/m')
ax.set_ylabel('$y$/m')
plt.savefig(os.path.join(out_dir, 'aperture_segmentation_voronoi_spacing.png'))
plt.close('all')


fig = plt.figure(figsize=(6, 6), dpi=320)
ax = fig.add_axes((0.1, 0.1, 0.9, 0.9))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
circ = plt.Circle((0, 0), radius=R, color='k', fill=False)
ax.add_patch(circ)
number_macro_paxel = 7
r = R/3
pax_x = [0.0]
pax_y = [0.0]
pax_r = r * np.ones(number_macro_paxel)
for i, phi in enumerate(
    np.linspace(0, 2*np.pi, number_macro_paxel - 1, endpoint=False)):
    pax_x.append(2*r*np.cos(phi))
    pax_y.append(2*r*np.sin(phi))
for i in range(number_macro_paxel):
    circ = plt.Circle(
        (pax_x[i], pax_y[i]),
        radius=pax_r[i],
        color='k',
        fill=False)
    ax.add_patch(circ)
ax.set_xlim(-R*1.03, R*1.03)
ax.set_ylim(-R*1.03, R*1.03)
ax.set_xlabel('$x$/m')
ax.set_ylabel('$y$/m')
plt.savefig(os.path.join(out_dir, 'aperture_segmentation_seve_telescopes.png'))
plt.close('all')
