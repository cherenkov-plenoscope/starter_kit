import numpy as np
import wrapp_mct_photon_propagation as mctw
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

out_dir = os.path.join('examples', 'phantom')
os.makedirs(out_dir, exist_ok=True)

prng = np.random.Generator(np.random.MT19937(seed=42))

def length_of_wire(vertices, edges):
    length = 0
    for edge in edges:
        length += np.linalg.norm(
            vertices[edge[0]] - vertices[edge[1]])
    return length


# phantom source
# --------------
number_photons = int(1e6)

# disk on ground to be illuminated
disc_x = 0.0
disc_y = 0.0
disc_z = 0.0
disc_radius = 40.0

# triangle at object-distance 5km
# -------------------------------
obj_tri = 2.5e3
tri_x = np.tan(np.deg2rad(-1.))*obj_tri
tri_y = np.tan(np.deg2rad(+1.3))*obj_tri

viewing_radius = np.deg2rad(1.8)
triangle_radius = obj_tri*np.tan(viewing_radius)
tr = triangle_radius

triangle_vertices = np.array([
    [
        tri_x + tr*np.cos(2*np.pi*0.25),
        tri_y + tr*np.sin(2*np.pi*0.25),
        obj_tri],
    [
        tri_x + tr*np.cos(2*np.pi*0.5833),
        tri_y + tr*np.sin(2*np.pi*0.5833),
        obj_tri],
    [
        tri_x + tr*np.cos(2*np.pi*0.9166),
        tri_y + tr*np.sin(2*np.pi*0.9166),
        obj_tri]])

triangle_edges = np.array([
    [0, 1],
    [1, 2],
    [2, 0]])
tri_len = length_of_wire(triangle_vertices, triangle_edges)

# spiral at object-distance 10km
# ------------------------------
obj_spi = 4.2e3
spi_x = np.tan(np.deg2rad(-1))*obj_spi
spi_y = np.tan(np.deg2rad(-1.3))*obj_spi

number_points_on_spiral = 110
spiral_azimuth = np.linspace(
    0,
    4*np.pi,
    number_points_on_spiral,
    endpoint=False)
spiral_viewing_radius = np.linspace(
    0,
    np.deg2rad(1.7),
    number_points_on_spiral)

spiral_vertices = []
spiral_edges = []

for n in range(number_points_on_spiral - 1):
    radius_s = obj_spi*np.tan(spiral_viewing_radius[n])
    azimuth_s = spiral_azimuth[n]
    start_vertex = [
        spi_x + np.cos(azimuth_s)*radius_s,
        spi_y + np.sin(azimuth_s)*radius_s,
        obj_spi]
    radius_e = obj_spi*np.tan(spiral_viewing_radius[n + 1])
    azimuth_e = spiral_azimuth[n + 1]
    end_vertex = [
        spi_x + np.cos(azimuth_e)*radius_e,
        spi_y + np.sin(azimuth_e)*radius_e,
        obj_spi]

    spiral_vertices.append(start_vertex)
    spiral_vertices.append(end_vertex)
    spiral_edges.append([
        len(spiral_vertices) - 2,
        len(spiral_vertices) - 1])

spiral_vertices = np.array(spiral_vertices)
spiral_edges = np.array(spiral_edges)
spi_len = length_of_wire(spiral_vertices, spiral_edges)

# sunny circle at object-distance 15km
# ------------------------------------
obj_sun = 7.1e3
sun_x = np.tan(np.deg2rad(1.7))*obj_sun
sun_y = np.tan(np.deg2rad(0))*obj_sun
sun_radius = np.tan(np.deg2rad(1))*obj_sun

number_edges_sun = 50
number_flares_sun = 11

sun_vertices = []
sun_edges = []

azimuths = np.linspace(
    0,
    2*np.pi,
    number_edges_sun)
for n in range(number_edges_sun - 1):
    start_vertex = [
        sun_x + sun_radius*np.cos(azimuths[n]),
        sun_y + sun_radius*np.sin(azimuths[n]),
        obj_sun]
    end_vertex = [
        sun_x + sun_radius*np.cos(azimuths[n + 1]),
        sun_y + sun_radius*np.sin(azimuths[n + 1]),
        obj_sun]
    sun_vertices.append(start_vertex)
    sun_vertices.append(end_vertex)
    sun_edges.append([
        len(sun_vertices) - 2,
        len(sun_vertices) - 1])

azimuths_flares = np.linspace(
    0,
    2*np.pi,
    number_flares_sun,
    endpoint=False)

for n in range(number_flares_sun):
    azimuth = azimuths_flares[n]
    start_vertex = [
        sun_x + 1.1*sun_radius*np.cos(azimuth),
        sun_y + 1.1*sun_radius*np.sin(azimuth),
        obj_sun]
    end_vertex = [
        sun_x + 1.4*sun_radius*np.cos(azimuth),
        sun_y + 1.4*sun_radius*np.sin(azimuth),
        obj_sun]
    sun_vertices.append(start_vertex)
    sun_vertices.append(end_vertex)
    sun_edges.append([
        len(sun_vertices) - 2,
        len(sun_vertices) - 1])

sun_vertices = np.array(sun_vertices)
sun_edges = np.array(sun_edges)
sun_len = length_of_wire(sun_vertices, sun_edges)

# Smiley
# -----
obj_smi = 11.9e3
smiley_x = np.tan(np.deg2rad(-1.))*obj_smi
smiley_y = np.tan(np.deg2rad(+1.3))*obj_smi
smiley_radius = np.tan(np.deg2rad(.6))*obj_smi

number_edges_smiley = 50

smiley_vertices = []
smiley_edges = []

azimuths = np.linspace(
    0,
    2*np.pi,
    number_edges_smiley)
# face
for n in range(number_edges_smiley - 1):
    start_vertex = [
        smiley_x + smiley_radius*np.cos(azimuths[n]),
        smiley_y + smiley_radius*np.sin(azimuths[n]),
        obj_smi]
    end_vertex = [
        smiley_x + smiley_radius*np.cos(azimuths[n + 1]),
        smiley_y + smiley_radius*np.sin(azimuths[n + 1]),
        obj_smi]
    smiley_vertices.append(start_vertex)
    smiley_vertices.append(end_vertex)
    smiley_edges.append([
        len(smiley_vertices) - 2,
        len(smiley_vertices) - 1])
# mouth
for n in range((number_edges_smiley - 1)//2):
    start_vertex = [
        smiley_x + .7*smiley_radius*np.cos(np.pi+azimuths[n]),
        smiley_y + .7*smiley_radius*np.sin(np.pi+azimuths[n]),
        obj_smi]
    end_vertex = [
        smiley_x + .7*smiley_radius*np.cos(np.pi+azimuths[n + 1]),
        smiley_y + .7*smiley_radius*np.sin(np.pi+azimuths[n + 1]),
        obj_smi]
    smiley_vertices.append(start_vertex)
    smiley_vertices.append(end_vertex)
    smiley_edges.append([
        len(smiley_vertices) - 2,
        len(smiley_vertices) - 1])

# eyes
smiley_vertices.append([
    smiley_x + smiley_radius*0.25,
    smiley_y + 0,
    obj_smi])
smiley_vertices.append([
    smiley_x + smiley_radius*0.25,
    smiley_y + smiley_radius*0.5,
    obj_smi])
smiley_edges.append([
        len(smiley_vertices) - 2,
        len(smiley_vertices) - 1])

smiley_vertices.append([
    smiley_x - smiley_radius*0.25,
    smiley_y + 0,
    obj_smi])
smiley_vertices.append([
    smiley_x - smiley_radius*0.25,
    smiley_y + smiley_radius*0.5,
    obj_smi])
smiley_edges.append([
        len(smiley_vertices) - 2,
        len(smiley_vertices) - 1])

smiley_vertices = np.array(smiley_vertices)
smiley_edges = np.array(smiley_edges)
smi_len = length_of_wire(smiley_vertices, smiley_edges)

# cross
# -----
obj_cro = 20e3
cross_x = np.tan(np.deg2rad(+1.))*obj_cro
cross_y = np.tan(np.deg2rad(-1.3))*obj_cro
cross_radius = np.tan(np.deg2rad(.35))*obj_cro

cross_vertices = []
cross_edges = []

cross_vertices.append([
    cross_x + cross_radius,
    cross_y + cross_radius,
    obj_cro])
cross_vertices.append([
    cross_x - cross_radius,
    cross_y - cross_radius,
    obj_cro])
cross_edges.append([
        len(cross_vertices) - 2,
        len(cross_vertices) - 1])

cross_vertices.append([
    cross_x - cross_radius,
    cross_y + cross_radius,
    obj_cro])
cross_vertices.append([
    cross_x - cross_radius*.1,
    cross_y + cross_radius*.1,
    obj_cro])
cross_edges.append([
        len(cross_vertices) - 2,
        len(cross_vertices) - 1])

cross_vertices.append([
    cross_x + cross_radius,
    cross_y - cross_radius,
    obj_cro])
cross_vertices.append([
    cross_x + cross_radius*.1,
    cross_y - cross_radius*.1,
    obj_cro])
cross_edges.append([
        len(cross_vertices) - 2,
        len(cross_vertices) - 1])


cross_vertices = np.array(cross_vertices)
cross_edges = np.array(cross_edges)
cro_len = length_of_wire(cross_vertices, cross_edges)

# Create photons
# --------------

total_length = tri_len + spi_len + sun_len + smi_len + cro_len

dist = obj_tri + obj_spi + obj_sun + obj_smi + obj_cro

number_photons_tri = int((tri_len/total_length)*number_photons)*(dist/obj_tri**1.3)
number_photons_spi = int((spi_len/total_length)*number_photons)*(dist/obj_spi**1.3)
number_photons_sun = int((sun_len/total_length)*number_photons)*(dist/obj_sun**1.3)
number_photons_smi = int((smi_len/total_length)*number_photons)*(dist/obj_smi**1.3)
number_photons_cro = int((cro_len/total_length)*number_photons)*(dist/obj_cro**1.3)

tri_sups, tri_dirs = mctw.vertex_wire_source_illuminating_xy_disc(
    number_photons=number_photons_tri,
    vertices=triangle_vertices,
    edges=triangle_edges,
    disc_x=disc_x,
    disc_y=disc_y,
    disc_z=disc_z,
    disc_radius=disc_radius,
    prng=prng,
)

spi_sups, spi_dirs = mctw.vertex_wire_source_illuminating_xy_disc(
    number_photons=number_photons_spi,
    vertices=spiral_vertices,
    edges=spiral_edges,
    disc_x=disc_x,
    disc_y=disc_y,
    disc_z=disc_z,
    disc_radius=disc_radius,
    prng=prng,
)

sun_sups, sun_dirs = mctw.vertex_wire_source_illuminating_xy_disc(
    number_photons=number_photons_sun,
    vertices=sun_vertices,
    edges=sun_edges,
    disc_x=disc_x,
    disc_y=disc_y,
    disc_z=disc_z,
    disc_radius=disc_radius,
    prng=prng,
)

smi_sups, smi_dirs = mctw.vertex_wire_source_illuminating_xy_disc(
    number_photons=number_photons_smi,
    vertices=smiley_vertices,
    edges=smiley_edges,
    disc_x=disc_x,
    disc_y=disc_y,
    disc_z=disc_z,
    disc_radius=disc_radius,
    prng=prng,
)

cro_sups, cro_dirs = mctw.vertex_wire_source_illuminating_xy_disc(
    number_photons=number_photons_cro,
    vertices=cross_vertices,
    edges=cross_edges,
    disc_x=disc_x,
    disc_y=disc_y,
    disc_z=disc_z,
    disc_radius=disc_radius,
    prng=prng,
)

sups = np.vstack([tri_sups, spi_sups, sun_sups, smi_sups, cro_sups])
dirs = np.vstack([tri_dirs, spi_dirs, sun_dirs, smi_dirs, cro_dirs])
wvls = 433e-9*np.ones(sups.shape[0])

ref_sups = mctw.supports_equal_dist_to_xy_plane(sups, dirs, 1e3)

mctw.write_ascii_table_of_photons(
    os.path.join(out_dir, 'phantom_photons.csv'),
    supports=ref_sups,
    directions=dirs,
    wavelengths=wvls)


def add_edges_to_ax3d(
    vertices,
    edges,
    ax3d,
    color='b',
):
    for e in edges:
        ax3d.plot(
            xs=[
                vertices[e[0], 0],
                vertices[e[1], 0]
            ],
            ys=[
                vertices[e[0], 1],
                vertices[e[1], 1]
            ],
            zs=[
                vertices[e[0], 2],
                vertices[e[1], 2]
            ],
            color=color
        )


def add_edges_to_ax_xy_projection(
    vertices,
    edges,
    ax,
    color='b',
):
    for e in edges:
        ax.plot(
            [
                vertices[e[0], 0],
                vertices[e[1], 0]
            ],
            [
                vertices[e[0], 1],
                vertices[e[1], 1]
            ],
            color=color
        )


def save_view(
    path,
    figsize=(12, 16),
    dpi=200,
    elev=5,
    azim=-45,
    zlabel=True
):
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax3d = fig.add_subplot(111, projection='3d')
    xy_radius = 0.4
    max_object_distance = 25
    add_edges_to_ax3d(triangle_vertices*1e-3, triangle_edges, ax3d, 'k')
    add_edges_to_ax3d(spiral_vertices*1e-3, spiral_edges, ax3d, 'k')
    add_edges_to_ax3d(sun_vertices*1e-3, sun_edges, ax3d, 'k')
    add_edges_to_ax3d(smiley_vertices*1e-3, smiley_edges, ax3d, 'k')
    add_edges_to_ax3d(cross_vertices*1e-3, cross_edges, ax3d, 'k')
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [0, max_object_distance]:
                ax3d.plot(xs=[x*xy_radius], ys=[y*xy_radius], zs=[z])
    ax3d.set_xlabel(r'$x$/km')
    ax3d.set_ylabel(r'$y$/km')
    ax3d.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax3d.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    if zlabel:
        ax3d.set_zlabel(r'object-distance $g$/km')
    else:
        ax3d.set_zticks([])
    ax3d.view_init(elev=elev, azim=azim)
    fig.savefig(path)

save_view(
    figsize=(6, 10),
    dpi=300,
    elev=10,
    azim=-45,
    path=os.path.join(out_dir, "phantom.png"))
save_view(
    figsize=(6, 10),
    dpi=300,
    elev=10,
    azim=-55,
    path=os.path.join(out_dir, "phantom1.png"))
save_view(
    figsize=(6, 10),
    dpi=300,
    elev=10,
    azim=-65,
    path=os.path.join(out_dir, "phantom2.png"))
save_view(
    figsize=(6, 10),
    dpi=300,
    elev=10,
    azim=-75,
    path=os.path.join(out_dir, "phantom3.png"))


def save_projection(
    vertices,
    edges,
    path
):
    fig = plt.figure(figsize=(2, 1.75), dpi=400)
    ax = fig.add_axes((0.3, 0.3, 0.7, 0.7))
    add_edges_to_ax_xy_projection(vertices, edges, ax, 'k')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x$/m')
    ax.set_ylabel(r'$y$/m')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.grid(color='k', linestyle='-', linewidth=0.66, alpha=0.1)
    fig.savefig(path)

save_projection(
    triangle_vertices,
    triangle_edges,
    os.path.join(out_dir, 'phantom_triangle.png'))
save_projection(
    sun_vertices,
    sun_edges,
    os.path.join(out_dir, 'phantom_sun.png'))
save_projection(
    spiral_vertices,
    spiral_edges,
    os.path.join(out_dir, 'phantom_spiral.png'))
save_projection(
    smiley_vertices,
    smiley_edges,
    os.path.join(out_dir, 'phantom_smiley.png'))
save_projection(
    cross_vertices,
    cross_edges,
    os.path.join(out_dir, 'phantom_cross.png'))

plt.close('all')
