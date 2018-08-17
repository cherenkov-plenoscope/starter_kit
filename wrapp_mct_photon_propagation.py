import numpy as np
import os
from os.path import join
import subprocess as sp
import shutil
import tempfile


def point_source_in_plenoscope(
    input_path,
    light_field_geometry_path,
    output_path,
    mct_propagate_raw_photons_path,
    config_path,
    random_seed=0,
):
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    mct_propagate_call = [
        mct_propagate_raw_photons_path,
        '-l', light_field_geometry_path,
        '-c', config_path,
        '-i', input_path,
        '-o', output_path,
        '--all_truth',
        '-r', str(random_seed),]

    o_path = output_path + '.stdout.txt'
    e_path = output_path + '.stderr.txt'
    with open(o_path, 'wt') as fo, open(e_path, 'wt') as fe:
        rc = sp.call(mct_propagate_call, stdout=fo, stderr=fe)
    return rc



def write_ascii_table_of_photons(path, supports, directions, wavelengths):
    number_photons = supports.shape[0]
    photons = np.zeros(shape=(number_photons, 8))
    photons[:, 0] = np.arange(number_photons)
    photons[:, 1:4] = supports
    photons[:, 4:7] = directions
    photons[:, 7] = wavelengths
    np.savetxt(path, photons, delimiter=' ', newline='\n')

"""
[0] id,
[1] [2] [3] support
[4] [5] [6] direction
[7] wavelength
"""

def sample_2D_points_within_radius(radius, size):
    rho = np.sqrt(np.random.uniform(0, 1, size)) * radius
    phi = np.random.uniform(0, 2 * np.pi, size)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def line_source_illuminating_xy_disc(
    number_photons,
    line_start_x,
    line_start_y,
    line_start_z,
    line_end_x,
    line_end_y,
    line_end_z,
    disc_x,
    disc_y,
    disc_z,
    disc_radius,
):
    supports = np.ones(shape=(number_photons, 3))

    line_start = np.array([line_start_x, line_start_y, line_start_z])
    line_end = np.array([line_end_x, line_end_y, line_end_z])
    line_direction = line_end - line_start
    line_length = np.linalg.norm(line_direction)
    line_direction = line_direction/line_length
    alphas = np.random.uniform(
        low=0,
        high=line_length,
        size=number_photons)
    supports = np.zeros(shape=(number_photons, 3))
    for i in range(number_photons):
        supports[i, :] = line_start + alphas[i]*line_direction

    intersections_on_disc = np.zeros(shape=(number_photons, 3))
    ix, iy = sample_2D_points_within_radius(
        radius=disc_radius,
        size=number_photons)
    intersections_on_disc[:, 0] = ix + disc_x
    intersections_on_disc[:, 1] = iy + disc_y
    intersections_on_disc[:, 2] = disc_z
    directions = intersections_on_disc - supports
    no = np.linalg.norm(directions, axis=1)
    directions[:, 0] /= no
    directions[:, 1] /= no
    directions[:, 2] /= no
    return supports, directions


def vertex_wire_source_illuminating_xy_disc(
    number_photons,
    vertices,
    edges,
    disc_x,
    disc_y,
    disc_z,
    disc_radius,
):
    vertices = np.array(vertices)
    edges = np.array(edges)
    edge_lengths = np.zeros(edges.shape[0])
    for e in range(edges.shape[0]):
        edge_lengths[e] = np.linalg.norm(
            vertices[edges[e, 0]] - vertices[edges[e, 1]])
    total_length = np.sum(edge_lengths)

    number_photons_on_edge = np.zeros(edges.shape[0], dtype=np.int64)
    for e in range(edges.shape[0]):
        number_photons_on_edge[e] = int(
            np.round(
                number_photons * edge_lengths[e]/total_length))

    sups = []
    dirs = []
    for e in range(edges.shape[0]):
        supp_ed, dirs_ed = line_source_illuminating_xy_disc(
            number_photons=number_photons_on_edge[e],
            line_start_x=vertices[edges[e, 0]][0],
            line_start_y=vertices[edges[e, 0]][1],
            line_start_z=vertices[edges[e, 0]][2],
            line_end_x=vertices[edges[e, 1]][0],
            line_end_y=vertices[edges[e, 1]][1],
            line_end_z=vertices[edges[e, 1]][2],
            disc_x=disc_x,
            disc_y=disc_y,
            disc_z=disc_z,
            disc_radius=disc_radius)
        sups.append(supp_ed)
        dirs.append(dirs_ed)

    return np.vstack(sups), np.vstack(dirs)
