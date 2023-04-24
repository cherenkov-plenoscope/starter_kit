import numpy as np
from . import transform


def init():
    return {"vertices": {}, "edges": []}


def transform_image_to_scneney(mesh):
    out = init()
    for vname in mesh["vertices"]:
        v = mesh["vertices"][vname]
        out["vertices"][vname] = transform.img2scn_3d(imgpos=v)
    for e in mesh["edges"]:
        out["edges"].append(e)

    return out


def scale(mesh, factor):
    out = init()
    for vname in mesh["vertices"]:
        v = mesh["vertices"][vname]
        out["vertices"][vname] = factor * np.array(mesh["vertices"][vname])
    for e in mesh["edges"]:
        out["edges"].append(e)

    return out


def len(vertices, edges):
    length = 0
    for edge in edges:
        length += np.linalg.norm(vertices[edge[0]] - vertices[edge[1]])
    return length


def split_long_edges_into_shorter_ones(mesh, max_length_of_edge):
    out = init()

    for edge in mesh["edges"]:
        vkey_start = edge[0]
        vkey_stop = edge[1]

        pos_start = np.array(mesh["vertices"][vkey_start])
        pos_stop = np.array(mesh["vertices"][vkey_stop])
        length = np.linalg.norm(pos_stop - pos_start)

        if length < max_length_of_edge:
            out["edges"].append(edge)

            if vkey_start not in out["vertices"]:
                out["vertices"][vkey_start] = mesh["vertices"][vkey_start]

            if vkey_stop not in out["vertices"]:
                out["vertices"][vkey_stop] = mesh["vertices"][vkey_stop]

        else:
            vkey_inter_template = str(vkey_start) + "_" + str(vkey_stop) + "_"
            num_inter_steps = int(np.ceil(length / max_length_of_edge))
            num_steps = 2 + num_inter_steps

            edge_support = pos_start
            edge_direction = pos_stop - pos_start
            edge_direction = edge_direction / length

            inter_vertices = {}
            for it, t in enumerate(np.linspace(0, length, num_steps)):
                vkey_inter = vkey_inter_template + "{:06d}".format(it)
                inter_pos = edge_support + t * edge_direction

                out["vertices"][vkey_inter] = inter_pos

            for it in range(num_steps - 1):
                vkey_inter_start = vkey_inter_template + "{:06d}".format(it)
                vkey_inter_stop = vkey_inter_template + "{:06d}".format(it + 1)
                inter_edge = (
                    vkey_inter_start,
                    vkey_inter_stop,
                    float(edge[2]),
                )
                out["edges"].append(inter_edge)
        return out


def triangle(pos, radius, density=1):
    x, y, z = pos
    r = radius
    vertices = {
        "0": [
            x + r * np.cos(2 * np.pi * 0.25),
            y + r * np.sin(2 * np.pi * 0.25),
            z,
        ],
        "1": [
            x + r * np.cos(2 * np.pi * 0.5833),
            y + r * np.sin(2 * np.pi * 0.5833),
            z,
        ],
        "2": [
            x + r * np.cos(2 * np.pi * 0.9166),
            y + r * np.sin(2 * np.pi * 0.9166),
            z,
        ],
    }

    edges = [("0", "1", density), ("1", "2", density), ("2", "0", density)]
    return {
        "vertices": vertices,
        "edges": edges,
    }


def spiral(pos, turns, outer_radius, density, fn=110):
    x, y, z = pos
    azimuth = np.linspace(0, turns * 2 * np.pi, fn, endpoint=False)
    radius = np.linspace(0, outer_radius, fn)

    vertices = {}
    edges = []

    def nkey(n):
        return "{:d}".format(n)

    for n in range(fn):
        radius_s = radius[n]
        azimuth_s = azimuth[n]
        vertex_s = [
            x + np.cos(azimuth_s) * radius_s,
            y + np.sin(azimuth_s) * radius_s,
            z,
        ]
        vertices[nkey(n)] = vertex_s

    for n in range(fn - 1):
        edges.append((nkey(n), nkey(n + 1), density))

    return {
        "vertices": vertices,
        "edges": edges,
    }


def sun(pos, radius, num_flares, fn=50, density=1):
    x, y, z = pos
    number_flares_sun = 11

    vertices = {}
    edges = []

    azimuths = np.linspace(0, 2 * np.pi, fn)
    for n in range(fn):
        start_vertex = [
            x + radius * np.cos(azimuths[n]),
            y + radius * np.sin(azimuths[n]),
            z,
        ]
        vertices["c{:d}".format(n)] = start_vertex

    for n in range(fn - 1):
        edges.append(("c{:d}".format(n), "c{:d}".format(n + 1), density))

    azimuths_flares = np.linspace(0, 2 * np.pi, num_flares, endpoint=False)

    for n in range(num_flares):
        az = azimuths_flares[n]
        start_vertex = [
            x + 1.1 * radius * np.cos(az),
            y + 1.1 * radius * np.sin(az),
            z,
        ]
        end_vertex = [
            x + 1.4 * radius * np.cos(az),
            y + 1.4 * radius * np.sin(az),
            z,
        ]
        vertices["fs{:d}".format(n)] = start_vertex
        vertices["fe{:d}".format(n)] = end_vertex

        edges.append(("fs{:d}".format(n), "fe{:d}".format(n), density))

    return {
        "vertices": vertices,
        "edges": edges,
    }


def cross(pos, radius, density=1):
    x, y, z = pos
    vertices = {}
    edges = []

    vertices["a0"] = [x + radius, y + radius, z]
    vertices["a1"] = [x - radius, y - radius, z]
    edges.append(("a0", "a1", density))

    vertices["b0"] = [x - radius, y + radius, z]
    vertices["b1"] = [x - radius * 0.1, y + radius * 0.1, z]
    edges.append(("b0", "b1", density))

    vertices["c0"] = [x + radius, y - radius, z]
    vertices["c1"] = [x + radius * 0.1, y - radius * 0.1, z]
    edges.append(("c0", "c1", density))

    return {
        "vertices": vertices,
        "edges": edges,
    }


def smiley(pos, radius, fn=50, density=1):
    x, y, z = pos
    vertices = {}
    edges = []

    azimuths = np.linspace(0, 2 * np.pi, fn)

    # face
    for n in range(fn):
        face_vertex = [
            x + radius * np.cos(azimuths[n]),
            y + radius * np.sin(azimuths[n]),
            z,
        ]
        vertices["face{:d}".format(n)] = face_vertex

    for n in range(fn - 1):
        edges.append(("face{:d}".format(n), "face{:d}".format(n + 1), density))

    # mouth
    for n in range(fn // 2):
        mouth_vertex = [
            x + 0.7 * radius * np.cos(np.pi + azimuths[n]),
            y + 0.7 * radius * np.sin(np.pi + azimuths[n]),
            z,
        ]
        vertices["mouth{:d}".format(n)] = mouth_vertex

    for n in range((fn // 2) - 1):
        edges.append(
            ("mouth{:d}".format(n), "mouth{:d}".format(n + 1), density)
        )

    # eyes
    vertices["eye00"] = [x + radius * 0.25, y + 0, z]
    vertices["eye01"] = [x + radius * 0.25, y + radius * 0.5, z]
    edges.append(("eye00", "eye01", density))

    vertices["eye10"] = [x - radius * 0.25, y + 0, z]
    vertices["eye11"] = [x - radius * 0.25, y + radius * 0.5, z]
    edges.append(("eye10", "eye11", density))

    return {
        "vertices": vertices,
        "edges": edges,
    }
