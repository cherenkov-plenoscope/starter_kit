import numpy as np
from . import mesh


def sample_2D_points_within_radius(prng, radius, size):
    rho = np.sqrt(prng.uniform(0, 1, size)) * radius
    phi = prng.uniform(0, 2 * np.pi, size)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def make_light_field_from_line(
    prng,
    number_photons,
    line_start_x,
    line_start_y,
    line_start_z,
    line_stop_x,
    line_stop_y,
    line_stop_z,
    aperture_radius,
):
    supports = np.ones(shape=(number_photons, 3))

    line_start = np.array([line_start_x, line_start_y, line_start_z])
    line_end = np.array([line_stop_x, line_stop_y, line_stop_z])
    line_direction = line_end - line_start
    line_length = np.linalg.norm(line_direction)
    line_direction = line_direction / line_length
    alphas = prng.uniform(low=0, high=line_length, size=number_photons)
    supports = np.zeros(shape=(number_photons, 3))
    for i in range(number_photons):
        supports[i, :] = line_start + alphas[i] * line_direction

    intersections_on_disc = np.zeros(shape=(number_photons, 3))
    ix, iy = sample_2D_points_within_radius(
        prng=prng, radius=aperture_radius, size=number_photons
    )
    intersections_on_disc[:, 0] = ix
    intersections_on_disc[:, 1] = iy
    intersections_on_disc[:, 2] = 0.0
    directions = intersections_on_disc - supports
    no = np.linalg.norm(directions, axis=1)
    directions[:, 0] /= no
    directions[:, 1] /= no
    directions[:, 2] /= no
    return supports, directions


def make_light_field_from_mesh(
    prng, mesh, aperture_radius,
):
    sups = []
    dirs = []
    aperture_pos = np.array([0, 0, 0])
    for edge in mesh["edges"]:

        start_pos = np.array(mesh["vertices"][edge[0]])
        stop_pos = np.array(mesh["vertices"][edge[1]])
        edge_length = np.linalg.norm(start_pos - stop_pos)

        center_pos = (start_pos + stop_pos) * 0.5

        distance_to_aperture = np.linalg.norm(aperture_pos - center_pos)
        assert (
            2 * aperture_radius
        ) < distance_to_aperture, "Diameter of aperture: {:f}, distance to object {:f}".format(
            2 * aperture_radius, distance_to_aperture
        )

        area_of_sphere_in_distance_of_aperture = (
            4.0 * np.pi * distance_to_aperture ** 2
        )
        area_of_aperture = np.pi * aperture_radius ** 2
        fraction_of_solid_angle_covered_by_aperture = (
            area_of_aperture / area_of_sphere_in_distance_of_aperture
        )

        photon_density = edge[2] * fraction_of_solid_angle_covered_by_aperture
        number_photons = int(np.ceil(photon_density * edge_length))

        supp_ed, dirs_ed = make_light_field_from_line(
            prng=prng,
            number_photons=number_photons,
            line_start_x=start_pos[0],
            line_start_y=start_pos[1],
            line_start_z=start_pos[2],
            line_stop_x=stop_pos[0],
            line_stop_y=stop_pos[1],
            line_stop_z=stop_pos[2],
            aperture_radius=aperture_radius,
        )
        sups.append(supp_ed)
        dirs.append(dirs_ed)

    return np.vstack(sups), np.vstack(dirs)


def make_supports_with_equal_distance_to_aperture(
    supports, directions, distance
):
    alpha = -supports[:, 2] / directions[:, 2]
    down_dirs = np.zeros(directions.shape)
    for i in range(down_dirs.shape[0]):
        down_dirs[i, :] = alpha[i] * directions[i, :]
    supports_xy = supports + down_dirs

    up_dirs = np.zeros(directions.shape)
    for i in range(up_dirs.shape[0]):
        up_dirs[i, :] = distance * directions[i, :]
    supports_up = supports_xy - up_dirs

    return supports_up


def make_light_fields_from_meshes(
    meshes,
    aperture_radius,
    emission_distance_to_aperture,
    prng,
):
    """
    meshes : list
            List of meshes.
    aperture_radius : float
            Radius of aperture.
    emission_distance_to_aperture : float
            Distance a photon has to travel to reach the aperture.
    prng: numpy, prng
    """
    light_fields = []
    for m in meshes:

        mm = mesh.split_long_edges_into_shorter_ones(
            mesh=m,
            max_length_of_edge=aperture_radius
        )

        lf = make_light_field_from_mesh(
            prng=prng, mesh=m, aperture_radius=aperture_radius,
        )
        photon_supports_on_aperture = lf[0]
        photon_directions = lf[1]

        photon_supports_at_emission = make_supports_with_equal_distance_to_aperture(
            supports=photon_supports_on_aperture,
            directions=photon_directions,
            distance=emission_distance_to_aperture,
        )

        light_fields.append((photon_supports_at_emission, photon_directions))
    return light_fields
