import os
import numpy as np
import collections
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", family="serif")


def sphere_intersection(xs, ys, xd, yd, r):
    div = xd ** 2 + yd ** 2
    p = (2 * xs * xd + 2 * ys * yd) / div
    q = (xs ** 2 + ys ** 2 - r ** 2) / div

    a_plus = -p / 2 + np.sqrt((p / 2) ** 2 - q)
    a_minus = -p / 2 - np.sqrt((p / 2) ** 2 - q)
    return a_plus, a_minus


def sphere_surface(x, y, r, theta_start, theta_end, number_points=1000):
    surf = np.zeros(shape=(number_points, 2))
    thetas = np.linspace(theta_start, theta_end, number_points)
    for i, theta in enumerate(thetas):
        surf[i, 0] = x + r * np.cos(theta)
        surf[i, 1] = y + r * np.sin(theta)
    return surf


def mirror_vector(dx, dy, nx, ny):
    dot = nx * dx + ny * dy
    rx = dx - 2 * dot * nx
    ry = dy - 2 * dot * ny
    return rx, ry


def vertical_line_intersection(sx, dx, lx):
    return (lx - sx) / dx


def propagate_photon(
    support_x, support_y, direction_x, direction_y, curvature_radius
):
    emission_offset = 10 * curvature_radius
    focal_length = curvature_radius / 2
    support_position = np.array([support_x, support_y])
    direction = np.array([direction_x, direction_y])
    direction = direction / np.linalg.norm(direction)

    ap, am = sphere_intersection(
        xs=support_position[0],
        ys=support_position[1],
        xd=direction[0],
        yd=direction[1],
        r=curvature_radius,
    )

    sphere_intersection_position = support_position + ap * direction
    sphere_surface_normal = sphere_intersection_position / np.linalg.norm(
        sphere_intersection_position
    )

    rx, ry = mirror_vector(
        dx=direction[0],
        dy=direction[1],
        nx=sphere_surface_normal[0],
        ny=sphere_surface_normal[1],
    )
    reflection_direction = np.array([rx, ry])

    beta = vertical_line_intersection(
        sx=sphere_intersection_position[0],
        dx=reflection_direction[0],
        lx=-curvature_radius + focal_length,
    )

    sensor_plane_intersection_position = (
        sphere_intersection_position + beta * reflection_direction
    )

    return (sphere_intersection_position, sensor_plane_intersection_position)


def _light_field_geometry_init(plenoscope_geometry):
    pg = plenoscope_geometry
    lfg = {}
    for key in ["y", "cy", "t"]:
        lfg[key] = []
        for pa in range(pg["paxel"]["num"]):
            pixs = []
            for pi in range(pg["sensor"]["smallcameras"]["num"]):
                pixs.append([])
            lfg[key].append(pixs)
    return lfg


def make_isochron_light_front_from_diffuse_source(
    incident_direction_start,
    incident_direction_stop,
    aperture_diameter,
    emission_distance,
    aperture_distance,
    num_rays,
):
    thetas = prng.uniform(
        low=incident_direction_start,
        high=incident_direction_stop,
        size=num_rays,
    )

    return _make_isochron_light_front(
        thetas=thetas,
        aperture_diameter=aperture_diameter,
        emission_distance=emission_distance,
        aperture_distance=aperture_distance,
    )


def make_isochron_light_front_from_point_source(
    true_incident_direction,
    point_source_spread,
    aperture_diameter,
    emission_distance,
    aperture_distance,
    num_rays,
):
    thetas = prng.normal(
        loc=true_incident_direction,
        scale=point_source_spread,
        size=num_rays,
    )

    return _make_isochron_light_front(
        thetas=thetas,
        aperture_diameter=aperture_diameter,
        emission_distance=emission_distance,
        aperture_distance=aperture_distance,
    )


def _make_isochron_light_front(
    thetas,
    aperture_diameter,
    emission_distance,
    aperture_distance,
):
    """
                                            /\ (0)
                    \                       |
                     \                      |
                      \                     |
    ------------------hh---ww-------------__0----------------------------> (1)
                        \  |   theta __---
                    bb   \ |ll  __---   aa
                          \|_---
                          PP

    sin(t) = geg/hyp

    cos(t) = ank/hyp

    """
    num_rays = len(thetas)

    hh = prng.uniform(
        low=-aperture_diameter / 2,
        high=aperture_diameter / 2,
        size=num_rays,
    )

    support_positions = np.zeros(shape=(num_rays, 2))
    support_positions[:, 0] = aperture_distance
    support_positions[:, 1] = hh

    directions = np.zeros(shape=(num_rays, 2))
    directions[:, 0] = np.cos(thetas)
    directions[:, 1] = np.sin(thetas)

    # hyp = hh
    # ank = aa
    aa = hh * np.cos(thetas)

    # hyp = aa
    # geg = ll
    # ank = ww
    ll = aa * np.sin(thetas)

    ww = aa * np.cos(thetas)

    emission_positions = np.zeros(shape=(num_rays, 2))
    emission_positions[:, 0] = ll
    emission_positions[:, 1] = ww
    emission_positions = support_positions + directions * emission_distance
    return emission_positions, support_positions




def _light_field_geometry_add_rays(
    light_field_geometry,
    plenoscope_geometry,
    max_off_axis_angle_deg,
    num_rays,
    max_num_rays_per_sensor,
    prng,
):
    lfg = light_field_geometry
    pg = plenoscope_geometry

    emission_positions, support_positions = make_isochron_light_front_from_diffuse_source(
        incident_direction_start=-np.deg2rad(max_off_axis_angle_deg),
        incident_direction_stop=np.deg2rad(max_off_axis_angle_deg),
        aperture_diameter=pg["mirror"]["diameter"],
        emission_distance=pg["sensor"]["distance"] * 1.25,
        aperture_distance=-pg["mirror"]["curvature_radius"],
        num_rays=num_rays,
    )

    incident_directions = emission_positions - support_positions
    incident_direction_norms = np.linalg.norm(incident_directions, axis=1)

    for ii in range(len(incident_directions)):
        incident_directions[ii] = (
            incident_directions[ii] / incident_direction_norms[ii]
        )

    sxs = support_positions[:, 0]
    sys = support_positions[:, 1]
    dxs = incident_directions[:, 0]
    dys = incident_directions[:, 1]

    """
    sxs = -pg["mirror"]["curvature_radius"] * np.ones(num_rays)

    sys = prng.uniform(
        low=-pg["mirror"]["diameter"] / 2,
        high=pg["mirror"]["diameter"] / 2,
        size=num_rays,
    )

    dxs = -np.ones(num_rays)

    dys = prng.uniform(
        low=-np.deg2rad(max_off_axis_angle_deg),
        high=np.deg2rad(max_off_axis_angle_deg),
        size=num_rays,
    )

    _norms = np.sqrt(dxs ** 2 + dys ** 2)
    dxs /= _norms
    dys /= _norms
    """

    for r in range(num_rays):

        (
            sphere_intersection_position,
            sensor_plane_intersection_position,
        ) = propagate_photon(
            support_x=sxs[r],
            support_y=sys[r],
            direction_x=dxs[r],
            direction_y=dys[r],
            curvature_radius=pg["mirror"]["curvature_radius"],
        )

        pixel_id = (
            np.digitize(
                sensor_plane_intersection_position[1],
                bins=pg["sensor"]["smallcameras"]["edges"],
            )
            - 1
        )

        paxel_id = np.digitize(sys[r], bins=pg["paxel"]["edges"]) - 1

        ddx = (sensor_plane_intersection_position[0])
        ddy = (sys[r] - sensor_plane_intersection_position[1])
        dd = np.hypot(ddx, ddy)

        if pixel_id >= 0 and pixel_id < pg["sensor"]["smallcameras"]["num"]:
            if paxel_id >= 0 and paxel_id < pg["paxel"]["num"]:
                num_rays_per_sensor = len(lfg["y"][paxel_id][pixel_id])
                if num_rays_per_sensor <= max_num_rays_per_sensor:
                    lfg["y"][paxel_id][pixel_id].append(sys[r])
                    lfg["cy"][paxel_id][pixel_id].append(dys[r])
                    lfg["t"][paxel_id][pixel_id].append(dd)
    return lfg


def _light_field_geometry_fraction_min_rays_per_sensor(
    light_field_geometry,
    min_num_rays_per_sensor
):
    lfg = light_field_geometry
    num_paxel = len(lfg["y"])
    num_smallcameras = len(lfg["y"][0])

    num_sensors_good = 0
    num_sensors = 0
    for key in ["y", "cy"]:
        for pa in range(num_paxel):
            for pi in range(num_smallcameras):
                num_sensors += 1
                num_rays_per_sensor = len(lfg[key][pa][pi])
                if num_rays_per_sensor > min_num_rays_per_sensor:
                    num_sensors_good += 1
    return num_sensors_good / num_sensors


def _light_field_geometry_condense(light_field_geometry):
    lfg = light_field_geometry
    num_paxel = len(lfg["y"])
    num_smallcameras = len(lfg["y"][0])

    out = {
        "y": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
        "cy": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
        "y_std": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
        "cy_std": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
        "t": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
        "t_std": np.nan * np.ones(shape=(num_paxel, num_smallcameras)),
    }
    for key in ["y", "cy", "t"]:
        for pa in range(num_paxel):
            for pi in range(num_smallcameras):
                arr = np.array(lfg[key][pa][pi])
                out[key][pa][pi] = np.mean(arr)
                out[key + "_std"][pa][pi] = np.std(arr)
    return out


def estimate_light_field_geometry(
    plenoscope_geometry,
    max_off_axis_angle_deg,
    num_rays,
    prng,
    max_num_loops=100,
    max_num_rays_per_sensor=1000,
    min_num_rays_per_sensor=25,
):
    pg = plenoscope_geometry

    lfg_temp = _light_field_geometry_init(plenoscope_geometry=pg)

    num_loops = 0
    good_statistics = 0.0
    while good_statistics < 0.99:
        if num_loops > max_num_loops:
            break

        print("light-field-geometry", int(good_statistics*100), "%")

        lfg_temp = _light_field_geometry_add_rays(
            light_field_geometry=lfg_temp,
            plenoscope_geometry=pg,
            max_off_axis_angle_deg=max_off_axis_angle_deg,
            num_rays=num_rays,
            max_num_rays_per_sensor=max_num_rays_per_sensor,
            prng=prng,
        )

        good_statistics = _light_field_geometry_fraction_min_rays_per_sensor(
            light_field_geometry=lfg_temp,
            min_num_rays_per_sensor=min_num_rays_per_sensor,
        )
        num_loops += 1

    lfg = _light_field_geometry_condense(light_field_geometry=lfg_temp)

    return lfg


def add2ax_optics_and_photons(ax, plenoscope_geometry, rc, max_num_rays=1000):
    pg = plenoscope_geometry
    num_rays = np.min([max_num_rays, rc["emission_positions"].shape[0]])

    alpha = np.max([1.0 / num_rays, 1.0 / 255.0])
    for i in range(num_rays):
        ax.plot(
            [
                rc["emission_positions"][i, 0],
                rc["sphere_intersection_positions"][i, 0],
            ],
            [
                rc["emission_positions"][i, 1],
                rc["sphere_intersection_positions"][i, 1],
            ],
            "k-",
            alpha=alpha,
        )

        ax.plot(
            [
                rc["sphere_intersection_positions"][i, 0],
                rc["sensor_plane_intersection_positions"][i, 0],
            ],
            [
                rc["sphere_intersection_positions"][i, 1],
                rc["sensor_plane_intersection_positions"][i, 1],
            ],
            "k-",
            alpha=alpha,
        )

    sensor_x = -pg["mirror"]["curvature_radius"] + pg["sensor"]["distance"]

    for pixel_edge in pg["sensor"]["smallcameras"]["edges"]:
        ax.plot(
            [sensor_x, sensor_x + 0.001],
            [pixel_edge, pixel_edge],
            "k-",
            alpha=0.2,
            linewidth=0.5,
        )

    ax.plot(
        [sensor_x, sensor_x],
        [-pg["sensor"]["diameter"] / 2, pg["sensor"]["diameter"] / 2,],
        "k-",
        alpha=0.2,
    )
    sphere = sphere_surface(
        x=0,
        y=0,
        r=pg["mirror"]["curvature_radius"],
        theta_start=np.deg2rad(180 - 12),
        theta_end=np.deg2rad(180 + 12),
    )
    ax.plot(sphere[:, 0], sphere[:, 1], "k-")
    ax.plot(
        [
            -pg["mirror"]["curvature_radius"] * 1.01,
            (-pg["mirror"]["curvature_radius"] + pg["sensor"]["distance"])
            * 0.98,
        ],
        [0, 0],
        "k-",
        alpha=0.2,
    )
    ax.set_aspect("equal")
    ax.set_axis_off()


def roi_limits(curvature_radius, roi_radius, incident_direction):
    focal_length = curvature_radius / 2
    sensor_x = -curvature_radius + focal_length
    sensor_y = -focal_length * np.tan(incident_direction)
    return (
        [sensor_x - 1.8 * roi_radius, sensor_x + 0.2 * roi_radius],
        [sensor_y - roi_radius, sensor_y + roi_radius],
    )


def paxel_alpha(num_paxel_idxs, paxel_idx):
    return np.linspace(0.25, 1, num_paxel_idxs)[paxel_idx] ** 2


def paxel_label_style_alpha(num_paxel, num_paxel_idxs, paxel_idx, marker=True):
    if num_paxel == 1:
        label = "image-sensor ({:d} paxel)".format(num_paxel)
        if marker:
            style = "k+:"
        else:
            style = "k"
        alpha = 1
    else:
        label = "light-field-sensor ({:d} paxels)".format(num_paxel)
        if marker:
            style = "kx-"
        else:
            style = "k"
        alpha = paxel_alpha(num_paxel_idxs, paxel_idx)
    return label, style, alpha


def _triangle(x, loc, width):
    """
    A triangle function.

    Parameters
    ----------
    x: array
            The x-axis where the triangle-function lives on.
    loc: float
            Location of the triangle's peak wrt. 'x'.
    width: float
            Width of the triangle's base.
    """
    height = 1/width

    _val = 1 - (1/width)*np.abs(x - loc)
    _comp = np.c_[_val, np.zeros(len(x))]
    return height * np.max(_comp, axis=1)


def histogram_image(cy, cy_std, edges):
    """
    Not only histograms the directions 'cy' into the bins defined by 'edges',
    but also adds to the neighboring bins when these are in reach of 'cy_std'.
    Actually, for each sample a triangle at location 'cy', and with a width
    of 'cy_std' is added to the bins.
    """
    assert len(cy) == len(cy_std)
    counts = np.zeros(len(edges) - 1)
    centers = (edges[1:] + edges[:-1])/2

    for i in range(len(cy)):
        count = _triangle(x=centers, loc=cy[i], width=cy_std[i])
        counts += count
    return counts


def simulate_point_source(
    plenoscope_geometry, true_incident_direction, num_rays, prng
):
    pg = plenoscope_geometry
    print(
        "direction", np.round(np.rad2deg(true_incident_direction), 2), "deg",
    )

    (
        emission_positions,
        support_positions
    ) = make_isochron_light_front_from_point_source(
        true_incident_direction=true_incident_direction,
        point_source_spread=np.deg2rad(POINT_SOURCE_SPREAD_DEG),
        aperture_diameter=pg["mirror"]["diameter"],
        emission_distance=pg["sensor"]["distance"] * 1.25,
        aperture_distance=-pg["mirror"]["curvature_radius"],
        num_rays=num_rays,
    )

    """
    support_positions = np.zeros(shape=(num_rays, 2))
    support_positions[:, 0] = -pg["mirror"]["curvature_radius"]

    support_positions[:, 1] = prng.uniform(
        low=-pg["mirror"]["diameter"] / 2,
        high=pg["mirror"]["diameter"] / 2,
        size=num_rays,
    )

    emission_distance = pg["sensor"]["distance"] * 1.25

    emission_positions = np.zeros(shape=(num_rays, 2))
    emission_positions[:, 0] = (
        emission_distance - pg["mirror"]["curvature_radius"]
    )
    true_incident_directions = prng.normal(
        loc=true_incident_direction,
        scale=np.deg2rad(POINT_SOURCE_SPREAD_DEG),
        size=num_rays,
    )
    emission_positions[:, 1] = support_positions[
        :, 1
    ] + emission_distance * np.tan(true_incident_directions)
    """

    directions = support_positions - emission_positions
    for i in range(num_rays):
        directions[i, :] = directions[i, :] / np.linalg.norm(directions[i, :])

    pixel_ids = np.zeros(num_rays, dtype=np.uint)
    paxel_ids = np.zeros(num_rays, dtype=np.uint)

    sphere_intersection_positions = np.zeros(shape=(num_rays, 2))
    sensor_plane_intersection_positions = np.zeros(shape=(num_rays, 2))
    path_length_emission_to_absorption = np.zeros(num_rays)

    for i in range(num_rays):

        (
            sphere_intersection_position,
            sensor_plane_intersection_position,
        ) = propagate_photon(
            support_x=support_positions[i, 0],
            support_y=support_positions[i, 1],
            direction_x=directions[i, 0],
            direction_y=directions[i, 1],
            curvature_radius=pg["mirror"]["curvature_radius"],
        )

        pixel_ids[i] = (
            np.digitize(
                sensor_plane_intersection_position[1],
                bins=pg["sensor"]["smallcameras"]["edges"],
            )
            - 1
        )

        paxel_ids[i] = (
            np.digitize(support_positions[i, 1], bins=pg["paxel"]["edges"]) - 1
        )

        sphere_intersection_positions[i, :] = sphere_intersection_position

        sensor_plane_intersection_positions[
            i, :
        ] = sensor_plane_intersection_position

        emi_to_sph = np.linalg.norm(
            emission_positions[i] - sphere_intersection_positions[i]
        )
        sph_to_sen = np.linalg.norm(
            sphere_intersection_positions[i] - sensor_plane_intersection_positions[i]
        )
        path_length_emission_to_absorption[i] = emi_to_sph + sph_to_sen

    light_field_calibrated_cy = []
    light_field_calibrated_cy_std = []
    light_field_calibrated_t = []
    light_field_calibrated_t_std = []
    for ray in range(num_rays):
        if (
            pixel_ids[ray] >= 0
            and pixel_ids[ray] < pg["sensor"]["smallcameras"]["num"]
        ):
            if paxel_ids[ray] >= 0 and paxel_ids[ray] < num_paxel:
                cy = -lfg["cy"][paxel_ids[ray], pixel_ids[ray]]
                cy_std = lfg["cy_std"][paxel_ids[ray], pixel_ids[ray]]

                # time
                # ----
                t = (
                    path_length_emission_to_absorption[ray]
                    - lfg["t"][paxel_ids[ray], pixel_ids[ray]]
                )
                t_std = lfg["t_std"][paxel_ids[ray], pixel_ids[ray]]
                if not np.isnan(cy):
                    light_field_calibrated_cy.append(cy)
                    light_field_calibrated_cy_std.append(cy_std)
                    light_field_calibrated_t.append(t)
                    light_field_calibrated_t_std.append(t_std)

    light_field_calibrated_image = histogram_image(
        cy=light_field_calibrated_cy,
        cy_std=light_field_calibrated_cy_std,
        edges=pg["computed_pixels"]["edges"],
    )
    light_field_calibrated_image_time = histogram_image(
        cy=light_field_calibrated_t,
        cy_std=light_field_calibrated_t_std,
        edges=pg["computed_timebins"]["edges"],
    )

    light_and_image = {
        "incident_direction": true_incident_direction,
        "support_positions": support_positions,
        "emission_positions": emission_positions,
        "sphere_intersection_positions": sphere_intersection_positions,
        "sensor_plane_intersection_positions": sensor_plane_intersection_positions,
        "path_length_emission_to_absorption": path_length_emission_to_absorption,
        "pixel_ids": pixel_ids,
        "paxel_ids": paxel_ids,
        "light_field_calibrated_directions": light_field_calibrated_cy,
        "light_field_calibrated_image": light_field_calibrated_image,
        "light_field_calibrated_image_time": light_field_calibrated_image_time,
    }

    return light_and_image


"""


 princ. aperture           sensor
--------|---------------------|---------------------|--------> x
      -2.5                  -1.25                   0

"""


RANDOM_SEED = 42
prng = np.random.Generator(np.random.MT19937(seed=RANDOM_SEED))

RELATIVE_RESOLUTION = 1
DPI = 320
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

PLOT_LIGHT_RAYS = False
SENSOR_DIAMETER = 0.5
NUM_SMALL_CAMERAS = 340
MIRROR_DIAMETER = 1.0
CURVATURE_RADIUS = 2.5
SENSOR_DISTANCE = 1.25
OFF_AXIS_ANGLES = np.deg2rad(np.linspace(0, 8.5, 3))
POINT_SOURCE_SPREAD_DEG = 0.025
NUM_TIME_BINS = 1000
PLENOSCOPES = [
    {"num_paxel": 1, "plot_image": True},
    {"num_paxel": 3, "plot_image": True},
    #{"num_paxel": 9, "plot_image": True},
    #{"num_paxel": 27, "plot_image": False},
]

res = []
for plenoscop_config in PLENOSCOPES:
    num_paxel = plenoscop_config["num_paxel"]
    print("------------- num_paxel", num_paxel, " -------------")

    pg = {}
    pg["mirror"] = {}
    pg["mirror"]["diameter"] = MIRROR_DIAMETER
    pg["mirror"]["curvature_radius"] = CURVATURE_RADIUS

    pg["sensor"] = {}
    pg["sensor"]["distance"] = SENSOR_DISTANCE
    pg["sensor"]["diameter"] = SENSOR_DIAMETER
    pg["sensor"]["smallcameras"] = {}
    pg["sensor"]["smallcameras"]["num"] = NUM_SMALL_CAMERAS
    pg["sensor"]["smallcameras"]["edges"] = np.linspace(
        -pg["sensor"]["diameter"] / 2,
        pg["sensor"]["diameter"] / 2,
        pg["sensor"]["smallcameras"]["num"] + 1,
    )
    pg["sensor"]["smallcameras"]["centers"] = (
        pg["sensor"]["smallcameras"]["edges"][:-1]
        + pg["sensor"]["smallcameras"]["edges"][1:]
    ) / 2

    _max_off_axis_angle_on_sensor_plane = np.arctan(
        (0.5 * pg["sensor"]["diameter"]) / pg["sensor"]["distance"]
    )
    pg["computed_pixels"] = {}
    pg["computed_pixels"]["edges"] = np.linspace(
        -_max_off_axis_angle_on_sensor_plane,
        _max_off_axis_angle_on_sensor_plane,
        pg["sensor"]["smallcameras"]["num"] + 1,
    )
    pg["computed_pixels"]["centers"] = (
        pg["computed_pixels"]["edges"][:-1]
        + pg["computed_pixels"]["edges"][1:]
    ) / 2

    pg["computed_timebins"] = {}
    pg["computed_timebins"]["edges"] = np.linspace(
        -10,
        10,
        NUM_TIME_BINS + 1,
    )
    pg["computed_timebins"]["centers"] = (
        pg["computed_timebins"]["edges"][:-1]
        + pg["computed_timebins"]["edges"][1:]
    ) / 2


    pg["paxel"] = {}
    pg["paxel"]["num"] = num_paxel
    pg["paxel"]["edges"] = np.linspace(
        -MIRROR_DIAMETER / 2, MIRROR_DIAMETER / 2, num_paxel + 1,
    )
    pg["paxel"]["centers"] = (
        pg["paxel"]["edges"][:-1] + pg["paxel"]["edges"][1:]
    ) / 2

    # light-field-geometry
    # --------------------
    lfg = estimate_light_field_geometry(
        plenoscope_geometry=pg,
        max_off_axis_angle_deg=12,
        num_rays=10 * 1000 * RELATIVE_RESOLUTION,
        prng=prng,
    )

    # point-source-example
    # --------------------
    image_results = []
    for ci, true_incident_direction in enumerate(OFF_AXIS_ANGLES):
        light_and_image = simulate_point_source(
            plenoscope_geometry=pg,
            true_incident_direction=true_incident_direction,
            num_rays=20 * 1000 * RELATIVE_RESOLUTION,
            prng=prng,
        )
        image_results.append(light_and_image)

    res.append(
        {
            "plenoscope_geometry": pg,
            "light_field_geometry": lfg,
            "images": image_results,
        }
    )


any_plenoscope_result = res[0]
for idir in range(len(any_plenoscope_result["images"])):
    pg = any_plenoscope_result["plenoscope_geometry"]
    images = any_plenoscope_result["images"]


# plot light-rays, (can use any num paxel)
# ========================================
if PLOT_LIGHT_RAYS:
    for idir in range(len(any_plenoscope_result["images"])):
        # overview
        # --------
        fig = plt.figure(figsize=(1080 / DPI, 1080 / DPI))
        ax = fig.add_axes([0, 0, 1, 1])
        add2ax_optics_and_photons(
            ax=ax, plenoscope_geometry=pg, rc=images[idir], max_num_rays=1000,
        )
        xlim, ylim = roi_limits(
            curvature_radius=pg["mirror"]["curvature_radius"],
            roi_radius=0.05,
            incident_direction=images[idir]["incident_direction"],
        )
        ax.plot(
            [xlim[0], xlim[1], xlim[1], xlim[0], xlim[0]],
            [ylim[0], ylim[0], ylim[1], ylim[1], ylim[0]],
            "k-",
            alpha=1,
            linewidth=0.3,
        )
        ax.set_xlim(
            [
                -pg["mirror"]["curvature_radius"] - 0.05,
                -pg["mirror"]["curvature_radius"]
                + pg["sensor"]["distance"]
                + 0.05,
            ]
        )
        ax.set_ylim(
            [-pg["mirror"]["diameter"] / 1.5, pg["mirror"]["diameter"] / 1.5]
        )
        fig.savefig(
            os.path.join(
                OUT_DIR, "spherical_aberrations_overview_{:01d}.jpg".format(idir),
            ),
            dpi=DPI,
        )
        plt.close("all")

        # close up
        # --------
        fig = plt.figure(figsize=(1080 / DPI, 1080 / DPI))
        ax = fig.add_axes([0, 0, 1, 1])
        add2ax_optics_and_photons(
            ax=ax, plenoscope_geometry=pg, rc=images[idir], max_num_rays=1000,
        )
        xlim, ylim = roi_limits(
            curvature_radius=pg["mirror"]["curvature_radius"],
            roi_radius=0.05,
            incident_direction=images[idir]["incident_direction"],
        )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        fig.savefig(
            os.path.join(
                OUT_DIR, "spherical_aberrations_close_up_{:01d}.jpg".format(idir),
            ),
            dpi=DPI,
        )
        plt.close("all")


# plot images
# ===========
num_plenoscopes = len(res)
num_angles = len(res[0]["images"])

for idir in range(len(any_plenoscope_result["images"])):
    fig = plt.figure(figsize=(960 / DPI, 540 / DPI))
    ax = fig.add_axes([0.2, 0.25, 0.75, 0.7])

    for iple in range(num_plenoscopes):
        if PLENOSCOPES[iple]["plot_image"]:
            pg = res[iple]["plenoscope_geometry"]

            normed_light_field_calibrated_image = np.array(
                res[iple]["images"][idir]["light_field_calibrated_image"]
            )
            normed_light_field_calibrated_image = (
                normed_light_field_calibrated_image
                / np.sum(normed_light_field_calibrated_image)
            )

            if pg["paxel"]["num"] == 1:
                ax.fill_between(
                    np.rad2deg(pg["computed_pixels"]["centers"]),
                    normed_light_field_calibrated_image,
                    step="pre",
                    alpha=0.33,
                    color="k",
                    linewidth=0.0,
                )
            else:
                ax.plot(
                    np.rad2deg(pg["computed_pixels"]["centers"]),
                    normed_light_field_calibrated_image,
                    drawstyle="steps",
                    alpha=1,
                    color="k",
                )

        ax.semilogy()
        ax.set_ylim([1e-3, 1e-0])
        roi_radius_deg = 2.5
        true_direction_deg = np.rad2deg(
            res[iple]["images"][idir]["incident_direction"]
        )
        ax.set_xlim(
            [
                true_direction_deg - roi_radius_deg,
                true_direction_deg + roi_radius_deg,
            ]
        )

        ax.set_ylabel(r"intensity / 1")
        ax.set_xlabel(r"off-axis-angle / deg")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        ax.plot(
            [true_direction_deg, true_direction_deg],
            [1e-3, 1e-0],
            c="k",
            alpha=0.5,
            linestyle="--",
            linewidth=0.8,
        )

    fig.savefig(
        os.path.join(OUT_DIR, "image_{idir:01d}_comb.jpg".format(idir=idir),),
        dpi=DPI,
    )
    plt.close("all")


# Plot summary: Spread VS off-axis-angle VS number paxel
# ======================================================
fig = plt.figure(figsize=(960 / DPI, 960 / DPI))
ax = fig.add_axes([0.17, 0.15, 0.8, 0.8])

num_plenoscopes = len(res)
for iple in range(num_plenoscopes):
    num_paxel = res[iple]["plenoscope_geometry"]["paxel"]["num"]
    label, style, alpha = paxel_label_style_alpha(
        num_paxel=num_paxel, num_paxel_idxs=num_plenoscopes, paxel_idx=iple,
    )
    psf_std_light_field = [
        np.std(res[iple]["images"][inci]["light_field_calibrated_directions"])
        for inci in range(len(res[iple]["images"]))
    ]
    ax.plot(
        np.rad2deg(OFF_AXIS_ANGLES),
        np.rad2deg(psf_std_light_field),
        style,
        label=label,
        alpha=alpha,
    )
ax.set_ylabel(r"Standard-deviation / deg")
ax.set_xlabel(r"off-axis-angle / deg")
_ylim = ax.get_ylim()
ax.set_ylim([0.0, _ylim[1]])
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.legend(loc="best", fontsize=10)
ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
fig.savefig(os.path.join(OUT_DIR, "imaging_cy_std_vs_cy.jpg"), dpi=DPI)
