import numpy as np
import os
import plenoirf
import json_utils
import subprocess
import homogeneous_transformation as htra
import sebastians_matplotlib_addons as sebplt


def sun_light_irradiation_vs_zenith(zenith_deg):
    # https://en.wikipedia.org/wiki/Air_mass_(solar_energy)#math_I.1
    #
    #   One empirical approximation model for solar intensity
    #   versus airmass is given by:[13][14]
    #
    #       I=1.1\times I_{\mathrm {o} }\times 0.7^{(AM^{0.678})}\,}
    #
    # [13]  PVCDROM retrieved 1 May 2011, Stuart Bowden and Christiana
    #       Honsberg, Solar Power Labs, Arizona State University
    # [14]  Meinel, A. B. and Meinel, M. P. (1976). Applied Solar Energy
    #       Addison Wesley Publishing Co.
    #
    zd_airmass_irradiance = np.array(
        [
            [0, 1, 1040],
            [23, 1.09, 1020],
            [30, 1.15, 1010],
            [45, 1.41, 950],
            [48.2, 1.5, 930],
            [60, 2, 840],
            [70, 2.9, 710],
            [75, 3.8, 620],
            [80, 5.6, 470],
            [85, 10, 270],
            [90, 38, 20],
        ]
    )
    zd_deg = zd_airmass_irradiance[:, 0]
    irradiance_W_per_m2 = zd_airmass_irradiance[:, 2]
    return np.interp(x=zenith_deg, xp=zd_deg, fp=irradiance_W_per_m2)


def make_bin_edges(edge_width, bin_width):
    num_bins = np.ceil(edge_width / bin_width)
    width_to_put_bins = num_bins * bin_width
    return np.linspace(
        -width_to_put_bins / 2, width_to_put_bins / 2, num_bins + 1
    )


def draw_x_y_in_radius(prng, radius, size):
    rho = np.sqrt(prng.uniform(low=0.0, high=1.0, size=size)) * radius
    phi = prng.uniform(low=0.0, high=2.0 * np.pi, size=size)
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def draw_direction_in_z_cone(prng, cone_half_angle_deg, size):
    # azimuth
    az = prng.uniform(low=0.0, high=(2.0 * np.pi), size=size)

    # zenith
    zd_min = 0.0
    zd_max = np.deg2rad(cone_half_angle_deg)

    z_min = (np.cos(zd_min) + 1.0) / 2.0
    z_range = (np.cos(zd_max) + 1.0) / 2.0 - z_min
    z = z_range * prng.uniform(low=0.0, high=1.0, size=size) + z_min
    zd = np.arccos(2.0 * z - 1.0)

    # direction vector
    sin_zd = np.sin(zd)
    return np.array(
        [
            sin_zd * np.cos(az),
            sin_zd * np.sin(az),
            np.cos(zd),
        ]
    ).T


def draw_photons_in_z_disk(prng, num, disk_radius, cone_half_angle_deg):
    photons = init_photons(num=num)

    photons["wavelengths"] = 433e-9 * np.ones(num)
    x, y = draw_x_y_in_radius(prng=prng, radius=disk_radius, size=num)
    photons["supports"][:, 0] = x
    photons["supports"][:, 1] = y
    photons["directions"] = draw_direction_in_z_cone(
        prng=prng, cone_half_angle_deg=cone_half_angle_deg, size=num
    )
    return photons


def init_photons(num):
    return {
        "supports": np.zeros(shape=(num, 3)),
        "directions": np.zeros(shape=(num, 3)),
        "wavelengths": np.zeros(num),
    }


def num_photons(photons):
    return photons["supports"].shape[0]


def transform_photons(homtra, photons):
    t = htra.compile(homtra)
    out = init_photons(num=num_photons(photons))
    out["supports"], out["directions"] = htra.transform_ray(
        t=t,
        ray_supports=photons["supports"],
        ray_directions=photons["directions"],
    )
    out["wavelengths"] = photons["wavelengths"]
    return out


def write_photons(path, photons):
    num = num_photons(photons)
    out = np.zeros(shape=(num, 8))
    out[:, 0] = np.arange(num)
    out[:, 1:4] = photons["supports"]
    out[:, 4:7] = photons["directions"]
    out[:, 7] = photons["wavelengths"]
    np.savetxt(path, out, delimiter=" ", newline="\n")


def read_sensor(path):
    result = np.genfromtxt(path)
    if result.shape[0] == 0:
        result = np.zeros(shape=[0, 6])
    r = {}
    r["x"] = result[:, 0]
    r["y"] = result[:, 1]
    r["cx"] = result[:, 2]
    r["cy"] = result[:, 3]
    r["wavelength"] = result[:, 4]
    r["arrival_time"] = result[:, 5]
    return r


def histogram_xy(sensor_response, bin_edges):
    return np.histogram2d(
        sensor_response["x"], sensor_response["y"], bins=bin_edges
    )[0]


def save_img(path, img, bin_edges, vmin=None, vmax=None):
    fig = sebplt.figure()
    ax = sebplt.add_axes(fig=fig, span=[0.1, 0.1, 0.8, 0.8])
    ax_cb = sebplt.add_axes(fig=fig, span=[0.92, 0.1, 0.02, 0.8])
    _cmap = ax.pcolormesh(
        bin_edges[0],
        bin_edges[1],
        np.transpose(img),
        cmap="magma_r",
        norm=sebplt.plt_colors.PowerNorm(gamma=0.5, vmin=vmin, vmax=vmax),
    )
    sebplt.plt.colorbar(_cmap, cax=ax_cb, extend="max")
    ax.set_aspect("equal")
    fig.savefig(path)
    sebplt.close(fig)


sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

SEED = 413372


mirror_diameter = 71.0

prng = np.random.Generator(np.random.PCG64(SEED))

space_truss_tower = {}
space_truss_tower["height"] = mirror_diameter * 2.277
space_truss_tower["width"] = mirror_diameter * 0.27722
space_truss_tower["pos"] = [2.37 * mirror_diameter, 0.0, 0.0]
space_truss_tower["bin_edges"] = [
    make_bin_edges(edge_width=space_truss_tower["height"], bin_width=1),
    make_bin_edges(edge_width=space_truss_tower["width"], bin_width=1),
]

concrete_tower = {}
concrete_tower["height"] = mirror_diameter * 0.9
concrete_tower["width"] = mirror_diameter * (1 / 20)
concrete_tower["pos"] = [0.9 * mirror_diameter, 0.0, 0.0]
concrete_tower["bin_edges"] = [
    make_bin_edges(edge_width=concrete_tower["height"], bin_width=1),
    make_bin_edges(edge_width=concrete_tower["width"], bin_width=1),
]

screen_bin_edges = [
    make_bin_edges(edge_width=13, bin_width=0.1),
    make_bin_edges(edge_width=13, bin_width=0.1),
]

merlict_propagate = {}
merlict_propagate["executable_path"] = os.path.join(
    "build", "merlict", "merlict-propagate"
)
merlict_propagate["config"] = {
    "max_num_interactions_per_photon": 1337,
    "use_multithread_when_possible": True,
}

portal_scenery = json_utils.read(
    os.path.join("resources", "acp", "71m", "scenery", "scenery.json")
)

mirror_reflectivity = portal_scenery["functions"][0]
assert mirror_reflectivity["name"] == "mirror_reflectivity_vs_wavelength"

portal_frame = portal_scenery["children"][0]
assert portal_frame["name"] == "Portal"

portal_mirror_frame = portal_frame["children"][0]
assert portal_mirror_frame["name"] == "reflector"

portal_mirror_scenery = {
    "functions": [mirror_reflectivity],
    "colors": [
        {"name": "orange", "rgb": [255, 91, 49]},
        {"name": "gray", "rgb": [100, 100, 100]},
        {"name": "green", "rgb": [30, 230, 50]},
    ],
    "children": [
        {
            "type": "Frame",
            "name": "Portal",
            "pos": [0, 0, 0],
            "rot": [0, 0, 0],
            "children": [
                portal_mirror_frame,
            ],
        },
        {
            "type": "Plane",
            "name": "space_truss_tower_surface",
            "sensor_id": 0,
            "pos": space_truss_tower["pos"]
            + np.array([0, 0, 0.5 * space_truss_tower["height"]]),
            "rot": np.deg2rad([0.0, 90.0, 0.0]),
            "x_width": space_truss_tower["height"],
            "y_width": space_truss_tower["width"],
            "surface": {
                "outer_color": "orange",
                "inner_color": "gray",
            },
            "children": [],
        },
        {
            "type": "Plane",
            "name": "concrete_tower_surface",
            "sensor_id": 1,
            "pos": concrete_tower["pos"]
            + np.array([0, 0, 0.5 * concrete_tower["height"]]),
            "rot": np.deg2rad([0.0, 90.0, 0.0]),
            "x_width": concrete_tower["height"],
            "y_width": concrete_tower["width"],
            "surface": {
                "outer_color": "green",
                "inner_color": "gray",
            },
            "children": [],
        },
        {
            "type": "Disc",
            "name": "screen",
            "sensor_id": 2,
            "pos": [0, 0, 106.5],
            "rot": [0, 0, 0],
            "radius": 6.5,
            "surface": {
                "outer_color": "gray",
                "inner_color": "orange",
            },
            "children": [],
        },
        {
            "type": "Disc",
            "name": "screen_shield",
            "pos": [0, 0, 106.6],
            "rot": [0, 0, 0],
            "radius": 6.7,
            "surface": {
                "outer_color": "gray",
                "inner_color": "gray",
            },
            "children": [],
        },
    ],
}


distance_of_light_to_mirror = mirror_diameter * 5
sun_light_cone_half_angle_deg = 0.25
sun_light_disk_radius = 0.8 * mirror_diameter
sun_light_disk_area = sun_light_disk_radius**2 * np.pi
sun_light_photons_areal_density = 100
sun_light_num_photons = int(
    sun_light_disk_area * sun_light_photons_areal_density
)

work_dir = "projection_of_sun"
os.makedirs(work_dir, exist_ok=True)

json_utils.write(
    os.path.join(work_dir, "scenery.json"),
    portal_mirror_scenery,
    indent=4,
)

json_utils.write(
    os.path.join(work_dir, "config.json"),
    merlict_propagate["config"],
    indent=4,
)

zenith_distances_deg = np.arange(30, 90, 1)
num_zenith_distances = len(zenith_distances_deg)
res = {"screen": [], "space_truss_tower": [], "concrete_tower": []}
for izenith in range(num_zenith_distances):
    zd_dir = os.path.join(
        work_dir, "zd_{:06d}deg".format(zenith_distances_deg[izenith])
    )

    if not os.path.exists(zd_dir):

        os.makedirs(zd_dir)

        sun_light_zenith_distance_rad = (-1.0) * np.deg2rad(
            zenith_distances_deg[izenith]
        )

        sun_light_homtra = {
            "pos": distance_of_light_to_mirror
            * np.array(
                [
                    np.sin(sun_light_zenith_distance_rad),
                    0.0,
                    np.cos(sun_light_zenith_distance_rad),
                ]
            ),
            "rot": {
                "repr": "axis_angle",
                "axis": np.array([0.0, 1.0, 0.0]),
                "angle_deg": 180 + np.rad2deg(sun_light_zenith_distance_rad),
            },
        }

        _sun_light_photons = draw_photons_in_z_disk(
            prng=prng,
            num=sun_light_num_photons,
            disk_radius=sun_light_disk_radius,
            cone_half_angle_deg=sun_light_cone_half_angle_deg,
        )

        sun_light_photons = transform_photons(
            homtra=sun_light_homtra, photons=_sun_light_photons
        )

        write_photons(
            path=os.path.join(zd_dir, "photons.json"),
            photons=sun_light_photons,
        )

        subprocess.call(
            [
                merlict_propagate["executable_path"],
                "--scenery",
                os.path.join(work_dir, "scenery.json"),
                "--config",
                os.path.join(work_dir, "config.json"),
                "--input",
                os.path.join(zd_dir, "photons.json"),
                "--output",
                os.path.join(zd_dir, "sensor"),
                "--random_seed",
                str(SEED + izenith),
            ]
        )

    response_screen = read_sensor(path=os.path.join(zd_dir, "sensor1_2"))
    response_space_truss_tower = read_sensor(
        path=os.path.join(zd_dir, "sensor1_0")
    )
    response_concrete_tower = read_sensor(
        path=os.path.join(zd_dir, "sensor1_1")
    )

    screen_img = histogram_xy(
        sensor_response=response_screen, bin_edges=screen_bin_edges
    )
    screen_img /= sun_light_photons_areal_density

    space_truss_tower_img = histogram_xy(
        sensor_response=response_space_truss_tower,
        bin_edges=space_truss_tower["bin_edges"],
    )
    space_truss_tower_img /= sun_light_photons_areal_density

    concrete_tower_img = histogram_xy(
        sensor_response=response_concrete_tower,
        bin_edges=concrete_tower["bin_edges"],
    )
    concrete_tower_img /= sun_light_photons_areal_density

    res["screen"].append(screen_img)
    res["space_truss_tower"].append(space_truss_tower_img)
    res["concrete_tower"].append(concrete_tower_img)


res["screen"] = np.array(res["screen"])
res["space_truss_tower"] = np.array(res["space_truss_tower"])
res["concrete_tower"] = np.array(res["concrete_tower"])

res["screen_vmax"] = np.max(res["screen"])
res["space_truss_tower_vmax"] = np.max(res["space_truss_tower"])
res["concrete_tower_vmax"] = np.max(res["concrete_tower"])

for izenith in range(num_zenith_distances):
    zd_str = "zd_{:06d}deg".format(zenith_distances_deg[izenith])

    save_img(
        path=os.path.join(work_dir, "screen" + zd_str + ".jpg"),
        img=res["screen"][izenith],
        bin_edges=screen_bin_edges,
        vmin=0.0,
        vmax=res["screen_vmax"],
    )
    save_img(
        path=os.path.join(work_dir, "space_truss_tower" + zd_str + ".jpg"),
        img=res["space_truss_tower"][izenith],
        bin_edges=space_truss_tower["bin_edges"],
        vmin=0.0,
        vmax=res["space_truss_tower_vmax"],
    )
    save_img(
        path=os.path.join(work_dir, "concrete_tower" + zd_str + ".jpg"),
        img=res["concrete_tower"][izenith],
        bin_edges=concrete_tower["bin_edges"],
        vmin=0.0,
        vmax=res["concrete_tower_vmax"],
    )

max_percentile = 99.5

concrete_tower_hot = []
space_truss_tower_hot = []
for izenith in range(num_zenith_distances):
    concrete_tower_hot.append(
        np.percentile(a=res["concrete_tower"][izenith], q=max_percentile)
    )
    space_truss_tower_hot.append(
        np.percentile(a=res["space_truss_tower"][izenith], q=max_percentile)
    )


direct_sun_light_irradiance_kW_per_m2 = (
    sun_light_irradiation_vs_zenith(zenith_distances_deg) * 1e-3
)

concrete_tower_irradiance_kW_per_m2 = (
    concrete_tower_hot
    * sun_light_irradiation_vs_zenith(zenith_distances_deg)
    * 1e-3
) + direct_sun_light_irradiance_kW_per_m2

space_truss_tower_irradiance_kW_per_m2 = (
    space_truss_tower_hot
    * sun_light_irradiation_vs_zenith(zenith_distances_deg)
    * 1e-3
) + direct_sun_light_irradiance_kW_per_m2

fig = sebplt.figure({"rows": 720, "cols": 1920, "fontsize": 1.5})
ax = sebplt.add_axes(fig=fig, span=[0.12, 0.22, 0.85, 0.75])
ax.plot(
    zenith_distances_deg,
    concrete_tower_irradiance_kW_per_m2,
    linestyle="-",
    color="gray",
    label=r"on mirror's towers",
)
ax.plot(
    zenith_distances_deg,
    concrete_tower_irradiance_kW_per_m2,
    "o",
    color="gray",
    markersize=2,
)
ax.plot(
    zenith_distances_deg,
    space_truss_tower_irradiance_kW_per_m2,
    linestyle="-",
    color="black",
    label=r"on camera's towers",
)
ax.plot(
    zenith_distances_deg,
    space_truss_tower_irradiance_kW_per_m2,
    "o",
    color="black",
    markersize=2,
)
ax.plot(
    zenith_distances_deg,
    direct_sun_light_irradiance_kW_per_m2,
    linestyle=":",
    color="black",
    alpha=0.2,
    label=r"direct light from sun",
)
ax.set_xlabel(r"sun's distance to zenith$\,/\,1^{\circ{}}$")
ax.set_ylabel(r"irradiance$\,/\,$kW$\,$m$^{-2}$")
ax.legend()
fig.savefig(os.path.join(work_dir, "hot.jpg"))
sebplt.close(fig)
