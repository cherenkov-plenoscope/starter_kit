import argparse
import phantom_source
import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as sebplt
import plenopy as pl

argparser = argparse.ArgumentParser(
    prog="make_technical_phantom_source",
    description=(
        "Create a thechnical phantom-source and export figures "
        "showing the geometry of the source as well as images of it "
        "taken by a plenoscope."
    ),
)
argparser.add_argument(
    "light_field_geometry",
    metavar="LIGHT_FIELD_GEOMETRY_PATH",
    type=str,
    help="Path to the light_field_geometry to be used for the plenoscope.",
)
argparser.add_argument(
    "out_dir",
    metavar="OUT_DIR",
    type=str,
    help="Path to directory to write results.",
)
argparser.add_argument(
    "--merlict_path",
    metavar="PATH",
    type=str,
    default=os.path.join(
        "build", "merlict", "merlict-plenoscope-raw-photon-propagation"
    ),
    help="Path to merlict executable to propagate table of raw photons.",
)
argparser.add_argument(
    "--merlict_config",
    metavar="PATH",
    type=str,
    default=os.path.join(
        "resources",
        "acp",
        "merlict_propagation_config_no_night_sky_background.json",
    ),
    help=(
        "Path to merlict config on night-sky-background "
        "and sensor-cofiguration."
    ),
)
argparser.add_argument(
    "--seed",
    metavar="RANDOM_SEED",
    type=int,
    default=43,
    help="Seed for pseudo-random-number-generator.",
)

args = argparser.parse_args()

SEED = args.seed
LIGHT_FIELD_GEOMETRY_PATH = args.light_field_geometry
MERLICT_PATH = args.merlict_path
MERLICT_CONFIG_PATH = args.merlict_config
out_dir = args.out_dir

os.makedirs(name=out_dir, exist_ok=True)
with open(os.path.join(out_dir, "config.json"), "wt") as f:
    f.write(
        json_numpy.dumps(
            {
                "seed": SEED,
                "light_field_geometry": LIGHT_FIELD_GEOMETRY_PATH,
                "merlict_path": MERLICT_PATH,
                "merlict_config_path": MERLICT_CONFIG_PATH,
            },
            indent=4,
        )
    )

prng = np.random.Generator(np.random.MT19937(seed=SEED))

aperture_radius = 40
N = 1e4
M = 120
EMISSION_DISTANCE_TO_APERTURE = 1e3
MAX_OBJECT_DISTANCE = 27e3
XY_RADIUS = 400

# primary track
"""
                                    A
                B0                                      B1
        C0               C1                 C2                      C3
                    E
                  F
                 H
                 I
"""
"""
sscn = phantom_source.mesh.init()
sscn["vertices"]["A"] = [0, 0, MAX_OBJECT_DISTANCE]
sscn["vertices"]["B"] = [10, 12, MAX_OBJECT_DISTANCE-1000]
sscn["edges"].append(("A", "B", N*(10**2)))

phantom_source.simple_shower.append_random_edge(
    mesh=sscn,
    start_vkey="B",
    radius_xy=10,
    min_depth=2000,
    max_particles=3,
    N=N,
    prng=prng,
)

simg = phantom_source.mesh.init()
simg["vertices"]["A"] = [-2, 1.8, MAX_OBJECT_DISTANCE]

simg["vertices"]["B0"] = [-1.7, 1.6, 18.6e3]
simg["edges"].append(("A", "B0", N))

simg["vertices"]["B1"] = [-1.5, 1.9, 18.6e3]
simg["edges"].append(("A", "B1", N))

simg["vertices"]["C0"] = [-1.3, 1.2, 12.8e3]
simg["edges"].append(("B0", "C0", N))

simg["vertices"]["C1"] = [-1.6, 2.0, 12.8e3]
simg["edges"].append(("B0", "C1", N))

simg["vertices"]["D"] = [-0.4, 0.2, 12.8e3]
simg["edges"].append(("B0", "D", N))

simg["vertices"]["E"] = [0.4, -0.9, 8.9e3]
simg["edges"].append(("D", "E", N))

simg["vertices"]["F"] = [0.2, 0.0, 6.1e3]
simg["edges"].append(("E", "F", N))

simg["vertices"]["G"] = [1.2, -0.4, 4.2e3]
simg["edges"].append(("F", "G", N))

simg["vertices"]["H"] = [-1.2, 0.8, 2.9e3]
simg["edges"].append(("G", "H", N))

simg["vertices"]["I"] = [-1.0, -0.4, 2.0e3]
simg["edges"].append(("H", "I", N))
"""

# transform shower to scenery
"""
Mimg = []
Mimg.append(simg)

Mimg.append(
    phantom_source.mesh.spiral(
        pos=simg["vertices"]["B0"],
        turns=1.5,
        outer_radius=0.5,
        density=M * N,
        fn=110,
    )
)

Mimg.append(
    phantom_source.mesh.sun(
        pos=simg["vertices"]["D"],
        num_flares=7,
        radius=0.4,
        density=M * N,
        fn=110,
    )
)

Mimg.append(
    phantom_source.mesh.triangle(
        pos=simg["vertices"]["G"], radius=0.4, density=M * N,
    )
)

Mimg.append(
    phantom_source.mesh.sun(
        pos=simg["vertices"]["I"], num_flares=13, radius=1.2, density=M * N,
    )
)
"""
RR = 1.0

Mimg = []
Mimg.append(
    phantom_source.mesh.triangle(
        pos=[-1.0, +1.3, 2.5e3], radius=1.8, density=M * N * (2.5 ** RR),
    )
)
Mimg.append(
    phantom_source.mesh.spiral(
        pos=[-1.0, -1.3, 4.2e3],
        turns=2.5,
        outer_radius=1.7,
        density=M * N * (4.2 ** RR),
        fn=110,
    )
)
Mimg.append(
    phantom_source.mesh.sun(
        pos=[1.7, 0.0, 7.1e3],
        num_flares=11,
        radius=1.0,
        density=M * N * (7.1 ** RR),
        fn=110,
    )
)
Mimg.append(
    phantom_source.mesh.smiley(
        pos=[-1.0, +1.3, 11.9e3],
        radius=0.9,
        density=M * N * (11.9 ** RR),
        fn=50,
    )
)
Mimg.append(
    phantom_source.mesh.cross(
        pos=[+1.3, -1.3, 20.0e3], radius=0.7, density=M * N * (20.0 ** RR),
    )
)

Mscn = []
for mimg in Mimg:
    mscn = phantom_source.mesh.transform_image_to_scneney(mesh=mimg)
    Mscn.append(mscn)


phantom_source.plot.save_3d_views_of_meshes(
    meshes=Mscn,
    scale=1e-3,
    xlim=[-XY_RADIUS * 1e-3, XY_RADIUS * 1e-3],
    ylim=[-XY_RADIUS * 1e-3, XY_RADIUS * 1e-3],
    zlim=[0 * 1e-3, MAX_OBJECT_DISTANCE * 1e-3],
    elevations=[23, 90, 0],
    azimuths=[-60, 0, 0],
    paths=[os.path.join(out_dir, "mesh_{:d}.jpg".format(i)) for i in range(3)],
)

light_fields = phantom_source.light_field.make_light_fields_from_meshes(
    meshes=Mscn,
    aperture_radius=aperture_radius,
    prng=prng,
    emission_distance_to_aperture=EMISSION_DISTANCE_TO_APERTURE,
)

(
    event,
    light_field_geometry,
) = phantom_source.merlict.make_plenopy_event_and_read_light_field_geometry(
    light_fields=light_fields,
    light_field_geometry_path=LIGHT_FIELD_GEOMETRY_PATH,
    merlict_propagate_photons_path=MERLICT_PATH,
    merlict_propagate_config_path=MERLICT_CONFIG_PATH,
    random_seed=SEED,
)

lf_t, lf_lixel_ids = event.photon_arrival_times_and_lixel_ids()

pl.plot.refocus.save_refocus_stack(
    light_field_geometry=light_field_geometry,
    photon_lixel_ids=lf_lixel_ids,
    output_path=out_dir,
    title=None,
    obj_dist_min=2000.0,
    obj_dist_max=MAX_OBJECT_DISTANCE,
    time_slices_window_radius=1,
    steps=16,
    use_absolute_scale=True,
    image_prefix="refocus_",
    figure_style={"rows": 720, "cols": 1280, "fontsize": 1},
)
