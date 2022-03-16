import phantom_source
import numpy as np
import os
import sebastians_matplotlib_addons as sebplt
import plenopy as pl

SEED = 43
light_field_geometry_path = "2021-12-20_run_mini/light_field_geometry"
merlict_path = "build/merlict/merlict-plenoscope-raw-photon-propagation"
merlict_config_path = (
    "resources/acp/merlict_propagation_config_no_night_sky_background.json"
)


prng = np.random.Generator(np.random.MT19937(seed=SEED))

aperture_radius = 40
N = 1e4
M = 20
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
"""
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
        pos=simg["vertices"]["G"],
        radius=0.4,
        density=M * N,
    )
)

Mimg.append(
    phantom_source.mesh.sun(
        pos=simg["vertices"]["I"],
        num_flares=13,
        radius=1.2,
        density=M * N,
    )
)



Mscn = []
for mimg in Mimg:
    mscn = phantom_source.mesh.transform_image_to_scneney(mesh=mimg)
    Mscn.append(mscn)
"""
Mscn = [sscn]
"""

output_path = "tmp_plenoscope_response"
os.makedirs(name=output_path, exist_ok=True)

phantom_source.plot.save_3d_views_of_meshes(
    meshes=Mscn,
    scale=1e-3,
    xlim=[-XY_RADIUS*1e-3, XY_RADIUS*1e-3],
    ylim=[-XY_RADIUS*1e-3, XY_RADIUS*1e-3],
    zlim=[0*1e-3, MAX_OBJECT_DISTANCE*1e-3],
    elevations=[23, 90, 0],
    azimuths=[-60, 0, 0],
    paths=[os.path.join(output_path, "mesh_{:d}.jpg".format(i)) for i in range(3)]
)

light_fields = phantom_source.light_field.make_light_fields_from_meshes(
    meshes=Mscn,
    aperture_radius=aperture_radius,
    prng=prng,
    emission_distance_to_aperture=EMISSION_DISTANCE_TO_APERTURE,
)

event, light_field_geometry = phantom_source.merlict.make_plenopy_event_and_read_light_field_geometry(
    light_fields=light_fields,
    light_field_geometry_path=light_field_geometry_path,
    merlict_propagate_photons_path=merlict_path,
    merlict_propagate_config_path=merlict_config_path,
    random_seed=SEED,
)

lf_t, lf_lixel_ids = event.photon_arrival_times_and_lixel_ids()


pl.plot.refocus.save_refocus_stack(
    light_field_geometry=light_field_geometry,
    photon_lixel_ids=lf_lixel_ids,
    output_path=output_path,
    title=None,
    obj_dist_min=2000.0,
    obj_dist_max=MAX_OBJECT_DISTANCE,
    time_slices_window_radius=1,
    steps=16,
    use_absolute_scale=True,
    image_prefix="refocus_",
    figure_style={"rows": 1080, "cols": 1920, "fontsize": 1},
)
