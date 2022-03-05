import optic_object_wavefronts as oow
import os
import subprocess
import io
from scipy import spatial
import scipy


tmp_dir = "TMP"

os.makedirs(tmp_dir, exist_ok=True)

FN_POLYGON = 71
FN_HEX_GRID = 31

MIRROR_RADIUS = 0.5
FOCAL_LENGTH = 1.5
SCREEN_RADIUS = 0.15
SENSOR_ID = 1337

c = {}
c["mirror"] = {}
c["mirror"]["outer_radius"] = MIRROR_RADIUS
c["mirror"]["curvature_radius"] = 2.0 * FOCAL_LENGTH

c["screen"] = {}
c["screen"]["outer_radius"] = SCREEN_RADIUS
c["screen"]["pos"] = [0, 0, FOCAL_LENGTH]

c["shield"] = {}
c["shield"]["outer_radius"] = 1.1 * c["screen"]["outer_radius"]
c["shield"]["pos"] = [0, 0, FOCAL_LENGTH * 1.01]


# Scenery tele- / plenoscope
# --------------------------
scope_path = os.path.join(tmp_dir, "scope.tar")
plane_path = os.path.join(tmp_dir, "principal_aperture_plane.tar")


scenery = oow.Scenery.init()

scenery["objects"]["mirror"] = oow.objects.spherical_cap.init(
    outer_polygon=oow.geometry.regular_polygon.make_vertices_xy(
        outer_radius=c["mirror"]["outer_radius"],
        fn=FN_POLYGON,
        ref="mirror_outer",
    ),
    curvature_radius=c["mirror"]["curvature_radius"],
    ref="mirror",
    fn_hex_grid=FN_HEX_GRID,
)

scenery["objects"]["screen"] = oow.objects.disc.init(
    outer_radius=c["screen"]["outer_radius"],
    fn=FN_POLYGON,
    rot=0.0,
    ref="screen",
    prevent_many_faces_share_same_vertex=True,
)

scenery["objects"]["shield"] = oow.objects.disc.init(
    outer_radius=c["shield"]["outer_radius"],
    fn=FN_POLYGON,
    rot=0.0,
    ref="shield",
    prevent_many_faces_share_same_vertex=True,
)

scenery["materials"]["media"]["vacuum"] = oow.Scenery.EXAMPLE_MEDIA_VACUUM
scenery["materials"]["default_medium"] = "vacuum"
scenery["materials"]["surfaces"]["specular"] = {
    "material": "Phong",
    "specular_reflection": [[200e-9, 1.0], [1.2e-6, 1.0]],
    "diffuse_reflection": [[200e-9, 0.0], [1.2e-6, 0.0]],
    "color": [200, 200, 200]
}
scenery["materials"]["surfaces"]["dull"] = {
    "material": "Phong",
    "specular_reflection": [[200e-9, 0.0], [1.2e-6, 0.0]],
    "diffuse_reflection": [[200e-9, 0.0], [1.2e-6, 0.0]],
    "color": [25, 25, 25]
}
scenery["materials"]["boundary_layers"]["vacu_dull_spec_vacu"] = {
    "inner": {"medium": "vacuum", "surface": "dull"},
    "outer": {"medium": "vacuum", "surface": "specular"}
}
scenery["materials"]["boundary_layers"]["vacu_dull_dull_vacu"] = {
    "inner": {"medium": "vacuum", "surface": "dull"},
    "outer": {"medium": "vacuum", "surface": "dull"}
}

scenery["tree"]["children"].append(
    {
        "id": 1001,
        "pos": [0, 0, 0,],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "mirror",
        "mtl": {"mirror": "vacu_dull_spec_vacu"}
    }
)
scenery["tree"]["children"].append(
    {
        "id": SENSOR_ID,
        "pos": c["screen"]["pos"],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "screen",
        "mtl": {"screen": "vacu_dull_dull_vacu"}
    }
)
scenery["tree"]["children"].append(
    {
        "id": 1003,
        "pos": c["shield"]["pos"],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "shield",
        "mtl": {"shield": "vacu_dull_dull_vacu"}
    }
)

oow.Scenery.write_to_merlict(
    scenery=scenery,
    path=scope_path,
)


# Scenery principal aperture plane
# --------------------------------

pap_scenery = oow.Scenery.init()
pap_scenery["objects"]["principal_aperture_plane"] = oow.objects.disc.init(
    outer_radius=5.0 * c["mirror"]["outer_radius"],
    fn=FN_POLYGON,
    rot=0.0,
    ref="disc",
    prevent_many_faces_share_same_vertex=True,
)
pap_scenery["materials"]["media"]["vacuum"] = scenery[
    "materials"]["media"]["vacuum"]
pap_scenery["materials"]["default_medium"] = "vacuum"

pap_scenery["materials"]["surfaces"]["dull"] = scenery[
    "materials"]["surfaces"]["dull"]
pap_scenery["materials"]["boundary_layers"]["vacu_dull_dull_vacu"] = scenery[
    "materials"]["boundary_layers"]["vacu_dull_dull_vacu"]

pap_scenery["tree"]["children"].append(
    {
        "id": SENSOR_ID,
        "pos": [0, 0, 0,],
        "rot": {"repr": "tait_bryan", "xyz_deg": [0, 0, 0]},
        "obj": "principal_aperture_plane",
        "mtl": {"disc": "vacu_dull_dull_vacu"}
    }
)

oow.Scenery.write_to_merlict(
    scenery=pap_scenery,
    path=plane_path,
)

# compile merlict
# ---------------
lfg_exe_path = os.path.join(tmp_dir, "lfg")
if os.path.exists(lfg_exe_path):
    os.remove(lfg_exe_path)
subprocess.call([
    "gcc",
    "make_light_field_geometry.c",
    "-o",
    lfg_exe_path,
    "-lm"
])
assert os.path.exists(lfg_exe_path)


draw_exe_path = os.path.join(tmp_dir, "draw_photons_wrt_principal_aperture_plane")
if os.path.exists(draw_exe_path):
    os.remove(draw_exe_path)
subprocess.call([
    "gcc",
    "draw_photons_wrt_principal_aperture_plane.c",
    "-o",
    draw_exe_path,
    "-lm"
])
assert os.path.exists(draw_exe_path)


def draw_photons_wrt_principal_aperture_plane(
    draw_exe_path,
    seed,
    num_photons,
    aperture_radius,
    opening_angle,
    distance_to_aperture,
    rotation_tait_brian,
    wavelength,
    cxlim,
    cylim,
):
    command = [
        draw_exe_path,
        str(seed),
        str(num_photons),
        str(aperture_radius),
        str(opening_angle),
        str(distance_to_aperture),
        str(rotation_tait_brian[0]),
        str(rotation_tait_brian[1]),
        str(rotation_tait_brian[2]),
        str(cxlim[0]),
        str(cxlim[1]),
        str(cylim[0]),
        str(cylim[1]),
    ]
    #print(command)
    server = subprocess.Popen(
        command,
        stdout=subprocess.PIPE
    )
    arr = np.genfromtxt(server.stdout, delimiter=",")
    photons = np.zeros(shape=(num_photons, 7))
    photons[:, 0:6] = arr
    photons[:, 6] = wavelength
    return photons


def lfg_server_init(lfg_exe_path, scenery_path, seed, sensor_id):
    return subprocess.Popen(
        [lfg_exe_path, scenery_path, str(seed), str(sensor_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE
    )


def lfg_server_propagate(server, photons):
    assert photons.shape[1] == 7
    assert photons.dtype == np.float64

    csv_lines = io.BytesIO()
    ii = 0
    for ph in photons:
        ii += 1
        server.stdin.write(ph.tobytes())
        server.stdin.flush()
        rr = server.stdout.readline()
        csv_lines.write(rr)

    csv_lines.seek(0)
    arr = np.genfromtxt(csv_lines, delimiter=",")
    return arr


def lfg_server_end(server):
    zeros = np.zeros(7, dtype=np.float64)
    server.stdin.write(zeros.tobytes())
    server.stdin.flush()


# populate light_field_geometry
# =============================

llffgg = {}
directional_binning = {}

PIXEL_OPENING_ANGLE_DIAGONAL_DEG = 0.1
PIXEL_EXTENT_DIAGONAL = c["screen"]["pos"][2] * np.tan(
    np.deg2rad(PIXEL_OPENING_ANGLE_DIAGONAL_DEG)
)
NUM_PIXEL_RADIUS = int(c["screen"]["outer_radius"] / PIXEL_EXTENT_DIAGONAL)
NUM_PAXEL_RADIUS = 2


def make_hexagonal_grids_xy(
    outer_radius,
    vertices_on_radius,
):
    grid = oow.geometry.hexagonal_grid.make_vertices_xy(
        outer_radius=outer_radius,
        fn=vertices_on_radius
    )
    out = []
    for key in grid:
        x = grid[key][0]
        y = grid[key][1]
        out.append([x, y])
    return np.array(out)


def limit_grid_xy(grid, xlim, ylim):
    out = []
    for c in grid:
        x = c[0]
        y = c[1]
        if xlim[0] <= x < xlim[1] and ylim[0] <= y < ylim[1]:
            out.append(c)
    return np.array(out)


def light_field_geometry_init(
    num_pixel_on_radius,
    num_paxel_on_radius,
    aperture_outer_radius,
    screen_outer_radius,
    focal_length,
    screen_xlim_relative=[-0.05, 1.0],
    screen_ylim_relative=[-0.05, 0.05],
    raw_max_samples=21,
):
    assert focal_length > 0
    assert aperture_outer_radius > 0
    assert screen_outer_radius > 0
    assert num_pixel_on_radius >= 0
    assert num_paxel_on_radius >= 0

    sx = screen_xlim_relative
    sy = screen_ylim_relative

    lfg = {}

    pixel_grid = make_hexagonal_grids_xy(
        outer_radius=screen_outer_radius,
        vertices_on_radius=num_pixel_on_radius,
    )
    lfg["screen_xlim"] = [sx[0] * screen_outer_radius, sx[1] * screen_outer_radius]
    lfg["screen_ylim"] = [sy[0] * screen_outer_radius, sy[1] * screen_outer_radius]
    assert lfg["screen_xlim"][0] < lfg["screen_xlim"][1]
    assert lfg["screen_ylim"][0] < lfg["screen_ylim"][1]

    pixel_grid = limit_grid_xy(
        grid=pixel_grid,
        xlim=lfg["screen_xlim"],
        ylim=lfg["screen_ylim"],
    )

    paxel_grid = make_hexagonal_grids_xy(
        outer_radius=aperture_outer_radius,
        vertices_on_radius=num_paxel_on_radius,
    )

    lfg["num_paxel_on_radius"] = num_paxel_on_radius
    lfg["num_pixel_on_radius"] = num_pixel_on_radius
    lfg["aperture_outer_radius"] = aperture_outer_radius
    lfg["screen_outer_radius"] = screen_outer_radius
    lfg["focal_length"] = focal_length
    lfg["opening_angle"] = np.arctan2(
        lfg["screen_outer_radius"],
        lfg["focal_length"]
    )
    open_margin = 0.1 * lfg["opening_angle"]
    lfg["opening_angle_cxlim"] = [
        -np.arctan2(
            lfg["screen_xlim"][1],
            lfg["focal_length"]
        ) - open_margin,
        -np.arctan2(
            lfg["screen_xlim"][0],
            lfg["focal_length"]
        ) + open_margin
    ]
    assert lfg["opening_angle_cxlim"][0] < lfg["opening_angle_cxlim"][1]

    lfg["opening_angle_cylim"] = [
        -np.arctan2(
            lfg["screen_ylim"][1],
            lfg["focal_length"]
        ) - open_margin,
        -np.arctan2(
            lfg["screen_ylim"][0],
            lfg["focal_length"]
        ) + open_margin
    ]
    assert lfg["opening_angle_cylim"][0] < lfg["opening_angle_cylim"][1]


    lfg["pixel_extent_radius"] = screen_outer_radius / num_pixel_on_radius
    lfg["pixel_tree"] = scipy.spatial.cKDTree(data=pixel_grid)
    lfg["num_pixel"] = lfg["pixel_tree"].data.shape[0]

    lfg["paxel_extent_radius"] = aperture_outer_radius / num_paxel_on_radius
    lfg["paxel_tree"] = scipy.spatial.cKDTree(data=paxel_grid)
    lfg["num_paxel"] = lfg["paxel_tree"].data.shape[0]

    lfg["num_lixel"] = (lfg["num_pixel"] * lfg["num_paxel"])

    lfg["raw_max_samples"] = raw_max_samples
    lfg["raw_cx"] = [[] for i in range(lfg["num_lixel"])]
    lfg["raw_cy"] = [[] for i in range(lfg["num_lixel"])]
    lfg["raw_x"] = [[] for i in range(lfg["num_lixel"])]
    lfg["raw_y"] = [[] for i in range(lfg["num_lixel"])]
    lfg["raw_t"] = [[] for i in range(lfg["num_lixel"])]
    return lfg


def light_field_geometry_add_photons(
    light_field_geometry,
    num_photons,
    scope_server,
    plane_server,
    draw_exe_path,
    seed,
):
    photons = draw_photons_wrt_principal_aperture_plane(
        draw_exe_path=draw_exe_path,
        seed=seed,
        num_photons=num_photons,
        aperture_radius=light_field_geometry["aperture_outer_radius"],
        opening_angle=lfg["opening_angle"],
        distance_to_aperture=10.0 * lfg["focal_length"],
        rotation_tait_brian=[0.0, 0.0, 0.0],
        cxlim=lfg["opening_angle_cxlim"],
        cylim=lfg["opening_angle_cylim"],
        wavelength=433e-9,
    )
    photon_directions = photons[:, 3:6]
    max_open_deg = np.rad2deg(np.max(photon_directions[:, 0]))
    min_open_deg = np.rad2deg(np.min(photon_directions[:, 0]))

    #print("photons", np.min(photons[:, 0]), np.max(photons[:, 0]), min_open_deg, max_open_deg, "deg")

    intersections_scope = lfg_server_propagate(scope_server, photons)
    intersections_plane = lfg_server_propagate(plane_server, photons)

    #plt.plot(intersections_scope[:, 0], intersections_scope[:, 1], "x")


    return light_field_geometry_add_photons_and_intersections(
        light_field_geometry=light_field_geometry,
        photon_directions=photon_directions,
        intersections_plane=intersections_plane,
        intersections_scope=intersections_scope,
    )


def light_field_geometry_add_photons_and_intersections(
    light_field_geometry,
    photon_directions,
    intersections_plane,
    intersections_scope,
):
    lfg = light_field_geometry
    assert intersections_plane.shape == intersections_scope.shape
    num_photons = intersections_plane.shape[0]

    # find intersection on aperture (paxel)
    for ph in range(num_photons):
        #print("ph", ph)

        (paxdist, paxelid) = lfg["paxel_tree"].query(
            x=intersections_plane[ph, 0:2],
            k=1,
            distance_upper_bound=lfg["paxel_extent_radius"],
        )
        #print("pax", paxelid, paxdist)
        if paxdist > lfg["paxel_extent_radius"]:
            continue

        (pixdist, pixelid) = lfg["pixel_tree"].query(
            x=intersections_scope[ph, 0:2],
            k=1,
            distance_upper_bound=lfg["pixel_extent_radius"],
        )
        #print("pix", pixelid, pixdist)
        if pixdist > lfg["pixel_extent_radius"]:
            continue

        lixelid = (paxelid * lfg["num_pixel"]) + pixelid
        #print("lix", lixelid)

        if len(lfg["raw_cx"][lixelid]) >= lfg["raw_max_samples"]:
            continue

        """
        print(
            photon_directions[ph, 0],
            photon_directions[ph, 1],
            intersections_plane[ph, 0],
            intersections_plane[ph, 1],
            intersections_plane[ph, 2])
        """

        lfg["raw_cx"][lixelid].append(photon_directions[ph, 0])
        lfg["raw_cy"][lixelid].append(photon_directions[ph, 1])
        lfg["raw_x"][lixelid].append(intersections_plane[ph, 0])
        lfg["raw_y"][lixelid].append(intersections_plane[ph, 1])
        lfg["raw_t"][lixelid].append(intersections_plane[ph, 2])

    return lfg


def light_field_geometry_raw_statistics_fill_status(light_field_geometry):
    lfg = light_field_geometry
    ratios = np.zeros(lfg["num_lixel"])

    for lix in range(lfg["num_lixel"]):
        ratios[lix] = len(lfg["raw_cx"][lix]) / lfg["raw_max_samples"]

    return np.median(ratios), np.mean(ratios)


def light_field_geometry_condense_statistics(light_field_geometry):
    lfg = light_field_geometry

    for key in ["cx", "cy", "x", "y", "t", "cx_std", "cy_std", "x_std", "y_std", "t_std"]:
        lfg[key] = np.zeros(lfg["num_lixel"])

    for lix in range(lfg["num_lixel"]):
        lfg["cx"][lix] = np.median(lfg["raw_cx"][lix])
        lfg["cy"][lix] = np.median(lfg["raw_cy"][lix])
        lfg["x"][lix] = np.median(lfg["raw_x"][lix])
        lfg["y"][lix] = np.median(lfg["raw_y"][lix])
        lfg["t"][lix] = np.median(lfg["raw_t"][lix])

        lfg["cx_std"][lix] = np.std(lfg["raw_cx"][lix])
        lfg["cy_std"][lix] = np.std(lfg["raw_cy"][lix])
        lfg["x_std"][lix] = np.std(lfg["raw_x"][lix])
        lfg["y_std"][lix] = np.std(lfg["raw_y"][lix])
        lfg["t_std"][lix] = np.std(lfg["raw_t"][lix])

    return lfg


SEED = 1

scope_server = lfg_server_init(
    lfg_exe_path=lfg_exe_path,
    scenery_path=scope_path,
    seed=SEED,
    sensor_id=SENSOR_ID,
)
plane_server = lfg_server_init(
    lfg_exe_path=lfg_exe_path,
    scenery_path=plane_path,
    seed=SEED,
    sensor_id=SENSOR_ID,
)

"""
photons = draw_photons_wrt_principal_aperture_plane(
    draw_exe_path=draw_exe_path,
    seed=SEED,
    num_photons=100,
    aperture_radius=0.5,
    opening_angle=np.deg2rad(3.0),
    distance_to_aperture=10.0,
    rotation_tait_brian=[np.deg2rad(3),0,0],
    wavelength=433e-9,
)

photons = np.ones(shape=(100, 7), dtype=np.float64)
photons[:, 6] = 433e-9

photons[:, 0] = 0.2
photons[:, 1] = 0.2
photons[:, 2] = 10.0

photons[:, 3] = 0.0
photons[:, 4] = 0.0
photons[:, 5] = -1.0

res = lfg_server_propagate(scope_server, photons)
res2 = lfg_server_propagate(scope_server, photons)
"""

lfg = light_field_geometry_init(
    num_pixel_on_radius=NUM_PIXEL_RADIUS,
    num_paxel_on_radius=NUM_PAXEL_RADIUS,
    aperture_outer_radius=c["mirror"]["outer_radius"],
    screen_outer_radius=c["screen"]["outer_radius"],
    focal_length=c["screen"]["pos"][2],
)

while True:
    SEED += 1
    lfg = light_field_geometry_add_photons(
        light_field_geometry=lfg,
        num_photons=20000,
        scope_server=scope_server,
        plane_server=plane_server,
        draw_exe_path=draw_exe_path,
        seed=SEED,
    )
    fill = light_field_geometry_raw_statistics_fill_status(
        light_field_geometry=lfg
    )
    print("fill", fill)
    if fill[0] > 0.5:
        break

lfg = light_field_geometry_condense_statistics(light_field_geometry=lfg)

lfg_server_end(scope_server)
lfg_server_end(plane_server)

print("done")
