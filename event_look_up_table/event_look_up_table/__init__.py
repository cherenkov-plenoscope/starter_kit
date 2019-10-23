import numpy as np
import os
import gzip
import plenopy as pl
import scipy
import acp_instrument_response_function as irf
from os import path as op
import simpleio
import corsika_wrapper as cw
import shutil
import tempfile
import subprocess
import glob
import array
import json
import cable_robo_mount as crm


def _merlict_simpleio(
    merlict_eventio_converter_path,
    evtio_run_path,
    output_path
):
    op = output_path
    with open(op+'.stdout', 'w') as out, open(op+'.stderr', 'w') as err:
        call = [
            merlict_eventio_converter_path,
            '-i', evtio_run_path,
            '-o', output_path]
        mct_rc = subprocess.call(call, stdout=out, stderr=err)
    return mct_rc


def _estimate_shower_maximum_altitude(photon_emission_altitudes):
    return np.median(photon_emission_altitudes)


example_aperture_binning_config = {
    "bin_edge_width": 64,
    "num_bins_radius": 16,
}


def make_aperture_binning(aperture_binning_config):
    t = aperture_binning_config.copy()
    t["num_bins_diameter"] = 2*t["num_bins_radius"]
    t["xy_bin_edges"] = np.linspace(
        -1 * t["bin_edge_width"]*t["num_bins_radius"],
        +1 * t["bin_edge_width"]*t["num_bins_radius"],
        t["num_bins_diameter"] + 1)
    t["xy_bin_centers"] = (t["xy_bin_edges"][0:-1] + t["xy_bin_edges"][1:])/2
    t["num_bins"] = t["num_bins_diameter"]*t["num_bins_diameter"]
    t["addressing_2D_to_1D"] = np.arange(t["num_bins"]).reshape(
        (t["num_bins_diameter"], t["num_bins_diameter"]),
        order="C")
    t["addressing_1D_to_2D"] = np.zeros(
        shape=(t["num_bins"], 2),
        dtype=np.int64)
    for b in range(t["num_bins"]):
        t["addressing_1D_to_2D"][b, 0] = b//t["num_bins_diameter"]
        t["addressing_1D_to_2D"][b, 1] = b % t["num_bins_diameter"]

    t["bin_centers"] = []
    for b in range(t["num_bins"]):
        ix = t["addressing_1D_to_2D"][b, 0]
        iy = t["addressing_1D_to_2D"][b, 1]
        center = [
            (t["xy_bin_edges"][ix] + t["xy_bin_edges"][ix + 1])/2,
            (t["xy_bin_edges"][iy] + t["xy_bin_edges"][iy + 1])/2]
        t["bin_centers"].append(center)
    t["bin_centers"] = np.array(t["bin_centers"])

    t["bin_centers_tree"] = scipy.spatial.cKDTree(t["bin_centers"])
    return t

LIGHT_FIELD_DTYPE = [
    ('x', np.float32),
    ('y', np.float32),
    ('cx', np.float32),
    ('cy', np.float32)]
PHOTON_DTYPE = [
    ('x', np.uint8),
    ('y', np.uint8),
    ('cx', np.uint16),
    ('cy', np.uint16)]
PHOTON_SIZE_IN_BYTE = int(
    np.sum([np.iinfo(t[1]).bits//8 for t in PHOTON_DTYPE]))
C_BIN_MAX = np.iinfo(PHOTON_DTYPE[2][1]).max
C_BIN_MAX_HALF = (C_BIN_MAX+1)//2

XY_BIN_MAX = np.iinfo(PHOTON_DTYPE[0][1]).max
XY_BIN_OVER = XY_BIN_MAX + 1
YX_BIN_OVER_HALF = (XY_BIN_OVER)//2


def compress_x_y(x, y, aperture_binning_config):
    bin_edge_width = aperture_binning_config["bin_edge_width"]
    num_bins_radius = aperture_binning_config["num_bins_radius"]
    ab = aperture_binning_config
    x_bin_idx = (x//bin_edge_width).astype(np.int) + num_bins_radius
    y_bin_idx = (y//bin_edge_width).astype(np.int) + num_bins_radius

    x_rest = x - (x_bin_idx - num_bins_radius)*bin_edge_width
    y_rest = y - (y_bin_idx - num_bins_radius)*bin_edge_width

    x_rest_bin = np.floor(x_rest/bin_edge_width*(XY_BIN_OVER)).astype(np.int64)
    y_rest_bin = np.floor(y_rest/bin_edge_width*(XY_BIN_OVER)).astype(np.int64)
    return x_bin_idx, y_bin_idx, x_rest_bin, y_rest_bin


def decompress_x_y(
    x_bin_idx,
    y_bin_idx,
    x_rest_bin,
    y_rest_bin,
    aperture_binning_config
):
    ab = aperture_binning_config
    x_bin = (x_bin_idx - ab["num_bins_radius"])*ab["bin_edge_width"]
    y_bin = (y_bin_idx - ab["num_bins_radius"])*ab["bin_edge_width"]
    x_fine = x_rest_bin/XY_BIN_OVER*ab["bin_edge_width"]
    y_fine = y_rest_bin/XY_BIN_OVER*ab["bin_edge_width"]
    return x_bin + x_fine, y_bin + y_fine


def compress_cx_cy(cx, cy, field_of_view_radius):
    field_of_view_diameter = 2.*field_of_view_radius
    c_bin_width = field_of_view_diameter/C_BIN_MAX
    field_of_view_radius_in_bins = c_bin_width*C_BIN_MAX_HALF
    cx_bin = (cx + field_of_view_radius_in_bins)//c_bin_width
    cy_bin = (cy + field_of_view_radius_in_bins)//c_bin_width
    return cx_bin.astype(np.int64), cy_bin.astype(np.int64)


def decompress_cx_cy(cx_bin, cy_bin, field_of_view_radius):
    field_of_view_diameter = 2.*field_of_view_radius
    c_bin_width = field_of_view_diameter/C_BIN_MAX
    field_of_view_radius_in_bins = c_bin_width*C_BIN_MAX_HALF
    cx = (cx_bin*c_bin_width) - field_of_view_radius_in_bins
    cy = (cy_bin*c_bin_width) - field_of_view_radius_in_bins
    return cx, cy


def compress_photons(
    x,
    y,
    cx,
    cy,
    aperture_binning_config,
    field_of_view_radius
):
    num_photons = x.shape[0]
    valid_photons = np.zeros(num_photons, dtype=np.bool)
    abc = aperture_binning_config
    fov_r = field_of_view_radius
    compressed_photons = []
    for b in range(abc["num_bins"]):
        compressed_photons.append({
            "xrests": array.array("B"),
            "yrests": array.array("B"),
            "cxbins": array.array("H"),
            "cybins": array.array("H")})

    xbins, ybins, xrests, yrests = compress_x_y(
        x=x,
        y=y,
        aperture_binning_config=abc)
    cxbins, cybins = compress_cx_cy(
        cx=cx,
        cy=cy,
        field_of_view_radius=fov_r)

    for ph in range(num_photons):
        if xbins[ph] < 0 or xbins[ph] >= abc["num_bins_diameter"]:
            continue
        if ybins[ph] < 0 or ybins[ph] >= abc["num_bins_diameter"]:
            continue
        if cxbins[ph] < 0 or cxbins[ph] > (C_BIN_MAX-1):
            continue
        if cybins[ph] < 0 or cybins[ph] > (C_BIN_MAX-1):
            continue
        valid_photons[ph] = True

        assert xrests[ph] >= 0
        assert xrests[ph] < XY_BIN_OVER
        assert yrests[ph] >= 0
        assert yrests[ph] < XY_BIN_OVER

        xy_bin = abc["addressing_2D_to_1D"][xbins[ph], ybins[ph]]
        compressed_photons[xy_bin]["xrests"].append(xrests[ph])
        compressed_photons[xy_bin]["yrests"].append(yrests[ph])
        compressed_photons[xy_bin]["cxbins"].append(cxbins[ph])
        compressed_photons[xy_bin]["cybins"].append(cybins[ph])

    for b in range(abc["num_bins"]):
        num_photons = len(compressed_photons[b]["xrests"])
        rec = np.recarray(
            shape=[num_photons],
            dtype=PHOTON_DTYPE)
        rec.x = compressed_photons[b]["xrests"]
        rec.y = compressed_photons[b]["yrests"]
        rec.cx = compressed_photons[b]["cxbins"]
        rec.cy = compressed_photons[b]["cybins"]
        compressed_photons[b] = rec

    return compressed_photons, valid_photons


APERTURE_BIN_FILENAME = "{:06d}.x_y_cx_cy"


def append_compressed_photons(path, compressed_photons):
    for xy_bin in range(len(compressed_photons)):
        xy_bin_path = os.path.join(path, APERTURE_BIN_FILENAME.format(xy_bin))
        with open(xy_bin_path, "ab") as fout:
            fout.write(compressed_photons[xy_bin].flatten(order="C").tobytes())


def _estimate_xy_bins_overlapping_with_circle(
    aperture_binning_config,
    circle_x,
    circle_y,
    circle_radius,
):
    abc = aperture_binning_config
    bin_half_edge_width = abc["bin_edge_width"]*.5
    bin_corner_radius = np.sqrt(2.)*bin_half_edge_width
    return abc["bin_centers_tree"].query_ball_point(
        x=np.array([circle_x, circle_y]),
        r=np.array([circle_radius + bin_corner_radius]),
        p=2,
        eps=0.)


def read_photons(
    path,
    aperture_binning_config,
    circle_x,
    circle_y,
    circle_radius,
    field_of_view_radius,
):
    abc = aperture_binning_config
    fov_r = field_of_view_radius
    bins_to_load = _estimate_xy_bins_overlapping_with_circle(
        abc,
        circle_x,
        circle_y,
        circle_radius)
    photon_blocks = []
    for bin_to_load in bins_to_load:
        xy_bin_path = os.path.join(
            path,
            APERTURE_BIN_FILENAME.format(bin_to_load))
        x_bin_idx = abc["addressing_1D_to_2D"][bin_to_load, 0]
        y_bin_idx = abc["addressing_1D_to_2D"][bin_to_load, 1]
        with open(xy_bin_path, "rb") as fin:
            compressed_photons = np.rec.array(
                np.frombuffer(fin.read(), dtype=PHOTON_DTYPE))
        num_photons_in_block = compressed_photons.shape[0]
        xs, ys = decompress_x_y(
            x_bin_idx=np.ones(num_photons_in_block, dtype=np.int)*x_bin_idx,
            y_bin_idx=np.ones(num_photons_in_block, dtype=np.int)*y_bin_idx,
            x_rest_bin=compressed_photons.x,
            y_rest_bin=compressed_photons.y,
            aperture_binning_config=abc)
        cxs, cys = decompress_cx_cy(
            cx_bin=compressed_photons.cx,
            cy_bin=compressed_photons.cy,
            field_of_view_radius=fov_r)
        photon_blocks.append(np.array([xs, ys, cxs, cys]))
    photons = np.concatenate(photon_blocks, axis=1).T
    off_x2 = (photons[:, 0] - circle_x)**2
    off_y2 = (photons[:, 1] - circle_y)**2
    valid = off_x2 + off_y2 < circle_radius**2
    lf = np.recarray(np.sum(valid), dtype=LIGHT_FIELD_DTYPE)
    lf.x = photons[valid, 0]
    lf.y = photons[valid, 1]
    lf.cx = photons[valid, 2]
    lf.cy = photons[valid, 3]
    return lf


def _print_image(image, v_max=None):
    num_cols = image.shape[0]
    num_rows = image.shape[1]
    if v_max is None:
        v_max = np.max(image)
    s = ""
    for row in range(num_rows):
        for col in range(num_cols):
            intensity = (image[col, row]/v_max)*255
            i_out = int(np.min([intensity, 255]))
            r = i_out
            g = i_out
            b = i_out
            s += "\033[48;2;{:d};{:d};{:d}m  \033[0m".format(r, g, b)
        s += "\n"
    print(s)


def _print_photons_cx_cy(
    photons_x_y_cx_cy,
    c_bin_edges=np.deg2rad(np.linspace(-4, 4, 97)),
    v_max=None
):
    image = np.histogram2d(
        photons_x_y_cx_cy[:, 2],
        photons_x_y_cx_cy[:, 3],
        bins=c_bin_edges)[0]
    if v_max is None:
        v_max = np.max(image)
    _print_image(image, v_max)


class Reader:
    def __init__(self, path):
        self.lookup_path = path
        with open(os.path.join(self.lookup_path, "lookup.json"), "rt") as f:
            config = json.loads(f.read())
        self.aperture_binning = make_aperture_binning(config["aperture"])
        self.altitude_bin_edges = np.array(config["altitude_bin_edges"])
        self.max_num_photons_in_bin = config["max_num_photons_in_bin"]
        self.field_of_view_radius = config["field_of_view"]["radius"]
        self.energy_bin_centers = config["energy_bin_centers"]
        self.__config = config

        self.__energy_altitude_paths = []
        for e in range(len(self.energy_bin_centers)):
            altitude_paths = []
            for a in range(len(self.altitude_bin_edges) - 1):
                p = os.path.join(
                    self.lookup_path,
                    "{:06d}_energy".format(e),
                    "{:06d}_altitude".format(a))
                altitude_paths.append(p)
            self.__energy_altitude_paths.append(altitude_paths)

        self.num_showers = []
        self.num_photons = []
        for e in range(len(self.energy_bin_centers)):
            p = os.path.join(
                self.lookup_path,
                "{:06d}_energy".format(e),
                "fill.json")
            if os.path.exists(p):
                with open(p, "rt") as fin:
                    energy_bin_fill = json.loads(fin.read())
                self.num_showers.append(energy_bin_fill["num_shower"])
                self.num_photons.append(energy_bin_fill["num_photons"])
            else:
                self.num_showers.append([])
                self.num_photons.append([])

    def read_light_field(
        self,
        energy_bin,
        altitude_bin,
        core_x,
        core_y,
        instrument_radius
    ):
        return read_photons(
            path=self.__energy_altitude_paths[energy_bin][altitude_bin],
            aperture_binning_config=self.aperture_binning,
            circle_x=core_x,
            circle_y=core_y,
            circle_radius=instrument_radius,
            field_of_view_radius=self.field_of_view_radius)


class Instrument:
    def __init__(self, path, light_field_geometry):
        self.reader = Reader(path)
        self.light_field_geometry = light_field_geometry
        self.instrument_radius = self.light_field_geometry. \
            sensor_plane2imaging_system. \
            expected_imaging_system_max_aperture_radius
        aperture_area = self.light_field_geometry. \
            sensor_plane2imaging_system. \
            expected_imaging_system_max_aperture_radius**2*np.pi
        paxel_area = aperture_area/self.light_field_geometry.number_paxel
        self.paxel_radius = np.sqrt(paxel_area/np.pi)
        self.pixel_radius = self.light_field_geometry. \
            sensor_plane2imaging_system.pixel_FoV_hex_flat2flat/2

    def response(
        self,
        energy_bin,
        altitude_bin,
        source_cx,
        source_cy,
        core_x,
        core_y
    ):
        projected_instrument_radius = (
            self.instrument_radius*
            _collection_sphere_scaling(
                source_cx_in_instrument_frame=source_cx,
                source_cy_in_instrument_frame=source_cy,))

        light_field_in_shower_frame = self.reader.read_light_field(
            energy_bin=energy_bin,
            altitude_bin=altitude_bin,
            core_x=core_x,
            core_y=core_y,
            instrument_radius=projected_instrument_radius)

        light_field = _transform_light_field_to_instrument_frame(
            light_field_in_shower_frame=light_field_in_shower_frame,
            source_cx_in_instrument_frame=source_cx,
            source_cy_in_instrument_frame=source_cy,
            shower_core_x_in_instrument_frame=core_x,
            shower_core_y_in_instrument_frame=core_y)


        num_shower = self.reader.num_showers[energy_bin][altitude_bin]

        paxel_dist, paxel_idx = self.light_field_geometry.paxel_pos_tree.query(
            np.c_[light_field.x, light_field.y],
            k=1)
        paxel_valid = paxel_dist <= self.paxel_radius
        pixel_dist, pixel_idx = self.light_field_geometry.pixel_pos_tree.query(
            np.c_[light_field.cx, light_field.cy],
            k=1)
        pixel_valid = pixel_dist <= self.pixel_radius
        valid = np.logical_and(paxel_valid, pixel_valid)

        valid_paxel_idx = paxel_idx[valid]
        valid_pixel_idx = pixel_idx[valid]

        valid_lixel_idx = (
            valid_pixel_idx*self.light_field_geometry.number_paxel +
            valid_paxel_idx)

        respo = np.zeros(self.light_field_geometry.number_lixel)
        for lix in valid_lixel_idx:
            respo[lix] += 1
        respo = respo * self.light_field_geometry.efficiency
        respo = respo / num_shower
        return respo


def _to_photon_lixel_ids(response):
    return np.concatenate(
        [i*np.ones(int(np.round(c*1000))) for i, c in enumerate(response)]
    ).astype(np.uint64)


def _collection_sphere_scaling(
    source_cx_in_instrument_frame,
    source_cy_in_instrument_frame
):
    approx_rotation_angle = np.hypot(
        source_cx_in_instrument_frame,
        source_cy_in_instrument_frame)
    return 1.0/np.cos(approx_rotation_angle)


def _transform_light_field_to_instrument_frame(
    light_field_in_shower_frame,
    source_cx_in_instrument_frame,
    source_cy_in_instrument_frame,
    shower_core_x_in_instrument_frame,
    shower_core_y_in_instrument_frame
):
    lf_SF = light_field_in_shower_frame
    num_photons = lf_SF.x.shape[0]

    # represent with 3D vecors
    # ------------------------
    support_SF = np.array([lf_SF.x, lf_SF.y, np.zeros(num_photons)])
    _dir_z = np.sqrt(1. - lf_SF.cx**2.0 - lf_SF.cy**2.0)
    direction_SF = np.array([lf_SF.cx, lf_SF.cy, _dir_z])

    # rotate
    # ------
    # The directional cosines x is the rotation-angles for axis Y,
    # and vice-versa.
    T_shower2instrument = crm.HomTra()
    T_shower2instrument.set_translation(pos=np.array([0., 0., 0.]))
    T_shower2instrument.set_rotation_tait_bryan_angles(
        Rx=-source_cy_in_instrument_frame,
        Ry=-source_cx_in_instrument_frame,
        Rz=0.)
    support_IF = T_shower2instrument.transformed_position(support_SF)
    direction_IF = T_shower2instrument.transformed_orientation(direction_SF)

    # represent support on obs.-level. z=0
    # ------------------------------------
    #  0 = sz + alpha*dz
    #  alpha*dz = -sz
    #  alpha = -sz/dz
    alpha = -support_IF[2, :]/direction_IF[2, :]
    support_zZero_IF = support_IF + alpha*direction_IF

    # translate
    # ---------
    lf_IF = np.recarray(shape=num_photons, dtype=LIGHT_FIELD_DTYPE)
    lf_IF.x = support_zZero_IF[0, :] + shower_core_x_in_instrument_frame
    lf_IF.y = support_zZero_IF[1, :] + shower_core_y_in_instrument_frame
    lf_IF.cx = direction_IF[0, :]
    lf_IF.cy = direction_IF[1, :]
    return lf_IF


def init_lookup(
    lookup_path,
    particle_config_path=op.join(
        "resources", "acp", "71m", "gamma_steering.json"),
    location_config_path=op.join(
        "resources", "acp", "71m", "chile_paranal.json"),
    aperture_binning_config={
        "bin_edge_width": 64,
        "num_bins_radius": 12},
    field_of_view_radius=np.deg2rad(10),
    altitude_bin_edges=np.linspace(5e3, 25e3, 21),
    energy_bin_centers=np.geomspace(0.25, 250, 16),
    max_num_photons_in_bin=1000*1000,
):
    os.makedirs(lookup_path)
    shutil.copy(
        particle_config_path,
        os.path.join(lookup_path, "particle_config.json"))
    shutil.copy(
        location_config_path,
        os.path.join(lookup_path, "location_config.json"))
    config = {}
    config["aperture"] = aperture_binning_config
    config["field_of_view"] = {"radius": field_of_view_radius}
    config["altitude_bin_edges"] = altitude_bin_edges.tolist()
    config["energy_bin_centers"] = energy_bin_centers.tolist()
    config["max_num_photons_in_bin"] = int(max_num_photons_in_bin)
    with open(os.path.join(lookup_path, "lookup.json"), "wt") as f:
        f.write(json.dumps(config, indent=4))


def _num_photons_in_bin(path, aperture_binning_config):
    total_size_in_byte = 0
    for idx in range(aperture_binning_config["num_bins"]):
        bin_path = os.path.join(path, APERTURE_BIN_FILENAME.format(idx))
        total_size_in_byte += os.stat(bin_path).st_size
    num_photons = total_size_in_byte/PHOTON_SIZE_IN_BYTE
    return num_photons


def _add_energy_to_lookup(
    lookup_path,
    energy_bin_center=0,
    num_showers_in_run=128,
    merlict_eventio_converter_path=op.join(
        'build',
        'merlict',
        'merlict-eventio-converter'),
    corsika_path=op.join(
        "build",
        "corsika",
        "corsika-75600",
        "run",
        "corsika75600Linux_QGSII_urqmd"),
):
    config = irf.__read_json(os.path.join(lookup_path, "lookup.json"))
    abc = make_aperture_binning(config["aperture"])
    altitude_bin_edges = np.array(config["altitude_bin_edges"])
    max_num_photons_in_bin = config["max_num_photons_in_bin"]
    cherenkov_collection_radius = float(np.max(abc["xy_bin_edges"]))
    fov_r = config["field_of_view"]["radius"]

    energy = config["energy_bin_centers"][energy_bin_center]
    energy_path = os.path.join(
        lookup_path,
        "{:06d}_energy".format(energy_bin_center))
    os.makedirs(energy_path)

    location_config = irf.__read_json(
        os.path.join(lookup_path, "location_config.json"))
    particle_config = irf.__read_json(
        os.path.join(lookup_path, "particle_config.json"))

    corsika_card_template = {}
    cct = corsika_card_template

    cct["num_events"] = num_showers_in_run
    cct["particle_id"] = irf.__particle_str_to_corsika_id(
            particle_config['primary_particle'])
    cct["energy_start"] = energy
    cct["energy_stop"] = energy
    cct["cone_zenith_deg"] = 0.
    cct["cone_azimuth_deg"] = 0.
    cct["cone_max_scatter_angle_deg"] = 0.
    cct['core_max_scatter_radius'] = 0.
    cct['instrument_x'] = 0.
    cct['instrument_y'] = 0.
    cct['instrument_radius'] = cherenkov_collection_radius
    cct['observation_level_altitude_asl'] = location_config[
        'observation_level_altitude_asl']
    cct['earth_magnetic_field_x_muT'] = location_config[
        'earth_magnetic_field_x_muT']
    cct['earth_magnetic_field_z_muT'] = location_config[
        'earth_magnetic_field_z_muT']
    cct['atmosphere_id'] = irf.__atmosphere_str_to_corsika_id(
        location_config["atmosphere"])

    num_altitude_bins = altitude_bin_edges.shape[0] - 1
    num_photons_in_altitude_bins = np.zeros(num_altitude_bins, dtype=np.int64)
    num_showers_in_altitude_bins = np.zeros(num_altitude_bins, dtype=np.int64)

    altitude_paths = {}
    for idx, altitude in enumerate(altitude_bin_edges[: -1]):
        altitude_bin_path = op.join(
            energy_path,
            "{:06d}_altitude".format(idx))
        altitude_paths[idx] = altitude_bin_path
        os.makedirs(altitude_bin_path)

    run_id = 1
    while np.sum(
        num_photons_in_altitude_bins < max_num_photons_in_bin
    ) > 0.5*num_altitude_bins:
        print(num_showers_in_altitude_bins)
        run = cct.copy()
        run["run_id"] = run_id

        with tempfile.TemporaryDirectory(prefix='plenoscope_lookup_') as tmp:
            corsika_card_path = op.join(tmp, 'corsika_card.txt')
            corsika_run_path = op.join(tmp, 'cherenkov_photons.evtio')
            simple_run_path = op.join(tmp, 'cherenkov_photons.simpleio')

            with open(corsika_card_path, "wt") as fout:
                card_str = irf.__make_corsika_steering_card_str(run=run)
                fout.write(card_str)

            cor_rc = cw.corsika(
                steering_card=cw.read_steering_card(corsika_card_path),
                output_path=corsika_run_path,
                save_stdout=True,
                corsika_path=corsika_path)

            mct_rc = _merlict_simpleio(
                merlict_eventio_converter_path=merlict_eventio_converter_path,
                evtio_run_path=corsika_run_path,
                output_path=simple_run_path)
            os.remove(corsika_run_path)

            events = simpleio.SimpleIoRun(simple_run_path)

            for event in events:
                num_bunches = event.cherenkov_photon_bunches.x.shape[0]
                if num_bunches == 0:
                    continue
                shower_maximum_altitude = _estimate_shower_maximum_altitude(
                    event.cherenkov_photon_bunches.emission_height)

                upper_altitude_bin_edge = int(np.digitize(
                    [shower_maximum_altitude],
                    altitude_bin_edges)[0])

                print(
                    "upper_altitude_bin_edge",
                    upper_altitude_bin_edge,
                    shower_maximum_altitude)
                if (
                    upper_altitude_bin_edge == 0 or
                    upper_altitude_bin_edge == altitude_bin_edges.shape[0]
                ):
                    continue

                assert (
                    shower_maximum_altitude <
                    altitude_bin_edges[upper_altitude_bin_edge])
                assert (
                    shower_maximum_altitude >
                    altitude_bin_edges[upper_altitude_bin_edge - 1])

                altitude_bin = upper_altitude_bin_edge - 1

                if (
                    num_photons_in_altitude_bins[altitude_bin] >=
                    max_num_photons_in_bin
                ):
                    continue

                comp_x_y_cx_cy, valid_photons = compress_photons(
                    x=event.cherenkov_photon_bunches.x,
                    y=event.cherenkov_photon_bunches.y,
                    cx=event.cherenkov_photon_bunches.cx,
                    cy=event.cherenkov_photon_bunches.cy,
                    aperture_binning_config=abc,
                    field_of_view_radius=fov_r)
                append_compressed_photons(
                    path=altitude_paths[altitude_bin],
                    compressed_photons=comp_x_y_cx_cy)

                num_showers_in_altitude_bins[altitude_bin] += 1
                num_photons_in_altitude_bins[altitude_bin] = \
                    _num_photons_in_bin(
                        path=altitude_paths[altitude_bin],
                        aperture_binning_config=abc)

                fill_path = os.path.join(energy_path, "fill.json")
                with open(fill_path, "wt") as fout:
                    fout.write(json.dumps({
                        "num_shower": num_showers_in_altitude_bins.tolist(),
                        "num_photons": num_photons_in_altitude_bins.tolist(),
                        },
                        indent=4))
        run_id += 1
