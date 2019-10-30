import functools
import tarfile
import io
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
from . import integrated


MERLICT_EVENTIO_CONVERTER_PATH = op.join(
    'build',
    'merlict',
    'merlict-eventio-converter')

CORSIKA_PATH = op.join(
    "build",
    "corsika",
    "corsika-75600",
    "run",
    "corsika75600Linux_QGSII_urqmd")

ALTITUDE_BIN_FILENAME = "{:06d}_altitude"
ENERGY_BIN_FILENAME = "{:06d}_energy"
APERTURE_BIN_FILENAME = "{:06d}.x_y_cx_cy"
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


@functools.lru_cache(maxsize=4096, typed=True)
def _read_aperture_bin(path):
    with open(path, "rb") as fin:
        compressed_photons = np.rec.array(
            np.frombuffer(fin.read(), dtype=PHOTON_DTYPE))
    return compressed_photons


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
        compressed_photons = _read_aperture_bin(xy_bin_path)
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


class Reader:
    def __init__(self, path):
        self.lookup_path = path
        _config_path = op.join(self.lookup_path, "binning_config.json")
        with open(_config_path, "rt") as f:
            config = json.loads(f.read())
        self.aperture_binning = make_aperture_binning(config["aperture"])
        self.altitude_bin_edges = np.array(config["altitude_bin_edges"])
        self.field_of_view_radius = config["field_of_view"]["radius"]
        self.energy_bin_centers = np.array(config["energy_bin_centers"])
        self._config = config

        self.__energy_altitude_paths = []
        for e in range(len(self.energy_bin_centers)):
            altitude_paths = []
            for a in range(len(self.altitude_bin_edges) - 1):
                p = os.path.join(
                    self.lookup_path,
                    ENERGY_BIN_FILENAME.format(e),
                    ALTITUDE_BIN_FILENAME.format(a))
                altitude_paths.append(p)
            self.__energy_altitude_paths.append(altitude_paths)

        self.num_showers = []
        self.num_photons = []
        for e in range(len(self.energy_bin_centers)):
            p = os.path.join(
                self.lookup_path,
                ENERGY_BIN_FILENAME.format(e),
                "fill.json")
            if os.path.exists(p):
                with open(p, "rt") as fin:
                    energy_bin_fill = json.loads(fin.read())
                self.num_showers.append(energy_bin_fill["num_showers"])
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


def init(
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
    energy_bin_centers=np.geomspace(0.25, 25, 8),
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
    with open(os.path.join(lookup_path, "binning_config.json"), "wt") as f:
        f.write(json.dumps(config, indent=4))
    with open(os.path.join(lookup_path, "filling_config.json"), "wt") as f:
        f.write(json.dumps({
            "max_num_photons_in_bin": int(max_num_photons_in_bin)}, indent=4))


def make_jobs(
    lookup_path,
    num_jobs=4,
    merlict_eventio_converter_path=MERLICT_EVENTIO_CONVERTER_PATH,
    corsika_path=CORSIKA_PATH,
    fraction_of_valid_altitude_bins=0.5,
    random_seed=1,
    random_seed_range_per_job=1000,
    energy_per_iteration=512,
):
    binning_config = irf.__read_json(
        os.path.join(lookup_path, "binning_config.json"))
    location_config = irf.__read_json(
        os.path.join(lookup_path, "location_config.json"))
    particle_config = irf.__read_json(
        os.path.join(lookup_path, "particle_config.json"))
    filling_config = irf.__read_json(
        os.path.join(lookup_path, "filling_config.json"))

    bc = binning_config
    num_energy_bins = len(bc["energy_bin_centers"])
    num_altitude_bins = len(bc["altitude_bin_edges"]) - 1
    num_jobs_in_energy_bin = int(np.ceil(num_jobs/num_energy_bins))
    max_num_photons_in_bin_job = np.ceil(
        filling_config["max_num_photons_in_bin"] /
        num_jobs_in_energy_bin)

    jobs = []
    for energy_bin_idx, energy in enumerate(bc["energy_bin_centers"]):
        random_seed_in_energy_bin = random_seed + energy_bin_idx
        for job_idx in range(num_jobs_in_energy_bin):
            job_seed = (
                random_seed_in_energy_bin
                + job_idx*random_seed_range_per_job)
            job = {}
            job["lookup_path"] = lookup_path
            job["energy_dirname"] = ENERGY_BIN_FILENAME.format(
                energy_bin_idx)+".jobs"
            job["job_dirname"] = "{:06d}".format(job_idx)
            job["location_config"] = location_config
            job["particle_config"] = particle_config
            job["binning_config"] = binning_config
            job["energy"] = energy
            job["energy_per_iteration"] = energy_per_iteration
            job["max_num_photons_in_bin"] = max_num_photons_in_bin_job
            job["min_num_altitude_bins_with_enough_num_photons"] = int(np.ceil(
                fraction_of_valid_altitude_bins*num_altitude_bins))
            job["merlict_eventio_converter_path"] = \
                merlict_eventio_converter_path
            job["corsika_path"] = corsika_path
            job["run_id_start"] = job_seed
            job["run_id_stop"] = job_seed + random_seed_range_per_job
            jobs.append(job)
    return jobs


def run_job(job):
    with tempfile.TemporaryDirectory(prefix='lookup_job_') as tmp:
        tmp_output_path = op.join(tmp, job["job_dirname"])

        _add_energy_to_lookup_job(
            output_path=tmp_output_path,
            location_config=job["location_config"],
            particle_config=job["particle_config"],
            binning_config=job["binning_config"],
            energy=job["energy"],
            energy_per_iteration=job["energy_per_iteration"],
            max_num_photons_in_bin=job["max_num_photons_in_bin"],
            min_num_altitude_bins_with_enough_num_photons=job[
                "min_num_altitude_bins_with_enough_num_photons"],
            merlict_eventio_converter_path=job[
                "merlict_eventio_converter_path"],
            corsika_path=job["corsika_path"],
            run_id_start=job["run_id_start"],
            run_id_stop=job["run_id_stop"])

        tmp_tar_path = op.join(tmp, job["job_dirname"]+".tar")
        subprocess.call([
            "tar",
            "-cf",
            tmp_tar_path,
            "-C",
            tmp_output_path,
            "."])

        output_dir = op.join(
            job["lookup_path"],
            job["energy_dirname"])
        os.makedirs(output_dir, exist_ok=True)
        output_path = op.join(output_dir, job["job_dirname"]+".tar")
        shutil.copy(tmp_tar_path, output_path+".part")
        shutil.move(output_path+".part", output_path)
    return 0


def reduce_jobs(lookup_path):
    binning_config = irf.__read_json(
        os.path.join(lookup_path, "binning_config.json"))
    num_energy_bins = len(binning_config["energy_bin_centers"])
    for energy_bin_idx in range(num_energy_bins):
        _reduce_energy_bin(
            lookup_path=lookup_path,
            energy_bin_idx=energy_bin_idx)


def _remove_incomplete_altitude_bins(lookup_path, energy_bin_idx):
    filling_config = irf.__read_json(
        os.path.join(lookup_path, "filling_config.json"))
    max_num_photons_in_bin = filling_config["max_num_photons_in_bin"]

    energy_bin_path = op.join(
        lookup_path,
        ENERGY_BIN_FILENAME.format(energy_bin_idx))
    energy_bin_part_path = energy_bin_path + ".part"

    shutil.copytree(energy_bin_path, energy_bin_part_path)

    with open(op.join(energy_bin_part_path, "fill.json"), "rt") as f:
        fill = json.loads(f.read())
    num_photons = np.array(fill["num_photons"])
    num_showers = np.array(fill["num_showers"])

    # from bottom
    altitude_bin = 0
    while num_photons[altitude_bin] < max_num_photons_in_bin:
        altitude_bin_path = op.join(
            energy_bin_part_path,
            ALTITUDE_BIN_FILENAME.format(altitude_bin))
        shutil.rmtree(altitude_bin_path)
        num_photons[altitude_bin] = 0
        num_showers[altitude_bin] = 0
        altitude_bin += 1

    # from top
    altitude_bin = num_photons.shape[0] - 1
    while num_photons[altitude_bin] < max_num_photons_in_bin:
        altitude_bin_path = op.join(
            energy_bin_part_path,
            ALTITUDE_BIN_FILENAME.format(altitude_bin))
        shutil.rmtree(altitude_bin_path)
        num_photons[altitude_bin] = 0
        num_showers[altitude_bin] = 0
        altitude_bin -= 1

    with open(op.join(energy_bin_part_path, "fill.json"), "wt") as fout:
        fout.write(json.dumps({
            "num_showers": num_showers.tolist(),
            "num_photons": num_photons.tolist()},
            indent=4))
    # shutil.move(energy_bin_part_path, energy_bin_path)


def _append_apperture_bin(
    input_altitude_bin_path,
    output_altitude_bin_path,
    num_aperture_bins,
):
    for aperture_bin in range(num_aperture_bins):
        aperture_bin_filename = APERTURE_BIN_FILENAME.format(aperture_bin)
        in_path = op.join(input_altitude_bin_path, aperture_bin_filename)
        out_path = op.join(output_altitude_bin_path, aperture_bin_filename)
        if op.exists(in_path):
            with open(in_path, "rb") as fi, open(out_path, "ab") as fo:
                fo.write(fi.read())


def _reduce_energy_bin(lookup_path, energy_bin_idx):
    bc = irf.__read_json(op.join(lookup_path, "binning_config.json"))
    num_altitude_bins = len(bc["altitude_bin_edges"]) - 1
    num_aperture_bins = make_aperture_binning(bc["aperture"])["num_bins"]

    energy_str = ENERGY_BIN_FILENAME.format(energy_bin_idx)
    energy_path = op.join(lookup_path, energy_str)
    energy_jobs_path = energy_path+".jobs"
    assert op.exists(energy_jobs_path)
    job_paths = glob.glob(op.join(energy_jobs_path, "*.tar"))

    with tempfile.TemporaryDirectory(prefix='lookup_reduce_') as tmp:
        tmp_energy_path = op.join(tmp, energy_str)
        os.makedirs(tmp_energy_path)
        for altitude_bin in range(num_altitude_bins):
            altitude_bin_dirname = ALTITUDE_BIN_FILENAME.format(altitude_bin)
            output_altitude_bin_path = op.join(
                tmp_energy_path,
                altitude_bin_dirname)
            os.makedirs(output_altitude_bin_path)

        num_showers_in_altitude_bins = np.zeros(
            num_altitude_bins,
            dtype=np.int64)
        num_photons_in_altitude_bins = np.zeros(
            num_altitude_bins,
            dtype=np.int64)

        for job_path in job_paths:
            tmp_job_path = op.join(tmp, op.basename(job_path))
            os.makedirs(tmp_job_path)
            subprocess.call([
                "tar",
                "-xf",
                job_path,
                "--directory",
                tmp_job_path])
            job_fill = irf.__read_json(op.join(tmp_job_path, "fill.json"))
            num_showers_in_altitude_bins += np.array(job_fill["num_showers"])
            num_photons_in_altitude_bins += np.array(job_fill["num_photons"])

            for altitude_bin in range(num_altitude_bins):
                altitude_bin_dirname = ALTITUDE_BIN_FILENAME.format(
                    altitude_bin)
                _append_apperture_bin(
                    input_altitude_bin_path=op.join(
                        tmp_job_path,
                        altitude_bin_dirname),
                    output_altitude_bin_path=op.join(
                        tmp_energy_path,
                        altitude_bin_dirname),
                    num_aperture_bins=num_aperture_bins)
            shutil.rmtree(tmp_job_path)

        with open(op.join(tmp_energy_path, "fill.json"), "wt") as fout:
            fout.write(json.dumps({
                "num_showers": num_showers_in_altitude_bins.tolist(),
                "num_photons": num_photons_in_altitude_bins.tolist()},
                indent=4))

        shutil.copytree(tmp_energy_path, energy_path+".part")
        shutil.move(energy_path+".part", energy_path)
    shutil.rmtree(energy_jobs_path)


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
    energy_per_iteration=512.,
    merlict_eventio_converter_path=MERLICT_EVENTIO_CONVERTER_PATH,
    corsika_path=CORSIKA_PATH,
    random_seed=1,
):
    binning_config = irf.__read_json(
        os.path.join(lookup_path, "binning_config.json"))
    energy = binning_config["energy_bin_centers"][energy_bin_center]
    energy_path = os.path.join(
        lookup_path,
        ENERGY_BIN_FILENAME.format(energy_bin_center))
    location_config = irf.__read_json(
        os.path.join(lookup_path, "location_config.json"))
    particle_config = irf.__read_json(
        os.path.join(lookup_path, "particle_config.json"))
    filling_config = irf.__read_json(
        os.path.join(lookup_path, "filling_config.json"))

    _add_energy_to_lookup_job(
        output_path=energy_path,
        location_config=location_config,
        particle_config=particle_config,
        binning_config=binning_config,
        energy=energy,
        energy_per_iteration=energy_per_iteration,
        max_num_photons_in_bin=filling_config["max_num_photons_in_bin"],
        min_num_altitude_bins_with_enough_num_photons=0.5*len(
            binning_config["altitude_bin_edges"]),
        merlict_eventio_converter_path=merlict_eventio_converter_path,
        corsika_path=corsika_path,
        run_id_start=random_seed,
        run_id_stop=random_seed+1000)


def _find_bin_in_edges(bin_edges, value):
    upper_bin_edge = int(np.digitize([value], bin_edges)[0])
    if upper_bin_edge == 0:
        return True, 0, False
    if upper_bin_edge == bin_edges.shape[0]:
        return False, upper_bin_edge - 1, True
    return False, upper_bin_edge - 1, False


def _find_bins_in_centers(bin_centers, value):
    underflow, lower_bin, overflow = _find_bin_in_edges(
        bin_edges=bin_centers,
        value=value)

    upper_bin = lower_bin + 1
    if underflow:
        lower_weight = 0.
    elif overflow:
        lower_weight = 1.
    else:
        dist_to_lower = value - bin_centers[lower_bin]
        dist_to_upper = bin_centers[upper_bin] - value
        bin_range = bin_centers[upper_bin] - bin_centers[lower_bin]
        lower_weight = 1 - dist_to_lower/bin_range

    return {
        "underflow": underflow,
        "overflow": overflow,
        "lower_bin": lower_bin,
        "upper_bin": lower_bin + 1,
        "lower_weight": lower_weight,
        "upper_weight": 1. - lower_weight,
    }


def _add_energy_to_lookup_job(
    output_path,
    location_config,
    particle_config,
    binning_config,
    energy,
    energy_per_iteration,
    max_num_photons_in_bin,
    min_num_altitude_bins_with_enough_num_photons,
    merlict_eventio_converter_path,
    corsika_path,
    run_id_start=1,
    run_id_stop=1001,
):
    abc = make_aperture_binning(binning_config["aperture"])
    altitude_bin_edges = np.array(binning_config["altitude_bin_edges"])
    cherenkov_collection_radius = float(np.max(abc["xy_bin_edges"]))
    fov_r = binning_config["field_of_view"]["radius"]

    num_showers_in_run = int(energy_per_iteration//energy)
    num_showers_in_run = np.max([num_showers_in_run, 16])
    num_showers_in_run = np.min([num_showers_in_run, 2048])

    os.makedirs(output_path)
    cct = {}
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
            output_path,
            ALTITUDE_BIN_FILENAME.format(idx))
        altitude_paths[idx] = altitude_bin_path
        os.makedirs(altitude_bin_path)

    run_id = run_id_start
    while True:
        num_altitude_bins_with_enough_photons = np.sum(
            num_photons_in_altitude_bins > max_num_photons_in_bin)
        if (
            num_altitude_bins_with_enough_photons >=
            min_num_altitude_bins_with_enough_num_photons
        ):
            # The expected break out of the loop.
            break

        if run_id >= run_id_stop:
            raise RuntimeError(
                "Ran out of run-ids, i.e. ran out of random-seeds.")

        run = cct.copy()
        run["run_id"] = run_id
        np.random.seed(run_id)

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

                underflow, altitude_bin, overflow = _find_bin_in_edges(
                    bin_edges=altitude_bin_edges,
                    value=shower_maximum_altitude)

                if underflow or overflow:
                    continue

                if (num_photons_in_altitude_bins[altitude_bin] >=
                        max_num_photons_in_bin):
                    continue

                bunch_weight = event. \
                    cherenkov_photon_bunches.\
                    probability_to_reach_observation_level
                assert np.sum(bunch_weight > 1) == 0
                passed_atmosphere = bunch_weight >= np.random.uniform(
                    size=num_bunches)

                photon_bunch_radius = np.hypot(
                    event.cherenkov_photon_bunches.x,
                    event.cherenkov_photon_bunches.y)
                on_disc = photon_bunch_radius <= cherenkov_collection_radius

                photon_incident = np.hypot(
                    event.cherenkov_photon_bunches.cx,
                    event.cherenkov_photon_bunches.cy)
                in_fov = photon_incident <= fov_r

                valid_geometry = np.logical_and(in_fov, on_disc)
                valid = np.logical_and(valid_geometry, passed_atmosphere)

                comp_x_y_cx_cy, valid_photons = compress_photons(
                    x=event.cherenkov_photon_bunches.x[valid],
                    y=event.cherenkov_photon_bunches.y[valid],
                    cx=event.cherenkov_photon_bunches.cx[valid],
                    cy=event.cherenkov_photon_bunches.cy[valid],
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

                fill_path = os.path.join(output_path, "fill.json")
                with open(fill_path, "wt") as fout:
                    fout.write(json.dumps({
                        "num_showers": num_showers_in_altitude_bins.tolist(),
                        "num_photons": num_photons_in_altitude_bins.tolist()},
                        indent=4))
        run_id += 1
