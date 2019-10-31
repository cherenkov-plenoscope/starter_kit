import PIL
import tarfile
import io
import numpy as np
import os
import gzip
import plenopy as pl
import acp_instrument_response_function as irf
from os import path as op
import shutil
import tempfile
import json
from . import unbinned

CONFIG_FILENAMES = [
    "particle_config.json",
    "location_config.json",
    "binning_config.json",
    "filling_config.json"]


def init_and_make_jobs(
    integrated_lookup_dir,
    unbinned_lookup_path,
    aperture_bin_radius=4.6,
    radius_stop=256,
    num_radius_bins=96,
    num_azimuth_bins=8,
    c_parallel_bin_edges_start=np.deg2rad(-.5),
    c_parallel_bin_edges_stop=np.deg2rad(2.5),
    num_c_parallel_bin_edges=3*64+1,
    c_perpendicular_bin_edges_start=np.deg2rad(-.5),
    c_perpendicular_bin_edges_stop=np.deg2rad(+.5),
    num_c_perpendicular_bin_edges=64+1,
):
    os.makedirs(integrated_lookup_dir)
    integrated_binning_path = op.join(
        integrated_lookup_dir,
        "integrated_binning_config.json")
    with open(integrated_binning_path, "wt") as fout:
        fout.write(json.dumps(
            {
                "aperture_bin_radius": float(aperture_bin_radius),
                "radius_stop": float(radius_stop),
                "num_radius_bins": int(num_radius_bins),
                "num_azimuth_bins": int(num_azimuth_bins),
                "c_parallel_bin_edges_start": float(
                    c_parallel_bin_edges_start),
                "c_parallel_bin_edges_stop": float(
                    c_parallel_bin_edges_stop),
                "num_c_parallel_bin_edges": int(
                    num_c_parallel_bin_edges),
                "c_perpendicular_bin_edges_start": float(
                    c_perpendicular_bin_edges_start),
                "c_perpendicular_bin_edges_stop": float(
                    c_perpendicular_bin_edges_stop),
                "num_c_perpendicular_bin_edges": int(
                    num_c_perpendicular_bin_edges),
            },
            indent=4))
    for config_filename in CONFIG_FILENAMES:
        shutil.copy(
            op.join(unbinned_lookup_path, config_filename),
            op.join(integrated_lookup_dir, config_filename))

    R = unbinned.Reader(unbinned_lookup_path)
    for energy_bin, _ in enumerate(R.energy_bin_centers):
        energy_bin_dirname = unbinned.ENERGY_BIN_FILENAME.format(energy_bin)
        os.makedirs(op.join(integrated_lookup_dir, energy_bin_dirname))
        shutil.copy(
            op.join(unbinned_lookup_path, energy_bin_dirname, "fill.json"),
            op.join(integrated_lookup_dir, energy_bin_dirname, "fill.json"))

    return _make_jobs(
        integrated_lookup_dir=integrated_lookup_dir,
        unbinned_lookup_path=unbinned_lookup_path)


def _make_integrated_binning_config(binning_construct):
    b = binning_construct.copy()
    b["c_parallel_bin_edges"] = np.linspace(
        b["c_parallel_bin_edges_start"],
        b["c_parallel_bin_edges_stop"],
        b["num_c_parallel_bin_edges"])
    b["c_perpendicular_bin_edges"] = np.linspace(
        b["c_perpendicular_bin_edges_start"],
        b["c_perpendicular_bin_edges_stop"],
        b["num_c_perpendicular_bin_edges"])
    b["azimuth_bin_centers"] = np.linspace(
        0.,
        2.*np.pi,
        b["num_azimuth_bins"],
        endpoint=False)
    b["radius_bin_centers"] = np.linspace(
        0.,
        b["radius_stop"],
        b["num_radius_bins"])
    return b


def _make_jobs(
    integrated_lookup_dir,
    unbinned_lookup_path,
):
    R = unbinned.Reader(unbinned_lookup_path)
    jobs = []
    for energy_bin in range(len(R.energy_bin_centers)):
        for altitude_bin in range(len(R.altitude_bin_edges) - 1):
            if R.num_showers[energy_bin][altitude_bin] > 0:
                job = {
                    "unbinned_lookup_path": unbinned_lookup_path,
                    "integrated_lookup_dir": integrated_lookup_dir,
                    "energy_bin": energy_bin,
                    "altitude_bin": altitude_bin}
                jobs.append(job)
    return jobs


def run_job(job):
    unbinned_reader = unbinned.Reader(job["unbinned_lookup_path"])

    altitude_bin = job["altitude_bin"]
    energy_bin = job["energy_bin"]

    integrated_binning_path = op.join(
        job["integrated_lookup_dir"],
        "integrated_binning_config.json")
    ib = _make_integrated_binning_config(
        irf.__read_json(integrated_binning_path))

    energy_dir = op.join(
        job["integrated_lookup_dir"],
        unbinned.ENERGY_BIN_FILENAME.format(energy_bin))
    os.makedirs(energy_dir, exist_ok=True)

    output_path = op.join(
        energy_dir,
        unbinned.ALTITUDE_BIN_FILENAME.format(altitude_bin)+".tar")
    part_output_path = output_path + ".part"

    num_showers = unbinned_reader.num_showers[energy_bin][altitude_bin]

    with tarfile.TarFile(part_output_path, "w") as tarf:
        for az_bin, az in enumerate(ib["azimuth_bin_centers"]):
            for r_bin, r in enumerate(ib["radius_bin_centers"]):

                aperture_x = np.cos(az)*r
                aperture_y = np.sin(az)*r

                light_field = unbinned_reader.read_light_field(
                    energy_bin=energy_bin,
                    altitude_bin=altitude_bin,
                    core_x=aperture_x,
                    core_y=aperture_y,
                    instrument_radius=ib["aperture_bin_radius"])

                integrated_image = _project_to_image(
                    light_field=light_field,
                    c_parallel_bin_edges=ib["c_parallel_bin_edges"],
                    c_perpendicular_bin_edges=ib["c_perpendicular_bin_edges"],
                    x=aperture_x,
                    y=aperture_y)

                integrated_image_per_shower = integrated_image/num_showers

                png_bytes, scale = _compress_histogram2d(
                    histogram2d=integrated_image_per_shower)

                scale_json_str = json.dumps(
                    {"photons_per_shower_scale": scale})
                img_path = "{:06d}_azimuth_{:06d}_radius".format(az_bin, r_bin)

                with io.BytesIO() as f:
                    f.write(png_bytes)
                    f.seek(0)
                    tarinfo = tarfile.TarInfo(name=img_path+".png")
                    tarinfo.size = len(f.getvalue())
                    tarf.addfile(tarinfo=tarinfo, fileobj=f)

                with io.BytesIO() as f:
                    f.write(str.encode(scale_json_str))
                    f.seek(0)
                    tarinfo = tarfile.TarInfo(name=img_path+".json")
                    tarinfo.size = len(f.getvalue())
                    tarf.addfile(tarinfo=tarinfo, fileobj=f)

    shutil.move(part_output_path, output_path)
    return 0


def _project_to_image(
    light_field,
    c_parallel_bin_edges,
    c_perpendicular_bin_edges,
    x,
    y,
):
    cxs = light_field.cy
    cys = light_field.cx

    azimuth = np.arctan2(y, x)

    cPara = np.cos(-azimuth)*cys - np.sin(-azimuth)*cxs
    cPerp = np.sin(-azimuth)*cys + np.cos(-azimuth)*cxs

    hist = np.histogram2d(
        x=cPara,
        y=cPerp,
        bins=(c_parallel_bin_edges, c_perpendicular_bin_edges))[0]
    return hist


def _read_energy_altitude_tar(path):
    scales = []
    png_images = []
    with tarfile.TarFile(path, "r") as tarf:
        for item in tarf:
            name = item.name
            az_bin = int(name[0:6])
            r_bin = int(name[15:21])
            is_png = name.split(".")[1] == "png"
            raw = tarf.extractfile(item).read()
            if is_png:
                png_images.append(
                    {"azimuth": az_bin, "radius": r_bin, "png": raw})
            else:
                scales.append(
                    {"azimuth": az_bin, "radius": r_bin, "scale": raw})

    for scale in scales:
        scale["scale"] = json.loads(scale["scale"])["photons_per_shower_scale"]

    azs = list(set([s["azimuth"] for s in scales]))
    ras = list(set([s["radius"] for s in scales]))
    num_azimuth_bin = np.max(azs) + 1
    num_radius_bins = np.max(ras) + 1

    out = []
    for a_idx in range(num_azimuth_bin):
        out.append([])

    comb = []
    for i in range(len(scales)):
        a_idx = png_images[i]["azimuth"]
        assert png_images[i]["azimuth"] == scales[i]["azimuth"]
        assert png_images[i]["radius"] == scales[i]["radius"]
        len_radius_bins = len(out[a_idx])
        assert len_radius_bins == scales[i]["radius"]
        out[a_idx].append({
            "scale": scales[i]["scale"],
            "png": png_images[i]["png"]})
    return out


class Reader:
    def __init__(self, path):
        self._read_configs(path)

        _nshow, _nphot = unbinned._read_num_showers_num_photons(
            lookup_path=path,
            num_energy_bins=len(self.energy_bin_centers))
        self.num_showers = _nshow
        self.num_photons = _nphot

        self.png_images = []
        for energy_bin, _ in enumerate(self.energy_bin_centers):
            altitude_png_images = []
            for altitude_bin, _ in enumerate(self.altitude_bin_edges):
                e_a_tar_path = op.join(
                    path,
                    unbinned.ENERGY_BIN_FILENAME.format(energy_bin),
                    unbinned.ALTITUDE_BIN_FILENAME.format(altitude_bin)+".tar")
                if op.exists(e_a_tar_path):
                    block = _read_energy_altitude_tar(e_a_tar_path)
                    altitude_png_images.append(block)
                else:
                    altitude_png_images.append(None)
            self.png_images.append(altitude_png_images)

        self.valid_energy_altitude_bins = _valid_energy_altitude_bins(self)

    def _read_configs(self, path):
        with open(op.join(path, "location_config.json"), "rt") as f:
            self.location = json.loads(f.read())
        with open(op.join(path, "particle_config.json"), "rt") as f:
            self.particle = json.loads(f.read())
        with open(op.join(path, "filling_config.json"), "rt") as f:
            self.filling = json.loads(f.read())
        with open(op.join(path, "binning_config.json"), "rt") as f:
            self.unbinned = json.loads(f.read())
        with open(op.join(path, "integrated_binning_config.json"), "rt") as f:
            self.integrated = _make_integrated_binning_config(
                json.loads(f.read()))

        self.energy_bin_centers = self.unbinned["energy_bin_centers"]
        self.altitude_bin_edges = self.unbinned["altitude_bin_edges"]

    def image(
        self,
        energy_bin,
        altitude_bin,
        azimuth_bin,
        radius_bin
    ):
        i = self.png_images[energy_bin][altitude_bin][azimuth_bin][radius_bin]
        return _decompress_histogram2d(
            png_bytes=i["png"],
            scale=i["scale"])


def _compress_histogram2d(histogram2d):
    assert np.sum(histogram2d < 0.) == 0, "histogram2d must be >= 0."
    scale = np.max(histogram2d)
    if scale > 0.:
        norm_hist = histogram2d/scale
    else:
        norm_hist = histogram2d
    norm_hist8 = norm_hist*255
    norm_hist8 = norm_hist8.astype(np.uint8)
    scale8 = np.float32(scale/255.)
    with io.BytesIO() as f:
        image = PIL.Image.fromarray(norm_hist8)
        image.save(f, format="PNG")
        png_bytes = f.getvalue()
    return png_bytes, scale8


def _decompress_histogram2d(png_bytes, scale):
    with io.BytesIO() as buf:
        buf.write(png_bytes)
        norm_hist8 = np.array(PIL.Image.open(buf), dtype=np.float32)
    histogram2d = norm_hist8*scale
    return histogram2d


def _benchmark(integrated_lookup, num_requests=1000, random_seed=0):
    il = integrated_lookup
    np.random.seed(random_seed)
    request = 0
    while request < num_requests:
        energy_bin = int(
            np.random.uniform()*len(il.energy_bin_centers))
        altitude_bin = int(
            np.random.uniform()*(len(il.altitude_bin_edges)-1))
        azimuth_bin = int(
            np.random.uniform()*il.integrated["num_azimuth_bins"])
        radius_bin = int(
            np.random.uniform()*il.integrated["num_radius_bins"])

        if not il.valid_energy_altitude_bins[energy_bin][altitude_bin]:
            continue

        image = il.image(
            energy_bin=energy_bin,
            altitude_bin=altitude_bin,
            azimuth_bin=azimuth_bin,
            radius_bin=radius_bin)
        request += 1
    return True


def _valid_energy_altitude_bins(integrated_lookup):
    il = integrated_lookup
    num_e_bins = len(il.energy_bin_centers)
    num_a_bins = len(il.altitude_bin_edges)-1
    mask = np.zeros((num_e_bins, num_a_bins), dtype=np.bool)
    for ebin in range(num_e_bins):
        for abin in range(num_a_bins):
            if il.png_images[ebin][abin] is not None:
                mask[ebin, abin] = True
    return mask
