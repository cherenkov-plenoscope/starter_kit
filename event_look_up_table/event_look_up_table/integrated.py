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

def init_and_make_jobs(
    integrated_lookup_dir,
    unbinned_lookup_path,
    aperture_bin_radius=4.6,
    radius_stop=256,
    num_radius_bins=96,
    num_azimuth_bin=8,
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
                "num_azimuth_bin": int(num_azimuth_bin),
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
    config_filenames = [
        "particle_config.json",
        "location_config.json",
        "binning_config.json",
        "filling_config.json"]
    for config_filename in config_filenames:
        shutil.copy(
            op.join(unbinned_lookup_path, config_filename),
            op.join(integrated_lookup_dir, config_filename))
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
        b["num_azimuth_bin"],
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

                num_showers = unbinned_reader.num_showers[
                    energy_bin][altitude_bin]
                integrated_image_per_shower = integrated_image/num_showers
                image_scale = np.max(integrated_image_per_shower)

                norm_image = integrated_image_per_shower/image_scale
                norm_image8 = norm_image*255
                norm_image8 = norm_image8.astype(np.uint8)
                image_scale8 = image_scale*255
                scale_json_str = json.dumps(
                    {"photons_per_shower_scale": image_scale8})

                img_path = "{:06d}_azimuth_{:06d}_radius".format(az_bin, r_bin)

                with io.BytesIO() as f:
                    image = PIL.Image.fromarray(norm_image8)
                    image.save(f, format="PNG")
                    f.seek(0)
                    tarinfo = tarfile.TarInfo(name=img_path+".png")
                    tarinfo.size=len(f.getvalue())
                    tarf.addfile(tarinfo=tarinfo, fileobj=f)

                with io.BytesIO() as f:
                    f.write(str.encode(scale_json_str))
                    f.seek(0)
                    tarinfo = tarfile.TarInfo(name=img_path+".json")
                    tarinfo.size=len(f.getvalue())
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
