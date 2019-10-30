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


def init(
    integrated_lookup_dir,
    aperture_bin_radius=4.6,
    radius_bin_centers=np.linspace(0., 256, 128),
    azimuth_bin_centers=np.linspace(0., 2*np.pi, 6),
    c_parallel_bin_edges=np.linspace(
        np.deg2rad(-.5),
        np.deg2rad(2.5),
        3*64 + 1),
    c_perpendicular_bin_edges=np.linspace(
        np.deg2rad(-.5),
        np.deg2rad(+.5),
        64 + 1),
):
    os.makedirs(integrated_lookup_dir)
    integrated_binning_path = op.join(
        integrated_lookup_dir,
        "integrated_binning_config.json")
    with open(integrated_binning_path, "wt") as fout:
        fout.write(json.dumps(
            {
                "aperture_bin_radius": float(aperture_bin_radius),
                "radius_bin_centers": radius_bin_centers.tolist(),
                "azimuth_bin_centers": azimuth_bin_centers.tolist(),
                "c_parallel_bin_edges": c_parallel_bin_edges.tolist(),
                "c_perpendicular_bin_edges": c_perpendicular_bin_edges.tolist()
            },
            indent=4))


def make_jobs(
    integrated_lookup_dir,
    unbinned_lookup_path,
):
    R = Reader(unbinned_lookup_path)
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
    unbinned_reader = Reader(job["unbinned_lookup_path"])

    altitude_bin = job["altitude_bin"]
    energy_bin = job["energy_bin"]

    integrated_binning_path = op.join(
        job["integrated_lookup_dir"],
        "integrated_binning_config.json")
    integrated_binning = irf.__read_json(integrated_binning_path)
    ib = integrated_binning

    energy_dir = op.join(
        job["integrated_lookup_dir"],
        ENERGY_BIN_FILENAME.format(energy_bin))
    os.makedirs(energy_dir, exist_ok=True)

    output_path = op.join(
        energy_dir,
        ALTITUDE_BIN_FILENAME.format(altitude_bin)+".tar")
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
