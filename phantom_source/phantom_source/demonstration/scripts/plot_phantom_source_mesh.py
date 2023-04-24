#!/usr/bin/python
import numpy as np
import plenopy as pl
import phantom_source
import json_numpy
import sebastians_matplotlib_addons as sebplt
import argparse

argparser = argparse.ArgumentParser(
    prog="plot_refocussed_images",
    description=(
        "Plot a stack of refocussed images from the "
        "observation of a phantom-source"
    ),
)

argparser.add_argument(
    "work_dir",
    metavar="WORK_DIR",
    type=str,
    help="Path to the work_dir of the phantom_source.demonstration.",
)

args = argparser.parse_args()

sebplt.matplotlib.rcParams.update(
    plenoirf.summary.figure.MATPLOTLIB_RCPARAMS_LATEX
)

work_dir = args.work_dir
plot_dir = os.path.join(work_dir, "plot")
os.makedirs(plot_dir, exist_ok=True)

config = json_numpy.read(os.path.join(work_dir, "config.json"))


phantom_source_meshes = json_numpy.read(
    os.path.join(work_dir, "phantom_source_meshes.json")
)

XY_RADIUS = 400
MAX_OBJECT_DISTANCE = 27e3

phantom_source.plot.save_3d_views_of_meshes(
    meshes=phantom_source_meshes,
    scale=1e-3,
    xlim=[-XY_RADIUS * 1e-3, XY_RADIUS * 1e-3],
    ylim=[-XY_RADIUS * 1e-3, XY_RADIUS * 1e-3],
    zlim=[0 * 1e-3, MAX_OBJECT_DISTANCE * 1e-3],
    elevations=[23, 90, 0],
    azimuths=[-60, 0, 0],
    paths=[
        os.path.join(plot_dir, "mesh_{:d}.jpg".format(i)) for i in range(3)
    ],
)

phantom_source_meshes_img = json_numpy.read(
    os.path.join(work_dir, "phantom_source_meshes_img.json")
)
