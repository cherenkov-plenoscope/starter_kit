import argparse
import phantom_source
import numpy as np
import json_numpy
import os
import sebastians_matplotlib_addons as sebplt
import plenopy as pl
import binning_utils as bu
import skimage
from skimage import filters
import gamma_ray_reconstruction as grr

argparser = argparse.ArgumentParser(
    prog="estimate_resolution_in_depth",
    description=(
        "Create a point-source and estimate how good the plenoscope "
        "can estimate its depth"
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


config = {}
config["seed"] = args.seed
config["light_field_geometry_path"] = args.light_field_geometry
config["merlict_propagate_photons_path"] = args.merlict_path
config["merlict_propagate_config_path"] = args.merlict_config
config["out_dir"] = args.out_dir

config["aperture_radius_m"] = 50
config["num_photons"] = 1e2
config["min_object_distance_m"] = 2.7e3
config["max_object_distance_m"] = 27.0e3
config["num_pixel_on_edge"] = 1024
config["image_field_of_view_deg"] = 6.5
config["image_containment_percentile"] = 95
config["auto_focus_step_rate"] = 0.5
config["oversampling_beam_spread"] = 1000

os.makedirs(name=config["out_dir"], exist_ok=True)
with open(os.path.join(config["out_dir"], "config.json"), "wt") as f:
    f.write(json_numpy.dumps(config, indent=4,))


c_radius = np.deg2rad(0.5 * config["image_field_of_view_deg"])
c_bin = bu.Binning(
    bin_edges=np.linspace(-c_radius, c_radius, config["num_pixel_on_edge"] + 1)
)

image_binning = {
    "cx": c_bin,
    "cy": c_bin,
}


prng = np.random.Generator(np.random.MT19937(seed=config["seed"]))


report, img = phantom_source.depth.estimate_resolution(
    cx_deg=1.0,
    cy_deg=-1.3,
    object_distance_m=35e3,
    aperture_radius_m=config["aperture_radius_m"],
    image_binning=image_binning,
    max_object_distance_m=config["max_object_distance_m"],
    min_object_distance_m=config["min_object_distance_m"],
    prng=prng,
    light_field_geometry_path=config["light_field_geometry_path"],
    merlict_propagate_photons_path=config["merlict_propagate_photons_path"],
    merlict_propagate_config_path=config["merlict_propagate_config_path"],
    image_containment_percentile=config["image_containment_percentile"],
    auto_focus_step_rate=config["auto_focus_step_rate"],
    oversampling_beam_spread=config["oversampling_beam_spread"],
    num_photons=config["num_photons"],
)


"""

# create response
# ---------------
Mimg = []
Mimg.append(
    phantom_source.mesh.triangle(
        pos=[-1.0, +1.3, 12.5e3], radius=0.01, density=DENSITY,
    )
)

Mscn = []
for mimg in Mimg:
    mscn = phantom_source.mesh.transform_image_to_scneney(mesh=mimg)
    Mscn.append(mscn)


light_fields = phantom_source.light_field.make_light_fields_from_meshes(
    meshes=Mscn,
    aperture_radius=APERTURE_RADIUS,
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
image_beams = pl.image.ImageRays(light_field_geometry=light_field_geometry)

lf_t, lf_lixel_ids = event.photon_arrival_times_and_lixel_ids()

# histogram photons
# -----------------
participating_beams = phantom_source.depth.make_participating_beams_from_lixel_ids(
    lixel_ids=lf_lixel_ids
)

# use response
# ------------

report = estimate_depth(
    image_beams=image_beams,
    participating_beams=participating_beams,
    image_binning=image_binning,
    max_object_distance=MAX_OBJECT_DISTANCE,
    min_object_distance=MIN_OBJECT_DISTANCE,
    image_containment_percentile=95,
    step_rate=0.5,
    oversampling_beam_spread=1000,
    num_max_iterations=1000,
)
"""
