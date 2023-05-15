import os
import numpy as np
import aberration_demo as abe
import json_numpy
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "work_dir", metavar="WORK_DIR", type=str,
)
args = argparser.parse_args()
work_dir = args.work_dir

config = json_numpy.read_tree(os.path.join(work_dir, "config"))

plot_iod_dir = os.path.join(
    work_dir, "plots", "impact_of_deformations", "guide_stars"
)


out = {}
for instrument_key in config["observations"]["instruments"]:
    if "star" in config["observations"]["instruments"][instrument_key]:

        star_analysis = json_numpy.read(
            os.path.join(work_dir, "analysis", instrument_key, "star.json")
        )

        instrument_max_img_intensity = 0.0
        for guide_star_idx in range(
            len(config["observations"]["star"]["guide_stars"])
        ):
            guide_star_key = "{:06d}".format(guide_star_idx)

            image_response = star_analysis[guide_star_key]

            img = abe.analysis.make_norm_image(image_response=image_response)
            instrument_max_img_intensity = np.max(
                [np.max(img), instrument_max_img_intensity]
            )

        out[instrument_key] = instrument_max_img_intensity


os.makedirs(plot_iod_dir, exist_ok=True)
json_numpy.write(
    os.path.join(plot_iod_dir, "maximum_image_intensity_for_guide_stars.json"),
    out,
)
