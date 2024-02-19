import numpy as np


def img2scn_1d(c_deg, depth_m):
    return np.tan(np.deg2rad(c_deg)) * depth_m


def img2scn_3d(imgpos):
    cx_deg, cy_deg, depth_m = imgpos
    return np.array(
        [
            img2scn_1d(cx_deg, depth_m),
            img2scn_1d(cy_deg, depth_m),
            depth_m,
        ]
    )
