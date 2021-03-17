import numpy as np
import datetime


def cone_solid_angle(cone_radial_opening_angle_rad):
    cap_hight = 1.0 - np.cos(cone_radial_opening_angle_rad)
    return 2.0 * np.pi * cap_hight


def contains_same_bytes(path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        a_bytes = fa.read()
        b_bytes = fb.read()
        return a_bytes == b_bytes


def date_dict_now():
    dt = datetime.datetime.now()
    out = {}
    for key in ["year", "month", "day", "hour", "minute", "second"]:
        out[key] = int(dt.__getattribute__(key))
    return out
