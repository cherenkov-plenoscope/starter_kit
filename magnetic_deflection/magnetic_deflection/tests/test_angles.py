import magnetic_deflection as mdfl
import numpy as np


def test_angles():
    for az_deg in np.linspace(-380, 380, 25):
        for zd_deg in np.linspace(0, 89, 25):

            cx, cy = mdfl._az_zd_to_cx_cy(
                azimuth_deg=az_deg,
                zenith_deg=zd_deg)

            az_deg_back, zd_deg_back = mdfl._cx_cy_to_az_zd_deg(cx=cx, cy=cy)

            dist_deg = mdfl._great_circle_distance_alt_zd_deg(
                az1_deg=az_deg,
                zd1_deg=zd_deg,
                az2_deg=az_deg_back,
                zd2_deg=zd_deg_back)

            print(az_deg, zd_deg, az_deg_back, zd_deg_back, cx, cy)
            assert dist_deg < 1
