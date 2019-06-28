import gev_limits as gli
import numpy as np
import tempfile
import os
from scipy import spatial

def draw_normal(mean, std, size):
    return np.random.normal(loc=mean, scale=std, size=size)


def draw_uniform_2d(radius, size):
    xys = []
    while len(xys) < size:
        x = (np.random.uniform() - .5)*radius
        y = (np.random.uniform() - .5)*radius
        if np.hypot(x, y) < radius:
            xys.append([x, y])
    return np.array(xys)


def test_point_cloud():
    weights = [2, 1, 1]
    FOV_RADIUS = 3.25
    APERTURE_RADIUS = 35.5
    np.random.seed(1)

    responses = []
    off_cxs = np.linspace(0., 1., 10)
    for off_cx in off_cxs:
        sims = []
        for i in  range(10):
            num_photons1 = 100
            xys1 = draw_uniform_2d(radius=APERTURE_RADIUS, size=num_photons1)
            lfs1 = np.array([
                draw_normal(mean=1.5, std=0.15, size=num_photons1),
                draw_normal(mean=.5, std=0.15, size=num_photons1),
                xys1[:, 0],
                xys1[:, 1],
                draw_normal(mean=0., std=5e-9, size=num_photons1)
            ]).T

            num_photons2 = 120
            xys2 = draw_uniform_2d(radius=APERTURE_RADIUS, size=num_photons2)
            lfs2 = np.array([
                draw_normal(mean=1.5, std=0.15, size=num_photons2),
                draw_normal(mean=0.5 + off_cx, std=0.15, size=num_photons2),
                xys2[:, 0],
                xys2[:, 1],
                draw_normal(mean=0., std=2e-9, size=num_photons2)
            ]).T

            lfs1_normed = np.array([
                weights[0]*lfs1[:, 0] / FOV_RADIUS,
                weights[0]*lfs1[:, 1] / FOV_RADIUS,
                weights[1]*lfs1[:, 2] / APERTURE_RADIUS,
                weights[1]*lfs1[:, 3] / APERTURE_RADIUS,
                weights[2]*lfs1[:, 4] / 5e-9
            ]).T

            lfs2_normed = np.array([
                weights[0]*lfs2[:, 0] / FOV_RADIUS,
                weights[0]*lfs2[:, 1] / FOV_RADIUS,
                weights[1]*lfs2[:, 2] / APERTURE_RADIUS,
                weights[1]*lfs2[:, 3] / APERTURE_RADIUS,
                weights[2]*lfs2[:, 4] / 5e-9
            ]).T

            distance_upper_bound_pc = 0.2

            sim_pc = gli.lookup.similarity_point_clouds(
                normalized_point_cloud_1=lfs1_normed,
                normalized_point_cloud_2=lfs2_normed,
                distance_upper_bound=distance_upper_bound_pc)

            sims.append(sim_pc)

        print(np.mean(sims))
        responses.append(np.mean(sims))

