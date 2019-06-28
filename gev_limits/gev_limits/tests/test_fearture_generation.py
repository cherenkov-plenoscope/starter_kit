import gev_limits as gli
import numpy as np


def test_hillas_ellipse():
    for seed in range(10):
        np.random.seed(seed)
        cov = np.random.uniform(0., 10., (2, 2))
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        if min_eig < 0:
            cov -= 10*min_eig * np.eye(*cov.shape)

        cx_mean = 10*(np.random.uniform() -.5)
        cy_mean = 10*(np.random.uniform() -.5)
        points = np.random.multivariate_normal(
            mean=(cx_mean, cy_mean),
            cov=cov,
            size=1000)
        cxs = points[:, 0]
        cys = points[:, 1]

        eli = gli.features.hillas_ellipse(cxs=cxs, cys=cys)

        """
        plt.plot(cxs, cys, 'xk')
        plt.plot(
            [eli.cx_mean, eli.cx_mean + eli.cx_major*eli.std_major],
            [eli.cy_mean, eli.cy_mean + eli.cy_major*eli.std_major],
            'r')
        plt.plot(
            [eli.cx_mean, eli.cx_mean + eli.cx_minor*eli.std_minor],
            [eli.cy_mean, eli.cy_mean + eli.cy_minor*eli.std_minor],
            'b')
        plt.axis('equal')
        plt.show()
        """
        assert np.abs(eli.cx_mean - cx_mean) < 1
        assert np.abs(eli.cy_mean - cy_mean) < 1
