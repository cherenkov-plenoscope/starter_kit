import event_look_up_table as elut
import numpy as np
import tempfile
import os
assert_close = np.testing.assert_almost_equal


def test_compress_image():
    np.random.seed(0)

    histograms = [
        np.ones((25, 25)),
        np.zeros((25, 25)),
        np.zeros((1, 1)),
        np.zeros((512, 4)),
        np.ones((25, 25))*1e3,
        np.ones((25, 25))*1e-5,
        np.eye(64)*1e6,
        np.reshape(np.random.uniform(size=100*100), newshape=(100, 100))
    ]

    for hist2d in histograms:
        png_bytes, scale = elut.integrated._compress_histogram2d(
            histogram2d=hist2d)

        hist2d_back = elut.integrated._decompress_histogram2d(
            png_bytes=png_bytes,
            scale=scale)

        assert len(hist2d.shape) == 2
        assert len(hist2d_back.shape) == 2
        assert hist2d.shape[0] == hist2d_back.shape[0]
        assert hist2d.shape[1] == hist2d_back.shape[1]

        for ix in range(hist2d.shape[0]):
            for iy in range(hist2d.shape[1]):
                assert_close(hist2d[ix, iy], hist2d_back[ix, iy], decimal=2)
