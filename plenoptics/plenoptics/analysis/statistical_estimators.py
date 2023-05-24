import numpy as np
import scipy
from scipy import spatial


def encirclement2d(
    x,
    y,
    x_std,
    y_std,
    weights,
    prng,
    percentile=80,
    iteration_shrinking_factor=0.99,
    num_sub_samples=1,
):
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    assert not np.any(np.isnan(x_std))
    assert not np.any(np.isnan(y_std))
    assert np.all(x_std >= 0.0)
    assert np.all(y_std >= 0.0)
    assert not np.any(np.isnan(weights))
    assert np.all(weights >= 0.0)
    assert 0 < percentile <= 100
    assert num_sub_samples > 0
    assert 0 < iteration_shrinking_factor < 1.0

    xy = []
    for i in range(len(x)):
        for w in range(weights[i]):
            for s in range(num_sub_samples):
                rx = prng.normal(loc=x[i], scale=x_std[i])
                ry = prng.normal(loc=y[i], scale=y_std[i])
                xy.append([rx, ry])
    xy = np.array(xy)

    required_fraction = percentile / 100.0
    integral = xy.shape[0]
    if integral == 0:
        return float("nan"), float("nan"), float("nan")

    assert integral > 0
    center_x = np.median(xy[:, 0])
    center_y = np.median(xy[:, 1])
    radii = np.hypot((xy[:, 0] - center_x), (xy[:, 1] - center_y))
    radius = np.max(radii)

    tree = scipy.spatial.cKDTree(xy)

    num_loops = 0

    while True:
        overlap = len(tree.query_ball_point(x=[center_x, center_y], r=radius))
        if overlap / integral >= required_fraction:
            radius = radius * iteration_shrinking_factor
        else:
            break

        if num_loops > 1000:
            assert False, "Can not converge."
        num_loops += 1

    return center_x, center_y, radius


def encirclement1d(x, f, percentile=80, oversample=137):
    assert len(x) == len(f)
    assert len(x) >= 3
    assert np.all(np.gradient(x) > 0.0)
    assert percentile > 0
    assert oversample >= 1
    num_bins_fine = len(x) * oversample

    start_fraction = 0.5 - 0.5 * (percentile / 100.0)
    stop_fraction = 0.5 + 0.5 * (percentile / 100.0)

    xfine = np.linspace(x[0], x[-1], num_bins_fine,)
    ffine = np.interp(x=xfine, xp=x, fp=f)

    ffine = ffine / np.sum(ffine)
    cumffine = np.cumsum(ffine)

    imax = np.argmax(ffine)

    istart = int(imax)
    istop = int(imax)

    while True:
        if cumffine[istart] < start_fraction:
            break
        elif istart == 0:
            break
        else:
            istart -= 1

    while True:
        if cumffine[istop] > stop_fraction:
            break
        elif istop == num_bins_fine - 1:
            break
        else:
            istop += 1

    return xfine[istart], xfine[istop]


def full_width_half_maximum(x, f, oversample=137):
    assert len(x) == len(f)
    assert len(x) >= 3
    assert np.all(np.gradient(x) > 0.0)
    assert oversample >= 1
    num_bins_fine = len(x) * oversample

    xfine = np.linspace(x[0], x[-1], num_bins_fine,)
    # print("xfine", xfine[0], xfine[-1])

    ffine = np.interp(x=xfine, xp=x, fp=f)
    ffine = ffine / np.max(ffine)
    imax = np.argmax(ffine)

    istart = int(imax)
    istop = int(imax)
    while True:
        if ffine[istart] < 0.5:
            break
        elif istart == 0:
            break
        else:
            istart -= 1

    while True:
        if ffine[istop] < 0.5:
            break
        elif istop == num_bins_fine - 1:
            break
        else:
            istop += 1

    return xfine[istart], xfine[istop]
