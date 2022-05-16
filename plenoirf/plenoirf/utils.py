import numpy as np
import propagate_uncertainties as pru
import binning_utils
import datetime
import io
import tarfile
import scipy.interpolate


def cone_solid_angle(cone_radial_opening_angle_rad):
    cap_hight = 1.0 - np.cos(cone_radial_opening_angle_rad)
    return 2.0 * np.pi * cap_hight


def sr2squaredeg(solid_angle_sr):
    return solid_angle_sr * (180.0 / np.pi) ** 2


def squaredeg2sr(solid_angle_squaredeg):
    return solid_angle_squaredeg * (np.pi / 180) ** 2


def contains_same_bytes(path_a, path_b):
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        a_bytes = fa.read()
        b_bytes = fb.read()
        return a_bytes == b_bytes


def tar_append(tarout, file_name, file_bytes):
    with io.BytesIO() as buff:
        info = tarfile.TarInfo(file_name)
        info.size = buff.write(file_bytes)
        buff.seek(0)
        tarout.addfile(info, buff)


def ray_plane_x_y_intersection(support, direction, plane_z):
    direction = np.array(direction)
    support = np.array(support)
    direction_norm = direction / np.linalg.norm(direction)
    ray_parameter = -(support[2] - plane_z) / direction_norm[2]
    intersection = support + ray_parameter * direction_norm
    assert np.abs(intersection[2] - plane_z) < 1e-3
    return intersection


def power10space_bin_edges(binning, fine):
    assert fine > 0
    space = binning_utils.power10.space(
        start_decade=binning["start"]["decade"],
        start_bin=binning["start"]["bin"] * fine,
        stop_decade=binning["stop"]["decade"],
        stop_bin=binning["stop"]["bin"] * fine,
        num_bins_per_decade=binning["num_bins_per_decade"] * fine,
    )
    return space, len(space) - 1


def _divide_silent(numerator, denominator, default):
    valid = denominator != 0
    division = np.ones(shape=numerator.shape) * default
    division[valid] = numerator[valid] / denominator[valid]
    return division


_10s = 10
_1M = 60
_1h = _1M * 60
_1d = _1h * 24
_1w = _1d * 7
_1m = _1d * 30
_1y = 365 * _1d


def make_civil_times_points_in_quasi_logspace():
    """
    time-points from 1s to 100y in the civil steps of:
    s, m, h, d, week, Month, year, decade
    """

    times = []
    for _secs in np.arange(1, _10s, 1):
        times.append(_secs)
    for _10secs in np.arange(_10s, _1M, _10s):
        times.append(_10secs)
    for _mins in np.arange(_1M, _1h, _1M):
        times.append(_mins)
    for _hours in np.arange(_1h, _1d, _1h):
        times.append(_hours)
    for _days in np.arange(_1d, _1w, _1d):
        times.append(_days)
    for _weeks in np.arange(_1w, 4 * _1w, _1w):
        times.append(_weeks)
    for _months in np.arange(_1m, 12 * _1m, _1m):
        times.append(_months)
    for _years in np.arange(_1y, 10 * _1y, _1y):
        times.append(_years)
    for _decades in np.arange(10 * _1y, 100 * _1y, 10 * _1y):
        times.append(_decades)
    return times


def make_civil_time_str(time_s, format_seconds="{:f}"):
    try:
        years = int(time_s // _1y)
        tr = time_s - years * _1y

        days = int(tr // _1d)
        tr = tr - days * _1d

        hours = int(tr // _1h)
        tr = tr - hours * _1h

        minutes = int(tr // _1M)
        tr = tr - minutes * _1M

        s = ""
        if years:
            s += "{:d}y ".format(years)
        if days:
            s += "{:d}d ".format(days)
        if hours:
            s += "{:d}h ".format(hours)
        if minutes:
            s += "{:d}min ".format(minutes)
        if tr:
            s += (format_seconds + "s").format(tr)
        if s[-1] == " ":
            s = s[0:-1]
        return s
    except Exception as err:
        print(str(err))
        return (format_seconds + "s").format(time_s)


def find_closest_index_in_array_for_value(arr, val, max_rel_error=0.1):
    arr = np.array(arr)
    idx = np.argmin(np.abs(arr - val))
    assert np.abs(arr[idx] - val) < max_rel_error * val
    return idx


def latex_scientific(real, format_template="{:e}", nan_template="nan"):
    if real != real:
        return nan_template
    assert format_template.endswith("e}")
    s = format_template.format(real)
    pos_e = s.find("e")
    assert pos_e >= 0
    mantisse = s[0:pos_e]
    exponent = str(int(s[pos_e + 1 :]))
    out = mantisse + r"\times{}10^{" + exponent + r"}"
    return out


def apply_confusion_matrix(x, confusion_matrix, x_unc=None):
    """
    Parameters
    ----------
    x : 1D-array
            E.g. Effective acceptance vs. true energy.
    confusion_matrix : 2D-array
            Confusion between e.g. true and reco. energy.
            The rows are expected to be notmalized:
            CM[i, :] == 1.0
    """
    cm = confusion_matrix
    n = cm.shape[0]
    assert cm.shape[1] == n
    assert x.shape[0] == n

    # assert confusion matrix is normalized
    for i in range(n):
        s = np.sum(cm[i, :])
        assert np.abs(s - 1) < 1e-3 or s < 1e-3

    y = np.zeros(shape=(n))
    for r in range(n):
        for t in range(n):
            y[r] += cm[t, r] * x[t]

    return y


def make_confusion_matrix(
    ax0_key,
    ax0_values,
    ax0_bin_edges,
    ax1_key,
    ax1_values,
    ax1_bin_edges,
    ax0_weights=None,
    min_exposure_ax0=100,
    default_low_exposure=np.nan,
):
    assert len(ax0_values) == len(ax1_values)
    if ax0_weights is not None:
        assert len(ax0_values) == len(ax0_weights)

    num_bins_ax0 = len(ax0_bin_edges) - 1
    assert num_bins_ax0 >= 1

    num_bins_ax1 = len(ax1_bin_edges) - 1
    assert num_bins_ax1 >= 1

    counts = np.histogram2d(
        ax0_values,
        ax1_values,
        weights=ax0_weights,
        bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    cb_rel_unc, cb_abs_unc = estimate_rel_abs_uncertainty_in_counts(
        counts=counts
    )

    exposure_no_weights = np.histogram2d(
        ax0_values, ax1_values, bins=[ax0_bin_edges, ax1_bin_edges],
    )[0]

    counts_normalized_on_ax0 = counts.copy()
    cbn_abs_unc = cb_abs_unc.copy()
    for i0 in range(num_bins_ax0):
        if np.sum(exposure_no_weights[i0, :]) >= min_exposure_ax0:
            axsum = np.sum(counts[i0, :])
            counts_normalized_on_ax0[i0, :] /= axsum
            cbn_abs_unc[i0, :] /= axsum
        else:
            counts_normalized_on_ax0[i0, :] = (
                np.ones(num_bins_ax1) * default_low_exposure
            )

    return {
        "ax0_key": ax0_key,
        "ax1_key": ax1_key,
        "ax0_bin_edges": ax0_bin_edges,
        "ax1_bin_edges": ax1_bin_edges,
        "counts": counts,
        "counts_rel_unc": cb_rel_unc,
        "counts_abs_unc": cb_abs_unc,
        "counts_normalized_on_ax0": counts_normalized_on_ax0,
        "counts_normalized_on_ax0_abs_unc": cbn_abs_unc,
        "exposure_ax0_no_weights": np.sum(exposure_no_weights, axis=1),
        "exposure_ax0": np.sum(counts, axis=1),
        "min_exposure_ax0": min_exposure_ax0,
    }


def estimate_rel_abs_uncertainty_in_counts(counts):
    cb = counts
    assert np.all(cb >= 0)
    shape = cb.shape

    rel_unc = np.nan * np.ones(shape=shape)
    abs_unc = np.nan * np.ones(shape=shape)

    has_expo = cb > 0
    no_expo = cb == 0

    # frequency regime
    # ----------------
    rel_unc[has_expo] = 1.0 / np.sqrt(cb[has_expo])
    abs_unc[has_expo] = cb[has_expo] * rel_unc[has_expo]

    # no frequency regime, have to approximate
    # ----------------------------------------
    _num_bins_with_exposure = np.sum(has_expo)
    _num_bins = shape[0] * shape[1]

    pseudocount = np.sqrt(_num_bins_with_exposure / _num_bins)
    assert pseudocount <= 1.0

    if pseudocount == 0:
        # this can not be saved
        return rel_unc, abs_unc

    rel_unc[no_expo] = 1.0 / np.sqrt(pseudocount)
    abs_unc[no_expo] = pseudocount

    return rel_unc, abs_unc


def integrate_rate_where_known(dRdE, dRdE_au, E_edges):
    unknown = np.isnan(dRdE_au)

    _dRdE = dRdE.copy()
    _dRdE_au = dRdE_au.copy()

    _dRdE[unknown] = 0.0
    _dRdE_au[unknown] = 0.0

    T, T_au = pru.integrate(f=(_dRdE, _dRdE_au), x_bin_edges=E_edges)
    return T, T_au


def filter_particles_with_electric_charge(particles):
    out = {}
    for pk in particles:
        if np.abs(particles[pk]["electric_charge_qe"]) > 0:
            out[pk] = dict(particles[pk])
    return out
