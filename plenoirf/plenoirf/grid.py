import numpy as np
import gzip
import os
import io
import tarfile
import shutil
import corsika_primary_wrapper as cpw
from . import table


def init(
    plenoscope_diameter=36,
    num_bins_radius=512,
):
    """
     num_bins_radius = 2
     _______^______
    /              -
    +-------+-------+-------+-------+-
    |       |       |       |       | |
    |       |       |       |       | |
    |       |       |       |       | |
    +-------+-------+-------+-------+ |
    |       |       |       |       |  > num_bins_radius = 2
    |       |       |       |       | |
    |       |       |(0,0)  |       | |
    +-------+-------+-------+-------+/
    |       |       |       |       |
    |       |       |       |       |
    |       |       |       |       |
    +-------+-------+-------+-------+ <-- bin edge
    |       |       |       |       |
    |       |       |       |   X <-- bin center
    |       |       |       |       |
    +-------+-------+-------+-------+

    """
    assert num_bins_radius > 0
    assert plenoscope_diameter > 0.0
    g = {}
    g["bin_width"] = plenoscope_diameter
    g["num_bins_radius"] = num_bins_radius
    g["xy_bin_edges"] = np.linspace(
        start=-g["bin_width"]*g["num_bins_radius"],
        stop=g["bin_width"]*g["num_bins_radius"],
        num=2*g["num_bins_radius"] + 1
    )
    g["num_bins_diameter"] = len(g["xy_bin_edges"]) - 1
    g["xy_bin_centers"] = .5*(g["xy_bin_edges"][:-1] + g["xy_bin_edges"][1:])
    g["total_area"] = (g["num_bins_diameter"]*g["bin_width"])**2
    return g


def _make_bunch_direction(cx, cy):
    d = np.zeros(shape=(cx.shape[0], 3))
    d[:, 0] = cx
    d[:, 1] = cy
    d[:, 2] = -1.0*np.sqrt(1.0 - cx**2 - cy**2)
    return d


def _normalize_rows_in_matrix(mat):
    return mat/np.sqrt(np.sum(mat**2, axis=1, keepdims=1))


def _make_angle_between(directions, direction):
    direction = direction/np.linalg.norm(direction)
    directions = _normalize_rows_in_matrix(mat=directions)
    return np.arccos(np.dot(directions, direction))


def cut_cherenkov_bunches_in_field_of_view(
    cherenkov_bunches,
    field_of_view_radius_deg,
    pointing_direction,
):
    bunch_directions = _make_bunch_direction(
        cx=cherenkov_bunches[:, cpw.ICX],
        cy=cherenkov_bunches[:, cpw.ICY]
    )
    bunch_incidents = -1.0*bunch_directions
    angle_bunch_pointing = _make_angle_between(
        directions=bunch_incidents,
        direction=pointing_direction
    )
    mask_inside_field_of_view = angle_bunch_pointing < np.deg2rad(
        field_of_view_radius_deg
    )
    return cherenkov_bunches[mask_inside_field_of_view, :]


def histogram2d_overflow_and_bin_idxs(
    x,
    y,
    weights,
    xy_bin_edges
):
    _xy_bin_edges = [-np.inf] + xy_bin_edges.tolist() + [np.inf]

    # histogram num photons, i.e. use bunchsize weights.
    grid_histogram_flow = np.histogram2d(
        x=x,
        y=y,
        bins=(_xy_bin_edges, _xy_bin_edges),
        weights=weights
    )[0]

    # cut out the inner grid, use outer rim to estimate under-, and overflow
    grid_histogram = grid_histogram_flow[1:-1, 1:-1]
    assert grid_histogram.shape[0] == len(xy_bin_edges) - 1
    assert grid_histogram.shape[1] == len(xy_bin_edges) - 1

    overflow = {}
    overflow["overflow_x"] = np.sum(grid_histogram_flow[-1, :])
    overflow["underflow_x"] = np.sum(grid_histogram_flow[0, :])
    overflow["overflow_y"] = np.sum(grid_histogram_flow[:, -1])
    overflow["underflow_y"] = np.sum(grid_histogram_flow[:, 0])

    # assignment
    x_bin_idxs = np.digitize(x, bins=xy_bin_edges)
    y_bin_idxs = np.digitize(y, bins=xy_bin_edges)

    return grid_histogram, overflow, x_bin_idxs, y_bin_idxs


def assign(
    cherenkov_bunches,
    field_of_view_radius_deg,
    pointing_direction,
    grid_geometry,
    grid_random_shift_x,
    grid_random_shift_y,
    grid_magnetic_deflection_shift_x,
    grid_magnetic_deflection_shift_y,
    threshold_num_photons,
):
    pgg = grid_geometry

    bunches_in_fov = cut_cherenkov_bunches_in_field_of_view(
        cherenkov_bunches=cherenkov_bunches,
        field_of_view_radius_deg=field_of_view_radius_deg,
        pointing_direction=pointing_direction,
    )

    # Supports
    # --------
    grid_shift_x = grid_random_shift_x - grid_magnetic_deflection_shift_x
    grid_shift_y = grid_random_shift_y - grid_magnetic_deflection_shift_y

    bunch_x_wrt_grid_m = cpw.CM2M*bunches_in_fov[:, cpw.IX] + grid_shift_x
    bunch_y_wrt_grid_m = cpw.CM2M*bunches_in_fov[:, cpw.IY] + grid_shift_y
    bunch_weight = bunches_in_fov[:, cpw.IBSIZE]

    (
        grid_histogram,
        grid_overflow,
        bunch_x_bin_idxs,
        bunch_y_bin_idxs
    ) = histogram2d_overflow_and_bin_idxs(
        x=bunch_x_wrt_grid_m,
        y=bunch_y_wrt_grid_m,
        xy_bin_edges=pgg["xy_bin_edges"],
        weights=bunch_weight
    )

    bin_idxs_above_threshold = np.where(grid_histogram > threshold_num_photons)
    num_bins_above_threshold = bin_idxs_above_threshold[0].shape[0]

    if num_bins_above_threshold == 0:
        choice = None
    else:
        _choice_bin = np.random.choice(np.arange(num_bins_above_threshold))
        bin_idx_x = bin_idxs_above_threshold[0][_choice_bin]
        bin_idx_y = bin_idxs_above_threshold[1][_choice_bin]
        num_photons_in_bin = grid_histogram[bin_idx_x, bin_idx_y]
        choice = {}
        choice["bin_idx_x"] = int(bin_idx_x)
        choice["bin_idx_y"] = int(bin_idx_y)
        choice["core_x_m"] = float(
            pgg["xy_bin_centers"][bin_idx_x] - grid_shift_x)
        choice["core_y_m"] = float(
            pgg["xy_bin_centers"][bin_idx_y] - grid_shift_y)
        match_bin_idx_x = bunch_x_bin_idxs - 1 == bin_idx_x
        match_bin_idx_y = bunch_y_bin_idxs - 1 == bin_idx_y
        match_bin = np.logical_and(match_bin_idx_x, match_bin_idx_y)
        num_photons_in_recovered_bin = np.sum(
            bunches_in_fov[match_bin, cpw.IBSIZE])
        abs_diff_num_photons = np.abs(
            num_photons_in_recovered_bin -
            num_photons_in_bin)
        if abs_diff_num_photons > 1e-2*num_photons_in_bin:
            msg = "".join([
                "num_photons_in_bin: {:E}\n".format(float(num_photons_in_bin)),
                "num_photons_in_recovered_bin: {:E}\n".format(float(
                    num_photons_in_recovered_bin)),
                "abs(diff): {:E}\n".format(abs_diff_num_photons),
                "bin_idx_x: {:d}\n".format(bin_idx_x),
                "bin_idx_y: {:d}\n".format(bin_idx_y),
                "sum(match_bin): {:d}\n".format(np.sum(match_bin)),
            ])
            assert False, msg
        choice["cherenkov_bunches"] = bunches_in_fov[match_bin, :].copy()
        choice["cherenkov_bunches"][:, cpw.IX] -= cpw.M2CM*choice["core_x_m"]
        choice["cherenkov_bunches"][:, cpw.IY] -= cpw.M2CM*choice["core_y_m"]

    out = {}
    out["random_choice"] = choice
    out["histogram"] = grid_histogram
    for overflow_key in grid_overflow:
        out[overflow_key] = grid_overflow[overflow_key]
    out["num_bins_above_threshold"] = num_bins_above_threshold
    return out


def histogram_to_bytes(img):
    img_f4 = img.astype('<f4')
    img_f4_flat_c = img_f4.flatten(order='c')
    img_f4_flat_c_bytes = img_f4_flat_c.tobytes()
    img_gzip_bytes = gzip.compress(img_f4_flat_c_bytes)
    return img_gzip_bytes


def bytes_to_histogram(img_bytes_gz):
    img_bytes = gzip.decompress(img_bytes_gz)
    arr = np.frombuffer(img_bytes, dtype='<f4')
    num_bins = arr.shape[0]
    num_bins_edge = int(np.sqrt(num_bins))
    assert num_bins_edge*num_bins_edge == num_bins
    return arr.reshape((num_bins_edge, num_bins_edge), order='c')


# histograms
# ----------
# A dict with the random_seed as key for the airshowers, containing the
# gzip-bytes to be read with bytes_to_histogram()

def read_all_histograms(path):
    grids = {}
    with tarfile.open(path, "r") as tarfin:
        for tarinfo in tarfin:
            idx = int(tarinfo.name[0:table.NUM_DIGITS_SEED])
            grids[idx] = tarfin.extractfile(tarinfo).read()
    return grids


def read_histograms(path, indices=None):
    if indices is None:
        return read_all_histograms(path)
    else:
        indices_set = set(indices)
        grids = {}
        with tarfile.open(path, "r") as tarfin:
            for tarinfo in tarfin:
                idx = int(tarinfo.name[0:table.NUM_DIGITS_SEED])
                if idx in indices_set:
                    grids[idx] = tarfin.extractfile(tarinfo).read()
        return grids


def write_histograms(path, grid_histograms):
    with tarfile.open(path+".tmp", "w") as tarfout:
        for idx in grid_histograms:
            filename = table.SEED_TEMPLATE_STR.format(seed=idx) + ".f4.gz"
            with io.BytesIO() as buff:
                info = tarfile.TarInfo(filename)
                info.size = buff.write(grid_histograms[idx])
                buff.seek(0)
                tarfout.addfile(info, buff)
    shutil.move(path+".tmp", path)


def reduce(list_of_grid_paths, out_path):
    with tarfile.open(out_path+".tmp", "w") as tarfout:
        for grid_path in list_of_grid_paths:
            with tarfile.open(grid_path, "r") as tarfin:
                for tarinfo in tarfin:
                    tarfout.addfile(
                        tarinfo=tarinfo,
                        fileobj=tarfin.extractfile(tarinfo))
    shutil.move(out_path+".tmp", out_path)
