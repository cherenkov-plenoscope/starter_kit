import numpy as np
import corsika_primary_wrapper as cpw
import gzip
import os
from . import table
import tarfile


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
    g = {
        "plenoscope_diameter": plenoscope_diameter,
        "num_bins_radius": num_bins_radius}
    g["xy_bin_edges"] = np.linspace(
        -g["plenoscope_diameter"]*g["num_bins_radius"],
        g["plenoscope_diameter"]*g["num_bins_radius"],
        2*g["num_bins_radius"] + 1)
    g["num_bins_diameter"] = len(g["xy_bin_edges"]) - 1
    g["xy_bin_centers"] = .5*(g["xy_bin_edges"][:-1] + g["xy_bin_edges"][1:])
    g["total_area"] = (g["num_bins_diameter"]*g["plenoscope_diameter"])**2
    return g


def _power2_bin_edges(power):
    be = np.geomspace(1, 2**power, power+1)
    beiz = np.zeros(shape=(be.shape[0] + 1))
    beiz[1:] = be
    return beiz


PH_BIN_EDGES = _power2_bin_edges(16)


def _make_bunch_direction(cx, cy):
    d = np.zeros(shape=(cx.shape[0], 3))
    d[:, 0] = cx
    d[:, 1] = cy
    d[:, 2] = -1.0*np.sqrt(1.0 - cx**2 - cy**2)
    return d


def _make_angle_between(directions, direction):
    # expect normalized
    return np.arccos(np.dot(directions, direction))


def assign(
    cherenkov_bunches,
    plenoscope_field_of_view_radius_deg,
    plenoscope_pointing_direction,
    plenoscope_grid_geometry,
    grid_random_shift_x,
    grid_random_shift_y,
    threshold_num_photons,
    FIELD_OF_VIEW_OVERHEAD=1.1,
):
    pgg = plenoscope_grid_geometry

    # Directions
    # ----------
    bunch_directions = _make_bunch_direction(
        cx=cherenkov_bunches[:, cpw.ICX],
        cy=cherenkov_bunches[:, cpw.ICY])
    bunch_incidents = -1.0*bunch_directions

    angle_bunch_pointing = _make_angle_between(
        directions=bunch_incidents,
        direction=plenoscope_pointing_direction)

    mask_inside_field_of_view = angle_bunch_pointing < np.deg2rad(
        plenoscope_field_of_view_radius_deg*FIELD_OF_VIEW_OVERHEAD)

    bunches_in_fov = cherenkov_bunches[mask_inside_field_of_view, :]

    # Supports
    # --------
    bunch_x_bin_idxs = np.digitize(
        cpw.CM2M*bunches_in_fov[:, cpw.IX] + grid_random_shift_x,
        bins=pgg["xy_bin_edges"])
    bunch_y_bin_idxs = np.digitize(
        cpw.CM2M*bunches_in_fov[:, cpw.IY] + grid_random_shift_y,
        bins=pgg["xy_bin_edges"])

    # Add under-, and overflow bin-edges
    _xy_bin_edges = [-np.inf] + pgg["xy_bin_edges"].tolist() + [np.inf]

    # histogram num photons, i.e. use bunchsize weights.
    grid_histogram_flow = np.histogram2d(
        x=cpw.CM2M*bunches_in_fov[:, cpw.IX] + grid_random_shift_x,
        y=cpw.CM2M*bunches_in_fov[:, cpw.IY] + grid_random_shift_y,
        bins=(_xy_bin_edges, _xy_bin_edges),
        weights=bunches_in_fov[:, cpw.IBSIZE])[0]

    # cut out the inner grid, use outer rim to estimate under-, and overflow
    grid_histogram = grid_histogram_flow[1:-1, 1:-1]
    assert grid_histogram.shape[0] == pgg["num_bins_diameter"]
    assert grid_histogram.shape[1] == pgg["num_bins_diameter"]

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
            pgg["xy_bin_centers"][bin_idx_x] - grid_random_shift_x)
        choice["core_y_m"] = float(
            pgg["xy_bin_centers"][bin_idx_y] - grid_random_shift_y)
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
    out["overflow_x"] = np.sum(grid_histogram_flow[-1, :])
    out["underflow_x"] = np.sum(grid_histogram_flow[0, :])
    out["overflow_y"] = np.sum(grid_histogram_flow[:, -1])
    out["underflow_y"] = np.sum(grid_histogram_flow[:, 0])
    out["intensity_histogram"] = np.histogram(
        grid_histogram.flatten(),
        bins=PH_BIN_EDGES)[0]
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


def reduce_histograms(
    feature_dir,
    wild_card='*_grid_images.tar',
    max_num_events_in_run=table.MAX_NUM_EVENTS_IN_RUN,
):
    run_ids = table._run_ids_in_dir(feature_dir, wild_card=wild_card)
    suffix = wild_card[1:]
    grids = {}
    for run_id in run_ids:
        tarpath = os.path.join(
            feature_dir,
            "{:06d}{:s}".format(run_id, suffix))
        with tarfile.open(tarpath, "r") as tarin:
            for tarinfo in tarin:
                airshower_id = int(tarinfo.name[0:6])
                unique_id = table.random_seed_based_on(
                    run_id=run_id,
                    airshower_id=airshower_id)
                grids[unique_id] = tarin.extractfile(tarinfo).read()
    return grids

