import plenoirf
import corsika_primary_wrapper as cpw
import numpy as np
import pytest


PLENOSCOPE_DIAMETER = 71.0
NUM_BINS_RADIUS = 512

PLENOSCOPE_GRID_GEOMETRY = plenoirf.grid.init_geometry(
    instrument_aperture_outer_diameter=PLENOSCOPE_DIAMETER,
    bin_width_overhead=1.0,
    instrument_field_of_view_outer_radius_deg=3.25,
    instrument_pointing_direction=[0, 0, 1],
    field_of_view_overhead=1.1,
    num_bins_radius=NUM_BINS_RADIUS,
)


def make_cherenkov_bunches(
    cx_deg,
    cx_std_deg,
    cy_deg,
    cy_std_deg,
    x_m,
    x_std_m,
    y_m,
    y_std_m,
    num_bunches,
):
    cherenkov_bunches = np.zeros(shape=(num_bunches, 8))
    cherenkov_bunches[:, cpw.IX] = np.random.normal(
        loc=x_m * cpw.M2CM, scale=x_std_m * cpw.M2CM, size=num_bunches
    )
    cherenkov_bunches[:, cpw.IY] = np.random.normal(
        loc=y_m * cpw.M2CM, scale=y_std_m * cpw.M2CM, size=num_bunches
    )
    cherenkov_bunches[:, cpw.ICX] = np.random.normal(
        loc=np.deg2rad(cx_deg), scale=np.deg2rad(cx_std_deg), size=num_bunches
    )
    cherenkov_bunches[:, cpw.ICY] = np.random.normal(
        loc=np.deg2rad(cy_deg), scale=np.deg2rad(cy_std_deg), size=num_bunches
    )
    cherenkov_bunches[:, cpw.ITIME] = np.random.normal(
        loc=100e-6, scale=10e-9, size=num_bunches
    )
    cherenkov_bunches[:, cpw.IZEM] = np.random.uniform(
        low=1e3 * cpw.M2CM, high=1e4 * cpw.M2CM, size=num_bunches
    )
    cherenkov_bunches[:, cpw.IBSIZE] = np.random.uniform(
        low=0.9, high=1.0, size=num_bunches
    )
    cherenkov_bunches[:, cpw.IWVL] = np.random.uniform(
        low=250e-9, high=700e-9, size=num_bunches
    )
    return cherenkov_bunches


def test_normalize_matrix_rows():
    np.random.seed(0)
    mat = np.zeros(shape=(100, 3))
    mat[:, 0] = np.random.uniform(size=100)
    mat[:, 1] = np.random.uniform(size=100)
    mat[:, 2] = np.random.uniform(size=100)
    for row in mat:
        assert np.abs(np.linalg.norm(row) - 1) > 1e-6
    norm_mat = plenoirf.grid._normalize_rows_in_matrix(mat=mat)
    for row in norm_mat:
        assert np.abs(np.linalg.norm(row) - 1) < 1e-6


def test_grid_assign_head_on():
    np.random.seed(0)
    cherenkov_bunches = make_cherenkov_bunches(
        cx_deg=0.0,
        cx_std_deg=1.0,
        cy_deg=0.0,
        cy_std_deg=1.0,
        x_m=0.0,
        x_std_m=100.0,
        y_m=0.0,
        y_std_m=100.0,
        num_bunches=10 * 1000,
    )
    result = plenoirf.grid.assign(
        cherenkov_bunches=cherenkov_bunches,
        grid_geometry=PLENOSCOPE_GRID_GEOMETRY,
        shift_x=0.0,
        shift_y=0.0,
        threshold_num_photons=50,
        bin_idxs_limitation=None,
    )
    assert result["random_choice"] is not None
    assert result["num_bins_above_threshold"] > 30
    assert (
        NUM_BINS_RADIUS - 2
        < result["random_choice"]["bin_idx_x"]
        < NUM_BINS_RADIUS + 2
    )
    assert (
        NUM_BINS_RADIUS - 2
        < result["random_choice"]["bin_idx_y"]
        < NUM_BINS_RADIUS + 2
    )


def test_shower_cx_moves_out_of_fov():
    np.random.seed(0)
    expectation = {
        0.0: 30,
        3.25: 10,
        6.5: 0,
    }
    for shower_cx_deg in expectation:
        cherenkov_bunches = make_cherenkov_bunches(
            cx_deg=shower_cx_deg,
            cx_std_deg=1.0,
            cy_deg=0.0,
            cy_std_deg=1.0,
            x_m=0.0,
            x_std_m=100.0,
            y_m=0.0,
            y_std_m=100.0,
            num_bunches=10 * 1000,
        )
        result = plenoirf.grid.assign(
            cherenkov_bunches=cherenkov_bunches,
            grid_geometry=PLENOSCOPE_GRID_GEOMETRY,
            shift_x=0.0,
            shift_y=0.0,
            threshold_num_photons=50,
            bin_idxs_limitation=None,
        )
        assert result["num_bins_above_threshold"] >= expectation[shower_cx_deg]


def test_shower_size_increases():
    np.random.seed(0)
    expectation = {
        1e3: 0,
        1e4: 30,
        1e5: 60,
    }
    for shower_size in expectation:
        cherenkov_bunches = make_cherenkov_bunches(
            cx_deg=0.0,
            cx_std_deg=1.0,
            cy_deg=0.0,
            cy_std_deg=1.0,
            x_m=0.0,
            x_std_m=100.0,
            y_m=0.0,
            y_std_m=100.0,
            num_bunches=int(shower_size),
        )
        result = plenoirf.grid.assign(
            cherenkov_bunches=cherenkov_bunches,
            grid_geometry=PLENOSCOPE_GRID_GEOMETRY,
            shift_x=0.0,
            shift_y=0.0,
            threshold_num_photons=50,
            bin_idxs_limitation=None,
        )
        assert result["num_bins_above_threshold"] >= expectation[shower_size]


def test_shower_x_moves_not_counteracted():
    """
    scenario 2
    ----------
                                 ccc                    CHERENKOV-POOL

                                --|--                   PLENSOCOPE
                                x = 0

    --|-------------|-------------|-------------|----   GRID
    x = 0          1e3           2e3           3e3

    --|-------------|-------------|-------------|----   OBSERVATION-LEVEL
    x = 0          1e3           2e3           3e3

    """
    np.random.seed(0)
    scenarios = {
        1: {"core_wrt_obslvl": 0e3, "num_bins_above_threshold": 2},
        2: {"core_wrt_obslvl": 1e3, "num_bins_above_threshold": 2},
        3: {"core_wrt_obslvl": 2e3, "num_bins_above_threshold": 2},
        4: {"core_wrt_obslvl": 3e3, "num_bins_above_threshold": 2},
    }
    for s in scenarios:
        cherenkov_bunches = make_cherenkov_bunches(
            cx_deg=0.0,
            cx_std_deg=1.0,
            cy_deg=0.0,
            cy_std_deg=1.0,
            x_m=scenarios[s]["core_wrt_obslvl"],
            x_std_m=10.0,
            y_m=0.0,
            y_std_m=10.0,
            num_bunches=10 * 1000,
        )
        result = plenoirf.grid.assign(
            cherenkov_bunches=cherenkov_bunches,
            grid_geometry=PLENOSCOPE_GRID_GEOMETRY,
            shift_x=0.0,
            shift_y=0.0,
            threshold_num_photons=50,
            bin_idxs_limitation=None,
        )
        assert (
            result["num_bins_above_threshold"]
            >= scenarios[s]["num_bins_above_threshold"]
        )

        cherenkov_wrt_plenoscope = result["random_choice"]["cherenkov_bunches"]
        x_wrt_plenoscope_m = cherenkov_wrt_plenoscope[:, cpw.IX] * cpw.CM2M
        y_wrt_plenoscope_m = cherenkov_wrt_plenoscope[:, cpw.IY] * cpw.CM2M

        median_x_wrt_plenoscope_m = np.median(x_wrt_plenoscope_m)
        median_y_wrt_plenoscope_m = np.median(y_wrt_plenoscope_m)

        assert np.abs(median_x_wrt_plenoscope_m) < 1e2
        assert np.abs(median_y_wrt_plenoscope_m) < 1e2

        core_x_wrt_plenoscope = result["random_choice"]["core_x_m"]
        core_y_wrt_plenoscope = result["random_choice"]["core_y_m"]
        assert (
            np.abs(core_x_wrt_plenoscope - scenarios[s]["core_wrt_obslvl"])
            < 1e2
        )
        assert np.abs(core_y_wrt_plenoscope) < 1e2

        assert 510 < result["random_choice"]["bin_idx_x"] < 560
        assert 510 < result["random_choice"]["bin_idx_y"] < 514


def test_shower_x_moves_but_counteracted():
    """
    scenario 2
    ----------
                                 ccc                    CHERENKOV-POOL

                                --|--                   PLENSOCOPE
                                x = 0

                                --|-------------|----   GRID
                                x = 0          1e3

    --|-------------|-------------|-------------|----   OBSERVATION-LEVEL
    x = 0          1e3           2e3           3e3

    """
    np.random.seed(0)
    scenarios = {
        1: {"core_wrt_obslvl": 0e3, "num_bins_above_threshold": 2},
        2: {"core_wrt_obslvl": 1e3, "num_bins_above_threshold": 2},
        3: {"core_wrt_obslvl": 2e3, "num_bins_above_threshold": 2},
        4: {"core_wrt_obslvl": 3e3, "num_bins_above_threshold": 2},
    }
    for s in scenarios:
        cherenkov_bunches = make_cherenkov_bunches(
            cx_deg=0.0,
            cx_std_deg=1.0,
            cy_deg=0.0,
            cy_std_deg=1.0,
            x_m=scenarios[s]["core_wrt_obslvl"],
            x_std_m=10.0,
            y_m=0.0,
            y_std_m=10.0,
            num_bunches=10 * 1000,
        )
        result = plenoirf.grid.assign(
            cherenkov_bunches=cherenkov_bunches,
            grid_geometry=PLENOSCOPE_GRID_GEOMETRY,
            shift_x=-1.0 * scenarios[s]["core_wrt_obslvl"],
            shift_y=0.0,
            threshold_num_photons=50,
            bin_idxs_limitation=None,
        )
        assert (
            result["num_bins_above_threshold"]
            >= scenarios[s]["num_bins_above_threshold"]
        )

        cherenkov_wrt_plenoscope = result["random_choice"]["cherenkov_bunches"]
        x_wrt_plenoscope_m = cherenkov_wrt_plenoscope[:, cpw.IX] * cpw.CM2M
        y_wrt_plenoscope_m = cherenkov_wrt_plenoscope[:, cpw.IY] * cpw.CM2M

        median_x_wrt_plenoscope_m = np.median(x_wrt_plenoscope_m)
        median_y_wrt_plenoscope_m = np.median(y_wrt_plenoscope_m)

        core_x_wrt_plenoscope = result["random_choice"]["core_x_m"]
        core_y_wrt_plenoscope = result["random_choice"]["core_y_m"]
        assert (
            np.abs(core_x_wrt_plenoscope - scenarios[s]["core_wrt_obslvl"])
            < 1e2
        )
        assert np.abs(core_y_wrt_plenoscope) < 1e2

        assert 510 < result["random_choice"]["bin_idx_x"] < 514
        assert 510 < result["random_choice"]["bin_idx_y"] < 514
