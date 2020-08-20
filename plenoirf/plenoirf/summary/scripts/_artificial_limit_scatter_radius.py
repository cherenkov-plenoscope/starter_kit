import plenoirf
import sparse_numeric_table as snt
import sys
import os
import shutil
import glob
import multiprocessing
import pandas

assert len(sys.argv) == 3

input_run_path = sys.argv[1]
output_run_path = sys.argv[2]

# the maximum scatter-radii from Sebastian's phd-thesis VS. energy.
phd_scatter_limits = {
    "gamma": {
        "energy_GeV": [0.23, 0.8, 3.0, 35, 81, 432, 1000],
        "max_scatter_radius_m": [150, 150, 460, 1100, 1235, 1410, 1660],
    },
    "electron": {
        "energy_GeV": [0.23, 1.0, 10, 100, 1000],
        "max_scatter_radius_m": [150, 150, 500, 1100, 2600],
    },
    "proton": {
        "energy_GeV": [5.0, 25, 250, 1000],
        "max_scatter_radius_m": [200, 350, 700, 1250],
    },
}

# Helium was not used in the phd-thesis
phd_scatter_limits["helium"] = phd_scatter_limits["proton"].copy()

assert not os.path.exists(output_run_path)
shutil.copytree(src=input_run_path, dst=output_run_path)

irf_config = plenoirf.summary.read_instrument_response_config(
    run_dir=input_run_path
)


def _estimate_num_grid_bins_with_scatter_limit(job):
    return estimate_num_grid_bins_with_scatter_limit(
        idx=job["idx"],
        _grid_intensity=job["_grid_intensity"],
        _grid=job["_grid"],
        _energy=job["_energy"],
        phd_scatter_limits=job["phd_scatter_limits"],
        irf_config=job["irf_config"],
    )


def estimate_num_grid_bins_with_scatter_limit(
    idx, _grid_intensity, _grid, _energy, phd_scatter_limits, irf_config,
):
    grid_bin_xy_width = irf_config["grid_geometry"]["plenoscope_diameter"]
    grid_xy_bin_centers = irf_config["grid_geometry"]["xy_bin_centers"]
    grid_num_bins_radius = irf_config["grid_geometry"]["num_bins_radius"]
    grid_num_bins_diameter = irf_config["grid_geometry"]["num_bins_diameter"]
    grid_threshold_num_photons = irf_config["config"]["grid"][
        "threshold_num_photons"
    ]

    # undo random-shift
    x_center_bin = 1 if _grid["random_shift_x_m"] > 0 else -1
    y_center_bin = 1 if _grid["random_shift_y_m"] > 0 else -1

    # max scatter-radius
    _scatter_radius_m = np.interp(
        x=_energy,
        xp=phd_scatter_limits["energy_GeV"],
        fp=phd_scatter_limits["max_scatter_radius_m"],
    )
    scatter_radius_bins = int(np.round(_scatter_radius_m / grid_bin_xy_width))

    # range of bins to search for
    x_bin_range = (
        grid_num_bins_radius
        + x_center_bin
        + np.arange(-scatter_radius_bins, scatter_radius_bins + 1)
    )
    y_bin_range = (
        grid_num_bins_radius
        + y_center_bin
        + np.arange(-scatter_radius_bins, scatter_radius_bins + 1)
    )

    # load grid intensity
    grid = plenoirf.grid.bytes_to_histogram(img_bytes_gz=_grid_intensity)

    num_bins_thrown = 0
    num_bins_above_threshold = 0
    for x_bin in x_bin_range:
        for y_bin in y_bin_range:
            if x_bin < 0 or x_bin >= grid_num_bins_diameter:
                continue
            if y_bin < 0 or y_bin >= grid_num_bins_diameter:
                continue
            rel_x_bin = x_bin - grid_num_bins_radius
            rel_y_bin = y_bin - grid_num_bins_radius
            if rel_x_bin ** 2 + rel_y_bin ** 2 >= scatter_radius_bins ** 2:
                continue
            num_bins_thrown += 1
            if grid[x_bin, y_bin] >= grid_threshold_num_photons:
                num_bins_above_threshold += 1

    return {
        "idx": idx,
        "scatter_num_bins_thrown": num_bins_thrown,
        "scatter_num_bins_above_threshold": num_bins_above_threshold,
    }


grid_bin_xy_width = irf_config["grid_geometry"]["plenoscope_diameter"]

for site_key in irf_config["config"]["sites"]:
    for particle_key in irf_config["config"]["particles"]:

        airshower_table = snt.read(
            path=os.path.join(
                input_run_path,
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            structure=plenoirf.table.STRUCTURE,
        )

        d_grid_intensities = plenoirf.grid.read_histograms(
            path=os.path.join(
                input_run_path,
                "event_table",
                site_key,
                particle_key,
                "grid.tar",
            )
        )
        d_grid = {}
        for ii in range(airshower_table["grid"]["idx"].shape[0]):
            d_grid[airshower_table["grid"]["idx"][ii]] = airshower_table[
                "grid"
            ][ii]
        d_energy = {}
        for ii in range(airshower_table["primary"]["idx"].shape[0]):
            d_energy[airshower_table["primary"]["idx"][ii]] = airshower_table[
                "primary"
            ]["energy_GeV"][ii]

        jobs = []
        for idx in airshower_table["primary"]["idx"]:
            job = {}
            job["idx"] = idx
            job["_grid_intensity"] = d_grid_intensities[idx]
            job["_grid"] = d_grid[idx]
            job["_energy"] = d_energy[idx]
            job["phd_scatter_limits"] = phd_scatter_limits[particle_key]
            job["irf_config"] = irf_config
            jobs.append(job)

        print(site_key, particle_key)
        pool = multiprocessing.Pool(8)
        result = pool.map(_estimate_num_grid_bins_with_scatter_limit, jobs)

        rr = pandas.merge(
            pandas.DataFrame(airshower_table["grid"]),
            pandas.DataFrame(result),
            on=["idx"],
        ).to_records(index=False)

        grid_out = {}
        grid_out["idx"] = rr["idx"]
        for column_key in plenoirf.table.STRUCTURE["grid"]:
            grid_out[column_key] = rr[column_key]

        grid_out["num_bins_radius"] = np.round(
            0.5 * np.sqrt(rr["scatter_num_bins_thrown"])
        ).astype(np.int64)
        grid_out["area_thrown_m2"] = (
            rr["scatter_num_bins_thrown"] * grid_bin_xy_width ** 2
        )
        grid_out["num_bins_above_threshold"] = rr[
            "scatter_num_bins_above_threshold"
        ]

        grid_out_recarray = pandas.DataFrame(grid_out).to_records(index=False)
        airshower_table["grid"] = grid_out_recarray

        snt.write(
            path=os.path.join(
                output_run_path,
                "event_table",
                site_key,
                particle_key,
                "event_table.tar",
            ),
            table=airshower_table,
            structure=plenoirf.table.STRUCTURE,
        )
