#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import scipy
import os
import json_numpy
import sparse_numeric_table as spt
import sebastians_matplotlib_addons as seb
import glob
import tempfile
import tarfile
import skimage
import multiprocessing
from skimage.draw import circle as skimage_draw_circle


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

NUM_EVENTS_PER_PARTICLE = 10
MIN_NUM_CHERENKOV_PHOTONS = 200
MAX_CORE_DISTANCE = 400
TOMO_NUM_ITERATIONS = 200
NUM_THREADS = 6

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)
FIELD_OF_VIEW_RADIUS_DEG = 0.5 * np.rad2deg(
    light_field_geometry.sensor_plane2imaging_system.max_FoV_diameter
)


def get_airshower_uids_in_directory(directory, wildcard="*.tar"):
    raw_sensor_paths = glob.glob(os.path.join(directory, wildcard))
    filenames = [os.path.split(raw_path)[1] for raw_path in raw_sensor_paths]
    airshower_uid_strs = [os.path.splitext(fn)[0] for fn in filenames]
    airshower_uids = [int(uidstr) for uidstr in airshower_uid_strs]
    return set(airshower_uids)


def read_simulation_truth_for_tomography(
    raw_sensor_response_dir, uid, tomography_binning
):
    with tempfile.TemporaryDirectory() as tmpdir:
        uid_str = irf.unique.UID_FOTMAT_STR.format(uid)
        response_path = os.path.join(tmpdir, uid_str)
        response_tar_path = os.path.join(
            raw_sensor_response_dir, uid_str + ".tar"
        )
        with tarfile.open(response_tar_path, "r") as tf:
            tf.extractall(response_path)
            event = pl.Event(
                path=response_path, light_field_geometry=light_field_geometry,
            )
            simulation_truth = pl.Tomography.Image_Domain.Simulation_Truth.init(
                event=event, binning=tomography_binning,
            )
    return simulation_truth


def make_binning_from_ligh_field_geometry(light_field_geometry):
    lfg = light_field_geometry
    design = lfg.sensor_plane2imaging_system

    NUM_CAMERAS_ON_DIAGONAL = int(
        0.707 * design.max_FoV_diameter / design.pixel_FoV_hex_flat2flat
    )
    NUM_CAMERAS_ON_DIAGONAL = 2 * (NUM_CAMERAS_ON_DIAGONAL // 2)

    binning = pl.Tomography.Image_Domain.Binning.init(
        focal_length=design.expected_imaging_system_focal_length,
        cx_min=-0.5 * design.max_FoV_diameter,
        cx_max=+0.5 * design.max_FoV_diameter,
        number_cx_bins=NUM_CAMERAS_ON_DIAGONAL,
        cy_min=-0.5 * design.max_FoV_diameter,
        cy_max=+0.5 * design.max_FoV_diameter,
        number_cy_bins=NUM_CAMERAS_ON_DIAGONAL,
        obj_min=40 * design.expected_imaging_system_focal_length,
        obj_max=240 * design.expected_imaging_system_focal_length,
        number_obj_bins=NUM_CAMERAS_ON_DIAGONAL // 4,
    )
    return binning


binning = make_binning_from_ligh_field_geometry(
    light_field_geometry=light_field_geometry
)

pl.Tomography.Image_Domain.Binning.write(
    binning=binning,
    path=os.path.join(
        pa["out_dir"], "binning_for_tomography_in_image_domain.json"
    ),
)

system_matrix_path = os.path.join(
    pa["out_dir"], "system_matrix_for_tomography_in_image_domain.bin"
)

pool = multiprocessing.Pool(NUM_THREADS)

if not os.path.exists(system_matrix_path):

    jobs = pl.Tomography.System_Matrix.make_jobs(
        light_field_geometry=light_field_geometry,
        sen_x_bin_edges=binning["sen_x_bin_edges"],
        sen_y_bin_edges=binning["sen_y_bin_edges"],
        sen_z_bin_edges=binning["sen_z_bin_edges"],
        random_seed=sum_config["random_seed"],
        num_lixels_in_job=1000,
        num_samples_per_lixel=10,
    )

    results = pool.map(pl.Tomography.System_Matrix.run_job, jobs)

    sparse_system_matrix = pl.Tomography.System_Matrix.reduce_results(results)
    pl.Tomography.System_Matrix.write(
        sparse_system_matrix=sparse_system_matrix, path=system_matrix_path
    )

sparse_system_matrix = pl.Tomography.System_Matrix.read(
    path=system_matrix_path
)

tomo_psf = pl.Tomography.Image_Domain.Point_Spread_Function.init(
    sparse_system_matrix=sparse_system_matrix
)

for sk in irf_config["config"]["sites"]:
    for pk in irf_config["config"]["particles"]:
        sk_pk_dir = os.path.join(pa["run_dir"], "event_table", sk, pk)

        print("===", sk, pk, "===")

        print("- read event_table")
        event_table = spt.read(
            path=os.path.join(sk_pk_dir, "event_table.tar",),
            structure=irf.table.STRUCTURE,
        )

        print("- find events with full output")
        raw_sensor_response_dir = os.path.join(sk_pk_dir, "past_trigger.map")
        have_raw_sensor_response_uids = get_airshower_uids_in_directory(
            directory=raw_sensor_response_dir
        )

        print("- read list-of-photons")
        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(sk_pk_dir, "cherenkov.phs.loph.tar")
        )

        event_counter = 0
        while event_counter < NUM_EVENTS_PER_PARTICLE:

            try:
                event = next(run)
            except StopIteration:
                break

            uid, loph_record = event

            num_cherenkov_photons = len(loph_record["photons"]["channels"])
            if num_cherenkov_photons < MIN_NUM_CHERENKOV_PHOTONS:
                print(uid, "not enough photons")
                continue

            if uid not in have_raw_sensor_response_uids:
                print(uid, "no full output avaiable")
                continue

            event_primary = spt.cut_and_sort_table_on_indices(
                table=event_table,
                common_indices=np.array([uid]),
                level_keys=["primary"],
            )["primary"]
            event_core_distance = np.hypot(
                event_primary["magnet_cherenkov_pool_x_m"][0],
                event_primary["magnet_cherenkov_pool_y_m"][0],
            )

            if event_core_distance > MAX_CORE_DISTANCE:
                print(uid, "distance to core too large")
                continue

            event_counter += 1

            print(sk, pk, uid, event_core_distance)

            simulation_truth = read_simulation_truth_for_tomography(
                raw_sensor_response_dir=raw_sensor_response_dir,
                uid=uid,
                tomography_binning=binning,
            )

            result_dir = os.path.join(
                pa["out_dir"], sk, pk, irf.unique.UID_FOTMAT_STR.format(uid)
            )

            os.makedirs(result_dir, exist_ok=True)
            reconstruction_path = os.path.join(
                result_dir, "reconstruction.json"
            )

            if not os.path.exists(reconstruction_path):
                reconstruction = pl.Tomography.Image_Domain.Reconstruction.init(
                    light_field_geometry=light_field_geometry,
                    photon_lixel_ids=loph_record["photons"]["channels"],
                    binning=binning,
                )

                for i in range(TOMO_NUM_ITERATIONS):
                    reconstruction = pl.Tomography.Image_Domain.Reconstruction.iterate(
                        reconstruction=reconstruction,
                        point_spread_function=tomo_psf,
                    )

                with open(reconstruction_path, "wt") as f:
                    f.write(json_numpy.dumps(reconstruction))

            with open(reconstruction_path, "rt") as f:
                reconstruction = json_numpy.loads(f.read())


            ivolrec = pl.Tomography.Image_Domain.Binning.volume_intensity_as_cube(
                volume_intensity=reconstruction['reconstructed_volume_intensity'],
                binning=binning,
            )
            ivolrec = ivolrec / np.sum(ivolrec)

            ivoltru = pl.Tomography.Image_Domain.Binning.volume_intensity_as_cube(
                volume_intensity=simulation_truth['true_volume_intensity'],
                binning=binning,
            )
            ivoltru = ivoltru / np.sum(ivoltru)
            imax = np.max([np.max(ivolrec), np.max(ivoltru)])
            imin = np.min([np.min(ivolrec[ivolrec > 0]), np.min(ivoltru[ivoltru > 0])])


            image_slices_path = os.path.join(result_dir, "image_slices")

            if True: #not os.path.exists(image_slices_path):
                #os.makedirs(image_slices_path)


                """
                xyzIrec = pl.plot.xyzI.hist3D_to_xyzI(
                    xyz_hist=ivolrec,
                    x_bin_centers=np.rad2deg(binning["cx_bin_centers"]),
                    y_bin_centers=np.rad2deg(binning["cy_bin_centers"]),
                    z_bin_centers=binning["sen_z_bin_centers"],
                    threshold=imin,
                )
                xyzItru = pl.plot.xyzI.hist3D_to_xyzI(
                    xyz_hist=ivoltru,
                    x_bin_centers=np.rad2deg(binning["cx_bin_centers"]),
                    y_bin_centers=np.rad2deg(binning["cy_bin_centers"]),
                    z_bin_centers=binning["sen_z_bin_centers"],
                    threshold=imin,
                )

                fig3 = plt.figure()
                ax3 = fig3.add_subplot(111, projection='3d')

                pl.plot.xyzI.add2ax_xyzI(
                    ax=ax3,
                    xyzIs=xyzIrec,
                    color='b',
                    steps=32,
                    alpha_max=0.2,
                    ball_size=50.0,
                )

                pl.plot.xyzI.add2ax_xyzI(
                    ax=ax3,
                    xyzIs=xyzItru,
                    color='r',
                    steps=32,
                    alpha_max=0.2,
                    ball_size=50.0,
                )
                fig3.savefig(os.path.join(image_slices_path, "3d.png"))
                seb.close(fig3)
                """


                for iz in range(binning["number_sen_z_bins"]):


                    ONLY_LOWEST_SLICE_HAS_AXES = True
                    axes_style_lowest = {"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}
                    axes_style_upper = {"spines": [], "axes": [], "grid": False}

                    if iz + 1 == binning["number_sen_z_bins"]:
                        axes_style = axes_style_lowest
                    else:
                        axes_style = axes_style_upper

                    RRR = 720
                    fig = seb.figure({"rows": RRR, "cols": RRR, "fontsize": 1})
                    ax = seb.add_axes(
                        fig=fig,
                        span=[0.0, 0.0, 1., 1.],
                        style=axes_style
                    )
                    """
                    seb.ax_add_circle(
                        ax=ax,
                        x=0.0,
                        y=0.0,
                        r=FIELD_OF_VIEW_RADIUS_DEG,
                        linewidth=1.0,
                        linestyle="-",
                        color="k",
                        alpha=1,
                        num_steps=180,
                    )
                    """
                    colorimg = np.zeros(
                        shape=(
                            binning["number_cx_bins"],
                            binning["number_cy_bins"],
                            4,
                        ),
                        dtype=np.float
                    )
                    colorimg[:, :, 0] = (ivolrec[:, :, iz]/imax)
                    colorimg[:, :, 1] = (ivoltru[:, :, iz]/imax)
                    """
                    colorimg *= 255.0
                    ax.imshow(
                        X=colorimg.astype(np.uint8),
                        extent=(
                            np.rad2deg(binning["cx_bin_edges"][0]),
                            np.rad2deg(binning["cx_bin_edges"][-1]),
                            np.rad2deg(binning["cy_bin_edges"][0]),
                            np.rad2deg(binning["cy_bin_edges"][-1]),
                        ),
                    )
                    """
                    UU = 0.5
                    transform = np.array(
                        [
                            [1,0],
                            [0,1]
                        ]
                    )
                    seb.ax_pcolormesh_fill(
                        ax=ax,
                        x_bin_edges=np.rad2deg(binning["cx_bin_edges"]),
                        y_bin_edges=np.rad2deg(binning["cy_bin_edges"]),
                        intensity_rgba=colorimg,
                        transform=transform,
                        edgecolor='none',
                        linewidth=0.0,
                        threshold=0.1,
                        circle_radius=3.25,
                    )

                    ax.set_xlim([np.rad2deg(binning["cx_bin_edges"][0]), np.rad2deg(binning["cx_bin_edges"][-1])])
                    ax.set_ylim([np.rad2deg(binning["cy_bin_edges"][0]), np.rad2deg(binning["cy_bin_edges"][-1])])


                    """
                    seb.ax_add_grid_with_explicit_ticks(
                        ax=ax,
                        xticks=np.arange(-3, 4, 1),
                        yticks=np.arange(-3, 4, 1),
                        color="k",
                        linestyle="-",
                        linewidth=0.66,
                        alpha=0.33,
                    )
                    """
                    ax.set_aspect("equal")
                    fig.savefig(
                        os.path.join(image_slices_path, "grid{:03d}.png".format(iz)),
                        transparent=True,
                    )
                    seb.close(fig)

                SKEW = 2
                WWW = RRR + RRR//SKEW
                HHH = RRR//SKEW
                full = np.zeros(
                    shape=(binning["number_sen_z_bins"]*HHH, WWW, 4),
                    dtype=np.uint8,
                )
                for iz in range(binning["number_sen_z_bins"]):

                    gskimg = skimage.io.imread(
                        fname=os.path.join(image_slices_path, "grid{:03d}.png".format(iz)),
                    )
                    gskimg_wide = np.zeros(shape=(RRR, WWW , 4), dtype=np.uint8)
                    gskimg_wide[0:RRR, 0:RRR] = gskimg

                    matrix_skew = skimage.transform.FundamentalMatrixTransform(
                        np.array([
                            [1.0, 0.0, -RRR//SKEW],
                            [0.0, SKEW, 0],
                            [0.0, 0.0, 1.0]
                        ])
                    )
                    gskimg_wide_skew = skimage.transform.warp(
                        gskimg_wide,
                        inverse_map=matrix_skew,
                    )
                    gskimg_wide_skew_cut = gskimg_wide_skew[0:HHH,:]
                    skimage.io.imsave(
                        fname=os.path.join(image_slices_path, "skew_grid{:03d}.png".format(iz)),
                        arr=gskimg_wide_skew_cut,
                        check_contrast=False,
                    )

                    off = iz*HHH//4
                    xoff = 0
                    yoff = 0
                    full[xoff+off:HHH+xoff+off, yoff:WWW+yoff, :] += (255*gskimg_wide_skew_cut).astype(np.uint8)

                skimage.io.imsave(
                    fname=os.path.join(image_slices_path, "skew_grid_full.png"),
                    arr=full,
                    check_contrast=False,
                )
            break
        break
    break
