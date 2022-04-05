#!/usr/bin/python
import sys
import plenoirf as irf
import plenopy as pl
import scipy
import os
import json_numpy
import sebastians_matplotlib_addons as seb
import glob
import tempfile
import tarfile
import skimage


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

NUM_EVENTS_PER_PARTICLE = 3
MIN_NUM_CHERENKOV_PHOTONS = 500

os.makedirs(pa["out_dir"], exist_ok=True)

light_field_geometry = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
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
            simulation_truth = pl.tomography.image_domain.simulation_truth.init(
                event=event, binning=tomography_binning,
            )
    return simulation_truth


def make_binning_from_ligh_field_geometry(light_field_geometry):
    lfg = light_field_geometry
    design = lfg.sensor_plane2imaging_system

    NUM_CAMERAS_ON_DIAGONAL = int(
        design.max_FoV_diameter / design.pixel_FoV_hex_flat2flat
    )
    NUM_CAMERAS_ON_DIAGONAL = 2 * (NUM_CAMERAS_ON_DIAGONAL // 2)

    binning = pl.tomography.image_domain.binning.init(
        focal_length=design.expected_imaging_system_focal_length,
        cx_min=-0.5 * design.max_FoV_diameter,
        cx_max=+0.5 * design.max_FoV_diameter,
        number_cx_bins=NUM_CAMERAS_ON_DIAGONAL,
        cy_min=-0.5 * design.max_FoV_diameter,
        cy_max=+0.5 * design.max_FoV_diameter,
        number_cy_bins=NUM_CAMERAS_ON_DIAGONAL,
        obj_min=40 * design.expected_imaging_system_focal_length,
        obj_max=240 * design.expected_imaging_system_focal_length,
        number_obj_bins=NUM_CAMERAS_ON_DIAGONAL // 2,
    )
    return binning


binning = make_binning_from_ligh_field_geometry(
    light_field_geometry=light_field_geometry
)

pl.tomography.image_domain.binning.write(
    binning=binning,
    path=os.path.join(
        pa["out_dir"], "binning_for_tomography_in_image_domain.json"
    ),
)

system_matrix_path = os.path.join(
    pa["out_dir"], "system_matrix_for_tomography_in_image_domain.bin"
)

if not os.path.exists(system_matrix_path):

    jobs = pl.tomography.system_matrix.make_jobs(
        light_field_geometry=light_field_geometry,
        sen_x_bin_edges=binning["sen_x_bin_edges"],
        sen_y_bin_edges=binning["sen_y_bin_edges"],
        sen_z_bin_edges=binning["sen_z_bin_edges"],
        random_seed=sum_config["random_seed"],
        num_lixels_in_job=1000,
        num_samples_per_lixel=10,
    )

    results = []
    for ijob, job in enumerate(jobs):
        print("system_matrix: job", ijob)
        job_result = pl.tomography.system_matrix.run_job(job)
        results.append(job_result)

    sparse_system_matrix = pl.tomography.system_matrix.reduce_results(results)
    pl.tomography.system_matrix.write(
        sparse_system_matrix=sparse_system_matrix, path=system_matrix_path
    )

sparse_system_matrix = pl.tomography.system_matrix.read(
    path=system_matrix_path
)

tomo_psf = pl.tomography.image_domain.point_spread_function.init(
    sparse_system_matrix=sparse_system_matrix
)

for sk in irf_config["config"]["sites"]:
    for pk in ["proton", "helium"]:
        sk_pk_dir = os.path.join(pa["run_dir"], "event_table", sk, pk)

        raw_sensor_response_dir = os.path.join(sk_pk_dir, "past_trigger.map")
        have_raw_sensor_response_uids = get_airshower_uids_in_directory(
            directory=raw_sensor_response_dir
        )

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
                continue

            if uid not in have_raw_sensor_response_uids:
                continue

            event_counter += 1

            print(sk, pk, uid)

            simulation_truth = read_simulation_truth_for_tomography(
                raw_sensor_response_dir=raw_sensor_response_dir,
                uid=uid,
                tomography_binning=binning,
            )

            result_dir = os.path.join(
                pa["out_dir"], sk, pk, irf.unique.UID_FOTMAT_STR.format(uid)
            )

            if os.path.exists(result_dir):
                continue

            os.makedirs(result_dir, exist_ok=True)
            reconstruction_path = os.path.join(
                result_dir, "reconstruction.json"
            )

            if not os.path.exists(reconstruction_path):
                reconstruction = pl.tomography.image_domain.reconstruction.init(
                    light_field_geometry=light_field_geometry,
                    photon_lixel_ids=loph_record["photons"]["channels"],
                    binning=binning,
                )

                for i in range(25):
                    reconstruction = pl.tomography.image_domain.reconstruction.iterate(
                        reconstruction=reconstruction,
                        point_spread_function=tomo_psf,
                    )

                with open(reconstruction_path, "wt") as f:
                    f.write(json_numpy.dumps(reconstruction))

            with open(reconstruction_path, "rt") as f:
                reconstruction = json_numpy.loads(f.read())

            pl.tomography.image_domain.save_imgae_slice_stack(
                binning=binning,
                reconstruction=reconstruction,
                simulation_truth=simulation_truth,
                out_dir=result_dir,
                sqrt_intensity=False,
                event_info_repr=None,
            )

            # plot_of_awesome
            # ---------------
            # uid 000008000100

            """
            ivolrec = pl.tomography.image_domain.reconstructed_volume_intensity_as_cube(
                reconstructed_volume_intensity=reconstruction['reconstructed_volume_intensity'],
                binning=binning,
            )

            ivoltru = pl.tomography.image_domain.reconstructed_volume_intensity_as_cube(
                reconstructed_volume_intensity=simulation_truth['true_volume_intensity'],
                binning=binning,
            )

            xyzIrec = pl.plot.xyzI.hist3D_to_xyzI(
                xyz_hist=ivolrec,
                x_bin_centers=binning["sen_x_bin_centers"],
                y_bin_centers=binning["sen_y_bin_centers"],
                z_bin_centers=binning["sen_z_bin_centers"],
                threshold=1/255
            )
            xyzItru = pl.plot.xyzI.hist3D_to_xyzI(
                xyz_hist=ivoltru,
                x_bin_centers=binning["sen_x_bin_centers"],
                y_bin_centers=binning["sen_y_bin_centers"],
                z_bin_centers=binning["sen_z_bin_centers"],
                threshold=1/255
            )
            fig = seb.figure({"rows": 2560, "cols": 1600, "fontsize": 1})
            ax3d = fig.add_subplot(111, projection='3d')

            pl.plot.xyzI.add2ax_xyzI(
                ax=ax3d,
                xyzIs=xyzItru,
                color='r',
                steps=32,
                alpha_max=0.2,
                ball_size=100.0,
            )
            pl.plot.xyzI.add2ax_xyzI(
                ax=ax3d,
                xyzIs=xyzIrec,
                color='b',
                steps=32,
                alpha_max=0.2,
                ball_size=100.0,
            )
            ax3d.view_init(elev=10, azim=32)
            fig.savefig(os.path.join(result_dir, "vol.jpg"))
            seb.close(fig)



            images = []

            for iz in range(binning["number_sen_z_bins"]):
                img_z_rec = pl.plot.slices.matrix_2_rgb_image(
                    matrix=ivolrec[:, :, iz],
                    color_channel=1,
                    I_vol_min1=np.min(ivolrec),
                    I_vol_max1=np.max(ivolrec),
                )
                img_z_tru = pl.plot.slices.matrix_2_rgb_image(
                    matrix=ivoltru[:, :, iz],
                    color_channel=0,
                    I_vol_min1=np.min(ivoltru),
                    I_vol_max1=np.max(ivoltru),
                )
                img_z = img_z_rec + img_z_tru
                images.append(img_z)
            images = np.array(images)
            images *= 255
            images = images.astype(np.uint8)

            for iz in range(binning["number_sen_z_bins"]):

                skimage.io.imsave(
                    fname=os.path.join(result_dir, "{:03d}.png".format(iz)),
                    arr=images[iz],
                )

            fig = seb.figure(seb.FIGURE_16_9)
            ax = seb.add_axes(
                fig=fig,
                span=[0.1, 0.1, 0.8, 0.8],
                style={"spines": ["left", "bottom"], "axes": ["x", "y"], "grid": False}
            )
            step = 0.1
            for iz in range(binning["number_sen_z_bins"]):
                ax.pcolormesh(
                    iz*step + binning["sen_x_bin_edges"],
                    iz*step + binning["sen_y_bin_edges"],
                    ivolrec[:, :, iz],
                    alpha=0.5, vmin=0, vmax=1, cmap="Reds"
                )
                ax.pcolormesh(
                    iz*step + binning["sen_x_bin_edges"],
                    iz*step + binning["sen_y_bin_edges"],
                    ivoltru[:, :, iz],
                    alpha=0.1, vmin=0, vmax=1, cmap="Blues"
                )

            ax.set_aspect("equal")
            fig.savefig(os.path.join(result_dir, "stack.jpg"))
            seb.close(fig)
            """
