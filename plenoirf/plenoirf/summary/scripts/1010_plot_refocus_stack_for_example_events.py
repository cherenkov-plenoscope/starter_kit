#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import plenopy as pl
import gamma_ray_reconstruction as gamrec
import sebastians_matplotlib_addons as seb
import json_numpy

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
seb.matplotlib.rcParams.update(sum_config["plot"]["matplotlib"])

passing_trigger = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)

lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)

SAMPLE = {
    "bin_edges_pe": np.geomspace(1e2, 1e5, 5),
    "count": 5 * np.ones(4),
}


def table_to_dict(ta):
    out = {}
    for col in ta:
        out[col] = recarray_to_dict(ta[col])
    return out


def recarray_to_dict(ra):
    out = {}
    for key in ra.dtype.names:
        out[key] = ra[key]
    return out


def counter_init(sample):
    return np.zeros(len(sample["count"]), dtype=np.int)


def counter_not_full(counter, sample):
    for i in range(len(counter)):
        if counter[i] < sample["count"][i]:
            return True
    return False


def counter_can_add(counter, pe, sample):
    b = np.digitize(x=pe, bins=sample["bin_edges_pe"])
    if b == 0:
        return False
    if b == len(sample["bin_edges_pe"]):
        return False
    bix = b - 1

    if counter[bix] < sample["count"][bix]:
        return True
    else:
        return False


def counter_add(counter, pe, sample):
    b = np.digitize(x=pe, bins=sample["bin_edges_pe"])
    assert b > 0
    assert b < len(sample["bin_edges_pe"])
    bix = b - 1
    counter[bix] += 1
    return counter


NUM_EVENTS_PER_PARTICLE = 10
MIN_NUM_PHOTONS = 1000
colormap = "inferno"

depths = np.geomspace(2.5e3, 25e3, 12)
number_depths = len(depths)

image_rays = pl.image.ImageRays(light_field_geometry=lfg)

# SITES = irf_config["config"]["sites"]
SITES = ["namibia"]

# PARTICLES= irf_config["config"]["particles"]
PARTICLES = ["gamma", "proton", "helium"]

for sk in SITES:
    sk_dir = os.path.join(pa["out_dir"], sk)
    os.makedirs(sk_dir, exist_ok=True)
    for pk in PARTICLES:
        pk_dir = os.path.join(sk_dir, pk)
        os.makedirs(pk_dir, exist_ok=True)

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        common_idx = spt.intersection(
            [passing_trigger[sk][pk]["idx"], passing_quality[sk][pk]["idx"]]
        )
        events_truth = spt.cut_and_sort_table_on_indices(
            event_table,
            common_indices=common_idx,
            level_keys=[
                "primary",
                "cherenkovsize",
                "grid",
                "cherenkovpool",
                "cherenkovsizepart",
                "cherenkovpoolpart",
                "core",
                "trigger",
                "pasttrigger",
                "cherenkovclassification",
            ],
        )

        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(
                pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar"
            )
        )

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        counter = counter_init(SAMPLE)
        while counter_not_full(counter, SAMPLE):
            try:
                event = next(run)
            except StopIteration:
                break

            airshower_id, loph_record = event

            # mandatory
            # ---------
            if airshower_id not in passing_trigger[sk][pk]["idx"]:
                continue

            if airshower_id not in passing_quality[sk][pk]["idx"]:
                continue

            # optional for cherry picking
            # ---------------------------
            num_pe = len(loph_record["photons"]["arrival_time_slices"])
            if not counter_can_add(counter, num_pe, SAMPLE):
                continue

            event_cx = np.median(
                lfg.cx_mean[loph_record["photons"]["channels"]]
            )
            event_cy = np.median(
                lfg.cy_mean[loph_record["photons"]["channels"]]
            )
            event_off_deg = np.rad2deg(np.hypot(event_cx, event_cy))
            if event_off_deg > 2.5:
                continue

            counter = counter_add(counter, num_pe, SAMPLE)

            event_truth = spt.cut_and_sort_table_on_indices(
                events_truth, common_indices=np.array([airshower_id]),
            )
            tabpath = os.path.join(
                pk_dir, "{:s}_{:012d}.json".format(pk, airshower_id)
            )
            json_numpy.write(
                path=tabpath, out_dict=table_to_dict(event_truth), indent=4,
            )

            # prepare image intensities
            # -------------------------
            image_stack = np.zeros(shape=(number_depths, lfg.number_pixel))

            for dek in range(number_depths):
                depth = depths[dek]
                (
                    pixel_indicies,
                    inside_fov,
                ) = image_rays.pixel_ids_of_lixels_in_object_distance(depth)

                # populate image:
                for channel_id in loph_record["photons"]["channels"]:
                    if inside_fov[channel_id]:
                        pixel_id = pixel_indicies[channel_id]
                        image_stack[dek, pixel_id] += 1

            # plot images
            # -----------
            for dek in range(number_depths):
                print(sk, pk, airshower_id, dek, counter)
                figpath = os.path.join(
                    pk_dir,
                    "{:s}_{:012d}_{:03d}.jpg".format(pk, airshower_id, dek),
                )
                if os.path.exists(figpath):
                    continue

                depth = depths[dek]

                fig = seb.figure(
                    style={"rows": 360, "cols": 640, "fontsize": 0.4}
                )
                ax = seb.add_axes(fig=fig, span=[0.3, 0.13, 0.6, 0.85])

                pl.plot.image.add2ax(
                    ax=ax,
                    I=image_stack[dek, :],
                    px=np.rad2deg(lfg.pixel_pos_cx),
                    py=np.rad2deg(lfg.pixel_pos_cy),
                    colormap=colormap,
                    hexrotation=30,
                    vmin=0,
                    vmax=np.max(image_stack),
                    colorbar=True,
                )
                ax.set_xlabel(r"$c_x\,/\,1^{\circ}$")
                ax.set_ylabel(r"$c_y\,/\,1^{\circ}$")

                axr = seb.add_axes(fig=fig, span=[0.15, 0.13, 0.1, 0.85])
                pl.plot.ruler.add2ax_object_distance_ruler(
                    ax=axr,
                    object_distance=depth,
                    object_distance_min=min(depths) * 0.9,
                    object_distance_max=max(depths) * 1.1,
                    label=r"depth$\,/\,$km",
                    print_value=False,
                    color="black",
                )

                fig.savefig(figpath)
                seb.close(fig)
