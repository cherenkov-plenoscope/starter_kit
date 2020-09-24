#!/usr/bin/python
import sys
import plenoirf as irf
import sparse_numeric_table as spt
import airshower_template_generator as atg
import os
import pandas
import plenopy as pl
from iminuit import Minuit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

# READ reconstruction
# ===================
_rec = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0206_fitting_light_field")
)
reconstruction = {}
for sk in _rec:
    reconstruction[sk] = {}
    for pk in _rec[sk]:
        _df = pandas.DataFrame(_rec[sk][pk]["reco"])
        reconstruction[sk][pk] = _df.to_records(index=False)
reco_by_index = {}
for sk in _rec:
    reco_by_index[sk] = {}
    for pk in _rec[sk]:
        reco_by_index[sk][pk] = {}
        _rr = _rec[sk][pk]["reco"]
        for ii in range(len(_rr[spt.IDX])):
            airshower_id = _rr[spt.IDX][ii]
            reco_by_index[sk][pk][airshower_id] = {
                "cx": _rr["cx"][ii],
                "cy": _rr["cy"][ii],
                "x": _rr["x"][ii],
                "y": _rr["y"][ii],
            }


# READ light-field-geometry
# =========================
lfg = pl.LightFieldGeometry(
    os.path.join(pa["run_dir"], "light_field_geometry")
)
image_rays = pl.image.ImageRays(lfg)


fov_radius_deg = (
    0.5 * irf_config["light_field_sensor_geometry"]["max_FoV_diameter_deg"]
)

c_bin_edges_deg = np.linspace(-fov_radius_deg, fov_radius_deg, 83)
object_distances = np.geomspace(5e3, 35e3, 3)

fig_16_by_9 = sum_config["plot"]["16_by_9"]

# plot
# ====
def write_refocus_images(
    path,
    image_rays,
    lixel_ids,
    true_cx,
    true_cy,
    true_x,
    true_y,
    reco_cx,
    reco_cy,
    reco_x,
    reco_y,
    c_bin_edges_deg,
    fov_radius_deg,
    object_distances=np.geomspace(5e3, 35e3, 3),
    fig_16_by_9=fig_16_by_9,
):
    imgs = []
    for obji, objd in enumerate(object_distances):

        refocus_cx, refocus_cy = image_rays.cx_cy_in_object_distance(objd)
        lfcx = refocus_cx[lixel_ids]
        lfcy = refocus_cy[lixel_ids]
        img = np.histogram2d(
            np.rad2deg(lfcx),
            np.rad2deg(lfcy),
            bins=[c_bin_edges_deg, c_bin_edges_deg],
        )[0].T
        imgs.append(img)

    imgs = np.array(imgs)
    vmax = np.max(imgs)

    for obji, objd in enumerate(object_distances):
        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        ax.pcolor(
            c_bin_edges_deg,
            c_bin_edges_deg,
            imgs[obji],
            cmap="Blues",
            vmax=vmax,
        )
        ax.plot(np.rad2deg(true_cx), np.rad2deg(true_cy), "or", alpha=0.5)
        ax.plot(np.rad2deg(reco_cx), np.rad2deg(reco_cy), "og", alpha=0.5)

        line_x = np.array([true_cx, true_x])
        line_y = np.array([true_cy, true_y])
        ax.plot(np.rad2deg(line_x), np.rad2deg(line_y), "-r", alpha=0.15)

        line_x = np.array([reco_cx, reco_x])
        line_y = np.array([reco_cy, reco_y])
        ax.plot(np.rad2deg(line_x), np.rad2deg(line_y), "-g", alpha=0.15)

        irf.summary.figure.ax_add_circle(
            ax=ax, x=0.0, y=0.0, r=fov_radius_deg, color="k"
        )
        ax.set_title(
            "      cx/deg    cy/deg    x/m   y/m\n"
            "true:   {: 3.1f}   {: 3.1f}   {: 6.1f}   {: 6.1f}\n".format(
                np.rad2deg(true_cx), np.rad2deg(true_cy), true_x, true_y
            )
            + "reco:   {: 3.1f}   {: 3.1f}   {: 6.1f}   {: 6.1f}".format(
                np.rad2deg(reco_cx), np.rad2deg(reco_cy), reco_x, reco_y
            ),
            family="monospace",
            fontsize=9,
        )
        fig.text(
            x=0.05,
            y=0.05,
            s="focus {:.1f}km".format(1e-3 * objd),
            family="monospace",
            fontsize=9,
        )
        ax.set_aspect("equal")
        ax.set_xlim([-1.01*fov_radius_deg, 1.01*fov_radius_deg])
        ax.set_ylim([-1.01*fov_radius_deg, 1.01*fov_radius_deg])
        ax.set_xlabel("cx / deg")
        ax.set_ylabel("cy / deg")
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["left"].set_color("none")
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        fig.savefig(path.format(refocus=obji))
        plt.close(fig)


truth_by_index = {}
for sk in reconstruction:
    truth_by_index[sk] = {}
    for pk in reconstruction[sk]:
        truth_by_index[sk][pk] = {}

        event_table = spt.read(
            path=os.path.join(
                pa["run_dir"], "event_table", sk, pk, "event_table.tar"
            ),
            structure=irf.table.STRUCTURE,
        )
        all_truth = spt.cut_table_on_indices(
            event_table,
            irf.table.STRUCTURE,
            common_indices=reconstruction[sk][pk][spt.IDX],
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
        all_truth = spt.sort_table_on_common_indices(
            table=all_truth, common_indices=reconstruction[sk][pk][spt.IDX]
        )
        (
            true_cx,
            true_cy,
        ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
            primary=all_truth["primary"],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )

        for ii in range(all_truth["primary"].shape[0]):
            airshower_id = all_truth["primary"][spt.IDX][ii]
            truth_by_index[sk][pk][airshower_id] = {
                "cx": true_cx[ii],
                "cy": true_cy[ii],
                "x": -all_truth["core"]["core_x_m"][ii],
                "y": -all_truth["core"]["core_y_m"][ii],
            }


for sk in reconstruction:
    for pk in reconstruction[sk]:
        run = pl.photon_stream.loph.LopfTarReader(
            os.path.join(
                pa["run_dir"], "event_table", sk, pk, "cherenkov.phs.loph.tar",
            )
        )

        site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
        os.makedirs(site_particle_dir, exist_ok=True)

        reco = []
        for event in run:
            airshower_id, phs = event
            lixel_ids = phs["photons"]["channels"]

            if airshower_id not in reco_by_index[sk][pk]:
                continue

            write_refocus_images(
                path=os.path.join(
                    pa["out_dir"],
                    sk,
                    pk,
                    "{:09d}".format(airshower_id) + "_{refocus:03d}.jpg",
                ),
                image_rays=image_rays,
                lixel_ids=lixel_ids,
                true_cx=truth_by_index[sk][pk][airshower_id]["cx"],
                true_cy=truth_by_index[sk][pk][airshower_id]["cy"],
                true_x=truth_by_index[sk][pk][airshower_id]["x"],
                true_y=truth_by_index[sk][pk][airshower_id]["y"],
                reco_cx=reco_by_index[sk][pk][airshower_id]["cx"],
                reco_cy=reco_by_index[sk][pk][airshower_id]["cy"],
                reco_x=reco_by_index[sk][pk][airshower_id]["x"],
                reco_y=reco_by_index[sk][pk][airshower_id]["y"],
                c_bin_edges_deg=c_bin_edges_deg,
                fov_radius_deg=fov_radius_deg,
                object_distances=object_distances,
            )
