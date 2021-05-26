#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import sparse_numeric_table as spt
import os
import sebastians_matplotlib_addons as seb

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

passing_trigger = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0055_passing_trigger")
)
passing_quality = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0056_passing_quality")
)

# feature correlation
# ===================
feature_correlations = [
    {
        "key": "reconstructed_trajectory/r_m",
        "label": "reco. core-radius / m",
        "bin_edges": np.linspace(0.0, 640, 17),
        "log": False,
    },
    {
        "key": "features/image_smallest_ellipse_object_distance",
        "label": "object-distance / m",
        "bin_edges": np.geomspace(5e3, 50e3, 17),
        "log": True,
    },
    {
        "key": "features/image_smallest_ellipse_solid_angle",
        "label": "smallest ellipse solid angle / sr",
        "bin_edges": np.geomspace(1e-7, 1e-3, 17),
        "log": True,
    },
    {
        "key": "features/num_photons",
        "label": "reco. num. photons / p.e.",
        "bin_edges": np.geomspace(1e1, 1e5, 17),
        "log": True,
    },
    {
        "key": "features/image_num_islands",
        "label": "num. islands / 1",
        "bin_edges": np.arange(1, 7),
        "log": False,
    },
    {
        "key": "features/image_half_depth_shift_c",
        "label": "image_half_depth_shift / rad",
        "bin_edges": np.deg2rad(np.linspace(0.0, 0.2, 17)),
        "log": False,
    },
]

quality_features = {
    "reconstructed_trajectory/r_m": {
        "scale": "linear",
        "trace": [[0, 0.25], [80, 1], [160, 1], [320, 0.25], [640, 0.0]],
        "weight": 1.0,
    },
    "features/num_photons": {
        "scale": "log10",
        "trace": [[1, 0.0], [4, 1.0],],
        "weight": 0.5,
    },
    "features/image_half_depth_shift_c": {
        "scale": "linear",
        "trace": [[0.0, 0.0], [1.5e-3, 1.0],],
        "weight": 1.0,
    },
    "features/image_smallest_ellipse_solid_angle": {
        "scale": "log10",
        "trace": [[-7, 0.0], [-5, 1.0],],
        "weight": 0.5,
    },
}


def write_correlation_figure(
    path,
    x,
    y,
    x_bin_edges,
    y_bin_edges,
    x_label,
    y_label,
    min_exposure_x,
    logx=False,
    logy=False,
    log_exposure_counter=False,
):
    valid = np.logical_and(
        np.logical_not((np.isnan(x))), np.logical_not((np.isnan(y)))
    )

    cm = irf.summary.figure.histogram_confusion_matrix_with_normalized_columns(
        x=x[valid],
        y=y[valid],
        x_bin_edges=x_bin_edges,
        y_bin_edges=y_bin_edges,
        min_exposure_x=min_exposure_x,
        default_low_exposure=0.0,
    )

    fig = seb.figure(seb.FIGURE_1_1)
    ax = seb.add_axes(fig=fig, span=[0.25, 0.27, 0.55, 0.65])
    ax_h = seb.add_axes(fig=fig, span=[0.25, 0.11, 0.55, 0.1])
    ax_cb = seb.add_axes(fig=fig, span=[0.85, 0.27, 0.02, 0.65])
    _pcm_confusion = ax.pcolormesh(
        cm["x_bin_edges"],
        cm["y_bin_edges"],
        np.transpose(cm["confusion_bins_normalized_columns"]),
        cmap="Greys",
        norm=seb.plt_colors.PowerNorm(gamma=0.5),
    )
    seb.plt.colorbar(_pcm_confusion, cax=ax_cb, extend="max")
    ax.set_title("normalized for each column")
    ax.set_ylabel(y_label)
    ax.set_xticklabels([])
    ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
    ax_h.set_xlim([np.min(cm["x_bin_edges"]), np.max(cm["x_bin_edges"])])
    ax_h.set_xlabel(x_label)
    ax_h.set_ylabel("num. events / 1")
    ax_h.axhline(cm["min_exposure_x"], linestyle=":", color="k")
    seb.ax_add_histogram(
        ax=ax_h,
        bin_edges=cm["x_bin_edges"],
        bincounts=cm["exposure_bins_x_no_weights"],
        linestyle="-",
        linecolor="k",
    )

    if logx:
        ax.semilogx()
        ax_h.semilogx()

    if logy:
        ax.semilogy()

    if log_exposure_counter:
        ax_h.semilogy()

    fig.savefig(path)
    seb.close_figure(fig)


for sk in irf_config["config"]["sites"]:
    pk = "gamma"
    the = "theta"

    site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
    os.makedirs(site_particle_dir, exist_ok=True)

    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"], "event_table", sk, pk, "event_table.tar"
        ),
        structure=irf.table.STRUCTURE,
    )
    idx_common = spt.intersection(
        [
            passing_trigger[sk][pk]["passed_trigger"]["idx"],
            passing_quality[sk][pk]["passed_quality"]["idx"],
        ]
    )
    event_table = spt.cut_and_sort_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=idx_common,
    )

    rectab = irf.reconstruction.trajectory_quality.make_rectangular_table(
        event_table=event_table,
        plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
    )

    for fk in feature_correlations:
        """
        write_correlation_figure(
            path=os.path.join(
                pa["out_dir"],
                "{:s}_{:s}_{:s}_vs_{:s}.jpg".format(
                    sk, pk, the, str.replace(fk["key"], "/", "-")
                ),
            ),
            x=rectab[fk["key"]],
            y=np.rad2deg(rectab["trajectory/" + the + "_rad"]),
            x_bin_edges=fk["bin_edges"],
            y_bin_edges=np.linspace(0.0, 3.0, 15),
            x_label=fk["label"],
            y_label=the + r" / $1^{\circ}$",
            min_exposure_x=100,
            logx=fk["log"],
            logy=False,
            log_exposure_counter=False,
        )
        """

        # estimate_quality
        # ================

    weight_sum = 0.0
    quality = np.zeros(rectab["idx"].shape[0])
    for qf_key in quality_features:
        weight_sum += quality_features[qf_key]["weight"]

    for qf_key in quality_features:
        qf = quality_features[qf_key]

        if qf["scale"] == "linear":
            w = rectab[qf_key]
        elif qf["scale"] == "log10":
            w = np.log10(rectab[qf_key])
        else:
            assert False, "Scaling unknown"

        trace = np.array(qf["trace"])
        q_comp = np.interp(x=w, xp=trace[:, 0], fp=trace[:, 1])
        q_comp *= qf["weight"] / weight_sum
        quality += q_comp

    write_correlation_figure(
        path=os.path.join(
            pa["out_dir"], "{:s}_{:s}_{:s}_vs_quality.jpg".format(sk, pk, the),
        ),
        x=quality,
        y=np.rad2deg(rectab["trajectory/" + the + "_rad"]),
        x_bin_edges=np.linspace(0, 1, 15),
        y_bin_edges=np.linspace(0.0, 3.0, 15),
        x_label="quality / 1",
        y_label=the + r" / $1^{\circ}$",
        min_exposure_x=100,
        logx=False,
        logy=False,
        log_exposure_counter=False,
    )

    write_correlation_figure(
        path=os.path.join(
            pa["out_dir"],
            "{:s}_{:s}_energy_vs_quality.jpg".format(sk, pk, the),
        ),
        x=quality,
        y=rectab["primary/energy_GeV"],
        x_bin_edges=np.linspace(0, 1, 15),
        y_bin_edges=np.geomspace(1, 1000, 15),
        x_label="quality / 1",
        y_label="energy / GeV",
        min_exposure_x=100,
        logx=False,
        logy=True,
        log_exposure_counter=False,
    )
