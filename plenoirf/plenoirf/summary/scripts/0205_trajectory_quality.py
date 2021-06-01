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
    os.path.join(pa["summary_dir"], "0056_passing_basic_quality")
)
weights_thrown2expected = irf.json_numpy.read_tree(
    os.path.join(
        pa["summary_dir"],
        "0040_weights_from_thrown_to_expected_energy_spectrum",
    )
)
min_trajectory_quality = sum_config["quality"]["min_trajectory_quality"]

theta_bin_edges_deg = np.linspace(0.0, 3.0, 15)

PARTICLES = irf_config["config"]["particles"]
SITES = irf_config["config"]["sites"]

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
        "trace": [
            [0, 0.25],
            [50, 0.8],
            [175, 1.0],
            [200, 0.8],
            [350, 0.25],
            [640, 0.0],
        ],
        "weight": 1.0,
    },
    "features/num_photons": {
        "scale": "log10",
        "trace": [[1, 0.0], [4, 1.0],],
        "weight": 0.0,
    },
    "features/image_half_depth_shift_c": {
        "scale": "linear",
        "trace": [[0.0, 0.0], [1.5e-3, 1.0],],
        "weight": 0.0,
    },
    "features/image_smallest_ellipse_solid_angle": {
        "scale": "log10",
        "trace": [[-7, 0.0], [-5, 1.0],],
        "weight": 0.0,
    },
}


def estimate_trajectory_quality(event_frame, quality_features):
    weight_sum = 0.0
    quality = np.zeros(event_frame["idx"].shape[0])
    for qf_key in quality_features:
        weight_sum += quality_features[qf_key]["weight"]

    for qf_key in quality_features:
        qf = quality_features[qf_key]

        if qf["scale"] == "linear":
            w = event_frame[qf_key]
        elif qf["scale"] == "log10":
            w = np.log10(event_frame[qf_key])
        else:
            assert False, "Scaling unknown"

        trace = np.array(qf["trace"])
        q_comp = np.interp(x=w, xp=trace[:, 0], fp=trace[:, 1])
        q_comp *= qf["weight"] / weight_sum
        quality += q_comp
    return quality


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
    x_cut=None,
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

    if x_cut is not None:
        for aa in [ax, ax_h]:
            aa.plot([x_cut, x_cut], aa.get_ylim(), "k:")

    if logx:
        ax.semilogx()
        ax_h.semilogx()

    if logy:
        ax.semilogy()

    if log_exposure_counter:
        ax_h.semilogy()

    fig.savefig(path)
    seb.close_figure(fig)


the = "theta"

QP = {}
QP["quality_cuts"] = np.linspace(0.0, 1.0, 137)
QP["fraction_passing"] = {}
QP["fraction_passing_w"] = {}
for sk in SITES:
    QP["fraction_passing"][sk] = {}
    QP["fraction_passing_w"][sk] = {}


for sk in SITES:
    for pk in PARTICLES:
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

        # estimate_quality
        # ================

        quality = estimate_trajectory_quality(
            event_frame=rectab, quality_features=quality_features
        )

        irf.json_numpy.write(
            os.path.join(site_particle_dir, "trajectory_quality.json"),
            {
                "comment": (
                    "Quality of reconstructed trajectory. "
                    "0 is worst, 1 is best."
                ),
                spt.IDX: rectab[spt.IDX],
                "unit": "1",
                "quality": quality,
            },
        )

        mask_passed_trajectory_quality = quality >= min_trajectory_quality
        idx_passed_trajectory_quality = rectab[spt.IDX][
            mask_passed_trajectory_quality
        ]

        irf.json_numpy.write(
            path=os.path.join(
                site_particle_dir, "passed_trajectory_quality.json"
            ),
            out_dict={spt.IDX: idx_passed_trajectory_quality},
        )

        write_correlation_figure(
            path=os.path.join(
                pa["out_dir"],
                "{:s}_{:s}_{:s}_vs_quality.jpg".format(sk, pk, the),
            ),
            x=quality,
            y=np.rad2deg(rectab["trajectory/" + the + "_rad"]),
            x_bin_edges=np.linspace(0, 1, 15),
            y_bin_edges=theta_bin_edges_deg,
            x_label="quality / 1",
            y_label=the + r" / $1^{\circ}$",
            min_exposure_x=100,
            logx=False,
            logy=False,
            log_exposure_counter=False,
            x_cut=min_trajectory_quality,
        )

        if pk == "gamma":
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
                x_cut=min_trajectory_quality,
            )

        if pk == "gamma":
            for fk in feature_correlations:

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
                    y_bin_edges=theta_bin_edges_deg,
                    x_label=fk["label"],
                    y_label=the + r" / $1^{\circ}$",
                    min_exposure_x=100,
                    logx=fk["log"],
                    logy=False,
                    log_exposure_counter=False,
                )

        # plot losses
        # ===========

        reweight_spectrum = np.interp(
            x=rectab["primary/energy_GeV"],
            xp=weights_thrown2expected[sk][pk]["weights_vs_energy"][
                "energy_GeV"
            ],
            fp=weights_thrown2expected[sk][pk]["weights_vs_energy"]["mean"],
        )

        fraction_passing = []
        fraction_passing_w = []
        for quality_cut in QP["quality_cuts"]:
            mask = quality >= quality_cut
            num_passing_cut = np.sum(mask)
            num_total = quality.shape[0]
            fraction_passing.append(num_passing_cut / num_total)

            num_passing_cut_w = np.sum(reweight_spectrum[mask])
            num_total_w = np.sum(reweight_spectrum)
            fraction_passing_w.append(num_passing_cut_w / num_total_w)

        QP["fraction_passing"][sk][pk] = np.array(fraction_passing)
        QP["fraction_passing_w"][sk][pk] = np.array(fraction_passing_w)


for sk in SITES:
    fig = seb.figure(seb.FIGURE_1_1)
    ax = seb.add_axes(fig=fig, span=[0.16, 0.11, 0.8, 0.8])
    for pk in PARTICLES:
        ax.plot(
            QP["quality_cuts"],
            QP["fraction_passing_w"][sk][pk],
            color=sum_config["plot"]["particle_colors"][pk],
        )
    ax.plot([min_trajectory_quality, min_trajectory_quality], [0.0, 1.0], "k:")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("trajectory-quality-cut / 1")
    ax.set_ylabel("passing cut / 1")
    fig.savefig(os.path.join(pa["out_dir"], "{:s}_passing.jpg".format(sk)))
    seb.close_figure(fig)
