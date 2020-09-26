#!/usr/bin/python
import sys
import numpy as np
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

os.makedirs(pa["out_dir"], exist_ok=True)

_passed_trigger_indices = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0066_passing_trigger")
)

fig_16_by_9 = sum_config["plot"]["16_by_9"]

theta_square_bin_edges_deg2 = np.linspace(
    0,
    sum_config["point_spread_function"]["theta_square"]["max_angle_deg"] ** 2,
    sum_config["point_spread_function"]["theta_square"]["num_bins"] * 2,
)
psf_containment_factor = sum_config["point_spread_function"][
    "containment_factor"
]

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

for sk in reconstruction:
    for pk in reconstruction[sk]:

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
        all_reco = reconstruction[sk][pk]
        (
            true_cx,
            true_cy,
        ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
            primary=all_truth["primary"],
            plenoscope_pointing=irf_config["config"]["plenoscope_pointing"],
        )
        true_x = -all_truth["core"]["core_x_m"]
        true_y = -all_truth["core"]["core_y_m"]


        delta_c_deg = np.rad2deg(
            np.hypot(all_reco["cx"] - true_cx, all_reco["cy"] - true_cy)
        )

        delta_hist = np.histogram(
            delta_c_deg ** 2,
            bins=theta_square_bin_edges_deg2
        )[0]


        theta_square_deg2 = irf.analysis.gamma_direction.integration_width_for_containment(
            bin_counts=delta_hist,
            bin_edges=theta_square_bin_edges_deg2,
            containment=psf_containment_factor,
        )

        fig = irf.summary.figure.figure(fig_16_by_9)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        irf.summary.figure.ax_add_hist(
            ax=ax,
            bin_edges=theta_square_bin_edges_deg2,
            bincounts=delta_hist,
            linestyle="-",
            linecolor="k",
            bincounts_upper=None,
            bincounts_lower=None,
            face_color="k",
            face_alpha=0.3,
        )
        ax.semilogy()
        ax.axvline(x=theta_square_deg2, color="k", linestyle="--", alpha=0.75)
        ax.set_xlabel("delta c**2 / deg**2")
        ax.set_ylabel("intensity / 1")
        ax.grid(color="k", linestyle="-", linewidth=0.66, alpha=0.1)
        ax.spines["top"].set_color("none")
        ax.spines["right"].set_color("none")
        fig.savefig(
            os.path.join(pa["out_dir"], sk + "_" + pk + "_benchmark" + ".jpg")
        )
        plt.close(fig)
