#!/usr/bin/python
import sys
import numpy as np
import plenoirf as irf
import os
import pandas
import sparse_numeric_table as spt

argv = irf.summary.argv_since_py(sys.argv)
pa = irf.summary.paths_from_argv(argv)

irf_config = irf.summary.read_instrument_response_config(run_dir=pa["run_dir"])
sum_config = irf.summary.read_summary_config(summary_dir=pa["summary_dir"])
run_key = os.path.dirname(pa["run_dir"])

os.makedirs(pa["out_dir"], exist_ok=True)

# Read reconstructions
_rec_fitlut = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0206_fitting_light_field")
)
_rec_params = irf.json_numpy.read_tree(
    os.path.join(pa["summary_dir"], "0200_gamma_point_spread_function")
)

def recordify_reconstructions(reco):
    reconstruction = {}
    for sk in reco:
        reconstruction[sk] = {}
        for pk in reco[sk]:
            _df = pandas.DataFrame(reco[sk][pk]["reco"])
            reconstruction[sk][pk] = _df.to_dict(orient="list")
    return reconstruction


def add_suffix_to_keys_in_dict(dic, suffix, except_keys=[spt.IDX]):
    except_keys = set(except_keys)
    out = {}
    for key in dic:
        if key in except_keys:
            out[key] = dic[key]
        else:
            out[key + suffix] = dic[key]
    return out

SIGN_PROBABLY_WRONG_IN_GRID_ALGO = -1.0

rec_fitlut = recordify_reconstructions(_rec_fitlut)
rec_params = recordify_reconstructions(_rec_params)

for sk in rec_fitlut:
    pk = "gamma"

    site_particle_dir = os.path.join(pa["out_dir"], sk, pk)
    os.makedirs(site_particle_dir, exist_ok=True)

    idx_both_reconstructions = spt.intersection(
        [rec_fitlut[sk][pk][spt.IDX], rec_params[sk][pk][spt.IDX]]
    )

    event_table = spt.read(
        path=os.path.join(
            pa["run_dir"], "event_table", sk, pk, "event_table.tar"
        ),
        structure=irf.table.STRUCTURE,
    )
    event_table = spt.cut_table_on_indices(
        table=event_table,
        structure=irf.table.STRUCTURE,
        common_indices=idx_both_reconstructions,
        level_keys=None,
    )
    event_table = spt.sort_table_on_common_indices(
        table=event_table, common_indices=idx_both_reconstructions
    )
    event_df = spt.make_rectangular_DataFrame(table=event_table)

    (
        true_cx,
        true_cy,
    ) = irf.analysis.gamma_direction.momentum_to_cx_cy_wrt_aperture(
        momentum_x_GeV_per_c=event_df["primary.momentum_x_GeV_per_c"],
        momentum_y_GeV_per_c=event_df["primary.momentum_y_GeV_per_c"],
        momentum_z_GeV_per_c=event_df["primary.momentum_z_GeV_per_c"],
        plenoscope_pointing=irf_config["config"][
            "plenoscope_pointing"
        ],
    )

    true_cxyy = {}
    true_cxyy[spt.IDX] = event_df[spt.IDX]
    true_cxyy["image_primary_cx_rad"] = true_cx
    true_cxyy["image_primary_cy_rad"] = true_cy

    pk_sk_rec_fitlut = add_suffix_to_keys_in_dict(
        dic=rec_fitlut[sk][pk],
        suffix="_reconstructed_probability_fit",
    )

    pk_sk_rec_params = add_suffix_to_keys_in_dict(
        dic=rec_params[sk][pk],
        suffix="_reconstructed_bright_spot",
    )

    event_df = pandas.merge(
        left=event_df, right=pandas.DataFrame(pk_sk_rec_fitlut), on=spt.IDX
    )
    event_df = pandas.merge(
        left=event_df, right=pandas.DataFrame(pk_sk_rec_params), on=spt.IDX
    )
    event_df = pandas.merge(
        left=event_df, right=pandas.DataFrame(true_cxyy), on=spt.IDX
    )

    minimal_df = pandas.DataFrame(
        {
            spt.IDX: event_df[spt.IDX],
            "true_cx_rad": event_df["image_primary_cx_rad"],
            "true_cy_rad": event_df["image_primary_cy_rad"],
            "true_impact_x_m": SIGN_PROBABLY_WRONG_IN_GRID_ALGO * event_df["core.core_x_m"],
            "true_impact_y_m": SIGN_PROBABLY_WRONG_IN_GRID_ALGO * event_df["core.core_y_m"],
            "true_energy_GeV": event_df["primary.energy_GeV"],
            "true_maximum_asl_m": event_df["cherenkovpool.maximum_asl_m"],

            "reco_cherenkov_pe": event_df["features.num_photons"],

            "reco_fit_cx_rad": event_df["cx_reconstructed_probability_fit"],
            "reco_fit_cy_rad": event_df["cy_reconstructed_probability_fit"],
            "reco_fit_impact_x_m": event_df["x_reconstructed_probability_fit"],
            "reco_fit_impact_y_m": event_df["y_reconstructed_probability_fit"],

            "reco_par_cx_rad": event_df["cx_reconstructed_bright_spot"],
            "reco_par_cy_rad": event_df["cy_reconstructed_bright_spot"],
            "reco_par_impact_x_m": event_df["x_reconstructed_bright_spot"],
            "reco_par_impact_y_m": event_df["y_reconstructed_bright_spot"],
        }
    )

    minimal_df.to_csv(
        os.path.join(
            pa["out_dir"], sk, pk, run_key + "_"+ sk + "_" + pk + ".csv"
        ),
        index=False,
        float_format="%.6f",
        na_rep="nan",
    )